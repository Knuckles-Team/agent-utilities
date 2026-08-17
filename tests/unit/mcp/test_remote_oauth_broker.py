"""Attack-rejection + lifecycle tests for the GOC-85 remote browser-OAuth broker.

CONCEPT:AU-ECO.mcp.remote-oauth-broker

Every fail-closed control in ``agent_utilities/mcp/remote_oauth_broker.py`` is proven
here against a KNOWN-BAD input, per the lane's own standing rule ("a security control
never demonstrated against a known-bad input is not evidence"): mismatched state,
replayed code/state, wrong redirect/endpoint, cross-user token access, non-allowlisted
resource, downgraded TLS, and scope widening.

No real network is used anywhere in this file — discovery and token-exchange calls are
served over ``httpx.MockTransport``, and secret storage uses an in-memory fake backend
implementing the exact ``SecretsBackend`` contract (including an atomic
``compare_and_set``) so the single-use-state and refresh-rotation races are meaningful
without requiring a live engine.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import httpx
import pytest

from agent_utilities.mcp.remote_oauth_broker import (
    OAuthBindingError,
    OAuthDiscoveryError,
    OAuthProviderError,
    OAuthRefreshRaceError,
    OAuthRevokedError,
    OAuthScopeError,
    OAuthStateError,
    OAuthTokenAbsentError,
    OAuthTokenStore,
    OAuthTransaction,
    ProviderDescriptor,
    ProviderRegistry,
    RemoteOAuthBroker,
    StoredToken,
    TransactionStore,
    discover_authorization_server,
    discover_protected_resource,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext
from agent_utilities.security.secrets_client import SecretsBackend, SecretsClient

RESOURCE_URL = "https://mcp.example.com/remote"
ISSUER = "https://idp.example.com"
AUTH_ENDPOINT = "https://idp.example.com/authorize"
TOKEN_ENDPOINT = "https://idp.example.com/token"
REDIRECT_URI = "https://broker.internal.example/oauth/callback"


# ---------------------------------------------------------------------------
# In-memory fake secrets backend — mirrors SecretsBackend's contract, with a
# genuinely atomic compare_and_set so the single-use/refresh races are real.
# ---------------------------------------------------------------------------
class FakeSecretsBackend(SecretsBackend):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: str, **metadata) -> None:
        with self._lock:
            self._store[key] = value

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def list_keys(self) -> list[str]:
        with self._lock:
            return sorted(self._store)

    def compare_and_set(self, key: str, expected: str, value: str, **metadata) -> bool:
        with self._lock:
            if self._store.get(key) != expected:
                return False
            self._store[key] = value
            return True


@pytest.fixture
def secrets_client() -> SecretsClient:
    return SecretsClient(backend=FakeSecretsBackend())


def verified_actor(actor_id: str = "user-a", tenant_id: str = "tenant-1") -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        tenant_id=tenant_id,
        authenticated=True,
        roles=("kg:read",),
    )


def unauthenticated_actor() -> ActorContext:
    return ActorContext(actor_id="anon", tenant_id="tenant-1", authenticated=False)


def enabled_provider(**overrides) -> ProviderDescriptor:
    fields = dict(
        provider_id="acme",
        resource_url=RESOURCE_URL,
        client_id="broker-public-client",
        redirect_uri=REDIRECT_URI,
        scopes=("mcp:read", "mcp:write"),
        enabled=True,
    )
    fields.update(overrides)
    return ProviderDescriptor(**fields)


def registry_with(provider: ProviderDescriptor) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(provider)
    return registry


# ---------------------------------------------------------------------------
# Mock discovery + token transport
# ---------------------------------------------------------------------------
def _resource_metadata_body(authorization_servers=(ISSUER,)) -> bytes:
    return json.dumps(
        {"resource": RESOURCE_URL, "authorization_servers": list(authorization_servers)}
    ).encode()


def _as_metadata_body(
    *,
    issuer=ISSUER,
    challenge_methods=("S256",),
    grant_types=("authorization_code",),
) -> bytes:
    return json.dumps(
        {
            "issuer": issuer,
            "authorization_endpoint": AUTH_ENDPOINT,
            "token_endpoint": TOKEN_ENDPOINT,
            "code_challenge_methods_supported": list(challenge_methods),
            "grant_types_supported": list(grant_types),
        }
    ).encode()


def happy_path_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.startswith("/.well-known/oauth-protected-resource"):
        return httpx.Response(200, content=_resource_metadata_body())
    if path.startswith("/.well-known/oauth-authorization-server"):
        return httpx.Response(200, content=_as_metadata_body())
    if str(request.url) == TOKEN_ENDPOINT:
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "access_token": "at-issued-by-mock-idp",
                    "refresh_token": "rt-issued-by-mock-idp",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "mcp:read mcp:write",
                }
            ).encode(),
        )
    return httpx.Response(404)


def client_factory_for(handler) -> callable:
    def _factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)

    return _factory


# ---------------------------------------------------------------------------
# 1. Provider registration
# ---------------------------------------------------------------------------
class TestProviderRegistration:
    def test_unknown_provider_fails_closed(self):
        registry = ProviderRegistry()
        with pytest.raises(OAuthProviderError):
            registry.require_enabled("does-not-exist")

    def test_disabled_provider_fails_closed(self):
        registry = registry_with(enabled_provider(enabled=False))
        with pytest.raises(OAuthProviderError):
            registry.require_enabled("acme")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("resource_url", "http://mcp.example.com/remote"),
            ("redirect_uri", "http://broker.internal.example/oauth/callback"),
        ],
    )
    def test_non_https_provider_fields_rejected(self, field, value):
        with pytest.raises(ValueError):
            enabled_provider(**{field: value})


# ---------------------------------------------------------------------------
# 2. Discovery — adversarial responses
# ---------------------------------------------------------------------------
class TestDiscovery:
    def test_happy_path(self):
        client = httpx.Client(transport=httpx.MockTransport(happy_path_handler))
        resource = discover_protected_resource(client, RESOURCE_URL)
        as_meta = discover_authorization_server(client, resource)
        assert as_meta.issuer == ISSUER
        assert "S256" in as_meta.code_challenge_methods_supported
        client.close()

    def test_malformed_json_rejected(self):
        def handler(request):
            return httpx.Response(200, content=b"not json at all {{{")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(OAuthDiscoveryError):
            discover_protected_resource(client, RESOURCE_URL)
        client.close()

    def test_oversized_response_rejected(self):
        def handler(request):
            return httpx.Response(200, content=b"0" * (300 * 1024))

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(OAuthDiscoveryError):
            discover_protected_resource(client, RESOURCE_URL)
        client.close()

    def test_slow_unreachable_server_rejected(self):
        def handler(request):
            raise httpx.ReadTimeout("simulated slow authorization server")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(OAuthDiscoveryError):
            discover_protected_resource(client, RESOURCE_URL)
        client.close()

    def test_issuer_mismatch_rejected(self):
        def handler(request):
            if request.url.path.startswith("/.well-known/oauth-protected-resource"):
                return httpx.Response(200, content=_resource_metadata_body())
            return httpx.Response(
                200, content=_as_metadata_body(issuer="https://a-different-idp.example.com")
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        resource = discover_protected_resource(client, RESOURCE_URL)
        with pytest.raises(OAuthDiscoveryError, match="issuer"):
            discover_authorization_server(client, resource)
        client.close()

    def test_unsupported_pkce_rejected(self):
        def handler(request):
            if request.url.path.startswith("/.well-known/oauth-protected-resource"):
                return httpx.Response(200, content=_resource_metadata_body())
            return httpx.Response(200, content=_as_metadata_body(challenge_methods=("plain",)))

        client = httpx.Client(transport=httpx.MockTransport(handler))
        resource = discover_protected_resource(client, RESOURCE_URL)
        with pytest.raises(OAuthDiscoveryError, match="PKCE"):
            discover_authorization_server(client, resource)
        client.close()

    def test_missing_authorization_code_grant_rejected(self):
        def handler(request):
            if request.url.path.startswith("/.well-known/oauth-protected-resource"):
                return httpx.Response(200, content=_resource_metadata_body())
            return httpx.Response(200, content=_as_metadata_body(grant_types=("implicit",)))

        client = httpx.Client(transport=httpx.MockTransport(handler))
        resource = discover_protected_resource(client, RESOURCE_URL)
        with pytest.raises(OAuthDiscoveryError, match="authorization_code"):
            discover_authorization_server(client, resource)
        client.close()

    def test_non_https_discovery_endpoint_rejected(self):
        client = httpx.Client(transport=httpx.MockTransport(happy_path_handler))
        with pytest.raises(OAuthDiscoveryError):
            discover_protected_resource(client, "http://mcp.example.com/remote")
        client.close()

    def test_redirect_response_is_never_followed(self):
        def handler(request):
            return httpx.Response(302, headers={"Location": "https://attacker.example/steal"})

        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        with pytest.raises(OAuthDiscoveryError):
            discover_protected_resource(client, RESOURCE_URL)
        client.close()


# ---------------------------------------------------------------------------
# 4/5. Transaction store — expiry, single-use, replay
# ---------------------------------------------------------------------------
class TestTransactionStore:
    def test_expired_transaction_rejected(self, secrets_client):
        store = TransactionStore(secrets_client)
        tx = OAuthTransaction(
            state="s1", nonce="n1", verifier="v1", provider_id="acme",
            tenant_id="tenant-1", principal_id="user-a", browser_session_id="sess-1",
            redirect_uri=REDIRECT_URI, scope="mcp:read", created_at=0.0, expires_at=100.0,
        )
        store.put(tx)
        with pytest.raises(OAuthStateError, match="expired"):
            store.consume_once("s1", now=200.0)

    def test_state_replay_rejected(self, secrets_client):
        store = TransactionStore(secrets_client)
        tx = OAuthTransaction(
            state="s2", nonce="n1", verifier="v1", provider_id="acme",
            tenant_id="tenant-1", principal_id="user-a", browser_session_id="sess-1",
            redirect_uri=REDIRECT_URI, scope="mcp:read", created_at=0.0, expires_at=1e12,
        )
        store.put(tx)
        first = store.consume_once("s2")
        assert first.state == "s2"
        with pytest.raises(OAuthStateError, match="replay|already-consumed"):
            store.consume_once("s2")

    def test_unknown_state_rejected(self, secrets_client):
        store = TransactionStore(secrets_client)
        with pytest.raises(OAuthStateError):
            store.consume_once("never-issued")

    def test_concurrent_replay_only_one_winner(self, secrets_client):
        store = TransactionStore(secrets_client)
        tx = OAuthTransaction(
            state="s3", nonce="n1", verifier="v1", provider_id="acme",
            tenant_id="tenant-1", principal_id="user-a", browser_session_id="sess-1",
            redirect_uri=REDIRECT_URI, scope="mcp:read", created_at=0.0, expires_at=1e12,
        )
        store.put(tx)

        results: list[str] = []

        def worker():
            try:
                store.consume_once("s3")
                results.append("ok")
            except OAuthStateError:
                results.append("rejected")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count("ok") == 1
        assert results.count("rejected") == 7


# ---------------------------------------------------------------------------
# 6. Token store — cross-principal / cross-tenant isolation
# ---------------------------------------------------------------------------
class TestTokenStoreIsolation:
    def test_round_trip(self, secrets_client):
        store = OAuthTokenStore(secrets_client)
        actor = verified_actor()
        token = StoredToken(
            access_token="at-1", refresh_token="rt-1", token_type="Bearer",
            expires_at=time.time() + 3600, granted_scope="mcp:read", key_version=1,
            audience=RESOURCE_URL,
        )
        store.put(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL, token=token)
        fetched = store.get(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL)
        assert fetched is not None
        assert fetched.access_token == "at-1"

    def test_cross_principal_read_denied(self, secrets_client):
        store = OAuthTokenStore(secrets_client)
        owner = verified_actor(actor_id="user-a", tenant_id="tenant-1")
        attacker = verified_actor(actor_id="user-b", tenant_id="tenant-1")
        token = StoredToken(
            access_token="owner-secret-token", refresh_token=None, token_type="Bearer",
            expires_at=time.time() + 3600, granted_scope="mcp:read", key_version=1,
            audience=RESOURCE_URL,
        )
        store.put(actor=owner, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL, token=token)

        assert store.get(actor=owner, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL) is not None
        assert store.get(actor=attacker, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL) is None

    def test_cross_tenant_read_denied(self, secrets_client):
        store = OAuthTokenStore(secrets_client)
        owner = verified_actor(actor_id="user-a", tenant_id="tenant-1")
        other_tenant_same_id = verified_actor(actor_id="user-a", tenant_id="tenant-2")
        token = StoredToken(
            access_token="tenant-1-secret", refresh_token=None, token_type="Bearer",
            expires_at=time.time() + 3600, granted_scope="mcp:read", key_version=1,
            audience=RESOURCE_URL,
        )
        store.put(actor=owner, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL, token=token)
        assert store.get(actor=other_tenant_same_id, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL) is None

    def test_unauthenticated_actor_rejected(self, secrets_client):
        store = OAuthTokenStore(secrets_client)
        with pytest.raises(PermissionError):
            store.get(actor=unauthenticated_actor(), provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL)


# ---------------------------------------------------------------------------
# Refresh / revoke — per-token lock, atomic rotation, fail-closed revoke
# ---------------------------------------------------------------------------
class TestRefreshAndRevoke:
    def _stored(self, store, secrets_client, actor, refresh_token="rt-0"):
        token = StoredToken(
            access_token="at-0", refresh_token=refresh_token, token_type="Bearer",
            expires_at=time.time() + 10, granted_scope="mcp:read", key_version=1,
            audience=RESOURCE_URL,
        )
        store.put(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL, token=token)

    def test_refresh_absent_token_fails_closed(self, secrets_client):
        store = OAuthTokenStore(secrets_client)
        provider = enabled_provider()
        client = httpx.Client(transport=httpx.MockTransport(happy_path_handler))
        with pytest.raises(OAuthTokenAbsentError):
            store.refresh(actor=verified_actor(), provider=provider, resource_url=RESOURCE_URL, audience=RESOURCE_URL, http_client=client)
        client.close()

    def test_refresh_revoked_token_fails_closed(self, secrets_client):
        store = OAuthTokenStore(secrets_client)
        actor = verified_actor()
        self._stored(store, secrets_client, actor)
        store.revoke(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL)
        provider = enabled_provider()
        client = httpx.Client(transport=httpx.MockTransport(happy_path_handler))
        with pytest.raises(OAuthRevokedError):
            store.refresh(actor=actor, provider=provider, resource_url=RESOURCE_URL, audience=RESOURCE_URL, http_client=client)
        client.close()

    def test_revoke_then_get_reads_as_absent(self, secrets_client):
        store = OAuthTokenStore(secrets_client)
        actor = verified_actor()
        self._stored(store, secrets_client, actor)
        store.revoke(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL)
        assert store.get(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL) is None

    def test_revoke_of_never_stored_token_still_fails_closed_on_next_read(self, secrets_client):
        # Ambiguous state (nothing was ever there) must still read as "not usable"
        # after a revoke call — never silently succeed as a no-op that could later
        # look like "not revoked" if a token were later stored under a stale window.
        store = OAuthTokenStore(secrets_client)
        actor = verified_actor()
        store.revoke(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL)
        assert store.get(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL) is None

    def test_concurrent_refresh_serializes_without_corruption(self, secrets_client):
        store = OAuthTokenStore(secrets_client)
        actor = verified_actor()
        self._stored(store, secrets_client, actor)
        provider = enabled_provider()

        call_count = {"n": 0}
        lock = threading.Lock()

        def handler(request: httpx.Request) -> httpx.Response:
            with lock:
                call_count["n"] += 1
                n = call_count["n"]
            return httpx.Response(
                200,
                content=json.dumps(
                    {
                        "access_token": f"at-rotated-{n}",
                        "refresh_token": f"rt-rotated-{n}",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    }
                ).encode(),
            )

        results: list[StoredToken] = []
        errors: list[Exception] = []

        def worker():
            client = httpx.Client(transport=httpx.MockTransport(handler))
            try:
                token = store.refresh(
                    actor=actor, provider=provider, resource_url=RESOURCE_URL,
                    audience=RESOURCE_URL, http_client=client,
                )
                results.append(token)
            except Exception as exc:  # noqa: BLE001 - collected for assertion below
                errors.append(exc)
            finally:
                client.close()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every thread serialized through the per-token lock: each saw a
        # self-consistent read-modify-write, so every attempt either succeeded
        # cleanly or failed with the declared race error -- never a corrupted
        # or partially-written record.
        assert not errors or all(isinstance(e, OAuthRefreshRaceError) for e in errors)
        final = store.get(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL)
        assert final is not None
        assert final.access_token.startswith("at-rotated-")


# ---------------------------------------------------------------------------
# Broker end-to-end: begin() / callback()
# ---------------------------------------------------------------------------
class TestBrokerBeginAndCallback:
    def _broker(self, secrets_client, provider):
        registry = registry_with(provider)
        return RemoteOAuthBroker(
            registry=registry,
            secrets_client=secrets_client,
            http_client_factory=client_factory_for(happy_path_handler),
        )

    def test_begin_requires_authenticated_actor(self, secrets_client):
        broker = self._broker(secrets_client, enabled_provider())
        with pytest.raises(PermissionError):
            broker.begin(provider_id="acme", actor=unauthenticated_actor(), browser_session_id="sess-1")

    def test_begin_rejects_unknown_or_disabled_provider(self, secrets_client):
        broker = self._broker(secrets_client, enabled_provider(enabled=False))
        with pytest.raises(OAuthProviderError):
            broker.begin(provider_id="acme", actor=verified_actor(), browser_session_id="sess-1")

    def test_begin_issues_authorization_url_and_stops_there(self, secrets_client):
        broker = self._broker(secrets_client, enabled_provider())
        url = broker.begin(provider_id="acme", actor=verified_actor(), browser_session_id="sess-1")
        assert url.startswith(AUTH_ENDPOINT + "?")
        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        assert "state=" in url
        # No token exchange happened yet -- begin() never reaches the token endpoint.
        assert (
            OAuthTokenStore(secrets_client).get(
                actor=verified_actor(), provider_id="acme",
                resource_url=RESOURCE_URL, audience=RESOURCE_URL,
            )
            is None
        )

    def test_callback_happy_path_stores_isolated_token(self, secrets_client):
        provider = enabled_provider()
        broker = self._broker(secrets_client, provider)
        actor = verified_actor()
        url = broker.begin(provider_id="acme", actor=actor, browser_session_id="sess-1")
        state = url.split("state=")[1].split("&")[0]

        token = broker.callback(code="auth-code-abc", state=state, actor=actor, browser_session_id="sess-1")
        assert token.access_token == "at-issued-by-mock-idp"

        fetched = broker.tokens.get(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL)
        assert fetched is not None and fetched.access_token == token.access_token

    def test_callback_rejects_replayed_state(self, secrets_client):
        provider = enabled_provider()
        broker = self._broker(secrets_client, provider)
        actor = verified_actor()
        url = broker.begin(provider_id="acme", actor=actor, browser_session_id="sess-1")
        state = url.split("state=")[1].split("&")[0]

        broker.callback(code="auth-code-abc", state=state, actor=actor, browser_session_id="sess-1")
        with pytest.raises(OAuthStateError):
            broker.callback(code="auth-code-abc-replayed", state=state, actor=actor, browser_session_id="sess-1")

    def test_callback_rejects_cross_user_binding(self, secrets_client):
        provider = enabled_provider()
        broker = self._broker(secrets_client, provider)
        initiator = verified_actor(actor_id="user-a")
        attacker = verified_actor(actor_id="user-b")
        url = broker.begin(provider_id="acme", actor=initiator, browser_session_id="sess-1")
        state = url.split("state=")[1].split("&")[0]

        with pytest.raises(OAuthBindingError):
            broker.callback(code="stolen-code", state=state, actor=attacker, browser_session_id="sess-1")

    def test_callback_rejects_session_mismatch(self, secrets_client):
        provider = enabled_provider()
        broker = self._broker(secrets_client, provider)
        actor = verified_actor()
        url = broker.begin(provider_id="acme", actor=actor, browser_session_id="sess-1")
        state = url.split("state=")[1].split("&")[0]

        with pytest.raises(OAuthBindingError):
            broker.callback(code="auth-code-abc", state=state, actor=actor, browser_session_id="sess-DIFFERENT")

    def test_callback_rejects_widened_scope(self, secrets_client):
        def widening_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.startswith("/.well-known/oauth-protected-resource"):
                return httpx.Response(200, content=_resource_metadata_body())
            if request.url.path.startswith("/.well-known/oauth-authorization-server"):
                return httpx.Response(200, content=_as_metadata_body())
            if str(request.url) == TOKEN_ENDPOINT:
                return httpx.Response(
                    200,
                    content=json.dumps(
                        {
                            "access_token": "at-with-extra-scope",
                            "token_type": "Bearer",
                            "expires_in": 3600,
                            "scope": "mcp:read mcp:write mcp:admin",
                        }
                    ).encode(),
                )
            return httpx.Response(404)

        provider = enabled_provider(scopes=("mcp:read", "mcp:write"))
        registry = registry_with(provider)
        broker = RemoteOAuthBroker(
            registry=registry, secrets_client=secrets_client,
            http_client_factory=client_factory_for(widening_handler),
        )
        actor = verified_actor()
        url = broker.begin(provider_id="acme", actor=actor, browser_session_id="sess-1")
        state = url.split("state=")[1].split("&")[0]

        with pytest.raises(OAuthScopeError):
            broker.callback(code="auth-code-abc", state=state, actor=actor, browser_session_id="sess-1")
        # A rejected callback must never persist a token.
        assert broker.tokens.get(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL) is None

    def test_callback_unknown_state_rejected(self, secrets_client):
        broker = self._broker(secrets_client, enabled_provider())
        with pytest.raises(OAuthStateError):
            broker.callback(code="c", state="never-issued", actor=verified_actor(), browser_session_id="sess-1")


# ---------------------------------------------------------------------------
# Endpoint-bound forwarding
# ---------------------------------------------------------------------------
class TestBearerHeadersFor:
    def test_rejects_non_registered_resource(self, secrets_client):
        provider = enabled_provider()
        registry = registry_with(provider)
        broker = RemoteOAuthBroker(registry=registry, secrets_client=secrets_client)
        with pytest.raises(OAuthProviderError):
            broker.bearer_headers_for(
                actor=verified_actor(), provider_id="acme",
                resource_url="https://attacker.example/different-resource",
            )

    def test_missing_token_fails_closed(self, secrets_client):
        provider = enabled_provider()
        registry = registry_with(provider)
        broker = RemoteOAuthBroker(registry=registry, secrets_client=secrets_client)
        with pytest.raises(OAuthTokenAbsentError):
            broker.bearer_headers_for(actor=verified_actor(), provider_id="acme", resource_url=RESOURCE_URL)

    def test_happy_path_returns_bound_bearer(self, secrets_client):
        provider = enabled_provider()
        registry = registry_with(provider)
        broker = RemoteOAuthBroker(registry=registry, secrets_client=secrets_client)
        actor = verified_actor()
        token = StoredToken(
            access_token="at-bound", refresh_token=None, token_type="Bearer",
            expires_at=time.time() + 3600, granted_scope="mcp:read", key_version=1,
            audience=RESOURCE_URL,
        )
        broker.tokens.put(actor=actor, provider_id="acme", resource_url=RESOURCE_URL, audience=RESOURCE_URL, token=token)
        headers = broker.bearer_headers_for(actor=actor, provider_id="acme", resource_url=RESOURCE_URL)
        assert headers == {"Authorization": "Bearer at-bound"}


# ---------------------------------------------------------------------------
# Audit sanitization
# ---------------------------------------------------------------------------
class TestAuditSanitization:
    def test_begin_and_callback_never_log_sensitive_values(self, secrets_client, caplog):
        provider = enabled_provider()
        registry = registry_with(provider)
        broker = RemoteOAuthBroker(
            registry=registry, secrets_client=secrets_client,
            http_client_factory=client_factory_for(happy_path_handler),
        )
        actor = verified_actor()
        with caplog.at_level(logging.DEBUG):
            url = broker.begin(provider_id="acme", actor=actor, browser_session_id="sess-1")
            state = url.split("state=")[1].split("&")[0]
            token = broker.callback(code="super-secret-code", state=state, actor=actor, browser_session_id="sess-1")

        forbidden = {state, "super-secret-code", token.access_token}
        # The verifier is internal; recompute is impractical here, so instead assert
        # the broad shape: no record contains the raw access token, state, or code.
        for record in caplog.records:
            rendered = record.getMessage()
            for secret in forbidden:
                assert secret not in rendered, f"log record leaked a sensitive value: {rendered!r}"
