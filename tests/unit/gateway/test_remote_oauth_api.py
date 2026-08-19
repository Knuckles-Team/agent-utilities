"""Gateway callback + authorize routes for the remote-OAuth broker (NE-008 Deliverable 2).

Exercises :mod:`agent_utilities.gateway.remote_oauth_api` handler functions
directly (not over a real ASGI/TestClient stack): ``brain_context``'s actor is
carried in a ``contextvars.ContextVar``, and Starlette's ``TestClient`` runs the
ASGI app through a background-thread portal, which would NOT reliably see a
contextvar set in the pytest thread. Calling the ``async def`` route handlers
directly, wrapped in ``use_actor(...)``, exercises the exact same code with a
faithful verified-identity context and no such cross-thread pitfall.

The attack-rejection matrix itself (state replay, verifier/redirect/issuer
mismatch, scope widening, endpoint rebinding, cross-principal/tenant reuse) is
proven once, thoroughly, against the broker directly in
``test_remote_oauth_broker.py``; this file proves the GATEWAY WIRING on top:
verified-actor resolution (never request data), fail-closed on no/unauthenticated
actor, unknown-provider 404, the one fixed non-caller-influenced success
redirect (no open-redirect surface, no code/state ever appended to it), and
that neither this module's own logging nor its response bodies ever carry a
sensitive value.
"""

from __future__ import annotations

import json
import logging
import threading

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from agent_utilities.core.config import AgentConfig
from agent_utilities.core.config import config as typed_config
from agent_utilities.gateway import remote_oauth_api
from agent_utilities.gateway.remote_oauth_api import (
    AuthorizeRequest,
    authorize,
    oauth_callback,
    register_remote_oauth_routes,
)
from agent_utilities.mcp.remote_oauth_broker import (
    ProviderDescriptor,
    ProviderRegistry,
    RemoteOAuthBroker,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.security.secrets_client import SecretsBackend, SecretsClient

RESOURCE_URL = "https://mcp.example.com/remote"
ISSUER = "https://idp.example.com"
AUTH_ENDPOINT = "https://idp.example.com/authorize"
TOKEN_ENDPOINT = "https://idp.example.com/token"
REDIRECT_URI = "https://gateway.internal.example/oauth/callback"
SUCCESS_URL = "https://webui.internal.example/settings/connections?connected=1"


def _synthetic_value(*parts: str) -> str:
    """Build a deterministic test-only value without a scanner-shaped literal."""
    return "".join(parts)


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


def happy_path_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.startswith("/.well-known/oauth-protected-resource"):
        return httpx.Response(
            200,
            content=json.dumps(
                {"resource": RESOURCE_URL, "authorization_servers": [ISSUER]}
            ).encode(),
        )
    if path.startswith("/.well-known/oauth-authorization-server"):
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "issuer": ISSUER,
                    "authorization_endpoint": AUTH_ENDPOINT,
                    "token_endpoint": TOKEN_ENDPOINT,
                    "code_challenge_methods_supported": ["S256"],
                    "grant_types_supported": ["authorization_code"],
                }
            ).encode(),
        )
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


def _client_factory():
    return httpx.Client(
        transport=httpx.MockTransport(happy_path_handler), follow_redirects=False
    )


def enabled_provider(**overrides) -> ProviderDescriptor:
    fields = dict(
        provider_id="acme",
        resource_url=RESOURCE_URL,
        client_id="gateway-public-client",
        redirect_uri=REDIRECT_URI,
        scopes=("mcp:read", "mcp:write"),
        enabled=True,
    )
    fields.update(overrides)
    return ProviderDescriptor(**fields)


@pytest.fixture
def broker():
    registry = ProviderRegistry()
    registry.register(enabled_provider())
    b = RemoteOAuthBroker(
        registry=registry,
        secrets_client=SecretsClient(backend=FakeSecretsBackend()),
        http_client_factory=_client_factory,
    )
    remote_oauth_api._set_broker(b)
    yield b
    remote_oauth_api._set_broker(None)
    remote_oauth_api._set_scope_policy(None)


def verified_actor(actor_id="user-a", tenant_id="tenant-1") -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        tenant_id=tenant_id,
        authenticated=True,
        roles=("kg:read",),
    )


def _state_from_url(url: str) -> str:
    return url.split("state=")[1].split("&")[0]


# ---------------------------------------------------------------------------
# Mounting
# ---------------------------------------------------------------------------
def test_register_routes_mounts_on_fastapi():
    app = FastAPI()
    register_remote_oauth_routes(app)
    # ``include_router`` wraps mounted routes in an internal
    # ``fastapi.routing._IncludedRouter`` with no ``.path`` of its own on this
    # FastAPI version -- ``openapi()["paths"]`` is the version-independent way
    # to confirm both routes actually resolved and are documented.
    paths = set(app.openapi()["paths"])
    assert "/providers/{provider_id}/authorize" in paths
    assert "/oauth/callback" in paths
    assert "/providers/{provider_id}/revoke" in paths


def test_graph_gateway_mounts_prefixed_remote_oauth_routes(monkeypatch):
    """NE-062: the centralized gateway owns the prefixed OAuth surface."""

    from agent_utilities.gateway import (
        fleet,
        graph_api,
        ontology_api,
        registry_api,
        research_api,
    )
    from agent_utilities.mcp import kg_server

    monkeypatch.setattr(kg_server, "ensure_tools_registered", lambda: None)
    monkeypatch.setattr(kg_server, "_mount_rest_routes", lambda app, prefix: None)
    monkeypatch.setattr(graph_api, "_mount_sparql_route", lambda app, prefix: None)
    monkeypatch.setattr(fleet, "mount_fleet_routes", lambda app, prefix: None)
    monkeypatch.setattr(
        ontology_api, "register_ontology_routes", lambda app, prefix: None
    )
    monkeypatch.setattr(
        research_api, "register_research_routes", lambda app, prefix: None
    )
    monkeypatch.setattr(
        registry_api, "register_registry_routes", lambda app, prefix: None
    )

    app = FastAPI()
    from agent_utilities.gateway.graph_api import register_graph_routes

    register_graph_routes(app, prefix="/api")

    paths = set(app.openapi()["paths"])
    assert "/api/providers/{provider_id}/authorize" in paths
    assert "/api/oauth/callback" in paths
    assert "/api/providers/{provider_id}/revoke" in paths
    assert "/providers/{provider_id}/authorize" not in paths


@pytest.mark.asyncio
async def test_revoke_uses_verified_actor_and_registry_binding(broker, monkeypatch):
    calls: dict[str, object] = {}

    def _revoke(*, actor, provider_id):
        calls.update(actor=actor, provider_id=provider_id)

    monkeypatch.setattr(broker, "revoke", _revoke)
    actor = verified_actor()

    with use_actor(actor):
        response = await remote_oauth_api._revoke("acme")

    assert response.status_code == 204
    assert calls == {"actor": actor, "provider_id": "acme"}


def test_default_registry_reads_typed_agent_config(monkeypatch):
    monkeypatch.setattr(
        typed_config,
        "remote_oauth_providers",
        [
            {
                "provider_id": "typed-config-provider",
                "resource_url": RESOURCE_URL,
                "client_id": "gateway-public-client",
                "redirect_uri": REDIRECT_URI,
                "scopes": ["mcp:read"],
                "enabled": True,
            }
        ],
    )

    registry = remote_oauth_api._default_provider_registry()

    assert (
        registry.require_enabled("typed-config-provider").resource_url == RESOURCE_URL
    )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "JSON array when set"),
        ("not-json", "valid JSON"),
        ("{}", "JSON array"),
        ("[1]", "JSON objects"),
    ],
)
def test_typed_config_rejects_malformed_provider_declarations(raw, message):
    with pytest.raises(ValueError, match=message):
        AgentConfig(REMOTE_OAUTH_PROVIDERS_JSON=raw)


def test_default_registry_disables_invalid_typed_config(monkeypatch, caplog):
    class BrokenConfig:
        @property
        def remote_oauth_providers(self):
            raise ValueError("invalid provider declaration")

    monkeypatch.setattr("agent_utilities.core.config.config", BrokenConfig())

    with caplog.at_level(logging.ERROR):
        registry = remote_oauth_api._default_provider_registry()

    assert registry.enabled_providers() == ()
    assert "registry disabled (ValueError)" in caplog.text


# ---------------------------------------------------------------------------
# Fail-closed identity resolution
#
# NOTE: this repo's ``tests/conftest.py`` binds an autouse, session-scoped,
# ALREADY-authenticated test actor for every test (so ordinary tests don't
# need identity boilerplate) -- so "no actor bound at all" is not a reachable
# state to simulate here. The real fail-closed boundary this module enforces
# (``actor.authenticated is False``) is exercised directly instead, by
# overriding that ambient actor with an explicitly unauthenticated one via
# ``use_actor(...)``.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_authorize_rejects_unauthenticated_actor_context(broker):
    unauth = ActorContext(actor_id="anon", tenant_id="tenant-1", authenticated=False)
    with use_actor(unauth):
        with pytest.raises(HTTPException) as exc_info:
            await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_callback_rejects_unauthenticated_actor_context(broker):
    unauth = ActorContext(actor_id="anon", tenant_id="tenant-1", authenticated=False)
    with use_actor(unauth):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_callback(code="c", state="never-issued", browser_session_id="b")
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Authorize
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_authorize_unknown_provider_returns_404(broker):
    with use_actor(verified_actor()):
        with pytest.raises(HTTPException) as exc_info:
            await authorize(
                "does-not-exist", AuthorizeRequest(browser_session_id="s-1")
            )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_authorize_uses_verified_actor_never_request_data(broker):
    """Nothing in ``AuthorizeRequest`` can name a tenant/principal -- confirmed
    structurally (the model has no such field) and behaviorally: two different
    verified actors calling the identical request body land in two DIFFERENT
    per-principal transactions (proven by each state being independently
    single-use -- see the cross-principal callback test below)."""
    with use_actor(verified_actor(actor_id="user-a")):
        url_a = await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
    with use_actor(verified_actor(actor_id="user-b")):
        url_b = await authorize("acme", AuthorizeRequest(browser_session_id="s-2"))
    assert url_a.authorization_url.startswith(AUTH_ENDPOINT)
    assert url_b.authorization_url.startswith(AUTH_ENDPOINT)
    assert _state_from_url(url_a.authorization_url) != _state_from_url(
        url_b.authorization_url
    )
    assert "client_id=gateway-public-client" in url_a.authorization_url


@pytest.mark.asyncio
async def test_authorize_rejects_scope_widening(broker):
    with use_actor(verified_actor()):
        with pytest.raises(HTTPException) as exc_info:
            await authorize(
                "acme",
                AuthorizeRequest(
                    browser_session_id="s-1", scopes=("mcp:read", "mcp:admin")
                ),
            )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_authorize_scope_policy_can_narrow_further(broker):
    remote_oauth_api._set_scope_policy(lambda actor, provider, requested: ("mcp:read",))
    with use_actor(verified_actor()):
        url = await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
    assert (
        "scope=mcp%3Aread" in url.authorization_url
        or "scope=mcp:read" in url.authorization_url
    )
    assert "mcp%3Awrite" not in url.authorization_url


# ---------------------------------------------------------------------------
# Callback + redirect
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_callback_round_trip_redirects_to_configured_success_url(
    broker, monkeypatch
):
    monkeypatch.setattr(typed_config, "remote_oauth_success_redirect_url", SUCCESS_URL)
    actor = verified_actor()
    with use_actor(actor):
        url = await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
        state = _state_from_url(url.authorization_url)
        response = await oauth_callback(
            code="auth-code-abc", state=state, browser_session_id="s-1"
        )
    assert response.status_code == 302
    # The Location is EXACTLY the configured URL -- no code/state/token ever
    # appended, no matter what arrived on the callback's own query string.
    assert response.headers["location"] == SUCCESS_URL
    assert "auth-code-abc" not in response.headers["location"]
    assert state not in response.headers["location"]


@pytest.mark.asyncio
async def test_callback_without_configured_success_url_returns_json(
    broker, monkeypatch
):
    monkeypatch.setattr(typed_config, "remote_oauth_success_redirect_url", None)
    actor = verified_actor()
    with use_actor(actor):
        url = await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
        state = _state_from_url(url.authorization_url)
        response = await oauth_callback(
            code="auth-code-abc", state=state, browser_session_id="s-1"
        )
    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "success"}


@pytest.mark.asyncio
async def test_callback_state_replay_rejected(broker):
    actor = verified_actor()
    with use_actor(actor):
        url = await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
        state = _state_from_url(url.authorization_url)
        await oauth_callback(
            code="auth-code-abc", state=state, browser_session_id="s-1"
        )
        with pytest.raises(HTTPException) as exc_info:
            await oauth_callback(
                code="auth-code-abc-again", state=state, browser_session_id="s-1"
            )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_callback_cross_principal_reuse_rejected(broker):
    with use_actor(verified_actor(actor_id="user-a")):
        url = await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
        state = _state_from_url(url.authorization_url)
    with use_actor(verified_actor(actor_id="user-b")):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_callback(
                code="stolen-code", state=state, browser_session_id="s-1"
            )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_callback_cross_tenant_reuse_rejected(broker):
    with use_actor(verified_actor(actor_id="user-a", tenant_id="tenant-1")):
        url = await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
        state = _state_from_url(url.authorization_url)
    with use_actor(verified_actor(actor_id="user-a", tenant_id="tenant-2")):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_callback(
                code="stolen-code", state=state, browser_session_id="s-1"
            )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_callback_unknown_state_rejected(broker):
    with use_actor(verified_actor()):
        with pytest.raises(HTTPException) as exc_info:
            await oauth_callback(
                code="c", state="never-issued-state", browser_session_id="s-1"
            )
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Secrets never leak into logs or response bodies
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_sensitive_values_in_logs_or_response_bodies(
    broker, monkeypatch, caplog
):
    monkeypatch.setattr(typed_config, "remote_oauth_success_redirect_url", SUCCESS_URL)
    actor = verified_actor()
    secret_code = _synthetic_value(
        "SUPER-",
        "SECRET-",
        "AUTH-",
        "CODE-",
        "ABC123",
    )
    with caplog.at_level(logging.DEBUG):
        with use_actor(actor):
            url = await authorize("acme", AuthorizeRequest(browser_session_id="s-1"))
            state = _state_from_url(url.authorization_url)
            response = await oauth_callback(
                code=secret_code, state=state, browser_session_id="s-1"
            )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_code not in log_text
    assert state not in log_text
    # The response is a bare redirect to the fixed success URL -- nothing
    # secret is echoed back to the caller either.
    assert secret_code not in response.headers["location"]
    assert state not in response.headers["location"]
