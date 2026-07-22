"""Tests for server-minted KG request identity (CONCEPT:AU-OS.identity.authenticated-identity-enforcement).

Covers:
- claims → ActorContext mapping (roles/tenant extraction, authenticated flag)
- ActorIdentityMiddleware (valid/invalid/missing token and health exemption)
- graph-os rejection of caller-supplied authority and missing GraphSession
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    suspend_session,
    use_session,
)
from agent_utilities.security.brain_context import (
    ActorContext,
    current_actor,
    use_actor,
)
from agent_utilities.security.request_identity import (
    ActorIdentityMiddleware,
    actor_from_claims,
    mint_graph_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    cfg = MagicMock()
    cfg.kg_auth_token_ref = overrides.get("kg_auth_token_ref", None)
    cfg.kg_identity_oauth2 = overrides.get("kg_identity_oauth2", None)
    cfg.auth_jwt_jwks_uri = overrides.get("auth_jwt_jwks_uri", None)
    cfg.auth_jwt_issuer = overrides.get("auth_jwt_issuer", None)
    cfg.auth_jwt_audience = overrides.get("auth_jwt_audience", "agent-services")
    cfg.auth_jwt_algorithms = overrides.get("auth_jwt_algorithms", ["RS256"])
    cfg.mcp_jwt_audience = overrides.get("mcp_jwt_audience", None)
    cfg.kg_policy_version = overrides.get("kg_policy_version", "policy-v1")
    cfg.graph_service_endpoints = overrides.get("graph_service_endpoints", [])
    cfg.deployment_profile = overrides.get("deployment_profile", "tiny")
    cfg.identity_group_capability_map = overrides.get(
        "identity_group_capability_map", None
    )
    return cfg


def _mint(actor: ActorContext):
    placement = SimpleNamespace(endpoint="unix://engine", group=0, epoch=1)
    with (
        mock.patch("agent_utilities.core.config.config", _make_config()),
        mock.patch(
            "agent_utilities.knowledge_graph.core.shard_topology.resolve_endpoints",
            return_value=["unix://engine"],
        ),
        mock.patch(
            "agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine.get_or_create"
        ),
        mock.patch(
            "agent_utilities.knowledge_graph.core.placement_catalog.resolve_placement",
            return_value=placement,
        ),
    ):
        return mint_graph_session(actor)


def _make_token_and_jwks(**claims):
    from joserfc import jwt as joserfc_jwt
    from joserfc.jwk import RSAKey

    key = RSAKey.generate_key(2048)
    jwks = {"keys": [key.as_dict(is_private=False)]}
    payload = {
        "sub": "principal:verified",
        "aud": "agent-services",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        **claims,
    }
    token = joserfc_jwt.encode({"alg": "RS256"}, payload, key)
    return token, jwks


def _make_inner_app(captured: dict):
    async def inner_app(scope, receive, send):  # noqa: ARG001
        from agent_utilities.knowledge_graph.core.session import current_session

        captured["actor"] = current_actor()
        captured["session"] = current_session()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return inner_app


async def _call(mw, path="/api/graph/query", headers=None, state=None):
    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request"}

    scope = {
        "type": "http",
        "path": path,
        "headers": headers or [],
        "state": state or {},
    }
    await mw(scope, receive, send)
    return sent


def _status(sent: list[dict]) -> int:
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


# ---------------------------------------------------------------------------
# actor_from_claims
# ---------------------------------------------------------------------------


class TestActorFromClaims:
    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    def test_basic_mapping_is_authenticated(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "roles": ["hr", "analyst"],
                "tenant_id": "tenant-a",
            }
        )
        assert actor.actor_id == "principal:verified"
        assert actor.roles == ("hr", "analyst")
        assert actor.tenant_id == "tenant-a"
        assert actor.authenticated is True

    def test_validated_claim_expiry_is_retained_for_runtime_enforcement(self):
        expiry = int(time.time()) + 300
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "tenant_id": "tenant-a",
                "exp": expiry,
            }
        )
        assert actor.credential_expires_at == expiry

    @pytest.mark.parametrize(
        "claims",
        [
            {},
            {"sub": ""},
            {"sub": "   "},
            {"sub": 42},
        ],
    )
    def test_missing_or_malformed_principal_is_rejected(self, claims):
        with pytest.raises(ValueError, match="subject claim"):
            actor_from_claims(claims)

    def test_client_id_is_an_explicit_service_principal_fallback(self):
        actor = actor_from_claims({"client_id": "service-client"})
        assert actor.actor_id == "service-client"

    @pytest.mark.parametrize(
        "expiry",
        [None, True, "123", "invalid", [], {}, float("nan"), float("inf"), -1],
    )
    def test_malformed_present_expiry_is_rejected(self, expiry):
        with pytest.raises(ValueError, match="invalid expiry"):
            actor_from_claims({"sub": "principal:verified", "exp": expiry})

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    def test_keycloak_realm_roles_and_tid(self):
        actor = actor_from_claims(
            {"sub": "svc:x", "realm_access": {"roles": ["kg-reader"]}, "tid": "t1"}
        )
        assert actor.roles == ("kg-reader",)
        assert actor.tenant_id == "t1"

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    def test_scope_string_split(self):
        actor = actor_from_claims({"sub": "svc:y", "scope": "kg:read kg:write"})
        assert actor.roles == ("kg:read", "kg:write")

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    def test_human_when_email_claim_present(self):
        from agent_utilities.models.company_brain import ActorType

        human = actor_from_claims(
            {"sub": "principal", "email": "principal@example.invalid"}
        )
        service = actor_from_claims({"sub": "principal"})
        assert human.actor_type == ActorType.HUMAN
        assert service.actor_type == ActorType.AUTOMATED_SERVICE

    def test_minted_writer_session_includes_precondition_read_scope(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "scope": "kg:write unrelated:claim",
                "tenant_id": "tenant-a",
            }
        )
        session = _mint(actor)
        assert session.scopes == frozenset({"kg:read", "kg:write"})

    def test_minted_admin_session_expands_only_the_kg_hierarchy(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "scope": "kg:admin unrelated:claim",
                "tenant_id": "tenant-a",
            }
        )
        session = _mint(actor)
        assert session.scopes == frozenset({"kg:read", "kg:write", "kg:admin"})

    def test_generic_admin_role_does_not_grant_graph_administration(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "roles": ["admin"],
                "tenant_id": "tenant-a",
            }
        )
        session = _mint(actor)
        assert session.scopes == frozenset()

    def test_configured_identity_mapping_can_grant_explicit_kg_admin(self):
        cfg = _make_config(
            identity_group_capability_map={"platform-operators": ["kg:admin"]}
        )
        with mock.patch("agent_utilities.core.config.config", cfg):
            actor = actor_from_claims(
                {
                    "sub": "principal:verified",
                    "groups": ["platform-operators"],
                    "tenant_id": "tenant-a",
                }
            )
        session = _mint(actor)
        assert session.scopes == frozenset({"kg:read", "kg:write", "kg:admin"})

    def test_authenticated_actor_without_tenant_cannot_mint_session(self):
        actor = actor_from_claims({"sub": "principal:verified", "scope": "kg:read"})
        with pytest.raises(PermissionError, match="verified tenant"):
            _mint(actor)

    def test_missing_audience_or_policy_cannot_mint_session(self):
        actor = actor_from_claims(
            {"sub": "principal:verified", "tenant_id": "tenant-a"}
        )
        for config in (
            _make_config(auth_jwt_audience=None),
            _make_config(kg_policy_version=None),
        ):
            with (
                mock.patch("agent_utilities.core.config.config", config),
                pytest.raises(PermissionError, match="audience or policy"),
            ):
                mint_graph_session(actor)

    def test_placement_lookup_uses_the_verified_unrouted_session(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "scope": "kg:admin",
                "tenant_id": "tenant-a",
            }
        )
        observed: dict[str, object] = {}

        def resolve_placement(graph, endpoints, _config):
            from agent_utilities.knowledge_graph.core.session import current_session

            session = current_session()
            assert session is not None
            observed["graph"] = graph
            observed["endpoint"] = session.endpoint
            observed["context"] = session.engine_verified_context()
            return type(
                "Placement",
                (),
                {"endpoint": endpoints[0], "group": 7, "epoch": 11},
            )()

        config = _make_config()
        with (
            mock.patch("agent_utilities.core.config.config", config),
            mock.patch(
                "agent_utilities.knowledge_graph.core.shard_topology.resolve_endpoints",
                return_value=["unix://engine"],
            ),
            mock.patch(
                "agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine.get_or_create"
            ) as bootstrap,
            mock.patch(
                "agent_utilities.knowledge_graph.core.placement_catalog.resolve_placement",
                side_effect=resolve_placement,
            ),
        ):
            session = mint_graph_session(actor)

        assert observed["graph"] == session.graph
        assert observed["endpoint"] is None
        assert observed["context"] == session.engine_verified_context()
        bootstrap.assert_called_once_with(graph_name=session.graph)
        assert session.endpoint == "unix://engine"
        assert session.placement_group == 7
        assert session.catalog_epoch == 11


# ---------------------------------------------------------------------------
# ActorIdentityMiddleware
# ---------------------------------------------------------------------------


class TestActorIdentityMiddleware:
    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_valid_token_mints_authenticated_actor(self):
        token, jwks = _make_token_and_jwks(roles=["hr"], tenant_id="tenant-a")
        cfg = _make_config(auth_jwt_jwks_uri="https://idp/jwks")
        captured: dict = {}
        mw = ActorIdentityMiddleware(_make_inner_app(captured))
        prior_actor = current_actor()

        async def fake_jwks(_uri):
            return jwks

        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch("agent_utilities.security.auth._fetch_jwks", fake_jwks),
            mock.patch(
                "agent_utilities.knowledge_graph.core.shard_topology.resolve_endpoints",
                return_value=["unix://engine"],
            ),
            mock.patch(
                "agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine.get_or_create"
            ),
            mock.patch(
                "agent_utilities.knowledge_graph.core.placement_catalog.resolve_placement",
                return_value=SimpleNamespace(
                    endpoint="unix://engine", group=0, epoch=1
                ),
            ),
        ):
            sent = await _call(
                mw, headers=[(b"authorization", f"Bearer {token}".encode())]
            )
        assert _status(sent) == 200
        actor = captured["actor"]
        assert actor.authenticated is True
        assert actor.actor_id == "principal:verified"
        assert actor.roles == ("hr",)
        assert actor.tenant_id == "tenant-a"
        session = captured["session"]
        assert session is not None
        assert session.tenant == "tenant-a"
        assert session.audience == "agent-services"
        assert session.policy_version == "policy-v1"
        # The request actor is reset and the caller's prior context is restored.
        assert current_actor() == prior_actor
        assert current_actor() != actor

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_invalid_token_is_401(self):
        _, jwks = _make_token_and_jwks()
        cfg = _make_config(auth_jwt_jwks_uri="https://idp/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))

        async def fake_jwks(_uri):
            return jwks

        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch("agent_utilities.security.auth._fetch_jwks", fake_jwks),
        ):
            sent = await _call(mw, headers=[(b"authorization", b"Bearer garbage")])
        assert _status(sent) == 401

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "headers",
        [
            [
                (b"authorization", b"Bearer first"),
                (b"authorization", b"Bearer second"),
            ],
            [(b"authorization", b"Basic opaque")],
            [(b"authorization", b"Bearer")],
            [(b"authorization", b"Bearer two tokens")],
            [(b"authorization", b"Bearer opaque\x00suffix")],
            [(b"authorization", b"Bearer \xff")],
        ],
    )
    async def test_ambiguous_or_malformed_authorization_is_401(self, headers):
        cfg = _make_config(auth_jwt_jwks_uri="https://idp.invalid/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw, headers=headers)
        assert _status(sent) == 401

    @pytest.mark.asyncio
    async def test_prevalidated_identity_cannot_bypass_duplicate_header_rejection(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://idp.invalid/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(
                mw,
                headers=[
                    (b"authorization", b"Bearer first"),
                    (b"authorization", b"Bearer second"),
                ],
                state={
                    "user_claims": {
                        "auth_type": "jwt",
                        "sub": "principal:verified",
                        "exp": int(time.time()) + 300,
                        "tenant_id": "tenant-a",
                    }
                },
            )
        assert _status(sent) == 401

    @pytest.mark.asyncio
    async def test_prevalidated_claims_without_principal_are_401(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://idp.invalid/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(
                mw,
                state={
                    "user_claims": {
                        "auth_type": "jwt",
                        "exp": int(time.time()) + 300,
                        "tenant_id": "tenant-a",
                    }
                },
            )
        assert _status(sent) == 401

    @pytest.mark.asyncio
    async def test_prevalidated_malformed_expiry_is_401(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://idp.invalid/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(
                mw,
                state={
                    "user_claims": {
                        "auth_type": "jwt",
                        "sub": "principal:verified",
                        "exp": "invalid",
                        "tenant_id": "tenant-a",
                    }
                },
            )
        assert _status(sent) == 401

    @pytest.mark.asyncio
    async def test_prevalidated_expired_credential_is_401(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://idp.invalid/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(
                mw,
                state={
                    "user_claims": {
                        "auth_type": "jwt",
                        "sub": "principal:verified",
                        "exp": int(time.time()) - 1,
                        "tenant_id": "tenant-a",
                    }
                },
            )
        assert _status(sent) == 401

    @pytest.mark.asyncio
    async def test_bearer_token_without_validator_is_rejected(self):
        cfg = _make_config(auth_jwt_jwks_uri=None)
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw, headers=[(b"authorization", b"Bearer opaque")])
        assert _status(sent) == 401

    @pytest.mark.asyncio
    async def test_token_validation_error_detail_is_not_reflected(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://issuer.invalid/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))

        async def fail_validation(_token):
            raise RuntimeError("sensitive validation context")

        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch(
                "agent_utilities.security.request_identity.actor_from_bearer_token",
                fail_validation,
            ),
        ):
            sent = await _call(mw, headers=[(b"authorization", b"Bearer opaque")])
        body = next(m["body"] for m in sent if m["type"] == "http.response.body")
        assert _status(sent) == 401
        assert b"sensitive validation context" not in body
        assert b"Token validation failed" in body

    @pytest.mark.asyncio
    async def test_valid_token_without_tenant_is_forbidden(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://issuer.invalid/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        actor = ActorContext(actor_id="principal:verified", authenticated=True)

        async def valid_without_tenant(_token):
            return actor

        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch(
                "agent_utilities.security.request_identity.actor_from_bearer_token",
                valid_without_tenant,
            ),
        ):
            sent = await _call(mw, headers=[(b"authorization", b"Bearer opaque")])
        assert _status(sent) == 403

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_missing_token_is_rejected(self):
        cfg = _make_config(auth_jwt_jwks_uri="https://idp/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw)
        assert _status(sent) == 401

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_health_is_the_only_unauthenticated_exemption(self):
        cfg = _make_config()
        captured: dict = {}
        mw = ActorIdentityMiddleware(_make_inner_app(captured))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw, path="/health")
        assert _status(sent) == 200
        assert captured["actor"].authenticated is False
        assert captured["session"] is None

    @pytest.mark.asyncio
    async def test_metrics_requires_identity(self):
        cfg = _make_config()
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw, path="/metrics")
        assert _status(sent) == 401

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_no_configuration_can_enable_anonymous_graph_access(self):
        cfg = _make_config()
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw)
        assert _status(sent) == 401


# ---------------------------------------------------------------------------
# kg_server identity resolution + read-only gate
# ---------------------------------------------------------------------------


class TestKgServerIdentityResolution:
    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    def test_caller_authority_fields_are_rejected(self):
        from agent_utilities.mcp.kg_server import _reject_caller_authority

        kwargs = {"_actor": "agent:mk", "_roles": "marketing", "_tenant": "t1", "x": 1}
        with pytest.raises(PermissionError, match="Caller-supplied"):
            _reject_caller_authority(kwargs)
        assert kwargs["_actor"] == "agent:mk"

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_missing_session_blocks_every_tool(self):
        from agent_utilities.mcp import kg_server

        async def fake_tool(**kwargs):  # noqa: ARG001
            return "ok"

        with (
            mock.patch.dict(
                kg_server.REGISTERED_TOOLS,
                {"graph_write": fake_tool, "graph_query": fake_tool},
            ),
            suspend_session(),
            mock.patch.object(kg_server, "_PROCESS_SESSION", None),
        ):
            with pytest.raises(PermissionError, match="GraphSession"):
                await kg_server._execute_tool("graph_write")
            with pytest.raises(PermissionError, match="GraphSession"):
                await kg_server._execute_tool("graph_query")

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_verified_session_passes_tool_gate(self):
        from agent_utilities.mcp import kg_server

        async def fake_tool(**kwargs):  # noqa: ARG001
            return "ok"

        actor = ActorContext(
            actor_id="principal:verified",
            tenant_id="tenant-a",
            roles=("kg:write",),
            authenticated=True,
        )
        session = GraphSession(
            actor=actor,
            tenant="tenant-a",
            scopes=frozenset({"kg:read", "kg:write"}),
            graph="tenant-a",
            audience="agent-services",
            policy_version="policy-v1",
        )
        with (
            mock.patch.dict(kg_server.REGISTERED_TOOLS, {"graph_write": fake_tool}),
            use_actor(actor),
            use_session(session),
        ):
            assert await kg_server._execute_tool("graph_write") == "ok"


# ---------------------------------------------------------------------------
# Served security profile (CONCEPT:AU-OS.identity.authenticated-identity-enforcement)
# ---------------------------------------------------------------------------


class TestServedSecurityProfile:
    """apply_served_security_profile() — fail-closed network MCP transports."""

    def test_stdio_is_noop(self):
        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        cfg = _make_config(auth_jwt_jwks_uri=None)
        # Stdio identity is validated by the process-identity startup boundary.
        apply_served_security_profile("stdio", config=cfg)

    def test_network_without_transport_auth_fails_loud(self):
        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        cfg = _make_config(auth_jwt_jwks_uri=None)
        with pytest.raises(RuntimeError, match="authentication provider"):
            apply_served_security_profile("streamable-http", config=cfg)

    def test_network_with_jwks_but_no_auth_provider_still_fails_loud(self):
        """AUTH_JWT_JWKS_URI alone must NOT satisfy the served-security gate.

        Regression guard for the live misconfiguration this closes: an
        operator wired every JWT/OIDC identity variable (JWKS, issuer,
        audience) but left the FastMCP auth-provider switch
        (``--auth-type``/``AUTH_TYPE``) unset — historically that combination
        was accepted as "configured" even though FastMCP never attached a
        token verifier, so the network endpoint served every request
        unauthenticated. JWKS being merely *present* must keep failing loud.
        """
        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        cfg = _make_config(auth_jwt_jwks_uri="https://kc/realms/x/certs")
        with pytest.raises(RuntimeError, match="authentication provider"):
            apply_served_security_profile("streamable-http", config=cfg)

    def test_network_with_jwks_accepts_mandatory_contract(self, monkeypatch):
        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        monkeypatch.setenv("KG_BRAIN_ENFORCE", "0")
        cfg = _make_config(auth_jwt_jwks_uri="https://kc/realms/x/certs")
        apply_served_security_profile(
            "streamable-http", config=cfg, transport_auth_configured=True
        )
        from agent_utilities.knowledge_graph.core.company_brain_runtime import (
            brain_enforcement_enabled,
        )

        assert brain_enforcement_enabled() is True

    def test_network_with_transport_auth_accepts_mandatory_contract(self):
        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        cfg = _make_config(auth_jwt_jwks_uri=None)
        apply_served_security_profile(
            "streamable-http",
            config=cfg,
            transport_auth_configured=True,
        )


class TestStdioProcessIdentity:
    def test_tiny_local_process_session_uses_neutral_ephemeral_authority(self):
        from agent_utilities.security.request_identity import (
            mint_local_process_session,
        )

        cfg = _make_config(auth_jwt_audience=None, kg_policy_version=None)
        placement = SimpleNamespace(endpoint="unix://engine", group=0, epoch=1)
        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch(
                "agent_utilities.knowledge_graph.core.shard_topology.resolve_endpoints",
                return_value=["unix://engine"],
            ),
            mock.patch(
                "agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine.get_or_create"
            ),
            mock.patch(
                "agent_utilities.knowledge_graph.core.placement_catalog.resolve_placement",
                return_value=placement,
            ),
        ):
            session = mint_local_process_session()

        assert session.actor.actor_id == "graph-os:local-process"
        assert session.actor.tenant_id == "local"
        assert session.actor.authenticated is True
        assert session.scopes == frozenset({"kg:read", "kg:write", "kg:admin"})
        assert session.audience == "graph-os-local"
        assert session.policy_version == "local-ephemeral-v1"
        assert session.actor.credential_expires_at is None

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, True),
            ({"deployment_profile": "single-node-prod"}, False),
            ({"graph_service_endpoints": ["https://engine.example.test"]}, False),
            ({"kg_auth_token_ref": "secret://graph/token"}, False),
            ({"kg_identity_oauth2": {"client_secret": "secret://graph/key"}}, False),
        ],
    )
    def test_local_process_authority_is_tiny_packaged_local_only(
        self, overrides, expected
    ):
        from agent_utilities.security.request_identity import (
            local_process_authority_enabled,
        )

        assert local_process_authority_enabled(_make_config(**overrides)) is expected

    def test_token_reference_is_resolved_at_runtime(self):
        from agent_utilities.security.request_identity import (
            acquire_process_identity_token,
        )

        cfg = _make_config(kg_auth_token_ref="secret://graph/process-token")
        with mock.patch(
            "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
            return_value="header.payload.signature",
        ):
            assert acquire_process_identity_token(cfg) == "header.payload.signature"

    def test_env_token_reference_does_not_construct_secret_backend(self):
        from agent_utilities.security.request_identity import (
            acquire_process_identity_token,
        )

        cfg = _make_config(kg_auth_token_ref="env://GRAPHOS_PROCESS_TOKEN")
        with (
            mock.patch.dict(
                "os.environ",
                {"GRAPHOS_PROCESS_TOKEN": "header.payload.signature"},
                clear=False,
            ),
            mock.patch(
                "agent_utilities.security.secrets_client.create_secrets_client"
            ) as create_backend,
        ):
            assert acquire_process_identity_token(cfg) == "header.payload.signature"
            create_backend.assert_not_called()

    def test_oauth2_source_mints_at_runtime(self):
        from agent_utilities.security.request_identity import (
            acquire_process_identity_token,
        )

        oauth2 = {
            "token_url": "https://identity.example.test/token",
            "client_id": "graph-os",
            "client_secret": "secret://graph/client-secret",
        }
        cfg = _make_config(kg_identity_oauth2=oauth2)
        provider = MagicMock()
        provider.get_token.return_value = "header.payload.signature"
        with mock.patch(
            "agent_utilities.security.oauth_client_credentials.build_provider_from_config",
            return_value=provider,
        ):
            assert acquire_process_identity_token(cfg) == "header.payload.signature"

    def test_process_identity_acquisition_error_is_transport_neutral(self):
        from agent_utilities.security.request_identity import (
            acquire_process_identity_token,
        )

        cfg = _make_config(kg_auth_token_ref="secret://graph/process-token")
        with (
            mock.patch(
                "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
                side_effect=RuntimeError("private backend detail"),
            ),
            pytest.raises(RuntimeError) as error,
        ):
            acquire_process_identity_token(cfg)

        assert str(error.value) == "Graph process identity acquisition failed"
        assert "Stdio" not in str(error.value)
        assert "private backend detail" not in str(error.value)

    @pytest.mark.parametrize(
        "cfg",
        [
            _make_config(),
            _make_config(
                kg_auth_token_ref="secret://graph/token",
                kg_identity_oauth2={"configured": True},
            ),
        ],
    )
    def test_identity_source_must_be_exactly_one(self, cfg):
        from agent_utilities.security.request_identity import (
            acquire_process_identity_token,
        )

        with pytest.raises(RuntimeError, match="exactly one"):
            acquire_process_identity_token(cfg)
