"""Tests for server-minted KG request identity (CONCEPT:AU-OS.identity.authenticated-identity-enforcement).

Covers:
- claims → ActorContext mapping (roles/tenant extraction, authenticated flag)
- ActorIdentityMiddleware (valid/invalid/missing token and health exemption)
- graph-os rejection of caller-supplied authority and missing GraphSession
"""

from __future__ import annotations

import time
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
    VerifiedLocalBearer,
    actor_from_claims,
    mint_graph_session,
    verify_local_bearer_token,
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
    """Mint exactly as a served request does — with nothing about the engine mocked.

    Deliberately no ``resolve_placement`` / ``GraphComputeEngine`` patches: this
    helper used to install them, which is precisely why the whole suite stayed
    green while every non-cluster-admin principal 500'd in production (D-SP-1).
    Minting must not reach the engine at all, so an unmocked mint is the test.
    """
    with mock.patch("agent_utilities.core.config.config", _make_config()):
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

    @pytest.mark.parametrize("expiry", [True, -1, 1 << 63, float("inf")])
    def test_validated_claim_expiry_is_bounded(self, expiry):
        with pytest.raises(ValueError, match="invalid expiry"):
            actor_from_claims(
                {
                    "sub": "principal:verified",
                    "tenant_id": "tenant-a",
                    "exp": expiry,
                }
            )

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
        from agent_utilities.security.actor_identity import ActorType

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
                "exp": int(time.time()) + 300,
            }
        )
        session = _mint(actor)
        assert session.scopes == frozenset({"kg:read", "kg:write"})

    @pytest.mark.spec("AU-SEC-R006")
    def test_minted_admin_session_expands_only_the_kg_hierarchy(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "scope": "kg:admin unrelated:claim",
                "tenant_id": "tenant-a",
                "exp": int(time.time()) + 300,
            }
        )
        session = _mint(actor)
        assert session.scopes == frozenset({"kg:read", "kg:write", "kg:admin"})

    @pytest.mark.spec("AU-SEC-R006")
    def test_the_elevation_approval_scope_reaches_the_session_only_as_itself(self):
        """AU-SEC requirement 006: an approver's realm role projects as the exact scope
        EG requires; ``kg:admin`` never implies it."""
        approver = actor_from_claims(
            {
                "sub": "principal:approver",
                "realm_access": {"roles": ["kg:read", "rbac:approve-elevation"]},
                "tenant_id": "tenant-a",
                "exp": int(time.time()) + 300,
            }
        )
        admin = actor_from_claims(
            {
                "sub": "principal:admin",
                "scope": "kg:admin",
                "tenant_id": "tenant-a",
                "exp": int(time.time()) + 300,
            }
        )
        assert "rbac:approve-elevation" in _mint(approver).scopes
        assert "rbac:approve-elevation" not in _mint(admin).scopes

    def test_generic_admin_role_does_not_grant_graph_administration(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "roles": ["admin"],
                "tenant_id": "tenant-a",
                "exp": int(time.time()) + 300,
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
                    "exp": int(time.time()) + 300,
                }
            )
        session = _mint(actor)
        assert session.scopes == frozenset({"kg:read", "kg:write", "kg:admin"})

    def test_authenticated_actor_without_tenant_cannot_mint_session(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "scope": "kg:read",
                "exp": int(time.time()) + 300,
            }
        )
        with pytest.raises(PermissionError, match="invalid tenant"):
            _mint(actor)

    def test_missing_audience_or_policy_cannot_mint_session(self):
        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "tenant_id": "tenant-a",
                "exp": int(time.time()) + 300,
            }
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

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    def test_minting_never_resolves_engine_placement(self):
        """Authentication establishes identity, never topology (D-SP-1).

        Placement is an ``admin:cluster-read`` engine read
        (``eg-capabilities/src/lib.rs:2274``) enforced against the engine's own
        ``IsolationLayer``, which no JWT claim can satisfy. Resolving it while
        minting made cluster-admin authority a precondition for authenticating
        ANY request on EVERY served surface. This observes the real
        ``resolve_placement`` and the real transport provisioner — pass-through,
        not replacement — and asserts the mint path never reaches either.
        """
        from agent_utilities.knowledge_graph.core import (
            graph_compute,
            placement_catalog,
        )
        from tests.wiring import observe

        actor = actor_from_claims(
            {
                "sub": "principal:verified",
                "scope": "kg:admin",
                "tenant_id": "tenant-a",
                "exp": int(time.time()) + 300,
            }
        )
        with (
            observe(placement_catalog, "resolve_placement") as resolved,
            observe(graph_compute.GraphComputeEngine, "get_or_create") as provisioned,
        ):
            session = _mint(actor)

        resolved.assert_not_called(
            why="authenticating a request must never perform the engine's "
            "admin:cluster-read PlacementRoute call"
        )
        provisioned.assert_not_called(
            why="authenticating a request must never provision an engine transport"
        )
        # `endpoint=None` is GraphSession's documented "resolve normally" value;
        # the data plane binds the authoritative route per call.
        assert session.endpoint is None
        assert session.placement_group is None
        assert session.catalog_epoch is None
        # Everything that constitutes authority is still bound.
        assert session.actor is actor
        assert session.tenant == "tenant-a"
        assert session.graph
        assert session.scopes == frozenset({"kg:read", "kg:write", "kg:admin"})
        assert session.policy_version == "policy-v1"
        assert session.audience == "agent-services"
        assert session.trace_context

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    def test_non_admin_principal_mints_a_full_session(self):
        """The case that 500'd today: a principal with no cluster authority.

        ``kg:read`` alone carries neither ``kg:admin`` nor ``admin:cluster-read``,
        so the pre-D-SP-1 minter's ``PlacementRoute`` call was rejected by the
        engine and the resulting ``PlacementAuthorityError`` (a ``RuntimeError``,
        so it matched none of the middleware's 401/403 arms) surfaced as a blanket
        HTTP 500 on every authenticated route.
        """
        actor = actor_from_claims(
            {
                "sub": "principal:agent-webui",
                "scope": "kg:read",
                "tenant_id": "tenant-a",
                "exp": int(time.time()) + 300,
            }
        )
        session = _mint(actor)
        assert session.scopes == frozenset({"kg:read"})
        assert "kg:admin" not in session.scopes
        assert session.tenant == "tenant-a"
        assert session.actor.actor_id == "principal:agent-webui"


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

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    @pytest.mark.asyncio
    async def test_jwt_dependency_fault_is_distinct_from_invalid_credential(self):
        """A verification-path dependency/config fault (the ``_decode_jwt``
        500 raised when its JWT stack is unavailable) must never collapse to
        the generic 401 "Token validation failed" — that collapse is exactly
        how a missing ``joserfc`` dependency was previously misreported as a
        rejected credential (a correctly issued token could never have been
        accepted, and the error blamed the credential instead of the missing
        dependency)."""
        from fastapi import HTTPException

        cfg = _make_config(auth_jwt_jwks_uri="https://issuer.invalid/jwks")
        mw = ActorIdentityMiddleware(_make_inner_app({}))

        async def dependency_missing(_token):
            raise HTTPException(
                status_code=500,
                detail="The base installation is incomplete: joserfc is required "
                "for JWT authentication.",
            )

        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch(
                "agent_utilities.security.request_identity.actor_from_bearer_token",
                dependency_missing,
            ),
        ):
            sent = await _call(mw, headers=[(b"authorization", b"Bearer opaque")])
        body = next(m["body"] for m in sent if m["type"] == "http.response.body")
        assert _status(sent) == 500
        assert b"Token validation failed" not in body
        assert b"joserfc" in body

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
    @pytest.mark.parametrize("path", ["/health", "/health/ready"])
    async def test_health_probes_are_unauthenticated_exemptions(self, path):
        cfg = _make_config()
        captured: dict = {}
        mw = ActorIdentityMiddleware(_make_inner_app(captured))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw, path=path)
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

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_non_admin_principal_completes_a_served_request(self):
        """The served edge that 500'd before D-SP-1, end to end.

        A real RS256 credential carrying only ``kg:read`` — no ``kg:admin``, no
        ``admin:cluster-read``, and a principal with no entry in the engine's
        ``IsolationLayer`` — is validated by the real JWKS path, minted by the
        real middleware, and reaches graph-os's real per-tool authority boundary
        (``kg_server.verified_tool_session_scope``), which admits it.

        The seams are observed, not replaced: ``resolve_placement`` is the real
        function and must simply never be reached, because reaching it is what
        required cluster-admin authority to authenticate at all.
        """
        from agent_utilities.knowledge_graph.core import placement_catalog
        from agent_utilities.mcp.kg_server import verified_tool_session_scope
        from tests.wiring import observe

        token, jwks = _make_token_and_jwks(scope="kg:read", tenant_id="tenant-a")
        cfg = _make_config(auth_jwt_jwks_uri="https://idp/jwks")
        captured: dict = {}

        async def graph_tool_app(scope, receive, send):  # noqa: ARG001
            # graph-os's real per-call authority gate, not a stand-in.
            with verified_tool_session_scope() as session:
                captured["session"] = session
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def fake_jwks(_uri):
            return jwks

        mw = ActorIdentityMiddleware(graph_tool_app)
        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch("agent_utilities.security.auth._fetch_jwks", fake_jwks),
            observe(placement_catalog, "resolve_placement") as resolved,
        ):
            sent = await _call(
                mw, headers=[(b"authorization", f"Bearer {token}".encode())]
            )

        assert _status(sent) == 200
        resolved.assert_not_called(
            why="a non-admin principal must authenticate without the engine's "
            "admin:cluster-read PlacementRoute call"
        )
        session = captured["session"]
        assert session is not None
        assert session.scopes == frozenset({"kg:read"})
        assert "kg:admin" not in session.scopes
        assert session.tenant == "tenant-a"
        # The verified engine claims a tool call forwards are complete without
        # any route being bound.
        assert session.engine_verified_context()["tenant"] == "tenant-a"

    @pytest.mark.concept("CONCEPT:AU-OS.identity.authenticated-identity-enforcement")
    @pytest.mark.asyncio
    async def test_unauthenticated_request_still_401s_and_never_reaches_the_app(self):
        """The other half of D-SP-1: no enforcement was traded away."""
        from agent_utilities.security.request_identity import (
            HEALTH_PATHS,
            UNAUTHENTICATED_PATHS,
        )

        # The exempt set is exactly the non-fingerprinting probes — unchanged.
        assert UNAUTHENTICATED_PATHS == HEALTH_PATHS
        assert UNAUTHENTICATED_PATHS == frozenset(
            {"/health", "/health/ready", "/healthz", "/api/health", "/api/healthz"}
        )

        reached: dict = {}

        async def graph_tool_app(scope, receive, send):  # noqa: ARG001
            reached["app"] = True
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = ActorIdentityMiddleware(graph_tool_app)
        with mock.patch(
            "agent_utilities.core.config.config",
            _make_config(auth_jwt_jwks_uri="https://idp/jwks"),
        ):
            sent = await _call(mw, path="/api/graph/query")
        assert _status(sent) == 401
        assert "app" not in reached


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

    @pytest.mark.spec("AU-SEC-R001")
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
        # No placement/transport patches: the stdio bootstrap goes through the
        # same minter, which no longer contacts the engine (D-SP-1).
        with mock.patch("agent_utilities.core.config.config", cfg):
            session = mint_local_process_session()

        assert session.actor.actor_id == "graph-os:local-process"
        assert session.actor.tenant_id == "local"
        assert session.actor.authenticated is True
        # CONCEPT:X1 -- least-privilege: kg:read/kg:write only, never
        # kg:admin. See test_tiny_local_process_session_default_has_no_admin_or_control_scope
        # and TestLocalProcessGrantChokepoint below for the mutation-proof
        # coverage of this specific fix.
        assert session.scopes == frozenset({"kg:read", "kg:write"})
        assert session.audience == "graph-os-local"
        assert session.policy_version == "local-ephemeral-v1"
        assert session.actor.credential_expires_at is not None
        assert session.actor.credential_lease is not None
        assert (
            session.actor.credential_lease.expires_at
            == session.actor.credential_expires_at
        )

    def test_tiny_local_process_session_default_has_no_admin_or_control_scope(self):
        """CONCEPT:X1 (a): the default tiny-profile local process must never be
        able to pass an admin/security/control gate. Checks both the aggregate
        scope set directly (mutation-proof against a changed default role) and
        that the coarse ``require_scope`` gate itself refuses ``kg:admin``."""
        from agent_utilities.knowledge_graph.core.session import ScopeError
        from agent_utilities.security.request_identity import (
            mint_local_process_session,
        )

        cfg = _make_config()
        with mock.patch("agent_utilities.core.config.config", cfg):
            session = mint_local_process_session()

        assert "kg:admin" not in session.scopes
        assert not any(
            scope == "*"
            or scope.startswith("admin:")
            or scope.startswith("security:")
            or scope.endswith(":control")
            for scope in session.scopes
        )
        with pytest.raises(ScopeError):
            session.require_scope("kg:admin")
        # The narrower scopes a normal local tool call/background write needs
        # must still be granted -- this is a least-privilege narrowing, not an
        # outage.
        session.require_scope("kg:read")
        session.require_scope("kg:write")

    def test_local_process_bootstrap_authority_is_a_distinct_graph_admin_mint(self):
        """CONCEPT:X1 (c): first-run local graph provisioning gets its own
        one-shot authority carrying exactly the engine's ``graph:admin``
        lifecycle scope -- never ``kg:admin`` -- and it is not the ambient
        subject used for ordinary tool calls, so it can never be mistaken for
        one in a provenance/audit trail."""
        from agent_utilities.knowledge_graph.core.session import ScopeError
        from agent_utilities.security.request_identity import (
            mint_local_process_bootstrap_authority,
            mint_local_process_session,
        )

        cfg = _make_config()
        with mock.patch("agent_utilities.core.config.config", cfg):
            ordinary = mint_local_process_session()
            bootstrap = mint_local_process_bootstrap_authority()

        assert bootstrap.scopes == frozenset({"kg:read", "graph:admin"})
        bootstrap.require_scope("graph:admin")
        with pytest.raises(ScopeError):
            bootstrap.require_scope("kg:admin")
        assert "graph:admin" not in ordinary.scopes
        assert bootstrap.actor.actor_id != ordinary.actor.actor_id
        assert bootstrap.actor.actor_id == "graph-os:local-process-bootstrap"

    def test_no_setting_widens_the_ambient_local_process_authority(self):
        """CONCEPT:X1: the former ``KG_LOCAL_PROCESS_ADMIN_SCOPE`` opt-in is
        gone. Neither the field nor a truthy stand-in on a config object can
        turn the ambient local mint into ``kg:admin``."""
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.security.request_identity import (
            mint_local_process_session,
        )

        assert "kg_local_process_admin_scope" not in AgentConfig.model_fields
        cfg = _make_config()
        cfg.kg_local_process_admin_scope = True
        with mock.patch("agent_utilities.core.config.config", cfg):
            session = mint_local_process_session()
        assert "kg:admin" not in session.scopes

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
        """BUG-PE-028: the outer message must stay sanitised (never echo the
        real transport/config detail), but -- unlike the `from None` this
        function used to raise with -- the real cause must still be reachable
        as ``__cause__`` for server-side logs/tracebacks. Same contract
        ``test_mint_actor_from_token_sync_preserves_cause`` already locks in
        for the sibling JWT-validation path."""
        from agent_utilities.security.request_identity import (
            acquire_process_identity_token,
        )

        underlying = RuntimeError("private backend detail")
        cfg = _make_config(kg_auth_token_ref="secret://graph/process-token")
        with (
            mock.patch(
                "agent_utilities.security.cli_secrets.resolve_runtime_secret_reference",
                side_effect=underlying,
            ),
            pytest.raises(RuntimeError) as error,
        ):
            acquire_process_identity_token(cfg)

        assert str(error.value) == "Graph process identity acquisition failed"
        assert "Stdio" not in str(error.value)
        assert "private backend detail" not in str(error.value)
        assert error.value.__cause__ is underlying

    @pytest.mark.concept("CONCEPT:AU-OS.config.secrets-authentication")
    def test_mint_actor_from_token_sync_preserves_cause(self):
        """``mint_actor_from_token_sync`` wraps every failure in a generic
        ``RuntimeError`` (never echoing the underlying detail to a caller),
        but must NOT discard the original exception as its ``__cause__`` —
        that would be the same swallowed-error pattern this change fixes in
        the JWT verification path (a dependency/config fault becoming an
        opaque failure with no signal of why)."""
        from agent_utilities.security.request_identity import (
            mint_actor_from_token_sync,
        )

        underlying = RuntimeError(
            "Cannot validate Bearer token: audience is not configured"
        )
        with (
            mock.patch(
                "agent_utilities.security.request_identity.actor_from_bearer_token",
                new=mock.AsyncMock(side_effect=underlying),
            ),
            pytest.raises(RuntimeError) as error,
        ):
            mint_actor_from_token_sync("irrelevant.token.value")

        assert str(error.value) == "Graph process identity token validation failed"
        assert error.value.__cause__ is underlying

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


# ---------------------------------------------------------------------------
# VerifiedLocalBearer / verify_local_bearer_token (GRAPHOS-IDENTITY-R003's
# agent-utilities consumer side: graph-os's local token issuer mints a
# bearer JWT; this is the one chokepoint that verifies it and returns its
# claims, never an authority).
# ---------------------------------------------------------------------------


def _local_bearer_claims(**overrides):
    now = int(time.time())
    claims = {
        "iss": "https://issuer.example.test",
        "aud": "agent-services",
        "sub": "principal:verified",
        "tenant_id": "tenant-a",
        "principal_kind": "service",
        "scope": "kg:read",
        "iat": now,
        "nbf": now,
        "exp": now + 300,
        "jti": "fixture-jti",
    }
    claims.update(overrides)
    return claims


def _signed_token_cfg(**claim_overrides):
    """A real RS256 token, its matching JWKS, and the config wired to verify it."""
    claims = _local_bearer_claims(**claim_overrides)
    token, jwks = _make_token_and_jwks(**claims)
    cfg = _make_config(
        auth_jwt_jwks_uri="https://issuer.example.test/jwks",
        auth_jwt_issuer=claims["iss"],
        auth_jwt_audience=claims["aud"],
    )

    async def fake_jwks(_uri):
        return jwks

    return claims, token, cfg, fake_jwks


class TestVerifiedLocalBearer:
    def test_rejects_non_dict_claims(self):
        with pytest.raises(PermissionError):
            VerifiedLocalBearer(
                claims="nope",  # type: ignore[arg-type]
                issuer="https://issuer.example.test",
                audience="agent-services",
            )

    def test_rejects_issuer_or_audience_drift(self):
        claims = _local_bearer_claims()
        with pytest.raises(PermissionError, match="drifted"):
            VerifiedLocalBearer(
                claims=claims, issuer="https://other.invalid", audience=claims["aud"]
            )
        with pytest.raises(PermissionError, match="drifted"):
            VerifiedLocalBearer(
                claims=claims, issuer=claims["iss"], audience="other-audience"
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("sub", ""),
            ("tenant_id", ""),
            ("scope", ""),
            ("jti", ""),
            ("principal_kind", "admin"),
            ("principal_kind", None),
            ("iat", "now"),
            ("nbf", True),
            ("exp", 1.5),
        ],
    )
    def test_rejects_malformed_claims(self, field, value):
        claims = _local_bearer_claims(**{field: value})
        with pytest.raises(PermissionError):
            VerifiedLocalBearer(
                claims=claims, issuer=claims["iss"], audience=claims["aud"]
            )

    def test_principal_reads_the_subject_claim(self):
        claims = _local_bearer_claims()
        bearer = VerifiedLocalBearer(
            claims=claims, issuer=claims["iss"], audience=claims["aud"]
        )
        assert bearer.principal == "principal:verified"

    def test_accepts_optional_fields_it_does_not_require(self):
        """A fuller claim set (roles/policy_version/agent_id/delegation, as
        graph-os's local issuer actually mints) must not be rejected just
        because this DTO does not itself require those optional fields."""
        claims = _local_bearer_claims(
            roles=["human"],
            policy_version="policy-v1",
            agent_id="principal:verified",
            delegation=[],
        )
        bearer = VerifiedLocalBearer(
            claims=claims, issuer=claims["iss"], audience=claims["aud"]
        )
        assert bearer.principal == "principal:verified"


class TestVerifyLocalBearerToken:
    @pytest.mark.asyncio
    async def test_verifies_a_real_signed_token(self):
        claims, token, cfg, fake_jwks = _signed_token_cfg()

        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch("agent_utilities.security.auth._fetch_jwks", fake_jwks),
        ):
            result = await verify_local_bearer_token(token)
        assert isinstance(result, VerifiedLocalBearer)
        assert result.principal == "principal:verified"
        assert result.issuer == claims["iss"]
        assert result.audience == claims["aud"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cfg",
        [
            _make_config(),
            _make_config(auth_jwt_jwks_uri="https://issuer.example.test/jwks"),
            _make_config(
                auth_jwt_jwks_uri="https://issuer.example.test/jwks",
                auth_jwt_issuer="https://issuer.example.test",
                auth_jwt_audience=None,
            ),
        ],
    )
    async def test_unconfigured_issuer_is_a_permission_error(self, cfg):
        with mock.patch("agent_utilities.core.config.config", cfg):
            with pytest.raises(PermissionError):
                await verify_local_bearer_token("whatever")

    @pytest.mark.asyncio
    async def test_invalid_token_is_a_permission_error_not_an_http_exception(self):
        _, _, cfg, fake_jwks = _signed_token_cfg()

        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch("agent_utilities.security.auth._fetch_jwks", fake_jwks),
        ):
            with pytest.raises(PermissionError):
                await verify_local_bearer_token("not-a-jwt")

    @pytest.mark.asyncio
    async def test_dependency_fault_is_distinct_from_a_rejected_credential(self):
        """Mirrors ``test_jwt_dependency_fault_is_distinct_from_invalid_credential``
        above for this local-bearer path: a verification-path fault that is
        NOT a credential rejection must surface loudly as a ``RuntimeError``,
        never collapse into the generic ``PermissionError`` denial."""
        from fastapi import HTTPException

        _, token, cfg, fake_jwks = _signed_token_cfg()

        def broken_decode(*_args, **_kwargs):
            raise HTTPException(status_code=500, detail="joserfc missing")

        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch("agent_utilities.security.auth._fetch_jwks", fake_jwks),
            mock.patch("agent_utilities.security.auth._decode_jwt", broken_decode),
        ):
            with pytest.raises(RuntimeError, match="verification path failed"):
                await verify_local_bearer_token(token)
