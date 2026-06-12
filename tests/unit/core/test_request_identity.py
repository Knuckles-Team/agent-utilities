"""Tests for server-minted KG request identity (CONCEPT:OS-5.14).

Covers:
- claims → ActorContext mapping (roles/tenant extraction, authenticated flag)
- ActorIdentityMiddleware (valid/invalid token, KG_AUTH_REQUIRED gating,
  health-path exemption, legacy pass-through)
- kg_server identity resolution (_actor_from_kwargs precedence) and the
  read-only tool gate for unauthenticated callers under KG_AUTH_REQUIRED
"""

from __future__ import annotations

import time
from unittest import mock
from unittest.mock import MagicMock

import pytest

from agent_utilities.security.brain_context import (
    ActorContext,
    current_actor,
    use_actor,
)
from agent_utilities.security.request_identity import (
    ActorIdentityMiddleware,
    actor_from_claims,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    cfg = MagicMock()
    cfg.kg_auth_required = overrides.get("kg_auth_required", False)
    cfg.kg_auth_token = overrides.get("kg_auth_token", None)
    cfg.auth_jwt_jwks_uri = overrides.get("auth_jwt_jwks_uri", None)
    cfg.auth_jwt_issuer = overrides.get("auth_jwt_issuer", None)
    cfg.auth_jwt_audience = overrides.get("auth_jwt_audience", None)
    return cfg


def _make_token_and_jwks(**claims):
    from joserfc import jwt as joserfc_jwt
    from joserfc.jwk import RSAKey

    key = RSAKey.generate_key(2048)
    jwks = {"keys": [key.as_dict(is_private=False)]}
    payload = {
        "sub": "user:ada",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        **claims,
    }
    token = joserfc_jwt.encode({"alg": "RS256"}, payload, key)
    return token, jwks


def _make_inner_app(captured: dict):
    async def inner_app(scope, receive, send):  # noqa: ARG001
        captured["actor"] = current_actor()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return inner_app


async def _call(mw, path="/api/graph/query", headers=None):
    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request"}

    scope = {"type": "http", "path": path, "headers": headers or []}
    await mw(scope, receive, send)
    return sent


def _status(sent: list[dict]) -> int:
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


# ---------------------------------------------------------------------------
# actor_from_claims
# ---------------------------------------------------------------------------


class TestActorFromClaims:
    @pytest.mark.concept("CONCEPT:OS-5.14")
    def test_basic_mapping_is_authenticated(self):
        actor = actor_from_claims(
            {"sub": "user:ada", "roles": ["hr", "analyst"], "tenant_id": "acme"}
        )
        assert actor.actor_id == "user:ada"
        assert actor.roles == ("hr", "analyst")
        assert actor.tenant_id == "acme"
        assert actor.authenticated is True

    @pytest.mark.concept("CONCEPT:OS-5.14")
    def test_keycloak_realm_roles_and_tid(self):
        actor = actor_from_claims(
            {"sub": "svc:x", "realm_access": {"roles": ["kg-reader"]}, "tid": "t1"}
        )
        assert actor.roles == ("kg-reader",)
        assert actor.tenant_id == "t1"

    @pytest.mark.concept("CONCEPT:OS-5.14")
    def test_scope_string_split(self):
        actor = actor_from_claims({"sub": "svc:y", "scope": "kg:read kg:write"})
        assert actor.roles == ("kg:read", "kg:write")

    @pytest.mark.concept("CONCEPT:OS-5.14")
    def test_human_when_email_claim_present(self):
        from agent_utilities.models.company_brain import ActorType

        human = actor_from_claims({"sub": "u", "email": "a@b.c"})
        service = actor_from_claims({"sub": "u"})
        assert human.actor_type == ActorType.HUMAN
        assert service.actor_type == ActorType.AUTOMATED_SERVICE


# ---------------------------------------------------------------------------
# ActorIdentityMiddleware
# ---------------------------------------------------------------------------


class TestActorIdentityMiddleware:
    @pytest.mark.concept("CONCEPT:OS-5.14")
    @pytest.mark.asyncio
    async def test_valid_token_mints_authenticated_actor(self):
        token, jwks = _make_token_and_jwks(roles=["hr"], tenant_id="acme")
        cfg = _make_config(auth_jwt_jwks_uri="https://idp/jwks")
        captured: dict = {}
        mw = ActorIdentityMiddleware(_make_inner_app(captured))

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
        assert actor.actor_id == "user:ada"
        assert actor.roles == ("hr",)
        assert actor.tenant_id == "acme"
        # The contextvar is reset after the request.
        assert current_actor().authenticated is False

    @pytest.mark.concept("CONCEPT:OS-5.14")
    @pytest.mark.asyncio
    async def test_invalid_token_is_401_even_when_auth_not_required(self):
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

    @pytest.mark.concept("CONCEPT:OS-5.14")
    @pytest.mark.asyncio
    async def test_auth_required_rejects_missing_token(self):
        cfg = _make_config(
            kg_auth_required=True, auth_jwt_jwks_uri="https://idp/jwks"
        )
        mw = ActorIdentityMiddleware(_make_inner_app({}))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw)
        assert _status(sent) == 401

    @pytest.mark.concept("CONCEPT:OS-5.14")
    @pytest.mark.asyncio
    async def test_auth_required_exempts_health(self):
        cfg = _make_config(kg_auth_required=True)
        captured: dict = {}
        mw = ActorIdentityMiddleware(_make_inner_app(captured))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw, path="/health")
        assert _status(sent) == 200
        assert captured["actor"].authenticated is False

    async def test_auth_required_exempts_metrics(self):
        # Prometheus scrapers cannot mint JWTs (CONCEPT:OS-5.23)
        cfg = _make_config(kg_auth_required=True)
        captured: dict = {}
        mw = ActorIdentityMiddleware(_make_inner_app(captured))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw, path="/metrics")
        assert _status(sent) == 200
        assert captured["actor"].authenticated is False

    @pytest.mark.concept("CONCEPT:OS-5.14")
    @pytest.mark.asyncio
    async def test_legacy_mode_passes_through_unauthenticated(self):
        cfg = _make_config()
        captured: dict = {}
        mw = ActorIdentityMiddleware(_make_inner_app(captured))
        with mock.patch("agent_utilities.core.config.config", cfg):
            sent = await _call(mw)
        assert _status(sent) == 200
        assert captured["actor"].authenticated is False


# ---------------------------------------------------------------------------
# kg_server identity resolution + read-only gate
# ---------------------------------------------------------------------------


class TestKgServerIdentityResolution:
    @pytest.mark.concept("CONCEPT:OS-5.14")
    def test_legacy_kwargs_build_actor(self):
        from agent_utilities.mcp.kg_server import _actor_from_kwargs

        cfg = _make_config()
        kwargs = {"_actor": "agent:mk", "_roles": "marketing", "_tenant": "t1", "x": 1}
        with mock.patch("agent_utilities.core.config.config", cfg):
            actor = _actor_from_kwargs(kwargs)
        assert actor is not None
        assert actor.actor_id == "agent:mk"
        assert actor.roles == ("marketing",)
        assert actor.authenticated is False
        assert kwargs == {"x": 1}  # identity kwargs always popped

    @pytest.mark.concept("CONCEPT:OS-5.14")
    def test_authenticated_ambient_actor_ignores_kwargs(self):
        from agent_utilities.mcp.kg_server import _actor_from_kwargs

        cfg = _make_config()
        minted = ActorContext(
            actor_id="user:jwt", roles=("hr",), authenticated=True
        )
        kwargs = {"_actor": "spoofed:admin", "_roles": "admin"}
        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            use_actor(minted),
        ):
            actor = _actor_from_kwargs(kwargs)
        assert actor is None  # ambient (server-minted) identity kept
        assert kwargs == {}

    @pytest.mark.concept("CONCEPT:OS-5.14")
    def test_auth_required_ignores_kwargs_entirely(self):
        from agent_utilities.mcp.kg_server import _actor_from_kwargs

        cfg = _make_config(kg_auth_required=True)
        kwargs = {"_actor": "spoofed:admin", "_roles": "admin,system"}
        with mock.patch("agent_utilities.core.config.config", cfg):
            actor = _actor_from_kwargs(kwargs)
        assert actor is None
        assert kwargs == {}

    @pytest.mark.concept("CONCEPT:OS-5.14")
    @pytest.mark.asyncio
    async def test_read_only_gate_blocks_writes_when_unauthenticated(self):
        from agent_utilities.mcp import kg_server

        async def fake_tool(**kwargs):  # noqa: ARG001
            return "ok"

        cfg = _make_config(kg_auth_required=True)
        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch.dict(
                kg_server.REGISTERED_TOOLS,
                {"graph_write": fake_tool, "graph_query": fake_tool},
            ),
        ):
            with pytest.raises(PermissionError, match="KG_AUTH_REQUIRED"):
                await kg_server._execute_tool("graph_write")
            # Read-only surface stays available.
            assert await kg_server._execute_tool("graph_query") == "ok"

    @pytest.mark.concept("CONCEPT:OS-5.14")
    @pytest.mark.asyncio
    async def test_authenticated_actor_passes_write_gate(self):
        from agent_utilities.mcp import kg_server

        async def fake_tool(**kwargs):  # noqa: ARG001
            return "ok"

        cfg = _make_config(kg_auth_required=True)
        minted = ActorContext(actor_id="user:jwt", authenticated=True)
        with (
            mock.patch("agent_utilities.core.config.config", cfg),
            mock.patch.dict(kg_server.REGISTERED_TOOLS, {"graph_write": fake_tool}),
            use_actor(minted),
        ):
            assert await kg_server._execute_tool("graph_write") == "ok"


# ---------------------------------------------------------------------------
# Served security profile (CONCEPT:OS-5.14)
# ---------------------------------------------------------------------------


class TestServedSecurityProfile:
    """apply_served_security_profile() — fail-closed network MCP transports."""

    def _clear_env(self, monkeypatch):
        for var in ("KG_SERVED_PROFILE", "KG_AUTH_REQUIRED", "KG_BRAIN_ENFORCE"):
            monkeypatch.delenv(var, raising=False)

    def test_stdio_is_noop(self, monkeypatch):
        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        self._clear_env(monkeypatch)
        cfg = _make_config(auth_jwt_jwks_uri=None)
        # stdio must never be touched even with no JWKS configured.
        apply_served_security_profile("stdio", config=cfg)
        assert "KG_AUTH_REQUIRED" not in __import__("os").environ
        assert "KG_BRAIN_ENFORCE" not in __import__("os").environ

    def test_network_without_jwks_fails_loud(self, monkeypatch):
        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        self._clear_env(monkeypatch)
        cfg = _make_config(auth_jwt_jwks_uri=None)
        with pytest.raises(RuntimeError, match="AUTH_JWT_JWKS_URI"):
            apply_served_security_profile("streamable-http", config=cfg)

    def test_network_with_jwks_turns_on_enforcement(self, monkeypatch):
        import os

        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        self._clear_env(monkeypatch)
        cfg = _make_config(auth_jwt_jwks_uri="https://kc/realms/x/certs")
        apply_served_security_profile("streamable-http", config=cfg)
        assert os.environ["KG_AUTH_REQUIRED"] == "1"
        assert os.environ["KG_BRAIN_ENFORCE"] == "1"
        assert cfg.kg_auth_required is True

    def test_explicit_opt_out_leaves_open(self, monkeypatch):
        import os

        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        self._clear_env(monkeypatch)
        monkeypatch.setenv("KG_SERVED_PROFILE", "0")
        cfg = _make_config(auth_jwt_jwks_uri=None)
        # Opt-out: no JWKS required, no flags forced, no raise.
        apply_served_security_profile("streamable-http", config=cfg)
        assert "KG_AUTH_REQUIRED" not in os.environ

    def test_operator_pinned_flag_is_honored(self, monkeypatch):
        import os

        from agent_utilities.security.request_identity import (
            apply_served_security_profile,
        )

        self._clear_env(monkeypatch)
        # Operator explicitly pinned auth off; the profile must not override it.
        monkeypatch.setenv("KG_AUTH_REQUIRED", "0")
        cfg = _make_config(auth_jwt_jwks_uri="https://kc/realms/x/certs")
        apply_served_security_profile("streamable-http", config=cfg)
        assert os.environ["KG_AUTH_REQUIRED"] == "0"
        assert cfg.kg_auth_required is False
        # Enforcement default still supplied since it was unset.
        assert os.environ["KG_BRAIN_ENFORCE"] == "1"
