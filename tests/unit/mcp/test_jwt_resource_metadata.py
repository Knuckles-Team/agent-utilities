"""Tests for RFC 9728 protected-resource metadata on the ``jwt`` auth path
(CONCEPT:AU-OS.identity.protected-resource-metadata).

A bare ``JWTVerifier``/``TokenVerifier`` (what plain ``AUTH_TYPE=jwt`` builds)
publishes zero OAuth routes and never appends ``resource_metadata`` to its 401
challenge, so an RFC 9728-aware MCP client (e.g. Claude Code) has no way to
discover the authorization server and refresh its own token — it is handed a
static JWT that silently expires. Setting ``MCP_PUBLIC_BASE_URL`` (``--public-
base-url``) now wraps the same verifier in ``fastmcp``'s ``RemoteAuthProvider``,
which publishes ``/.well-known/oauth-protected-resource`` and wires the metadata
URL into every 401 — with no MCP_PUBLIC_BASE_URL configured, the verifier is
returned unwrapped (today's exact behavior, additive and backward compatible).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.mcp.server_factory import _configure_jwt_auth


def _args(**kw: object) -> SimpleNamespace:
    # NOTE: fake issuer/JWKS URLs below use https:// — ``_secure_auth_url``
    # requires HTTPS for any non-loopback hostname (AU-OS.identity hardening).
    base: dict[str, object] = {
        "token_jwks_uri": "https://kc.test/realms/master/protocol/openid-connect/certs",
        "token_issuer": "https://kc.test/realms/master",
        "token_audience": "agent-services",
        "token_algorithm": None,
        "token_secret": None,
        "token_public_key": None,
        "required_scopes": None,
        "public_base_url": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_public_base_url_unset_returns_bare_verifier() -> None:
    """Backward compatible: no MCP_PUBLIC_BASE_URL -> today's exact behavior."""
    from fastmcp.server.auth import RemoteAuthProvider
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    verifier = _configure_jwt_auth(_args())
    assert isinstance(verifier, JWTVerifier)
    assert not isinstance(verifier, RemoteAuthProvider)
    # TokenVerifier's default get_routes() publishes nothing (the root cause).
    assert verifier.get_routes(mcp_path="/mcp") == []


def test_public_base_url_unset_omitted_attr_returns_bare_verifier() -> None:
    """A caller whose Namespace never set ``public_base_url`` at all (e.g. an
    older test double) must behave identically -- ``getattr(..., "")`` covers it."""
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    args = _args()
    del args.public_base_url
    verifier = _configure_jwt_auth(args)
    assert isinstance(verifier, JWTVerifier)


def test_public_base_url_set_wraps_in_remote_auth_provider() -> None:
    """MCP_PUBLIC_BASE_URL set -> RemoteAuthProvider advertising RFC 9728."""
    from fastmcp.server.auth import RemoteAuthProvider

    verifier = _configure_jwt_auth(_args(public_base_url="https://graph-os.arpa"))
    assert isinstance(verifier, RemoteAuthProvider)

    routes = verifier.get_routes(mcp_path="/mcp")
    paths = [getattr(route, "path", None) for route in routes]
    assert "/.well-known/oauth-protected-resource/mcp" in paths

    assert [str(server) for server in verifier.authorization_servers] == [
        "https://kc.test/realms/master"
    ]


def test_public_base_url_set_401_carries_resource_metadata() -> None:
    """The 401 WWW-Authenticate challenge names the resource_metadata URL --
    this is the concrete signal an RFC 9728-aware client (Claude Code) follows
    to discover Keycloak and refresh its own token."""
    from mcp.server.auth.middleware.bearer_auth import RequireAuthMiddleware
    from mcp.server.auth.routes import build_resource_metadata_url

    verifier = _configure_jwt_auth(_args(public_base_url="https://graph-os.arpa"))
    verifier.get_routes(mcp_path="/mcp")  # binds mcp_path / resource url
    resource_url = verifier._get_resource_url("/mcp")
    assert resource_url is not None
    metadata_url = build_resource_metadata_url(resource_url)
    assert str(metadata_url) == (
        "https://graph-os.arpa/.well-known/oauth-protected-resource/mcp"
    )

    async def _inner_app(
        scope: Any, receive: Any, send: Any
    ) -> None:  # pragma: no cover
        raise AssertionError("inner app must not run for an unauthenticated request")

    sent: list[dict[str, Any]] = []

    async def _capture_send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = RequireAuthMiddleware(
        _inner_app, required_scopes=[], resource_metadata_url=metadata_url
    )
    # No "user" in scope == unauthenticated request -> the middleware's own
    # missing-auth 401 path, independent of any real bearer-token handling.
    asyncio.run(middleware(scope={"type": "http"}, receive=None, send=_capture_send))

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 401
    headers = dict(start["headers"])
    www_authenticate = headers[b"www-authenticate"].decode()
    assert (
        'resource_metadata="https://graph-os.arpa/.well-known/'
        'oauth-protected-resource/mcp"' in www_authenticate
    )


def test_public_base_url_unset_401_has_no_resource_metadata() -> None:
    """Root-cause regression guard: with no public base URL, the resource URL
    (and therefore the metadata URL) resolves to None -- the exact bare-Bearer
    behavior this fix closes."""
    verifier = _configure_jwt_auth(_args())
    assert verifier._get_resource_url("/mcp") is None


def test_public_base_url_set_multi_realm_also_wraps() -> None:
    """The multi-realm branch (_MultiIssuerVerifier) gets the same treatment."""
    from fastmcp.server.auth import RemoteAuthProvider

    verifier = _configure_jwt_auth(
        _args(
            token_jwks_uri=(
                "https://kc.test/realms/master/protocol/openid-connect/certs,"
                "https://kc.test/realms/homelab/protocol/openid-connect/certs"
            ),
            token_issuer="https://kc.test/realms/master,https://kc.test/realms/homelab",
            public_base_url="https://graph-os.arpa",
        )
    )
    assert isinstance(verifier, RemoteAuthProvider)
    assert hasattr(verifier.token_verifier, "_verifiers")
    assert len(verifier.token_verifier._verifiers) == 2  # type: ignore[attr-defined]
    assert len(verifier.authorization_servers) == 2
    assert [str(server) for server in verifier.authorization_servers] == [
        "https://kc.test/realms/master",
        "https://kc.test/realms/homelab",
    ]


def test_public_base_url_unset_multi_realm_stays_plain() -> None:
    """Backward compatible for the multi-realm branch too."""
    from fastmcp.server.auth import RemoteAuthProvider

    verifier = _configure_jwt_auth(
        _args(
            token_jwks_uri=(
                "https://kc.test/realms/master/protocol/openid-connect/certs,"
                "https://kc.test/realms/homelab/protocol/openid-connect/certs"
            ),
            token_issuer="https://kc.test/realms/master,https://kc.test/realms/homelab",
        )
    )
    assert not isinstance(verifier, RemoteAuthProvider)
    assert hasattr(verifier, "_verifiers")


def test_invalid_public_base_url_exits_closed() -> None:
    """A malformed MCP_PUBLIC_BASE_URL fails closed (consistent with every other
    auth-URL policy check in this module), never silently ignored."""
    with pytest.raises(SystemExit):
        _configure_jwt_auth(_args(public_base_url="not a url"))


def test_http_public_base_url_off_loopback_rejected() -> None:
    """Same HTTPS-outside-loopback policy as every sibling auth URL flag."""
    with pytest.raises(SystemExit):
        _configure_jwt_auth(_args(public_base_url="http://graph-os.arpa"))
