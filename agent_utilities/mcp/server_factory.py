from __future__ import annotations

"""MCP Server Factory.

Handles CLI argument parsing for MCP servers, automated server initialization
with middleware stacks, and the ``create_mcp_server`` convenience constructor.

CONCEPT:AU-ECO.mcp.standardized-interfaces — MCP Standardized Interfaces
"""


import argparse
import ipaddress
import logging
import os
import re
import sys
from typing import Any
from urllib.parse import urlsplit

from agent_utilities._version import __version__
from agent_utilities.base_utilities import to_boolean
from agent_utilities.core.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    setting,
)
from agent_utilities.security.cli_secrets import (
    RuntimeSecretReferenceAction,
    RuntimeSecretReferenceError,
    resolve_runtime_secret_reference,
)

logger = logging.getLogger(__name__)


# MCP-specific auth/delegation config (not in config.py)
mcp_auth_config = {
    "enable_delegation": to_boolean(setting("ENABLE_DELEGATION", "False")),
    "audience": setting("AUDIENCE", None),
    "delegated_scopes": setting("DELEGATED_SCOPES", "api"),
    "token_endpoint": None,
    "oidc_client_id": setting("OIDC_CLIENT_ID", None),
    "oidc_client_secret": None,
    "oidc_config_url": setting("OIDC_CONFIG_URL", None),
    "jwt_jwks_uri": setting("FASTMCP_SERVER_AUTH_JWT_JWKS_URI", None),
    "jwt_issuer": setting("FASTMCP_SERVER_AUTH_JWT_ISSUER", None),
    "jwt_audience": setting("FASTMCP_SERVER_AUTH_JWT_AUDIENCE", None),
    "jwt_algorithm": setting("FASTMCP_SERVER_AUTH_JWT_ALGORITHM", None),
    "jwt_secret": setting("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY", None),
    "jwt_required_scopes": setting("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES", None),
}  # nosec B105

DEFAULT_TRANSPORT = setting("TRANSPORT", "stdio")

_NETWORK_TRANSPORTS = frozenset({"streamable-http", "sse"})
_ALL_TRANSPORTS = ("stdio", "streamable-http", "sse")
_FILTER_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_NO_HTTP_REQUEST = "No active HTTP request found."


def _bounded_filter_values(values: list[str]) -> list[str]:
    """Parse bounded client visibility filters; these may only narrow access."""
    parsed: list[str] = []
    for raw in values:
        if len(str(raw).encode("utf-8")) > 16_384:
            raise ValueError("MCP visibility filter is too large")
        for part in str(raw).split(","):
            value = part.strip()
            if not value:
                continue
            if value.lower() == "all" or not _FILTER_VALUE_RE.fullmatch(value):
                raise ValueError("MCP visibility filter is invalid")
            parsed.append(value)
            if len(parsed) > 256:
                raise ValueError("MCP visibility filter has too many values")
    return list(dict.fromkeys(parsed))


def _narrow_values(current: list[str], requested: list[str]) -> list[str]:
    """Intersect an existing allowlist, or establish one when none exists."""
    if not requested:
        return current
    if not current:
        return requested
    requested_set = set(requested)
    narrowed = [value for value in current if value in requested_set]
    return narrowed or ["__empty_filter_intersection__"]


def _union_values(current: list[str], additions: list[str]) -> list[str]:
    """Union deny lists without allowing a caller to remove server policy."""
    return list(dict.fromkeys([*current, *additions]))


def _optional_http_request() -> Any | None:
    """Return the active FastMCP HTTP request, if this is an HTTP invocation.

    FastMCP deliberately raises when ``list_tools`` is called in-process (for
    example by connector certification) or over stdio because neither path has
    an HTTP request context.  That expected absence is not a rejected client
    filter.  Every other exception still propagates to the fail-closed filter
    guard below.
    """
    from fastmcp.server.dependencies import get_http_request

    try:
        return get_http_request()
    except RuntimeError as exc:
        if exc.args == (_NO_HTTP_REQUEST,):
            return None
        raise


def _hardened_jwt_verifier(**kwargs: Any) -> Any:
    """Build FastMCP's verifier with pinned JWKS I/O and privacy-safe logging."""
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    class _Verifier(JWTVerifier):
        async def _fetch_jwks(self) -> dict[str, Any]:
            if not self.jwks_uri:
                raise ValueError("JWKS URI is not configured")
            from agent_utilities.security.auth import _fetch_jwks

            document = await _fetch_jwks(str(self.jwks_uri))
            return dict(document)

        async def verify_token(self, token: str) -> Any:
            if not isinstance(token, str) or not token or len(token.encode()) > 16_384:
                return None
            result = await super().verify_token(token)
            if result is None:
                return None
            claims = getattr(result, "claims", None)
            if not isinstance(claims, dict):
                return None
            import math
            import time

            now = time.time()
            exp = claims.get("exp")
            if (
                isinstance(exp, bool)
                or not isinstance(exp, (int, float))
                or not math.isfinite(float(exp))
                or float(exp) < now
            ):
                return None
            for field in ("nbf", "iat"):
                value = claims.get(field)
                if value is None:
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) > now + 30.0
                ):
                    return None
            return result

    class _PrivacyLogger:
        def debug(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.debug("JWT verifier diagnostic")

        def info(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.info("JWT verifier event")

        def warning(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.warning("JWT verifier rejected a credential")

        def error(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.error("JWT verifier error")

        def exception(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            logger.error("JWT verifier error")

        def isEnabledFor(self, level: int) -> bool:  # noqa: N802
            return logger.isEnabledFor(level)

    verifier = _Verifier(**kwargs)
    verifier.logger = _PrivacyLogger()
    return verifier


def _secure_auth_url(value: Any, *, field: str) -> str:
    rendered = str(value or "").strip()
    if len(rendered) > 8_192:
        raise ValueError(f"{field} is too large")
    parsed = urlsplit(rendered)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{field} is invalid")
    if parsed.scheme.lower() == "http" and parsed.hostname.lower() not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError(f"{field} requires HTTPS outside loopback")
    return rendered


def _validated_redirect_uris(value: str | None) -> list[str] | None:
    if not value:
        return None
    uris = [part.strip() for part in value.split(",") if part.strip()]
    if not uris or len(uris) > 32:
        raise ValueError("redirect URI allowlist is invalid")
    return [_secure_auth_url(uri, field="redirect URI") for uri in uris]


def _is_loopback_bind(host: Any) -> bool:
    """Return true only for an explicit loopback bind target.

    Hostnames other than ``localhost`` are deliberately treated as remote. We
    do not resolve DNS while validating startup configuration because a name
    can be rebound after validation.
    """
    value = str(host or "").strip().lower()
    if value in {"localhost", "localhost."}:
        return True
    value = value.strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _validate_network_exposure(args: argparse.Namespace) -> None:
    """Fail closed unless a remote MCP listener has auth and a TLS boundary."""
    transport = str(getattr(args, "transport", "stdio") or "stdio").lower()
    auth_type = str(getattr(args, "auth_type", "none") or "none").lower()
    if transport not in _NETWORK_TRANSPORTS:
        return
    if _is_loopback_bind(getattr(args, "host", "")):
        return
    if auth_type == "none":
        logger.error(
            "Refusing an unauthenticated MCP network listener outside loopback"
        )
        raise SystemExit(1)

    certfile = str(getattr(args, "tls_certfile", "") or "").strip()
    keyfile = str(getattr(args, "tls_keyfile", "") or "").strip()
    if bool(certfile) != bool(keyfile):
        logger.error("MCP server TLS requires both certificate and key material")
        raise SystemExit(1)
    direct_tls = bool(certfile and keyfile)
    if direct_tls and not (os.path.isfile(certfile) and os.path.isfile(keyfile)):
        logger.error("MCP server TLS material is unavailable")
        raise SystemExit(1)

    terminated = bool(getattr(args, "tls_terminated", False))
    proxy_cidrs = str(getattr(args, "trusted_proxy_cidrs", "") or "").strip()
    if terminated and not proxy_cidrs:
        logger.error(
            "MCP_TLS_TERMINATED requires MCP_TRUSTED_PROXY_CIDRS so plaintext "
            "traffic is accepted only from the TLS ingress"
        )
        raise SystemExit(1)
    if not direct_tls and not terminated:
        logger.error(
            "A non-loopback MCP listener requires direct TLS or an explicitly "
            "trusted TLS-terminating ingress"
        )
        raise SystemExit(1)

    hosts = [
        value.strip()
        for value in str(getattr(args, "allowed_hosts", "") or "").split(",")
        if value.strip()
    ]
    if not hosts or any("*" in value for value in hosts):
        logger.error("A non-loopback MCP listener requires exact MCP_ALLOWED_HOSTS")
        raise SystemExit(1)


def create_mcp_parser(
    *, transport_choices: tuple[str, ...] = _ALL_TRANSPORTS
) -> argparse.ArgumentParser:
    """Create a standard argument parser for MCP servers.

    Defines a comprehensive set of CLI flags for transport selection,
    host/port configuration, authentication (JWT, OIDC, OAuth), and Eunomia
    policy enforcement. Callers may narrow ``transport_choices`` when a server
    intentionally exposes a smaller current transport surface.

    Returns:
        An argparse.ArgumentParser instance.

    """
    # Keycloak defaults integration
    keycloak_url = setting("KEYCLOAK_URL")
    keycloak_realm = setting("KEYCLOAK_REALM", "master")
    default_oidc_config = setting("OIDC_CONFIG_URL")
    if not default_oidc_config and keycloak_url:
        default_oidc_config = (
            f"{keycloak_url}/realms/{keycloak_realm}/.well-known/openid-configuration"
        )

    default_oidc_client_id = setting("OIDC_CLIENT_ID") or setting("KEYCLOAK_CLIENT_ID")
    default_oidc_client_secret_ref = setting("OIDC_CLIENT_SECRET_REF") or setting(
        "KEYCLOAK_CLIENT_SECRET_REF"
    )
    try:
        default_oidc_client_secret = (
            resolve_runtime_secret_reference(default_oidc_client_secret_ref)
            if default_oidc_client_secret_ref
            else None
        )
        oauth_upstream_client_secret_ref = setting("OAUTH_UPSTREAM_CLIENT_SECRET_REF")
        default_oauth_upstream_client_secret = (
            resolve_runtime_secret_reference(oauth_upstream_client_secret_ref)
            if oauth_upstream_client_secret_ref
            else None
        )
        openapi_password_ref = setting("OPENAPI_PASSWORD_REF")
        default_openapi_password = (
            resolve_runtime_secret_reference(openapi_password_ref)
            if openapi_password_ref
            else None
        )
        openapi_client_secret_ref = setting("OPENAPI_CLIENT_SECRET_REF")
        default_openapi_client_secret = (
            resolve_runtime_secret_reference(openapi_client_secret_ref)
            if openapi_client_secret_ref
            else None
        )
    except RuntimeSecretReferenceError as exc:
        raise RuntimeError("configured authentication secret is unavailable") from exc

    if not transport_choices or any(
        transport not in _ALL_TRANSPORTS for transport in transport_choices
    ):
        raise ValueError(
            "transport_choices must be a non-empty subset of supported transports"
        )

    parser = argparse.ArgumentParser(add_help=False, description="MCP Server")
    parser.add_argument(
        "-t",
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=list(transport_choices),
        help=f"Transport method: {', '.join(transport_choices)} (default: {DEFAULT_TRANSPORT})",
    )
    parser.add_argument(
        "-H",
        "--host",
        default=DEFAULT_HOST,
        help="Host address for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--tls-certfile",
        default=setting("MCP_TLS_CERTFILE"),
        help="Runtime path to the inbound TLS certificate (prefer secret projection)",
    )
    parser.add_argument(
        "--tls-keyfile",
        default=setting("MCP_TLS_KEYFILE"),
        help="Runtime path to the inbound TLS private key (prefer secret projection)",
    )
    parser.add_argument(
        "--tls-terminated",
        action="store_true",
        default=to_boolean(setting("MCP_TLS_TERMINATED", "False")),
        help="TLS is terminated by a peer restricted with --trusted-proxy-cidrs",
    )
    parser.add_argument(
        "--trusted-proxy-cidrs",
        default=setting("MCP_TRUSTED_PROXY_CIDRS"),
        help="Comma-separated exact ingress peer CIDRs used with --tls-terminated",
    )
    parser.add_argument(
        "--allowed-hosts",
        default=setting("MCP_ALLOWED_HOSTS"),
        help="Comma-separated exact Host-header allowlist for network transports",
    )
    parser.add_argument(
        "--allowed-origins",
        default=setting("MCP_ALLOWED_ORIGINS"),
        help="Comma-separated exact browser/WebSocket origins; unset blocks Origin requests",
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=int(setting("MCP_MAX_REQUEST_BYTES", str(4 * 1024 * 1024))),
        help="Maximum buffered MCP HTTP request body",
    )
    parser.add_argument(
        "--auth-type",
        default=setting("AUTH_TYPE", "none"),
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    parser.add_argument(
        "--static-tokens-ref",
        default=setting("FASTMCP_SERVER_AUTH_STATIC_TOKENS_REF"),
        help="Secret reference containing the JSON map used by static authentication",
    )
    parser.add_argument(
        "--token-jwks-uri",
        default=setting("FASTMCP_SERVER_AUTH_JWT_JWKS_URI"),
        help="JWKS URI for JWT verification",
    )
    parser.add_argument(
        "--token-issuer",
        default=setting("FASTMCP_SERVER_AUTH_JWT_ISSUER"),
        help="Issuer for JWT verification",
    )
    parser.add_argument(
        "--token-audience",
        default=setting("FASTMCP_SERVER_AUTH_JWT_AUDIENCE"),
        help="Audience for JWT verification",
    )
    parser.add_argument(
        "--token-algorithm",
        default=setting("FASTMCP_SERVER_AUTH_JWT_ALGORITHM"),
        choices=[
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
        ],
        help="JWT signing algorithm (required for HMAC or static key). Auto-detected for JWKS.",
    )
    parser.add_argument(
        "--token-secret-ref",
        default=setting("FASTMCP_SERVER_AUTH_JWT_SECRET_REF"),
        help="Runtime secret reference for an HMAC JWT verification secret",
    )
    parser.add_argument(
        "--token-public-key",
        default=setting("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Path to PEM public key file or inline PEM string (for static asymmetric keys).",
    )
    parser.add_argument(
        "--required-scopes",
        default=setting("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES"),
        help="Comma-separated list of required scopes (e.g., gitlab.read,gitlab.write).",
    )
    parser.add_argument(
        "--oauth-upstream-auth-endpoint",
        default=None,
        help="Upstream authorization endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-token-endpoint",
        default=None,
        help="Upstream token endpoint for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-id",
        default=None,
        help="Upstream client ID for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-upstream-client-secret-ref",
        dest="oauth_upstream_client_secret",
        action=RuntimeSecretReferenceAction,
        default=default_oauth_upstream_client_secret,
        help="Runtime secret reference for the OAuth proxy client secret",
    )
    parser.add_argument(
        "--oauth-base-url", default=None, help="Base URL for OAuth Proxy"
    )
    parser.add_argument(
        "--oidc-config-url",
        default=default_oidc_config,
        help="OIDC configuration URL",
    )
    parser.add_argument(
        "--oidc-client-id",
        default=default_oidc_client_id,
        help="OIDC client ID",
    )
    parser.add_argument(
        "--oidc-client-secret-ref",
        dest="oidc_client_secret",
        action=RuntimeSecretReferenceAction,
        default=default_oidc_client_secret,
        help="Runtime secret reference for the OIDC client secret",
    )
    parser.add_argument(
        "--oidc-base-url",
        default=setting("OIDC_BASE_URL"),
        help="Base URL for OIDC Proxy",
    )
    parser.add_argument(
        "--remote-auth-servers",
        default=None,
        help="Comma-separated list of authorization servers for Remote OAuth",
    )
    parser.add_argument(
        "--remote-base-url", default=None, help="Base URL for Remote OAuth"
    )
    parser.add_argument(
        "--allowed-client-redirect-uris",
        default=None,
        help="Comma-separated list of allowed client redirect URIs",
    )
    parser.add_argument(
        "--eunomia-type",
        default=setting("EUNOMIA_TYPE", "none"),
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default=setting("EUNOMIA_POLICY_FILE", "mcp_policies.json"),
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url",
        default=setting("EUNOMIA_REMOTE_URL", None),
        help="URL for remote Eunomia server",
    )
    parser.add_argument(
        "--eunomia-api-key-ref",
        default=setting("EUNOMIA_API_KEY_REF", None),
        help="Runtime secret reference for the remote Eunomia API key",
    )
    parser.add_argument(
        "--enable-delegation",
        action="store_true",
        default=to_boolean(setting("ENABLE_DELEGATION", "False")),
        help="Enable OIDC token delegation",
    )
    parser.add_argument(
        "--audience",
        default=setting("AUDIENCE", None),
        help="Audience for the delegated token",
    )
    parser.add_argument(
        "--delegated-scopes",
        default=setting("DELEGATED_SCOPES", "api"),
        help="Scopes for the delegated token (space-separated)",
    )
    parser.add_argument(
        "--openapi-file",
        default=None,
        help="Path to the OpenAPI JSON file to import additional tools from",
    )
    parser.add_argument(
        "--openapi-base-url",
        default=None,
        help="Base URL for the OpenAPI client (overrides instance URL)",
    )
    parser.add_argument(
        "--openapi-use-token",
        action="store_true",
        help="Use the incoming Bearer token (from MCP request) to authenticate OpenAPI import",
    )

    parser.add_argument(
        "--openapi-username",
        default=setting("OPENAPI_USERNAME"),
        help="Username for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-password-ref",
        dest="openapi_password",
        action=RuntimeSecretReferenceAction,
        default=default_openapi_password,
        help="Runtime secret reference for OpenAPI basic authentication",
    )

    parser.add_argument(
        "--openapi-client-id",
        default=setting("OPENAPI_CLIENT_ID"),
        help="OAuth client ID for OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-secret-ref",
        dest="openapi_client_secret",
        action=RuntimeSecretReferenceAction,
        default=default_openapi_client_secret,
        help="Runtime secret reference for the OpenAPI OAuth client secret",
    )

    parser.add_argument(
        "--tools",
        "--toolsets",
        default=setting("MCP_ENABLED_TOOLS"),
        help="Comma-separated list of enabled tools or toolsets to expose",
    )
    parser.add_argument(
        "--disabled-tools",
        "--disabled-toolsets",
        default=setting("MCP_DISABLED_TOOLS"),
        help="Comma-separated list of disabled tools or toolsets to exclude",
    )

    parser.add_argument("--help", action="store_true", help="Show usage")
    return parser


def _configure_auth(args: argparse.Namespace) -> Any:
    """Configure authentication provider based on parsed CLI args.

    Returns the auth provider instance or None.
    """
    if args.auth_type == "none" or not args.auth_type:
        return None

    import sys as _sys

    from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
    from fastmcp.server.auth.oidc_proxy import OIDCProxy

    mcp_auth_config["enable_delegation"] = args.enable_delegation
    mcp_auth_config["audience"] = args.audience or mcp_auth_config["audience"]
    mcp_auth_config["delegated_scopes"] = (
        args.delegated_scopes or mcp_auth_config["delegated_scopes"]
    )

    if hasattr(args, "oidc_config_url"):
        mcp_auth_config["oidc_config_url"] = (
            args.oidc_config_url or mcp_auth_config["oidc_config_url"]
        )
    if hasattr(args, "oidc_client_id"):
        mcp_auth_config["oidc_client_id"] = (
            args.oidc_client_id or mcp_auth_config["oidc_client_id"]
        )
    if hasattr(args, "oidc_client_secret"):
        mcp_auth_config["oidc_client_secret"] = (
            args.oidc_client_secret or mcp_auth_config["oidc_client_secret"]
        )

    if mcp_auth_config["enable_delegation"]:
        if args.auth_type != "oidc-proxy":
            logger.error("Error: Token delegation requires auth-type=oidc-proxy")
            _sys.exit(1)
        if not mcp_auth_config["audience"]:
            logger.error("Error: audience is required for delegation")
            _sys.exit(1)
        if not all(
            [
                mcp_auth_config["oidc_config_url"],
                mcp_auth_config["oidc_client_id"],
                mcp_auth_config["oidc_client_secret"],
            ]
        ):
            logger.error("Error: Delegation requires complete OIDC configuration")
            _sys.exit(1)
        try:
            config_url = mcp_auth_config["oidc_config_url"]
            if not isinstance(config_url, str):
                raise ValueError("oidc_config_url must be a string")
            suffix = "/.well-known/openid-configuration"
            issuer = (
                config_url[: -len(suffix)]
                if config_url.endswith(suffix)
                else config_url
            )
            from agent_utilities.security.oidc_discovery import discover

            oidc_config = discover(issuer)
            mcp_auth_config["token_endpoint"] = oidc_config.get("token_endpoint")
            if not mcp_auth_config["token_endpoint"]:
                raise ValueError("No token_endpoint found in OIDC configuration")
        except Exception as exc:
            logger.error(
                "Failed to fetch OIDC configuration (exception_type=%s)",
                type(exc).__name__,
            )
            _sys.exit(1)

    try:
        allowed_uris = _validated_redirect_uris(args.allowed_client_redirect_uris)
    except ValueError:
        logger.error("Error: allowed client redirect URI policy is invalid")
        _sys.exit(1)

    if args.auth_type == "none":
        return None
    elif args.auth_type == "static":
        import json as _json

        reference = str(getattr(args, "static_tokens_ref", "") or "").strip()
        if not reference:
            logger.error(
                "Error: static auth requires --static-tokens-ref; inline and "
                "built-in tokens are not permitted"
            )
            _sys.exit(1)
        try:
            if reference.startswith("env://"):
                raw_tokens = setting(reference[len("env://") :])
            else:
                from agent_utilities.security.secrets_client import (
                    create_secrets_client,
                )

                raw_tokens = create_secrets_client().resolve_ref(reference)
            token_map = _json.loads(str(raw_tokens or ""))
            if not isinstance(token_map, dict) or not token_map:
                raise ValueError("empty token map")
            validated: dict[str, dict[str, Any]] = {}
            for token, claims in token_map.items():
                if not isinstance(token, str) or len(token) < 32 or len(token) > 4096:
                    raise ValueError("invalid token")
                if not isinstance(claims, dict):
                    raise ValueError("invalid claims")
                client_id = claims.get("client_id")
                scopes = claims.get("scopes", [])
                if not isinstance(client_id, str) or not 1 <= len(client_id) <= 256:
                    raise ValueError("invalid client id")
                if not isinstance(scopes, list) or not all(
                    isinstance(scope, str) and 1 <= len(scope) <= 256
                    for scope in scopes
                ):
                    raise ValueError("invalid scopes")
                expires_at = claims.get("expires_at")
                if expires_at is not None:
                    import math

                    if (
                        isinstance(expires_at, bool)
                        or not isinstance(expires_at, (int, float))
                        or not math.isfinite(float(expires_at))
                    ):
                        raise ValueError("invalid expiry")
                validated[token] = {
                    "client_id": client_id,
                    "scopes": scopes,
                    **({"expires_at": expires_at} if expires_at is not None else {}),
                }
        except Exception as exc:
            logger.error(
                "Error: static authentication token reference is unavailable or invalid (%s)",
                type(exc).__name__,
            )
            _sys.exit(1)
        import hashlib as _hashlib
        import hmac as _hmac
        import secrets as _secrets
        import time as _time

        from fastmcp.server.auth import AccessToken, TokenVerifier

        class _ConstantTimeStaticVerifier(TokenVerifier):
            def __init__(self, tokens: dict[str, dict[str, Any]]) -> None:
                super().__init__()
                self._key = _secrets.token_bytes(32)
                self._entries = [
                    (
                        _hmac.new(
                            self._key,
                            token.encode("utf-8"),
                            _hashlib.sha256,
                        ).digest(),
                        dict(claims),
                    )
                    for token, claims in tokens.items()
                ]

            async def verify_token(self, token: str) -> Any:
                if not isinstance(token, str) or len(token) > 4_096:
                    return None
                candidate = _hmac.new(
                    self._key,
                    token.encode("utf-8"),
                    _hashlib.sha256,
                ).digest()
                matched: dict[str, Any] | None = None
                for expected, claims in self._entries:
                    if _secrets.compare_digest(candidate, expected):
                        matched = claims
                if matched is None:
                    return None
                expires_at = matched.get("expires_at")
                if expires_at is not None and float(expires_at) < _time.time():
                    return None
                return AccessToken(
                    token=token,
                    client_id=matched["client_id"],
                    scopes=list(matched.get("scopes", [])),
                    expires_at=int(expires_at) if expires_at is not None else None,
                    claims=matched,
                )

        return _ConstantTimeStaticVerifier(validated)
    elif args.auth_type == "jwt":
        return _configure_jwt_auth(args)
    elif args.auth_type == "oauth-proxy":
        if not all(
            [
                args.oauth_upstream_auth_endpoint,
                args.oauth_upstream_token_endpoint,
                args.oauth_upstream_client_id,
                args.oauth_upstream_client_secret,
                args.oauth_base_url,
                args.token_jwks_uri,
                args.token_issuer,
                args.token_audience,
            ]
        ):
            logger.error(
                "Error: oauth-proxy requires all upstream endpoints and JWT params"
            )
            _sys.exit(1)
        try:
            upstream_auth = _secure_auth_url(
                args.oauth_upstream_auth_endpoint,
                field="OAuth authorization endpoint",
            )
            upstream_token = _secure_auth_url(
                args.oauth_upstream_token_endpoint,
                field="OAuth token endpoint",
            )
            base_url = _secure_auth_url(args.oauth_base_url, field="OAuth base URL")
        except ValueError:
            logger.error("Error: OAuth proxy URL policy is invalid")
            _sys.exit(1)
        token_verifier = _hardened_jwt_verifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        return OAuthProxy(
            upstream_authorization_endpoint=upstream_auth,
            upstream_token_endpoint=upstream_token,
            upstream_client_id=args.oauth_upstream_client_id,
            upstream_client_secret=args.oauth_upstream_client_secret,
            token_verifier=token_verifier,
            base_url=base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "oidc-proxy":
        if not all(
            [
                args.oidc_config_url,
                args.oidc_client_id,
                args.oidc_client_secret,
                args.oidc_base_url,
            ]
        ):
            logger.error(
                "Error: oidc-proxy requires OIDC metadata, client identity, "
                "a resolved client-secret reference, and base URL"
            )
            _sys.exit(1)
        inbound_audience = args.token_audience or args.audience
        if not inbound_audience:
            logger.error("Error: oidc-proxy requires an explicit token audience")
            _sys.exit(1)
        try:
            config_url = _secure_auth_url(
                args.oidc_config_url, field="OIDC configuration URL"
            )
            base_url = _secure_auth_url(args.oidc_base_url, field="OIDC base URL")
        except ValueError:
            logger.error("Error: OIDC proxy URL policy is invalid")
            _sys.exit(1)
        return OIDCProxy(
            config_url=config_url,
            client_id=args.oidc_client_id,
            client_secret=args.oidc_client_secret,
            audience=inbound_audience,
            base_url=base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "remote-oauth":
        if not all(
            [
                args.remote_auth_servers,
                args.remote_base_url,
                args.token_jwks_uri,
                args.token_issuer,
                args.token_audience,
            ]
        ):
            logger.error(
                "Error: remote-oauth requires remote-auth-servers, remote-base-url, and JWT params"
            )
            _sys.exit(1)
        try:
            auth_servers = [
                _secure_auth_url(url.strip(), field="authorization server")
                for url in args.remote_auth_servers.split(",")
                if url.strip()
            ]
            remote_base_url = _secure_auth_url(
                args.remote_base_url, field="remote OAuth base URL"
            )
        except ValueError:
            logger.error("Error: remote OAuth URL policy is invalid")
            _sys.exit(1)
        if not auth_servers or len(auth_servers) > 16:
            logger.error("Error: remote authorization-server policy is invalid")
            _sys.exit(1)
        token_verifier = _hardened_jwt_verifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        return RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=auth_servers,
            base_url=remote_base_url,
        )
    return None


def _configure_jwt_auth(args: argparse.Namespace) -> Any:
    """Configure JWT authentication from CLI args."""
    import sys as _sys

    jwks_uri = args.token_jwks_uri or setting("FASTMCP_SERVER_AUTH_JWT_JWKS_URI")
    # OIDC_ISSUER is the canonical, provider-agnostic var; FASTMCP_SERVER_AUTH_JWT_ISSUER
    # is the per-server alias. Either may be a comma-separated multi-issuer list (OS-5.45).
    issuer = (
        setting("OIDC_ISSUER")
        or args.token_issuer
        or setting("FASTMCP_SERVER_AUTH_JWT_ISSUER")
    )
    audience = args.token_audience or setting("FASTMCP_SERVER_AUTH_JWT_AUDIENCE")
    algorithm = args.token_algorithm
    secret_or_key = args.token_public_key
    public_key_pem = None

    # CONCEPT:AU-OS.identity.resolve-token-endpoint-from — IdP-agnostic: with no explicit JWKS URI (and not a static key),
    # resolve it from each issuer's OIDC discovery document, so config carries only the
    # issuer (Keycloak, Okta, Auth0, Entra, …) — never a vendor-specific path.
    if not jwks_uri and not secret_or_key and not args.token_secret_ref and issuer:
        from agent_utilities.security.oidc_discovery import jwks_uri_for

        _resolved = [
            jwks_uri_for(i.strip()) for i in str(issuer).split(",") if i.strip()
        ]
        if _resolved and all(_resolved):
            jwks_uri = ",".join(u for u in _resolved if u)
        elif any(_resolved):
            logger.warning(
                "OIDC discovery resolved JWKS for only part of the issuer policy; "
                "configure the JWKS policy explicitly"
            )

    if not (jwks_uri or secret_or_key or args.token_secret_ref):
        logger.error(
            "Error: JWT auth requires --token-jwks-uri, --token-public-key, "
            "or --token-secret-ref"
        )
        _sys.exit(1)
    if not (issuer and audience):
        logger.error("Error: JWT requires --token-issuer and --token-audience")
        _sys.exit(1)
    if (
        not isinstance(audience, str)
        or not 1 <= len(audience) <= 512
        or any(character in audience for character in "\r\n\x00")
    ):
        logger.error("Error: JWT audience is invalid")
        _sys.exit(1)

    try:
        issuer_values = [
            _secure_auth_url(value.strip(), field="JWT issuer")
            for value in str(issuer).split(",")
            if value.strip()
        ]
        if not issuer_values:
            raise ValueError("missing issuer")
        issuer = ",".join(issuer_values)
        if jwks_uri:
            jwks_values = [
                _secure_auth_url(value.strip(), field="JWKS URI")
                for value in str(jwks_uri).split(",")
                if value.strip()
            ]
            if not jwks_values:
                raise ValueError("missing JWKS URI")
            jwks_uri = ",".join(jwks_values)
    except ValueError:
        logger.error("Error: JWT issuer/JWKS URL policy is invalid")
        _sys.exit(1)

    if args.token_public_key and os.path.isfile(args.token_public_key):
        try:
            with open(args.token_public_key) as f:
                public_key_pem = f.read()
        except Exception as exc:
            logger.error(
                "Failed to read public key file (exception_type=%s)",
                type(exc).__name__,
            )
            _sys.exit(1)
    elif args.token_public_key:
        public_key_pem = args.token_public_key

    if algorithm and algorithm.startswith("HS"):
        secret_ref = str(getattr(args, "token_secret_ref", "") or "").strip()
        if not secret_ref:
            logger.error("Error: HMAC JWT verification requires --token-secret-ref")
            _sys.exit(1)
        try:
            from agent_utilities.security.secrets_client import create_secrets_client

            public_key = create_secrets_client().resolve_ref(secret_ref)
            if not isinstance(public_key, str) or len(public_key.encode()) < 32:
                raise ValueError("weak secret")
        except Exception:
            logger.error("Error: HMAC JWT secret reference is unavailable")
            _sys.exit(1)
    else:
        public_key = public_key_pem

    required_scopes = None
    if args.required_scopes:
        required_scopes = [
            s.strip() for s in args.required_scopes.split(",") if s.strip()
        ]

    # CONCEPT:AU-OS.identity.native-multi-realm-jwt — native multi-realm JWT trust. FASTMCP_SERVER_AUTH_JWT_ISSUER and
    # _JWKS_URI may each be a comma-separated, aligned-by-index list (one Keycloak realm per
    # entry). FastMCP's JWTVerifier accepts an issuer list but only a SINGLE jwks_uri (one
    # realm's signing keys), so during a realm migration the fleet must trust two realms'
    # signing keys at once. Build one JWTVerifier per realm and accept a token if ANY verifies
    # it — enabling a zero-downtime issuer cutover (add new realm → flip the minter → drop the
    # old realm) with no signature gap and no auth lock-out window. A single value (the common
    # case) falls through unchanged to the single-verifier path below.
    _issuers = (
        [s.strip() for s in str(issuer).split(",") if s.strip()] if issuer else []
    )
    _jwks_uris = (
        [s.strip() for s in str(jwks_uri).split(",") if s.strip()] if jwks_uri else []
    )
    if len(_jwks_uris) > 1 or len(_issuers) > 1:
        if len(_jwks_uris) != len(_issuers):
            logger.error(
                "Multi-realm JWT auth requires FASTMCP_SERVER_AUTH_JWT_ISSUER and "
                "FASTMCP_SERVER_AUTH_JWT_JWKS_URI to be comma-separated lists of EQUAL length "
                "(one entry per realm, aligned by index)."
            )
            _sys.exit(1)
        from fastmcp.server.auth import TokenVerifier

        class _MultiIssuerVerifier(TokenVerifier):
            """Accept a JWT validated by ANY wrapped per-realm JWTVerifier (CONCEPT:AU-OS.identity.native-multi-realm-jwt)."""

            def __init__(self, verifiers: list, *, required_scopes: Any = None) -> None:
                super().__init__(required_scopes=required_scopes)
                self._verifiers = list(verifiers)

            async def verify_token(self, token: str) -> Any:
                for _v in self._verifiers:
                    try:
                        _result = await _v.verify_token(token)
                    except Exception:  # noqa: BLE001 - try the next realm
                        _result = None
                    if _result is not None:
                        return _result
                return None

        try:
            _verifiers = [
                _hardened_jwt_verifier(
                    jwks_uri=_u,
                    issuer=_i,
                    audience=audience,
                    algorithm=algorithm or "RS256",
                    required_scopes=required_scopes,
                )
                for _u, _i in zip(_jwks_uris, _issuers, strict=False)
            ]
            logger.info("JWT auth: native multi-issuer trust configured")
            return _MultiIssuerVerifier(_verifiers, required_scopes=required_scopes)
        except Exception as exc:
            logger.error(
                "Failed to initialize multi-realm JWTVerifier (exception_type=%s)",
                type(exc).__name__,
            )
            _sys.exit(1)

    try:
        return _hardened_jwt_verifier(
            jwks_uri=jwks_uri,
            public_key=public_key,
            issuer=issuer,
            audience=audience,
            algorithm=algorithm or "RS256",
            required_scopes=required_scopes,
        )
    except Exception as exc:
        logger.error(
            "Failed to initialize JWTVerifier (exception_type=%s)",
            type(exc).__name__,
        )
        _sys.exit(1)


def _configure_middleware(args: argparse.Namespace) -> list[Any]:
    """Build the standard middleware stack for an MCP server."""
    from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
    from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

    try:
        from agent_utilities.mcp.middlewares import (
            ActorContextMiddleware,
            EntityLinkingMiddleware,
            JWTClaimsLoggingMiddleware,
            ToolMetricsMiddleware,
            UserTokenMiddleware,
        )
    except ImportError:
        UserTokenMiddleware = None  # type: ignore
        JWTClaimsLoggingMiddleware = None  # type: ignore
        EntityLinkingMiddleware = None  # type: ignore
        ToolMetricsMiddleware = None  # type: ignore
        ActorContextMiddleware = None  # type: ignore

    middlewares: list[Any] = [
        ErrorHandlingMiddleware(include_traceback=False, transform_errors=True),
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20),
    ]

    # Scope every tool call to the caller's validated OIDC identity so
    # servers can auto-load resources and inherit authz per-caller
    # (CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance). No-op when the
    # request carries no validated token (loopback/stdio trust only).
    if ActorContextMiddleware is not None:
        middlewares.append(ActorContextMiddleware())

    # Per-tool Prometheus metrics (count/latency/error) for this server, scraped
    # from its own /metrics route (CONCEPT:AU-OS.observability.no-op-without-metrics). No-op without the metrics extra.
    if ToolMetricsMiddleware is not None:
        middlewares.append(ToolMetricsMiddleware())

    if JWTClaimsLoggingMiddleware is not None:
        pass  # Also do not add this as it logs to stdout

    if EntityLinkingMiddleware is not None:
        middlewares.append(EntityLinkingMiddleware())

    if mcp_auth_config["enable_delegation"]:
        if UserTokenMiddleware is not None:
            # Keep the privacy-safe error middleware outermost so a delegation
            # rejection cannot bypass the standardized exception surface.
            middlewares.insert(1, UserTokenMiddleware(config=mcp_auth_config))

    if args.eunomia_type in ["embedded", "remote"]:
        try:
            from agent_utilities.mcp.eunomia_principal import (
                create_eunomia_middleware,
            )

            require_verified = str(getattr(args, "auth_type", "none")) != "none"
            if args.eunomia_type == "remote":
                eunomia_mw = create_eunomia_middleware(
                    policy_file=None,
                    use_remote_eunomia=True,
                    eunomia_endpoint=args.eunomia_remote_url,
                    api_key_ref=getattr(args, "eunomia_api_key_ref", None),
                    require_verified_principal=require_verified,
                )
            else:
                policy_file = args.eunomia_policy_file or "mcp_policies.json"
                eunomia_mw = create_eunomia_middleware(
                    policy_file=policy_file,
                    use_remote_eunomia=False,
                    require_verified_principal=require_verified,
                )
            middlewares.append(eunomia_mw)
        except Exception as exc:
            logger.error(
                "Failed to load Eunomia middleware (exception_type=%s)",
                type(exc).__name__,
            )
            import sys

            sys.exit(1)

    return middlewares


_STDIO_PROTECTED = False


def protect_stdio_jsonrpc() -> None:
    """Redirect ``print``/warnings to stderr for a process serving MCP over stdio.

    On the stdio transport the JSON-RPC framing IS stdout, so any stray ``print``
    corrupts the protocol. Call this immediately before ``mcp.run(transport="stdio")``
    — NOT at server-build time — because it permanently monkeypatches the
    process-global ``builtins.print``; doing it for a process that does not actually
    own stdout (tests, embedded/sse/http hosts) silently breaks stdout capture
    everywhere downstream. Idempotent.
    """
    global _STDIO_PROTECTED
    if _STDIO_PROTECTED:
        return
    _STDIO_PROTECTED = True

    import builtins
    import warnings

    original_print = builtins.print

    def stderr_print(*print_args, **kwargs):
        if (
            "file" not in kwargs
            or kwargs["file"] is None
            or kwargs["file"] == sys.stdout
        ):
            kwargs["file"] = sys.stderr
        original_print(*print_args, **kwargs)

    builtins.print = stderr_print

    def stderr_showwarning(message, category, filename, lineno, file=None, line=None):
        sys.stderr.write(f"[{category.__name__}] {message} ({filename}:{lineno})\n")
        sys.stderr.flush()

    warnings.showwarning = stderr_showwarning


def mcp_network_run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return the one hardened FastMCP/Uvicorn network-serving configuration."""
    from starlette.middleware import Middleware

    from agent_utilities.security.http_boundary import (
        BoundedRequestBodyMiddleware,
        ExactHostAuthorityMiddleware,
        OriginPolicyMiddleware,
        TrustedProxyPeerMiddleware,
        normalize_host_authorities,
        normalize_origins,
        parse_cidrs,
    )

    transport = str(getattr(args, "transport", "stdio") or "stdio").lower()
    if transport not in _NETWORK_TRANSPORTS:
        return {}
    host = str(getattr(args, "host", "") or "")
    loopback = _is_loopback_bind(host)
    hosts = [
        value.strip()
        for value in str(getattr(args, "allowed_hosts", "") or "").split(",")
        if value.strip()
    ]
    if not hosts:
        if not loopback:
            raise RuntimeError("MCP_ALLOWED_HOSTS is required")
        port = int(getattr(args, "port", 8000))
        hosts = [
            f"localhost:{port}",
            f"127.0.0.1:{port}",
            f"[::1]:{port}",
            "testserver",
        ]
    try:
        hosts = sorted(normalize_host_authorities(hosts))
    except ValueError:
        raise RuntimeError("MCP_ALLOWED_HOSTS must contain exact authorities") from None

    origin_values = [
        value.strip()
        for value in str(getattr(args, "allowed_origins", "") or "").split(",")
        if value.strip()
    ]
    normalize_origins(origin_values)  # validate before the listener starts

    max_bytes = int(getattr(args, "max_request_bytes", 4 * 1024 * 1024))
    middleware = [
        Middleware(ExactHostAuthorityMiddleware, allowed_hosts=hosts),
        Middleware(OriginPolicyMiddleware, allowed_origins=origin_values),
        Middleware(BoundedRequestBodyMiddleware, max_bytes=max_bytes),
    ]
    if bool(getattr(args, "tls_terminated", False)):
        cidrs = [
            value.strip()
            for value in str(getattr(args, "trusted_proxy_cidrs", "") or "").split(",")
            if value.strip()
        ]
        parse_cidrs(cidrs)
        middleware.insert(
            0, Middleware(TrustedProxyPeerMiddleware, trusted_cidrs=cidrs)
        )

    max_connections = int(setting("MCP_MAX_CONNECTIONS", "128"))
    backlog = int(setting("MCP_LISTEN_BACKLOG", "256"))
    if not 1 <= max_connections <= 10_000 or not 1 <= backlog <= 65_535:
        raise RuntimeError("MCP listener resource bounds are invalid")
    uvicorn_config: dict[str, Any] = {
        # Never trust Forwarded/X-Forwarded-* from the network. The direct peer
        # address remains authoritative for the trusted-ingress CIDR gate.
        "proxy_headers": False,
        "timeout_keep_alive": 5,
        "timeout_graceful_shutdown": 15,
        "limit_concurrency": max_connections,
        "backlog": backlog,
        "h11_max_incomplete_event_size": 65_536,
    }
    certfile = str(getattr(args, "tls_certfile", "") or "").strip()
    keyfile = str(getattr(args, "tls_keyfile", "") or "").strip()
    if certfile and keyfile:
        uvicorn_config.update(ssl_certfile=certfile, ssl_keyfile=keyfile)
    return {"middleware": middleware, "uvicorn_config": uvicorn_config}


def create_mcp_server(
    name: str = "MCP Server",
    version: str = __version__,
    instructions: str = "",
    command_args: list[str] | None = None,
    transport_choices: tuple[str, ...] = _ALL_TRANSPORTS,
):
    """Initialize a FastMCP server with a standard middleware and auth stack.

    This helper consolidates the steps of creating a parser, configuring
    authentication providers (JWT, OIDC, etc.), and assembling standard
    middleware (Logging, Timing, Rate Limiting). It handles CLI flag
    parsing and will exit the process if help is requested or configuration
    is invalid.

    Args:
        name: The human-readable name of the MCP server.
        version: Semantic version string for the server.
        instructions: System instructions specific to this MCP server's
            tools, providing context for the LLM.
        command_args: Optional list of CLI arguments (default: sys.argv).
        transport_choices: Current transports exposed by this server.

    Returns:
        A tuple containing:
            - args: The parsed argparse.Namespace object.
            - mcp: The initialized FastMCP server instance.
            - middlewares: A list of configured middleware instances.

    """
    import logging
    import sys

    from fastmcp import FastMCP

    # Force all logging to stderr to prevent JSON-RPC corruption over stdio
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

    parser = create_mcp_parser(transport_choices=transport_choices)
    args, _ = parser.parse_known_args(command_args)
    if args.transport not in transport_choices:
        parser.error(
            f"transport {args.transport!r} is not supported by this server; "
            f"choose from {', '.join(transport_choices)}"
        )

    # NOTE: the stdout/print → stderr protection is applied by ``protect_stdio_jsonrpc()``
    # at the actual stdio *serve* sites (mcp.run(transport="stdio")), NOT here at build
    # time. Building a server does not dedicate the process to stdio, and applying the
    # permanent ``builtins.print`` override here broke stdout capture for every in-process
    # caller (tests, embedded/sse/http hosts) created afterwards.

    if hasattr(args, "help") and args.help:
        parser.print_help()
        import sys

        sys.exit(0)

    if args.port < 0 or args.port > 65535:
        logger.error(f"Error: Port {args.port} is out of valid range (0-65535).")
        import sys

        sys.exit(1)

    _validate_network_exposure(args)

    auth = _configure_auth(args)
    middlewares = _configure_middleware(args)

    remote_network = str(
        getattr(args, "transport", "stdio") or "stdio"
    ).lower() in _NETWORK_TRANSPORTS and not _is_loopback_bind(
        getattr(args, "host", "")
    )
    metrics_token: str | None = None
    metrics_ref = str(setting("MCP_METRICS_TOKEN_REF", "") or "").strip()
    if metrics_ref:
        try:
            if metrics_ref.startswith("env://"):
                metrics_token = str(setting(metrics_ref[len("env://") :], "") or "")
            else:
                from agent_utilities.security.secrets_client import (
                    create_secrets_client,
                )

                metrics_token = str(
                    create_secrets_client().resolve_ref(metrics_ref) or ""
                )
            if not 32 <= len(metrics_token) <= 4_096 or any(
                character in metrics_token for character in "\r\n\x00"
            ):
                metrics_token = None
        except Exception:
            metrics_token = None

    import os

    os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"
    mcp = FastMCP(name, version=version, auth=auth, instructions=instructions)

    # Operational routes live outside the tool authorization path. Health is a
    # generic readiness result. Metrics are local-only unless a remote listener
    # has a runtime-resolved bearer token; otherwise that route is absent.
    # Wrapped defensively so older FastMCP builds without custom_route still
    # produce a working server.
    try:
        from starlette.requests import Request as _Request
        from starlette.responses import JSONResponse as _JSONResponse
        from starlette.responses import Response as _Response

        from agent_utilities.observability.gateway_metrics import (
            render_metrics as _render_metrics,
        )

        if not remote_network or metrics_token is not None:

            @mcp.custom_route("/metrics", methods=["GET"])
            async def _metrics_route(request: _Request) -> _Response:
                if metrics_token is not None:
                    import secrets as _secrets

                    values = [
                        value.decode("latin-1")
                        for key, value in request.scope.get("headers", ())
                        if key.lower() == b"authorization"
                    ]
                    expected = f"Bearer {metrics_token}"
                    if len(values) != 1 or not _secrets.compare_digest(
                        values[0], expected
                    ):
                        return _Response(
                            status_code=401,
                            headers={
                                "WWW-Authenticate": "Bearer",
                                "Cache-Control": "no-store",
                            },
                        )
                body, content_type = _render_metrics()
                return _Response(
                    content=body,
                    media_type=content_type,
                    headers={"Cache-Control": "no-store"},
                )

        @mcp.custom_route("/health", methods=["GET"])
        async def _health_route(request: _Request) -> _JSONResponse:  # noqa: ARG001
            return _JSONResponse(
                {"status": "ok"}, headers={"Cache-Control": "no-store"}
            )

    except Exception as _route_exc:  # pragma: no cover - defensive
        logger.warning(
            "Could not register metrics and health routes (exception_type=%s)",
            type(_route_exc).__name__,
        )

    # Inject dynamic visibility transform for dynamic tag/tool filtering
    try:
        from fastmcp.server.transforms import Transform

        class DynamicVisibilityTransform(Transform):
            """Enforces environment-variable and header-based tag and tool filters dynamically across all components."""

            def _filter_components(self, components):
                enabled_tools_list = []
                disabled_tools_list = []
                enabled_tags_list = []
                disabled_tags_list = []
                query_filter = None
                reject_all = False

                # 1. Start with environment variable defaults
                enabled_tags_env = setting("MCP_ENABLED_TAGS")
                disabled_tags_env = setting("MCP_DISABLED_TAGS")
                enabled_components_env = setting("MCP_ENABLED_TOOLS")
                disabled_components_env = setting("MCP_DISABLED_TOOLS")

                if enabled_components_env:
                    enabled_tools_list.extend(
                        [
                            x.strip()
                            for x in enabled_components_env.split(",")
                            if x.strip()
                        ]
                    )
                if disabled_components_env:
                    disabled_tools_list.extend(
                        [
                            x.strip()
                            for x in disabled_components_env.split(",")
                            if x.strip()
                        ]
                    )
                if enabled_tags_env:
                    enabled_tags_list.extend(
                        [x.strip() for x in enabled_tags_env.split(",") if x.strip()]
                    )
                if disabled_tags_env:
                    disabled_tags_list.extend(
                        [x.strip() for x in disabled_tags_env.split(",") if x.strip()]
                    )

                # 1.5. Override/Append with parsed CLI args if specified
                cli_tools = getattr(args, "tools", None)
                if cli_tools:
                    enabled_tools_list = [
                        x.strip() for x in cli_tools.split(",") if x.strip()
                    ]
                cli_disabled_tools = getattr(args, "disabled_tools", None)
                if cli_disabled_tools:
                    disabled_tools_list = [
                        x.strip() for x in cli_disabled_tools.split(",") if x.strip()
                    ]

                # 2. Extract request query parameters and headers if HTTP/SSE transport is active
                try:
                    req = _optional_http_request()
                    if req:
                        # Extract query parameters (supporting list getlist syntax and CSV syntax)
                        q_params = req.query_params

                        # Gather all enabled tools/toolsets from query params
                        q_tools = []
                        for key in ["tools", "toolsets"]:
                            if hasattr(q_params, "getlist"):
                                vals = q_params.getlist(key)
                            else:
                                vals = (
                                    [q_params.get(key) or ""]
                                    if q_params.get(key)
                                    else []
                                )
                            for val in vals:
                                if val:
                                    q_tools.extend(
                                        [x.strip() for x in val.split(",") if x.strip()]
                                    )
                        if q_tools:
                            enabled_tools_list = _narrow_values(
                                enabled_tools_list,
                                _bounded_filter_values(q_tools),
                            )

                        # Gather all disabled tools/toolsets from query params
                        q_disabled = []
                        for key in ["disabled_tools", "disabled_toolsets"]:
                            if hasattr(q_params, "getlist"):
                                vals = q_params.getlist(key)
                            else:
                                vals = (
                                    [q_params.get(key) or ""]
                                    if q_params.get(key)
                                    else []
                                )
                            for val in vals:
                                if val:
                                    q_disabled.extend(
                                        [x.strip() for x in val.split(",") if x.strip()]
                                    )
                        if q_disabled:
                            disabled_tools_list = _union_values(
                                disabled_tools_list,
                                _bounded_filter_values(q_disabled),
                            )

                        # Gather all enabled tags from query params
                        q_tags = []
                        if hasattr(q_params, "getlist"):
                            vals = q_params.getlist("tags")
                        else:
                            vals = (
                                [q_params.get("tags") or ""]
                                if q_params.get("tags")
                                else []
                            )
                        for val in vals:
                            if val:
                                q_tags.extend(
                                    [x.strip() for x in val.split(",") if x.strip()]
                                )
                        if q_tags:
                            enabled_tags_list = _narrow_values(
                                enabled_tags_list,
                                _bounded_filter_values(q_tags),
                            )

                        # Gather all disabled tags from query params
                        q_disabled_tags = []
                        if hasattr(q_params, "getlist"):
                            vals = q_params.getlist("disabled_tags")
                        else:
                            vals = (
                                [q_params.get("disabled_tags") or ""]
                                if q_params.get("disabled_tags")
                                else []
                            )
                        for val in vals:
                            if val:
                                q_disabled_tags.extend(
                                    [x.strip() for x in val.split(",") if x.strip()]
                                )
                        if q_disabled_tags:
                            disabled_tags_list = _union_values(
                                disabled_tags_list,
                                _bounded_filter_values(q_disabled_tags),
                            )

                        # Gather query/search keyword from query params (e.g. q=dns or query=dns)
                        for key in ["q", "query", "search"]:
                            if q_params.get(key):
                                query_filter = q_params.get(key)
                                break

                        # Extract request headers (headers take top precedence)
                        headers = req.headers
                        if headers:
                            # Gather all enabled tools/components from headers
                            h_tools = []
                            for key in [
                                "x-mcp-enabled-tools",
                                "x-mcp-enabled-components",
                            ]:
                                if hasattr(headers, "getlist"):
                                    vals = headers.getlist(key)
                                else:
                                    vals = (
                                        [headers.get(key) or ""]
                                        if headers.get(key)
                                        else []
                                    )
                                for val in vals:
                                    if val:
                                        h_tools.extend(
                                            [
                                                x.strip()
                                                for x in val.split(",")
                                                if x.strip()
                                            ]
                                        )
                            if h_tools:
                                enabled_tools_list = _narrow_values(
                                    enabled_tools_list,
                                    _bounded_filter_values(h_tools),
                                )

                            # Gather all disabled tools/components from headers
                            h_disabled = []
                            for key in [
                                "x-mcp-disabled-tools",
                                "x-mcp-disabled-components",
                            ]:
                                if hasattr(headers, "getlist"):
                                    vals = headers.getlist(key)
                                else:
                                    vals = (
                                        [headers.get(key) or ""]
                                        if headers.get(key)
                                        else []
                                    )
                                for val in vals:
                                    if val:
                                        h_disabled.extend(
                                            [
                                                x.strip()
                                                for x in val.split(",")
                                                if x.strip()
                                            ]
                                        )
                            if h_disabled:
                                disabled_tools_list = _union_values(
                                    disabled_tools_list,
                                    _bounded_filter_values(h_disabled),
                                )

                            # Gather all enabled tags from headers
                            h_tags = []
                            if hasattr(headers, "getlist"):
                                vals = headers.getlist("x-mcp-enabled-tags")
                            else:
                                vals = (
                                    [headers.get("x-mcp-enabled-tags") or ""]
                                    if headers.get("x-mcp-enabled-tags")
                                    else []
                                )
                            for val in vals:
                                if val:
                                    h_tags.extend(
                                        [x.strip() for x in val.split(",") if x.strip()]
                                    )
                            if h_tags:
                                enabled_tags_list = _narrow_values(
                                    enabled_tags_list,
                                    _bounded_filter_values(h_tags),
                                )

                            # Gather all disabled tags from headers
                            h_disabled_tags = []
                            if hasattr(headers, "getlist"):
                                vals = headers.getlist("x-mcp-disabled-tags")
                            else:
                                vals = (
                                    [headers.get("x-mcp-disabled-tags") or ""]
                                    if headers.get("x-mcp-disabled-tags")
                                    else []
                                )
                            for val in vals:
                                if val:
                                    h_disabled_tags.extend(
                                        [x.strip() for x in val.split(",") if x.strip()]
                                    )
                            if h_disabled_tags:
                                disabled_tags_list = _union_values(
                                    disabled_tags_list,
                                    _bounded_filter_values(h_disabled_tags),
                                )

                            # Gather query/search keyword from headers
                            for key in ["x-mcp-query", "x-mcp-search"]:
                                if headers.get(key):
                                    query_filter = headers.get(key)
                                    break
                except Exception as exc:
                    # A malformed client filter must never widen the surface or
                    # silently fall back to the server's full component set.
                    logger.warning(
                        "Rejected MCP visibility filter (exception_type=%s)",
                        type(exc).__name__,
                    )
                    reject_all = True

                # A requested semantic filter is a least-privilege boundary. No
                # match, no active graph, or any resolution failure exposes no
                # components; it must never widen back to the configured surface.
                if query_filter and not reject_all:
                    try:
                        from agent_utilities.knowledge_graph.core.engine import (
                            IntelligenceGraphEngine,
                        )
                        from agent_utilities.tools.dynamic_tool_orchestrator import (
                            DynamicToolOrchestrator,
                        )

                        engine = IntelligenceGraphEngine.get_active()
                        kg_matched: list[str] = []
                        if engine:
                            orchestrator = DynamicToolOrchestrator(engine)
                            kg_matched = list(
                                orchestrator.resolve_mcp_tools(
                                    query_filter, server_name=name
                                )
                            )
                        if not kg_matched:
                            reject_all = True
                        elif enabled_tools_list and "all" not in enabled_tools_list:
                            enabled_tools_list = [
                                tool
                                for tool in enabled_tools_list
                                if tool in kg_matched
                            ]
                            if not enabled_tools_list:
                                reject_all = True
                        else:
                            enabled_tools_list = kg_matched
                    except Exception as exc:
                        logger.debug(
                            "Failed to filter components using Knowledge Graph "
                            "(exception_type=%s)",
                            type(exc).__name__,
                        )
                        reject_all = True

                if reject_all:
                    return []

                # 3. Convert lists to sets
                enabled_tags = set(enabled_tags_list) if enabled_tags_list else None
                disabled_tags = set(disabled_tags_list) if disabled_tags_list else None
                enabled_names = set(enabled_tools_list) if enabled_tools_list else None
                disabled_names = (
                    set(disabled_tools_list) if disabled_tools_list else None
                )

                # Fallback: if enabled_names contains "all" or is empty, expose all tools
                if enabled_names is not None:
                    if "all" in enabled_names or not enabled_names:
                        enabled_names = None

                filtered = []
                for c in components:
                    c_name = getattr(c, "name", None) or getattr(c, "uri", None)
                    if not c_name:
                        filtered.append(c)
                        continue

                    if enabled_names is not None and c_name not in enabled_names:
                        continue
                    if disabled_names is not None and c_name in disabled_names:
                        continue

                    if hasattr(c, "tags") and c.tags:
                        if enabled_tags is not None:
                            if not (c.tags & enabled_tags):
                                continue
                        if disabled_tags is not None:
                            if c.tags & disabled_tags:
                                continue
                    else:
                        if enabled_tags is not None:
                            continue

                    filtered.append(c)
                return filtered

            async def list_tools(self, tools):
                return self._filter_components(tools)

            async def get_tool(self, name, call_next, *, version=None):
                tool = await call_next(name, version=version)
                if tool is None:
                    return None
                filtered = self._filter_components([tool])
                return filtered[0] if filtered else None

            async def list_resources(self, resources):
                return self._filter_components(resources)

            async def get_resource(self, uri, call_next, *, version=None):
                res = await call_next(uri, version=version)
                if res is None:
                    return None
                filtered = self._filter_components([res])
                return filtered[0] if filtered else None

            async def list_resource_templates(self, templates):
                return self._filter_components(templates)

            async def get_resource_template(self, uri, call_next, *, version=None):
                tmpl = await call_next(uri, version=version)
                if tmpl is None:
                    return None
                filtered = self._filter_components([tmpl])
                return filtered[0] if filtered else None

            async def list_prompts(self, prompts):
                return self._filter_components(prompts)

            async def get_prompt(self, name, call_next, *, version=None):
                prompt = await call_next(name, version=version)
                if prompt is None:
                    return None
                filtered = self._filter_components([prompt])
                return filtered[0] if filtered else None

        mcp.add_transform(DynamicVisibilityTransform())
    except Exception as exc:
        logger.warning(
            "Could not register dynamic visibility transform (exception_type=%s)",
            type(exc).__name__,
        )

    return args, mcp, middlewares
