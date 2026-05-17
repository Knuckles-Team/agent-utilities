from __future__ import annotations

"""MCP Server Factory.

Handles CLI argument parsing for MCP servers, automated server initialization
with middleware stacks, and the ``create_mcp_server`` convenience constructor.

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""


import argparse
import logging
import os
from typing import Any

from agent_utilities.base_utilities import GET_DEFAULT_SSL_VERIFY, to_boolean
from agent_utilities.core.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
)

logger = logging.getLogger(__name__)


# MCP-specific auth/delegation config (not in config.py)
mcp_auth_config = {
    "enable_delegation": to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
    "audience": os.environ.get("AUDIENCE", None),
    "delegated_scopes": os.environ.get("DELEGATED_SCOPES", "api"),
    "token_endpoint": None,
    "oidc_client_id": os.environ.get("OIDC_CLIENT_ID", None),
    "oidc_client_secret": os.environ.get("OIDC_CLIENT_SECRET", None),
    "oidc_config_url": os.environ.get("OIDC_CONFIG_URL", None),
    "jwt_jwks_uri": os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI", None),
    "jwt_issuer": os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER", None),
    "jwt_audience": os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE", None),
    "jwt_algorithm": os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM", None),
    "jwt_secret": os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY", None),
    "jwt_required_scopes": os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES", None),
}  # nosec B105

DEFAULT_TRANSPORT = os.environ.get("TRANSPORT", "stdio")
DEFAULT_SSL_VERIFY = GET_DEFAULT_SSL_VERIFY()
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER") or os.getenv("PROVIDER") or "openai"
DEFAULT_LLM_MODEL_ID = (
    os.getenv("LLM_MODEL_ID")
    or os.getenv("MODEL_ID")
    or "text-embedding-nomic-embed-text-v2-moe"
)
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "llama")

__version__ = "0.3.0"


def create_mcp_parser() -> argparse.ArgumentParser:
    """Create a standard argument parser for MCP servers.

    Defines a comprehensive set of CLI flags for transport selection (stdio,
    sse, http), host/port configuration, authentication (JWT, OIDC, OAuth),
    and Eunomia policy enforcement.

    Returns:
        An argparse.ArgumentParser instance.

    """
    parser = argparse.ArgumentParser(add_help=False, description="MCP Server")
    parser.add_argument(
        "-t",
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=["stdio", "streamable-http", "sse"],
        help="Transport method: 'stdio', 'streamable-http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-H",
        "--host",
        default=DEFAULT_HOST,
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--auth-type",
        default="none",
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    parser.add_argument(
        "--token-jwks-uri", default=None, help="JWKS URI for JWT verification"
    )
    parser.add_argument(
        "--token-issuer", default=None, help="Issuer for JWT verification"
    )
    parser.add_argument(
        "--token-audience", default=None, help="Audience for JWT verification"
    )
    parser.add_argument(
        "--token-algorithm",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_ALGORITHM"),
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
        "--token-secret",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Shared secret for HMAC (HS*) or PEM public key for static asymmetric verification.",
    )
    parser.add_argument(
        "--token-public-key",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_PUBLIC_KEY"),
        help="Path to PEM public key file or inline PEM string (for static asymmetric keys).",
    )
    parser.add_argument(
        "--required-scopes",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_REQUIRED_SCOPES"),
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
        "--oauth-upstream-client-secret",
        default=None,
        help="Upstream client secret for OAuth Proxy",
    )
    parser.add_argument(
        "--oauth-base-url", default=None, help="Base URL for OAuth Proxy"
    )
    parser.add_argument(
        "--oidc-config-url", default=None, help="OIDC configuration URL"
    )
    parser.add_argument("--oidc-client-id", default=None, help="OIDC client ID")
    parser.add_argument("--oidc-client-secret", default=None, help="OIDC client secret")
    parser.add_argument("--oidc-base-url", default=None, help="Base URL for OIDC Proxy")
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
        default="none",
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default="mcp_policies.json",
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url", default=None, help="URL for remote Eunomia server"
    )
    parser.add_argument(
        "--enable-delegation",
        action="store_true",
        default=to_boolean(os.environ.get("ENABLE_DELEGATION", "False")),
        help="Enable OIDC token delegation",
    )
    parser.add_argument(
        "--audience",
        default=os.environ.get("AUDIENCE", None),
        help="Audience for the delegated token",
    )
    parser.add_argument(
        "--delegated-scopes",
        default=os.environ.get("DELEGATED_SCOPES", "api"),
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
        default=os.getenv("OPENAPI_USERNAME"),
        help="Username for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-password",
        default=os.getenv("OPENAPI_PASSWORD"),
        help="Password for basic auth during OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-id",
        default=os.getenv("OPENAPI_CLIENT_ID"),
        help="OAuth client ID for OpenAPI import",
    )

    parser.add_argument(
        "--openapi-client-secret",
        default=os.getenv("OPENAPI_CLIENT_SECRET"),
        help="OAuth client secret for OpenAPI import",
    )

    parser.add_argument("--help", action="store_true", help="Show usage")
    return parser


def _configure_auth(args: argparse.Namespace) -> Any:
    """Configure authentication provider based on parsed CLI args.

    Returns the auth provider instance or None.
    """
    import sys as _sys

    import requests as _requests
    from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
    from fastmcp.server.auth.oidc_proxy import OIDCProxy
    from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier

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
            oidc_config_resp = _requests.get(config_url, timeout=30)
            oidc_config_resp.raise_for_status()
            oidc_config = oidc_config_resp.json()
            mcp_auth_config["token_endpoint"] = oidc_config.get("token_endpoint")
            if not mcp_auth_config["token_endpoint"]:
                raise ValueError("No token_endpoint found in OIDC configuration")
        except Exception as e:
            logger.error(f"Failed to fetch OIDC configuration: {e}")
            _sys.exit(1)

    allowed_uris = (
        args.allowed_client_redirect_uris.split(",")
        if args.allowed_client_redirect_uris
        else None
    )

    if args.auth_type == "none":
        return None
    elif args.auth_type == "static":
        return StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
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
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        return OAuthProxy(
            upstream_authorization_endpoint=args.oauth_upstream_auth_endpoint,
            upstream_token_endpoint=args.oauth_upstream_token_endpoint,
            upstream_client_id=args.oauth_upstream_client_id,
            upstream_client_secret=args.oauth_upstream_client_secret,
            token_verifier=token_verifier,
            base_url=args.oauth_base_url,
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
                "Error: oidc-proxy requires oidc-config-url, oidc-client-id, oidc-client-secret, oidc-base-url"
            )
            _sys.exit(1)
        return OIDCProxy(
            config_url=args.oidc_config_url,
            client_id=args.oidc_client_id,
            client_secret=args.oidc_client_secret,
            base_url=args.oidc_base_url,
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
        auth_servers = [url.strip() for url in args.remote_auth_servers.split(",")]
        token_verifier = JWTVerifier(
            jwks_uri=args.token_jwks_uri,
            issuer=args.token_issuer,
            audience=args.token_audience,
        )
        return RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=auth_servers,
            base_url=args.remote_base_url,
        )
    return None


def _configure_jwt_auth(args: argparse.Namespace) -> Any:
    """Configure JWT authentication from CLI args."""
    import sys as _sys

    from fastmcp.server.auth.providers.jwt import JWTVerifier

    jwks_uri = args.token_jwks_uri or os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI")
    issuer = args.token_issuer or os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER")
    audience = args.token_audience or os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE")
    algorithm = args.token_algorithm
    secret_or_key = args.token_secret or args.token_public_key
    public_key_pem = None

    if not (jwks_uri or secret_or_key):
        logger.error(
            "Error: JWT auth requires either --token-jwks-uri or --token-secret/--token-public-key"
        )
        _sys.exit(1)
    if not (issuer and audience):
        logger.error("Error: JWT requires --token-issuer and --token-audience")
        _sys.exit(1)

    if args.token_public_key and os.path.isfile(args.token_public_key):
        try:
            with open(args.token_public_key) as f:
                public_key_pem = f.read()
        except Exception as e:
            logger.error(f"Failed to read public key file: {e}")
            _sys.exit(1)
    elif args.token_public_key:
        public_key_pem = args.token_public_key

    if algorithm and algorithm.startswith("HS"):
        if not secret_or_key:
            logger.error(f"Error: HMAC algorithm {algorithm} requires --token-secret")
            _sys.exit(1)
        public_key = secret_or_key
    else:
        public_key = public_key_pem

    required_scopes = None
    if args.required_scopes:
        required_scopes = [
            s.strip() for s in args.required_scopes.split(",") if s.strip()
        ]

    try:
        return JWTVerifier(
            jwks_uri=jwks_uri,
            public_key=public_key,
            issuer=issuer,
            audience=audience,
            algorithm=(algorithm if algorithm and algorithm.startswith("HS") else None),
            required_scopes=required_scopes,
        )
    except Exception as e:
        logger.error(f"Failed to initialize JWTVerifier: {e}")
        _sys.exit(1)


def _configure_middleware(args: argparse.Namespace) -> list[Any]:
    """Build the standard middleware stack for an MCP server."""
    from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
    from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

    try:
        from agent_utilities.mcp.middlewares import (
            EntityLinkingMiddleware,
            JWTClaimsLoggingMiddleware,
            UserTokenMiddleware,
        )
    except ImportError:
        UserTokenMiddleware = None  # type: ignore
        JWTClaimsLoggingMiddleware = None  # type: ignore
        EntityLinkingMiddleware = None  # type: ignore

    middlewares: list[Any] = [
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True),
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20),
    ]

    if JWTClaimsLoggingMiddleware is not None:
        pass  # Also do not add this as it logs to stdout

    if EntityLinkingMiddleware is not None:
        middlewares.append(EntityLinkingMiddleware())

    if mcp_auth_config["enable_delegation"] or args.auth_type == "jwt":
        if UserTokenMiddleware is not None:
            middlewares.insert(0, UserTokenMiddleware(config=mcp_auth_config))

    if args.eunomia_type in ["embedded", "remote"]:
        try:
            from eunomia_mcp import create_eunomia_middleware

            policy_file = args.eunomia_policy_file or "mcp_policies.json"
            eunomia_endpoint = (
                args.eunomia_remote_url if args.eunomia_type == "remote" else None
            )
            eunomia_mw = create_eunomia_middleware(
                policy_file=policy_file, eunomia_endpoint=eunomia_endpoint
            )
            middlewares.append(eunomia_mw)
        except Exception as e:
            logger.error(f"Failed to load Eunomia middleware: {e}")
            import sys

            sys.exit(1)

    return middlewares


def create_mcp_server(
    name: str = "MCP Server",
    version: str = "0.1.0",
    instructions: str = "",
    command_args: list[str] | None = None,
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

    parser = create_mcp_parser()
    args, _ = parser.parse_known_args(command_args)

    if hasattr(args, "help") and args.help:
        parser.print_help()
        import sys

        sys.exit(0)

    if args.port < 0 or args.port > 65535:
        logger.error(f"Error: Port {args.port} is out of valid range (0-65535).")
        import sys

        sys.exit(1)

    auth = _configure_auth(args)
    middlewares = _configure_middleware(args)

    import os

    os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"
    mcp = FastMCP(name, auth=auth, instructions=instructions)

    return args, mcp, middlewares
