from __future__ import annotations

"""MCP Server Factory.

Handles CLI argument parsing for MCP servers, automated server initialization
with middleware stacks, and the ``create_mcp_server`` convenience constructor.

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""


import argparse
import logging
import os
import sys
import warnings
from typing import Any

# Global suppression for Authlib deprecations to prevent standard output pollution
# that breaks JSON-RPC protocol parser and leads to "0 tools" or "EOF" errors
warnings.filterwarnings("ignore", category=DeprecationWarning, module="authlib")
warnings.filterwarnings("ignore", message=".*authlib.jose module is deprecated.*")

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

__version__ = "0.3.0"


def create_mcp_parser() -> argparse.ArgumentParser:
    """Create a standard argument parser for MCP servers.

    Defines a comprehensive set of CLI flags for transport selection (stdio,
    sse, http), host/port configuration, authentication (JWT, OIDC, OAuth),
    and Eunomia policy enforcement.

    Returns:
        An argparse.ArgumentParser instance.

    """
    # Keycloak defaults integration
    keycloak_url = os.environ.get("KEYCLOAK_URL")
    keycloak_realm = os.environ.get("KEYCLOAK_REALM", "master")
    default_oidc_config = os.environ.get("OIDC_CONFIG_URL")
    if not default_oidc_config and keycloak_url:
        default_oidc_config = (
            f"{keycloak_url}/realms/{keycloak_realm}/.well-known/openid-configuration"
        )

    default_oidc_client_id = os.environ.get("OIDC_CLIENT_ID") or os.environ.get(
        "KEYCLOAK_CLIENT_ID"
    )
    default_oidc_client_secret = os.environ.get("OIDC_CLIENT_SECRET") or os.environ.get(
        "KEYCLOAK_CLIENT_SECRET"
    )

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
        default=os.environ.get("AUTH_TYPE", "none"),
        choices=["none", "static", "jwt", "oauth-proxy", "oidc-proxy", "remote-oauth"],
        help="Authentication type for MCP server: 'none' (disabled), 'static' (internal), 'jwt' (external token verification), 'oauth-proxy', 'oidc-proxy', 'remote-oauth' (external) (default: none)",
    )
    parser.add_argument(
        "--token-jwks-uri",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_JWKS_URI"),
        help="JWKS URI for JWT verification",
    )
    parser.add_argument(
        "--token-issuer",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_ISSUER"),
        help="Issuer for JWT verification",
    )
    parser.add_argument(
        "--token-audience",
        default=os.getenv("FASTMCP_SERVER_AUTH_JWT_AUDIENCE"),
        help="Audience for JWT verification",
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
        "--oidc-client-secret",
        default=default_oidc_client_secret,
        help="OIDC client secret",
    )
    parser.add_argument(
        "--oidc-base-url",
        default=os.environ.get("OIDC_BASE_URL"),
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
        default=os.environ.get("EUNOMIA_TYPE", "none"),
        choices=["none", "embedded", "remote"],
        help="Eunomia authorization type: 'none' (disabled), 'embedded' (built-in), 'remote' (external) (default: none)",
    )
    parser.add_argument(
        "--eunomia-policy-file",
        default=os.environ.get("EUNOMIA_POLICY_FILE", "mcp_policies.json"),
        help="Policy file for embedded Eunomia (default: mcp_policies.json)",
    )
    parser.add_argument(
        "--eunomia-remote-url",
        default=os.environ.get("EUNOMIA_REMOTE_URL", None),
        help="URL for remote Eunomia server",
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

    parser.add_argument(
        "--tools",
        "--toolsets",
        default=os.getenv("MCP_ENABLED_TOOLS"),
        help="Comma-separated list of enabled tools or toolsets to expose",
    )
    parser.add_argument(
        "--disabled-tools",
        "--disabled-toolsets",
        default=os.getenv("MCP_DISABLED_TOOLS"),
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

            if args.eunomia_type == "remote":
                eunomia_mw = create_eunomia_middleware(
                    policy_file=None,
                    use_remote_eunomia=True,
                    eunomia_endpoint=args.eunomia_remote_url,
                )
            else:
                policy_file = args.eunomia_policy_file or "mcp_policies.json"
                eunomia_mw = create_eunomia_middleware(
                    policy_file=policy_file,
                    use_remote_eunomia=False,
                )
            middlewares.append(eunomia_mw)
        except Exception as e:
            logger.error(f"Failed to load Eunomia middleware: {e}")
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

    auth = _configure_auth(args)
    middlewares = _configure_middleware(args)

    import os

    os.environ["FASTMCP_LOG_LEVEL"] = "CRITICAL"
    mcp = FastMCP(name, auth=auth, instructions=instructions)

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

                # 1. Start with environment variable defaults
                enabled_tags_env = os.environ.get("MCP_ENABLED_TAGS")
                disabled_tags_env = os.environ.get("MCP_DISABLED_TAGS")
                enabled_components_env = os.environ.get(
                    "MCP_ENABLED_TOOLS"
                ) or os.environ.get("MCP_ENABLED_COMPONENTS")
                disabled_components_env = os.environ.get(
                    "MCP_DISABLED_TOOLS"
                ) or os.environ.get("MCP_DISABLED_COMPONENTS")

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
                    from fastmcp.server.dependencies import get_http_request

                    req = get_http_request()
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
                            enabled_tools_list = (
                                q_tools  # Query params override env/CLI if provided
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
                            disabled_tools_list = q_disabled

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
                            enabled_tags_list = q_tags

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
                            disabled_tags_list = q_disabled_tags

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
                                enabled_tools_list = h_tools

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
                                disabled_tools_list = h_disabled

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
                                enabled_tags_list = h_tags

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
                                disabled_tags_list = h_disabled_tags

                            # Gather query/search keyword from headers
                            for key in ["x-mcp-query", "x-mcp-search"]:
                                if headers.get(key):
                                    query_filter = headers.get(key)
                                    break
                except Exception:
                    pass

                # If query_filter is active, resolve matching tools from the Knowledge Graph!
                if query_filter:
                    try:
                        from agent_utilities.knowledge_graph.core.engine import (
                            IntelligenceGraphEngine,
                        )
                        from agent_utilities.tools.dynamic_tool_orchestrator import (
                            DynamicToolOrchestrator,
                        )

                        engine = IntelligenceGraphEngine.get_active()
                        if engine:
                            orchestrator = DynamicToolOrchestrator(engine)
                            kg_matched = orchestrator.resolve_mcp_tools(
                                query_filter, server_name=name
                            )
                            if kg_matched:
                                if enabled_tools_list:
                                    enabled_tools_list = [
                                        t for t in enabled_tools_list if t in kg_matched
                                    ]
                                else:
                                    enabled_tools_list = kg_matched
                    except Exception as e:
                        logger.debug(
                            "Failed to filter components using Knowledge Graph: %s", e
                        )

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
    except Exception as e:
        logger.warning(f"Could not register dynamic visibility transform: {e}")

    return args, mcp, middlewares
