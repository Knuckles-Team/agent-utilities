#!/usr/bin/python
"""MCP Utilities Module.

This module provides boilerplate and helper functions for working with the
Model Context Protocol (MCP). It handles CLI argument parsing for MCP servers,
automated server initialization with middleware stacks, and robust loading
of MCP configurations from JSON files with environment variable expansion.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from .base_utilities import GET_DEFAULT_SSL_VERIFY, to_boolean
from .config import (
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
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "text-embedding-nomic-embed-text-v2-moe")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "llama")

__version__ = "0.2.40"


def create_mcp_parser():
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

    Returns:
        A tuple containing:
            - args: The parsed argparse.Namespace object.
            - mcp: The initialized FastMCP server instance.
            - middlewares: A list of configured middleware instances.

    """
    import requests as _requests
    from fastmcp import FastMCP
    from fastmcp.server.auth import OAuthProxy, RemoteAuthProvider
    from fastmcp.server.auth.oidc_proxy import OIDCProxy
    from fastmcp.server.auth.providers.jwt import JWTVerifier, StaticTokenVerifier
    from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
    from fastmcp.server.middleware.logging import LoggingMiddleware
    from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
    from fastmcp.server.middleware.timing import TimingMiddleware

    try:
        from agent_utilities.middlewares import (
            JWTClaimsLoggingMiddleware,
            UserTokenMiddleware,
        )
    except ImportError:
        UserTokenMiddleware = None  # type: ignore
        JWTClaimsLoggingMiddleware = None  # type: ignore

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
        import sys as _sys

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

    auth = None
    allowed_uris = (
        args.allowed_client_redirect_uris.split(",")
        if args.allowed_client_redirect_uris
        else None
    )

    if args.auth_type == "none":
        auth = None
    elif args.auth_type == "static":
        auth = StaticTokenVerifier(
            tokens={
                "test-token": {"client_id": "test-user", "scopes": ["read", "write"]},
                "admin-token": {"client_id": "admin", "scopes": ["admin"]},
            }
        )
    elif args.auth_type == "jwt":
        import sys as _sys

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
                logger.error(
                    f"Error: HMAC algorithm {algorithm} requires --token-secret"
                )
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
            auth = JWTVerifier(
                jwks_uri=jwks_uri,
                public_key=public_key,
                issuer=issuer,
                audience=audience,
                algorithm=(
                    algorithm if algorithm and algorithm.startswith("HS") else None
                ),
                required_scopes=required_scopes,
            )
        except Exception as e:
            logger.error(f"Failed to initialize JWTVerifier: {e}")
            _sys.exit(1)
    elif args.auth_type == "oauth-proxy":
        import sys as _sys

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
        auth = OAuthProxy(
            upstream_authorization_endpoint=args.oauth_upstream_auth_endpoint,
            upstream_token_endpoint=args.oauth_upstream_token_endpoint,
            upstream_client_id=args.oauth_upstream_client_id,
            upstream_client_secret=args.oauth_upstream_client_secret,
            token_verifier=token_verifier,
            base_url=args.oauth_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "oidc-proxy":
        import sys as _sys

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
        auth = OIDCProxy(
            config_url=args.oidc_config_url,
            client_id=args.oidc_client_id,
            client_secret=args.oidc_client_secret,
            base_url=args.oidc_base_url,
            allowed_client_redirect_uris=allowed_uris,
        )
    elif args.auth_type == "remote-oauth":
        import sys as _sys

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
        auth = RemoteAuthProvider(
            token_verifier=token_verifier,
            authorization_servers=auth_servers,
            base_url=args.remote_base_url,
        )

    middlewares = [
        ErrorHandlingMiddleware(include_traceback=True, transform_errors=True),
        RateLimitingMiddleware(max_requests_per_second=10.0, burst_capacity=20),
        TimingMiddleware(),
        LoggingMiddleware(),
    ]

    if JWTClaimsLoggingMiddleware is not None:
        middlewares.append(JWTClaimsLoggingMiddleware())

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

    mcp = FastMCP(name, auth=auth, instructions=instructions)

    return args, mcp, middlewares


def load_mcp_servers_from_config(config_path: str | Path) -> list[Any]:
    """Load and expand environment variables in an MCP config file.

    Reads the specified mcp_config.json, expands any environment variable
    placeholders (e.g., ${API_KEY}), performs robust pre-validation of
    executable commands in the PATH, and initializes the server objects.

    Args:
        config_path: Path to the mcp_config.json file.

    Returns:
        A list of initialized pydantic_ai.mcp.MCPServer objects (technically
        MCPToolSet in newer versions, but returned as list of servers here).

    """
    import json
    import shutil
    import tempfile

    from pydantic_ai.mcp import load_mcp_servers

    from .base_utilities import expand_env_vars

    try:
        path = Path(config_path)
        if not path.exists():
            return []

        content = path.read_text()
        expanded_content = expand_env_vars(content)

        # Robust Validation: Check if commands exist before pydantic-ai tries to start them
        try:
            config_data = json.loads(expanded_content)
            mcp_servers = config_data.get("mcpServers", {})
            modified = False

            for name, cfg in mcp_servers.items():
                command = cfg.get("command")
                if command:
                    # Resolve command path with explicit ~/.local/bin support
                    search_path = os.environ.get("PATH", "")
                    local_bin = str(Path.home() / ".local" / "bin")
                    if local_bin not in search_path:
                        search_path = f"{local_bin}:{search_path}"

                    resolved = shutil.which(command, path=search_path)
                    if not resolved:
                        logger.warning(
                            f"MCP Config: Command '{command}' for server '{name}' NOT FOUND in PATH ({search_path}). Startup will likely fail."
                        )
                    else:
                        logger.debug(
                            f"MCP Config: Resolved command '{command}' to '{resolved}'"
                        )

                    # Ensure PATH and PYTHONPATH are preserved if not explicitly set
                    if "env" not in cfg:
                        cfg["env"] = {}

                    if "PATH" not in cfg["env"]:
                        cfg["env"]["PATH"] = search_path
                    if "PYTHONPATH" not in cfg["env"] and "PYTHONPATH" in os.environ:
                        cfg["env"]["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")

                    # Suppress RequestsDependencyWarning in subprocesses
                    if "PYTHONWARNINGS" not in cfg["env"]:
                        cfg["env"]["PYTHONWARNINGS"] = (
                            "ignore:urllib3 (2.3.0) or chardet"
                        )
                    else:
                        if "ignore:urllib3" not in cfg["env"]["PYTHONWARNINGS"]:
                            cfg["env"]["PYTHONWARNINGS"] += (
                                ",ignore:urllib3 (2.3.0) or chardet"
                            )

                    # Token forwarding: propagate user session token to
                    # MCP subprocesses for delegated authentication.
                    # CONCEPT:AU-011 — Secrets & Authentication
                    if "AGENT_USER_TOKEN" not in cfg["env"]:
                        _user_token = os.environ.get("AGENT_USER_TOKEN")
                        if not _user_token:
                            try:
                                from .secrets_client import create_secrets_client

                                _sc = create_secrets_client()
                                _user_token = _sc.get("session_token")
                            except Exception:
                                pass
                        if _user_token:
                            cfg["env"]["AGENT_USER_TOKEN"] = _user_token

                    modified = True

            if modified:
                expanded_content = json.dumps(config_data)
        except Exception as e:
            logger.warning(f"MCP Config: Pre-validation failed: {e}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(expanded_content)
            tmp_path = tmp.name

        try:
            servers = load_mcp_servers(tmp_path)
            # Re-attach IDs from config
            config_data = json.loads(expanded_content)
            mcp_servers_cfg = config_data.get("mcpServers", {})

            # Match by command and args as a heuristic if pydantic-ai doesn't preserve order or names
            for ts in servers:
                # pydantic-ai objects might not have a clean way to match back,
                # but they usually follow the order in the JSON.
                pass

            # Better: If we have a list, and the config had a dict, they MIGHT match by order
            # However, pydantic-ai load_mcp_servers is internal.
            # I'll just set the .id if they are list components.
            for i, (name, cfg) in enumerate(mcp_servers_cfg.items()):
                if i < len(servers):
                    servers[i].id = name
                    logger.debug(f"MCP Config: Loaded server '{name}'")

            return servers
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.error(f"Failed to load MCP config {config_path}: {e}")
        return []


# ---------------------------------------------------------------------------
# FastMCP Context (ctx) Helper Utilities
# ---------------------------------------------------------------------------
# Standardized helpers for the FastMCP ``Context`` object.  Every MCP server
# in the agent-packages ecosystem should import these instead of hand-rolling
# ctx interactions.  All helpers are safe when *ctx* is ``None`` (backward
# compatible with callers that do not inject a context).
# ---------------------------------------------------------------------------


async def ctx_progress(ctx: Any, progress: int, total: int = 100) -> None:
    """Report progress to the MCP client if a context is available.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        progress: Current progress value.
        total: Total steps (default 100 for percentage-style reporting).

    """
    if ctx:
        await ctx.report_progress(progress=progress, total=total)


async def ctx_confirm_destructive(
    ctx: Any,
    action_description: str,
) -> bool:
    """Standard elicitation guard for destructive operations.

    When a ``Context`` is available this asks the human user to confirm
    before proceeding.  If no context is provided (e.g. headless / test
    invocation) the operation is allowed by default.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        action_description: Human-readable description of the action,
            e.g. ``"delete stack 'production'"``.

    Returns:
        ``True`` if the operation should proceed, ``False`` if cancelled.

    """
    if not ctx:
        return True  # No context → allow (headless / test mode)
    try:
        result = await ctx.elicit(
            f"⚠️ Are you sure you want to {action_description}?",
            response_type=bool,
        )
        return result.action == "accept" and bool(result.data)
    except Exception as exc:
        logger.warning("Elicitation failed (%s); allowing operation by default.", exc)
        return True


def ctx_log(
    ctx: Any,
    server_logger: logging.Logger,
    level: str,
    message: str,
) -> None:
    """Dual-log a message to *both* the server-side logger and the MCP client.

    This ensures that diagnostic output is visible in two places:
    • The server process logs (for operators / container logs).
    • The MCP client log console (for the AI agent / human user).

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        server_logger: A standard Python :class:`logging.Logger`.
        level: Log level string — ``"debug"``, ``"info"``, ``"warning"``,
            or ``"error"``.
        message: The log message.

    """
    getattr(server_logger, level, server_logger.info)(message)
    if ctx:
        client_fn = getattr(ctx, level, None) or getattr(ctx, "info", None)
        if client_fn:
            try:
                client_fn(message)
            except Exception:
                pass  # nosec: B110 - Never let client-side logging break tool execution


async def ctx_set_state(
    ctx: Any,
    project: str,
    key: str,
    value: Any,
) -> None:
    """Store a value in the MCP session state with a standardized key.

    Keys are namespaced as ``"{project}_{key}"`` to prevent collisions
    across different MCP servers sharing the same session.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        project: Project namespace (e.g. ``"portainer"``, ``"gitlab"``).
        key: State key (e.g. ``"auth_token"``, ``"active_context"``).
        value: The value to store.

    """
    if ctx and hasattr(ctx, "session"):
        try:
            namespaced = f"{project}_{key}"
            await ctx.session.set_state(namespaced, value)
        except Exception as exc:
            logger.debug("ctx_set_state failed for %s_%s: %s", project, key, exc)


async def ctx_get_state(
    ctx: Any,
    project: str,
    key: str,
    default: Any = None,
) -> Any:
    """Retrieve a value from the MCP session state.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        project: Project namespace.
        key: State key.
        default: Fallback if the key is missing or ctx is unavailable.

    Returns:
        The stored value, or *default*.

    """
    if ctx and hasattr(ctx, "session"):
        try:
            namespaced = f"{project}_{key}"
            val = await ctx.session.get_state(namespaced)
            return val if val is not None else default
        except Exception:
            pass  # nosec: B110
    return default


async def ctx_sample(
    ctx: Any,
    prompt: str,
    system_prompt: str | None = None,
) -> str | None:
    """Ask the client LLM to generate a response (sampling).

    This is an optional capability — it only works when the connected MCP
    client supports the ``sampling`` feature.  Returns ``None`` silently
    when sampling is unavailable.

    Args:
        ctx: FastMCP ``Context`` (may be ``None``).
        prompt: The user-turn prompt to send to the LLM.
        system_prompt: Optional system prompt to guide the LLM.

    Returns:
        The LLM-generated text, or ``None`` if sampling is unavailable.

    """
    if not ctx:
        return None
    try:
        from mcp.types import SamplingMessage, TextContent

        messages = [
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt),
            )
        ]
        result = await ctx.sample(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=2048,
        )
        if result and hasattr(result, "content"):
            return (
                result.content.text
                if hasattr(result.content, "text")
                else str(result.content)
            )
        return None
    except ImportError:
        logger.debug("mcp.types not available — sampling disabled.")
        return None
    except Exception as exc:
        logger.debug("ctx_sample failed: %s", exc)
        return None


# Maintain backward compatibility for local usage
config = mcp_auth_config
load_mcp_config = load_mcp_servers_from_config

__all__ = [
    "create_mcp_parser",
    "create_mcp_server",
    "load_mcp_servers_from_config",
    "mcp_auth_config",
    "DEFAULT_TRANSPORT",
    "DEFAULT_SSL_VERIFY",
    "DEFAULT_PROVIDER",
    "DEFAULT_MODEL_ID",
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_LLM_API_KEY",
    # ctx helpers
    "ctx_progress",
    "ctx_confirm_destructive",
    "ctx_log",
    "ctx_set_state",
    "ctx_get_state",
    "ctx_sample",
]
