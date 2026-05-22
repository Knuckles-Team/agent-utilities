#!/usr/bin/python
"""MCP Utilities Module — backward-compatible re-exports.

New code should import from the submodules directly:
- ``agent_utilities.mcp.server_factory`` — parser, server creation, auth config
- ``agent_utilities.mcp.context_helpers`` — ctx_* helpers
- ``agent_utilities.mcp.delegated_auth`` — OAuth 2.0 delegation helpers

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

__version__ = "0.12.0"

# Re-export server factory (parser, server, config loading, defaults)
# Re-export config loader (still defined here — too coupled to move without
# broader refactoring of pydantic-ai integration)
from agent_utilities.core.config import (  # noqa: F401
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
)
from agent_utilities.mcp.config_loader import (  # noqa: F401
    load_mcp_servers_from_config,
)

# Re-export context helpers
from agent_utilities.mcp.context_helpers import (  # noqa: F401
    ctx_confirm_destructive,
    ctx_get_state,
    ctx_log,
    ctx_progress,
    ctx_sample,
    ctx_set_state,
)

# Re-export delegated auth helpers
from agent_utilities.mcp.delegated_auth import (  # noqa: F401
    exchange_authorization_code,
    get_3lo_authorization_url,
    get_delegated_token,
    get_user_claims,
    get_user_identity,
    get_user_token,
    is_delegation_enabled,
    refresh_access_token,
)
from agent_utilities.mcp.server_factory import (  # noqa: F401
    DEFAULT_SSL_VERIFY,
    DEFAULT_TRANSPORT,
    create_mcp_parser,
    create_mcp_server,
    mcp_auth_config,
)

# Maintain backward compatibility aliases
config = mcp_auth_config
load_mcp_config = load_mcp_servers_from_config

__all__ = [
    "create_mcp_parser",
    "create_mcp_server",
    "load_mcp_servers_from_config",
    "mcp_auth_config",
    "DEFAULT_TRANSPORT",
    "DEFAULT_SSL_VERIFY",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_LLM_MODEL_ID",
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_LLM_API_KEY",
    # ctx helpers
    "ctx_progress",
    "ctx_confirm_destructive",
    "ctx_log",
    "ctx_set_state",
    "ctx_get_state",
    "ctx_sample",
    # delegated auth
    "get_delegated_token",
    "get_user_token",
    "get_user_claims",
    "get_user_identity",
    "is_delegation_enabled",
    "get_3lo_authorization_url",
    "exchange_authorization_code",
    "refresh_access_token",
]
