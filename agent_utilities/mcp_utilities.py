#!/usr/bin/python
# ruff: noqa: F822
"""MCP Utilities Module — backward-compatible re-exports.

New code should import from the submodules directly:
- ``agent_utilities.mcp.server_factory`` — parser, server creation, auth config
- ``agent_utilities.mcp.context_helpers`` — ctx_* helpers
- ``agent_utilities.mcp.delegated_auth`` — OAuth 2.0 delegation helpers

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

__version__ = "0.25.0"


def __getattr__(name: str):
    # LLM Config Defaults
    if name in (
        "DEFAULT_LLM_API_KEY",
        "DEFAULT_LLM_BASE_URL",
        "DEFAULT_LLM_MODEL_ID",
        "DEFAULT_LLM_PROVIDER",
    ):
        import agent_utilities.core.config as config_mod

        return getattr(config_mod, name)

    # Config Loader
    if name == "load_mcp_servers_from_config" or name == "load_mcp_config":
        import agent_utilities.mcp.config_loader as loader_mod

        return loader_mod.load_mcp_servers_from_config

    # Context Helpers
    if name in (
        "ctx_progress",
        "ctx_confirm_destructive",
        "ctx_log",
        "ctx_set_state",
        "ctx_get_state",
        "ctx_sample",
    ):
        import agent_utilities.mcp.context_helpers as ctx_mod

        return getattr(ctx_mod, name)

    # Delegated Auth
    if name in (
        "get_delegated_token",
        "get_user_token",
        "get_user_claims",
        "get_user_identity",
        "is_delegation_enabled",
        "get_3lo_authorization_url",
        "exchange_authorization_code",
        "refresh_access_token",
    ):
        import agent_utilities.mcp.delegated_auth as auth_mod

        return getattr(auth_mod, name)

    # Server Factory
    if name in (
        "DEFAULT_TRANSPORT",
        "DEFAULT_SSL_VERIFY",
        "create_mcp_parser",
        "create_mcp_server",
        "mcp_auth_config",
        "config",
    ):
        import agent_utilities.mcp.server_factory as factory_mod

        if name == "config":
            return factory_mod.mcp_auth_config
        return getattr(factory_mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
