#!/usr/bin/python
# ruff: noqa: F822
"""MCP Utilities Module — backward-compatible re-exports.

New code should import from the submodules directly:
- ``agent_utilities.mcp.server_factory`` — parser, server creation, auth config
- ``agent_utilities.mcp.context_helpers`` — ctx_* helpers
- ``agent_utilities.mcp.delegated_auth`` — OAuth 2.0 delegation helpers

CONCEPT:AU-ECO.mcp.standardized-interfaces — MCP Standardized Interfaces
"""

__version__ = "1.23.2"


def __getattr__(name: str):
    # LLM Config Defaults
    if name in (
        "DEFAULT_LLM_API_KEY",
        "DEFAULT_LLM_BASE_URL",
        "DEFAULT_LLM_MODEL_ID",
        "DEFAULT_LLM_PROVIDER",
    ):
        import importlib

        config_mod = importlib.import_module("agent_utilities.core.config")

        return getattr(config_mod, name)

    # Config Loader
    if name == "load_mcp_servers_from_config" or name == "load_mcp_config":
        import importlib

        loader_mod = importlib.import_module("agent_utilities.core.config")

        return getattr(loader_mod, name)

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

    # Action dispatch (standardized list_actions / aliases / did-you-mean)
    if name in (
        "DISCOVERY_ACTIONS",
        "public_actions",
        "suggest",
        "canonicalize",
        "unknown_action_error",
        "resolve_action",
        "dispatch",
    ):
        import agent_utilities.mcp.action_dispatch as dispatch_mod

        return getattr(dispatch_mod, name)

    # Concurrency (offload blocking sync work off the event loop)
    if name == "run_blocking":
        import agent_utilities.mcp.concurrency as concurrency_mod

        return concurrency_mod.run_blocking

    # Verbose 1:1 tool surface + the MCP_TOOL_MODE knob + central surface wiring
    if name in (
        "tool_mode",
        "register_verbose_tools",
        "register_tool_surface",
        "register_action_provider",
        "VALID_TOOL_MODES",
    ):
        import agent_utilities.mcp.verbose_tools as verbose_mod

        return getattr(verbose_mod, name)

    # Shared config loader — agents call this in place of load_dotenv(find_dotenv())
    if name == "load_config":
        import importlib

        config_mod = importlib.import_module("agent_utilities.core.config")

        return config_mod.load_config

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
    # action dispatch
    "DISCOVERY_ACTIONS",
    "public_actions",
    "suggest",
    "canonicalize",
    "unknown_action_error",
    "resolve_action",
    "dispatch",
    # concurrency
    "run_blocking",
    # tool-mode surface + config loader
    "tool_mode",
    "register_verbose_tools",
    "register_tool_surface",
    "register_action_provider",
    "VALID_TOOL_MODES",
    "load_config",
]
