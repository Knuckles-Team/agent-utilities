#!/usr/bin/python
"""Agent Utilities Core Module.

This module serves as the primary entry point for the agent-utilities package,
providing a unified interface for agent creation, graph orchestration, workspace
management, and various helper utilities.

Warning suppression is centralized here so every downstream import inherits
the filters without needing per-file boilerplate.
"""

import os
import warnings

# ruff: noqa: E402, F401

# ── Centralized warning suppression ──────────────────────────────────
# All library-level noise is filtered once at package init so that
# downstream modules (server, mcp_utilities, base_utilities, etc.)
# don't need their own copies.

# 1. requests/urllib3 version-mismatch noise
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")
warnings.filterwarnings("ignore", message=r".*urllib3 v2.*only supports OpenSSL.*")

# 2. InsecureRequestWarning (emitted when ssl_verify=False)
try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # nosec B110
    pass
warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")

# 3. DeprecationWarnings from third-party libs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastmcp")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="httpx")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# 4. PydanticDeprecatedSince20 (noisy in older pydantic shims)
try:
    from pydantic import PydanticDeprecatedSince20

    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except ImportError:
    pass

# ── End warning suppression ──────────────────────────────────────────


# Lazy imports for all modules to avoid heavy import chains
def __getattr__(name):
    if name in [
        "get_logger",
        "optional_import_block",
        "require_optional_import",
        "retrieve_package_name",
        "to_boolean",
        "to_dict",
        "to_float",
        "to_integer",
        "to_list",
        "ensure_package_installed",
    ]:
        from .base_utilities import (
            ensure_package_installed,
            get_logger,
            optional_import_block,
            require_optional_import,
            retrieve_package_name,
            to_boolean,
            to_dict,
            to_float,
            to_integer,
            to_list,
        )

        return locals()[name]
    elif name in ["create_agent", "create_agent_parser"]:
        from .agent_factory import create_agent, create_agent_parser

        return create_agent if name == "create_agent" else create_agent_parser
    elif name == "DEFAULT_GRAPH_PERSISTENCE_PATH":
        from .config import DEFAULT_GRAPH_PERSISTENCE_PATH

        return DEFAULT_GRAPH_PERSISTENCE_PATH
    elif name in ["discover_agents", "discover_all_specialists"]:
        from .discovery import discover_agents, discover_all_specialists

        return (
            discover_agents if name == "discover_agents" else discover_all_specialists
        )
    elif name == "create_embedding_model":
        from .embedding_utilities import create_embedding_model

        return create_embedding_model
    elif name in [
        "GraphState",
        "build_tag_env_map",
        "create_graph_agent",
        "create_master_graph",
        "get_graph_mermaid",
        "initialize_graph_from_workspace",
        "register_on_enter_hook",
        "register_on_exit_hook",
        "run_graph",
        "run_graph_stream",
        "run_orthogonal_regions",
        "validate_graph",
    ]:
        from .graph import (
            GraphState,
            build_tag_env_map,
            create_graph_agent,
            create_master_graph,
            get_graph_mermaid,
            initialize_graph_from_workspace,
            register_on_enter_hook,
            register_on_exit_hook,
            run_graph,
            run_graph_stream,
            run_orthogonal_regions,
            validate_graph,
        )

        return locals()[name]
    elif name == "create_model":
        from .model_factory import create_model

        return create_model
    elif name in ["build_system_prompt_from_workspace", "load_identity"]:
        from .prompt_builder import build_system_prompt_from_workspace, load_identity

        return (
            build_system_prompt_from_workspace
            if name == "build_system_prompt_from_workspace"
            else load_identity
        )
    elif name in ["create_agent_server", "create_graph_agent_server"]:
        from .server import create_agent_server, create_graph_agent_server

        return (
            create_agent_server
            if name == "create_agent_server"
            else create_graph_agent_server
        )
    elif name == "CodemapGenerator":
        from .knowledge_graph.codemaps import CodemapGenerator

        return CodemapGenerator
    elif name == "parse_codemap_mentions":
        from .agent_chat.parser import parse_codemap_mentions

        return parse_codemap_mentions
    elif name in [
        "CORE_FILES",
        "append_to_md_file",
        "get_mcp_config_path",
        "get_workspace_path",
        "initialize_workspace",
        "list_workspace_files",
        "load_workspace_file",
        "read_md_file",
        "write_md_file",
        "write_workspace_file",
    ]:
        from .workspace import (
            CORE_FILES,
            append_to_md_file,
            get_mcp_config_path,
            get_workspace_path,
            initialize_workspace,
            list_workspace_files,
            load_workspace_file,
            read_md_file,
            write_md_file,
            write_workspace_file,
        )

        return locals()[name]
    elif name in [
        "delete_chat_from_disk",
        "get_chat_from_disk",
        "list_chats_from_disk",
        "save_chat_to_disk",
    ]:
        from .chat_persistence import (
            delete_chat_from_disk,
            get_chat_from_disk,
            list_chats_from_disk,
            save_chat_to_disk,
        )

        return locals()[name]
    elif name in [
        "CodemapArtifact",
        "CodemapNode",
        "DiscoveredSpecialist",
        "ImplementationPlan",
        "NestedStructure",
        "PeriodicTask",
        "ProjectConstitution",
        "Spec",
        "StructuredPrompt",
        "Task",
        "Tasks",
    ]:
        from .models import (
            CodemapArtifact,
            CodemapNode,
            DiscoveredSpecialist,
            ImplementationPlan,
            NestedStructure,
            PeriodicTask,
            ProjectConstitution,
            Spec,
            StructuredPrompt,
            Task,
            Tasks,
        )

        return locals()[name]
    elif name == "SDDManager":
        from .sdd import SDDManager

        return SDDManager
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Set environment variables without using to_boolean
os.environ.setdefault("OTEL_ENABLE_OTEL", "false")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
if os.environ.get("ENABLE_OTEL", "True").lower() in ["true", "1", "yes"]:
    os.environ.setdefault("OTEL_ENABLE_OTEL", "True")

# ── Graph Integration ────────────────────────────────────────────────
# Disabled by default to avoid import overhead during testing
# Can be enabled by setting ENABLE_GRAPH_INTEGRATION=true and calling initialize_graph_integration() explicitly

__version__ = "0.2.40"

__all__ = [
    # Agent creation
    "create_agent",
    "create_agent_parser",
    "create_agent_server",
    "create_graph_agent_server",
    # Graph orchestration
    "GraphState",
    "create_graph_agent",
    "create_master_graph",
    "run_graph",
    "run_graph_stream",
    "build_tag_env_map",
    "get_graph_mermaid",
    "validate_graph",
    "initialize_graph_from_workspace",
    # Workspace
    "CORE_FILES",
    "get_workspace_path",
    "get_mcp_config_path",
    "initialize_workspace",
    "load_workspace_file",
    "write_workspace_file",
    "list_workspace_files",
    "read_md_file",
    "write_md_file",
    "append_to_md_file",
    # Prompt / Identity
    "load_identity",
    "build_system_prompt_from_workspace",
    # Model factory
    "create_model",
    # A2A
    "discover_agents",
    "discover_all_specialists",
    # Chat persistence
    "save_chat_to_disk",
    "list_chats_from_disk",
    "get_chat_from_disk",
    "delete_chat_from_disk",
    # Config
    "DEFAULT_GRAPH_PERSISTENCE_PATH",
    # Base utilities
    "to_boolean",
    "to_integer",
    "to_float",
    "to_list",
    "to_dict",
    "retrieve_package_name",
    "get_logger",
    "ensure_package_installed",
    "optional_import_block",
    "require_optional_import",
    # Embedding
    "create_embedding_model",
    # HSM hooks
    "register_on_enter_hook",
    "register_on_exit_hook",
    "run_orthogonal_regions",
    # Models
    "PeriodicTask",
    "DiscoveredSpecialist",
    "ProjectConstitution",
    "Spec",
    "ImplementationPlan",
    "Tasks",
    "Task",
    "StructuredPrompt",
    "NestedStructure",
    # SDD
    "SDDManager",
    # Codemaps
    "CodemapNode",
    "CodemapArtifact",
    "CodemapGenerator",
    "parse_codemap_mentions",
]
