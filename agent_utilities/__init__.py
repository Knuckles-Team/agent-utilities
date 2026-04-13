#!/usr/bin/python
# coding: utf-8
"""Agent Utilities Core Module.

This module serves as the primary entry point for the agent-utilities package,
providing a unified interface for agent creation, graph orchestration, workspace
management, and various helper utilities. It filters common library warnings
and initializes environment-level observability settings.
"""

import os
import warnings

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

# General urllib3/chardet mismatch warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")

from .base_utilities import (
    to_boolean,
    to_integer,
    to_float,
    to_list,
    to_dict,
    retrieve_package_name,
    get_logger,
    optional_import_block,
    require_optional_import,
)

os.environ.setdefault("OTEL_ENABLE_OTEL", "false")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
if to_boolean(os.environ.get("ENABLE_OTEL", "True")):
    os.environ.setdefault("OTEL_ENABLE_OTEL", "True")

from .agent_factory import create_agent, create_agent_parser
from .server import create_agent_server, create_graph_agent_server
from .graph import (
    GraphState,
    create_graph_agent,
    create_master_graph,
    run_graph,
    run_graph_stream,
    build_tag_env_map,
    get_graph_mermaid,
    validate_graph,
    initialize_graph_from_workspace,
    register_on_enter_hook,
    register_on_exit_hook,
    run_orthogonal_regions,
)

from .prompt_builder import (
    load_identity,
    build_system_prompt_from_workspace,
)

from .model_factory import (
    create_model,
)

from .a2a import (
    discover_agents,
)

from .workspace import (
    CORE_FILES,
    get_workspace_path,
    get_mcp_config_path,
    initialize_workspace,
    load_workspace_file,
    write_workspace_file,
    list_workspace_files,
    read_md_file,
    write_md_file,
    append_to_md_file,
)

from .config import DEFAULT_GRAPH_PERSISTENCE_PATH

from .base_utilities import (
    to_boolean,
    ensure_package_installed,
)

from .embedding_utilities import create_embedding_model

from .models import PeriodicTask

from .chat_persistence import (
    save_chat_to_disk,
    list_chats_from_disk,
    get_chat_from_disk,
    delete_chat_from_disk,
)

__version__ = "0.2.39"

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
]
