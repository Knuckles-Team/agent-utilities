from __future__ import annotations

"""MCP Configuration Loader.

Reads mcp_config.json, expands environment variable placeholders,
validates executable commands, and initializes pydantic-ai MCP server objects.

CONCEPT:OS-5.1 — Secrets & Authentication
"""


import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    from pydantic_ai.mcp import load_mcp_servers

    from agent_utilities.base_utilities import expand_env_vars

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
                    # CONCEPT:OS-5.1 — Secrets & Authentication
                    if "AGENT_USER_TOKEN" not in cfg["env"]:
                        _user_token = os.environ.get("AGENT_USER_TOKEN")
                        if not _user_token:
                            try:
                                from agent_utilities.security.secrets_client import (
                                    create_secrets_client,
                                )

                                _sc = create_secrets_client()
                                _user_token = _sc.get("session_token")
                            except Exception:  # nosec B110
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
