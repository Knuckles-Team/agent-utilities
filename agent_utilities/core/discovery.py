#!/usr/bin/python
"""Cross-IDE Discovery — Unified KG MCP endpoint registration.

CONCEPT:OS-5.0 — Agent OS Infrastructure (Extension)

Manages MCP config generation and registration across multiple IDE environments
so that all agents (Antigravity, Claude Code, OpenCode, Devin) can discover
and connect to the shared Knowledge Graph MCP server.

On first run or explicit call, generates ``mcp_config.json`` entries for
the canonical config location and known IDE config paths.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .paths import config_dir, data_dir, kg_db_path

logger = logging.getLogger(__name__)

# Known IDE MCP config locations
KNOWN_IDE_CONFIGS: dict[str, Path] = {
    "antigravity": Path.home() / ".gemini" / "antigravity" / "mcp_config.json",
    "claude_code": Path.home() / ".claude" / "mcp_config.json",
    "opencode": Path.home() / ".opencode" / "mcp_config.json",
}

# The canonical MCP server entry for the KG server
KG_SERVER_ENTRY = {
    "agent-utilities-kg": {
        "command": "uv",
        "args": ["run", "python", "-m", "agent_utilities.mcp.kg_server"],
        "env": {},
    }
}


def generate_mcp_config() -> dict:
    """Generate the canonical MCP config for both KG and Harness servers.

    Returns:
        A dictionary suitable for writing to ``mcp_config.json``.
    """
    db_path = str(kg_db_path())
    config = {
        "mcpServers": {
            "agent-utilities-kg": {
                "command": "uv",
                "args": ["run", "python", "-m", "agent_utilities.mcp.kg_server"],
                "env": {
                    "GRAPH_DB_PATH": db_path,
                },
            },
            "agent-utilities-harness": {
                "command": "uv",
                "args": [
                    "run",
                    "python",
                    "-m",
                    "agent_utilities.mcp.harness_server",
                ],
                "env": {
                    "GRAPH_DB_PATH": db_path,
                },
            },
        }
    }
    return config


def write_canonical_config(overwrite: bool = False) -> Path:
    """Write the canonical MCP config to the XDG config directory.

    Args:
        overwrite: If True, overwrite existing config. If False, only write
            if the file doesn't exist.

    Returns:
        Path to the written config file.
    """
    config_path = config_dir() / "mcp_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists() and not overwrite:
        # Merge: add our server entry without overwriting existing servers
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        servers = existing.get("mcpServers", {})
        if "agent-utilities-kg" not in servers:
            servers.update(generate_mcp_config()["mcpServers"])
            existing["mcpServers"] = servers
            config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            logger.info("Added agent-utilities-kg to existing %s", config_path)
    else:
        config_path.write_text(
            json.dumps(generate_mcp_config(), indent=2), encoding="utf-8"
        )
        logger.info("Created canonical MCP config at %s", config_path)

    return config_path


def register_with_ides(force: bool = False) -> dict[str, str]:
    """Register the KG MCP server with known IDE config files.

    For each known IDE that has a config directory, adds the
    ``agent-utilities-kg`` server entry if not already present.

    Args:
        force: If True, overwrite existing entries.

    Returns:
        Dictionary mapping IDE names to their status ('added', 'exists', 'skipped').
    """
    results: dict[str, str] = {}

    for ide_name, config_path in KNOWN_IDE_CONFIGS.items():
        if not config_path.parent.exists():
            results[ide_name] = "skipped (directory not found)"
            continue

        try:
            if config_path.exists():
                existing = json.loads(config_path.read_text(encoding="utf-8"))
                servers = existing.get("mcpServers", {})
                if "agent-utilities-kg" in servers and not force:
                    results[ide_name] = "exists"
                    continue

                servers.update(generate_mcp_config()["mcpServers"])
                existing["mcpServers"] = servers
                config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                results[ide_name] = "added"
            else:
                config_path.write_text(
                    json.dumps(generate_mcp_config(), indent=2), encoding="utf-8"
                )
                results[ide_name] = "created"
        except Exception as e:
            results[ide_name] = f"error: {e}"
            logger.warning("Failed to register with %s: %s", ide_name, e)

    return results


from typing import Any


def get_discovery_status() -> dict:
    """Get the current status of KG MCP server discovery across IDEs.

    Returns:
        Dictionary with canonical config status and per-IDE registration status.
    """
    canonical = config_dir() / "mcp_config.json"
    status: dict[str, Any] = {
        "canonical_config": str(canonical),
        "canonical_exists": canonical.exists(),
        "kg_db_path": str(kg_db_path()),
        "data_dir": str(data_dir()),
        "ides": {},
    }

    for ide_name, config_path in KNOWN_IDE_CONFIGS.items():
        if not config_path.exists():
            status["ides"][ide_name] = {
                "status": "not_configured",
                "path": str(config_path),
            }
        else:
            try:
                content = json.loads(config_path.read_text(encoding="utf-8"))
                has_kg = "agent-utilities-kg" in content.get("mcpServers", {})
                status["ides"][ide_name] = {
                    "status": "registered" if has_kg else "missing_entry",
                    "path": str(config_path),
                }
            except Exception:
                status["ides"][ide_name] = {
                    "status": "parse_error",
                    "path": str(config_path),
                }

    return status
