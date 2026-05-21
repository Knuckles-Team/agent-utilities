#!/usr/bin/python
"""XDG-Compliant Path Resolution Module.

CONCEPT:OS-5.0 — Agent OS Infrastructure (Extension)

Centralizes all file path resolution for the agent-utilities ecosystem using
the XDG Base Directory Specification via ``platformdirs``. This replaces 10+
scattered hardcoded ``knowledge_graph.db`` references with a single source
of truth for path resolution.

Architecture:
    - ``config_dir()``: ``~/.config/agent-utilities/`` — config files,
      mcp_config.json, a2a_config.json
    - ``data_dir()``: ``~/.local/share/agent-utilities/`` — KG database,
      ontologies, vector indexes, runtime artifacts
    - ``cache_dir()``: ``~/.cache/agent-utilities/`` — embedding caches,
      similarity indexes, skill graph cache

All paths can be overridden via environment variables:
    - ``AGENT_UTILITIES_CONFIG_DIR``
    - ``AGENT_UTILITIES_DATA_DIR``
    - ``AGENT_UTILITIES_CACHE_DIR``
    - ``GRAPH_DB_PATH`` (specific KG database override)

Backward Compatibility:
    If a ``knowledge_graph.db`` exists in the workspace root, it takes
    priority over the XDG data directory. This ensures existing setups
    continue to work without migration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

APP_NAME = "agent-utilities"
APP_AUTHOR = "knuckles-team"


def config_dir() -> Path:
    """Return the XDG config directory for agent-utilities.

    Default: ``~/.config/agent-utilities/``

    Override via ``AGENT_UTILITIES_CONFIG_DIR`` environment variable.

    Contains:
        - ``mcp_config.json`` — MCP server discovery (cross-IDE)
        - ``a2a_config.json`` — A2A agent discovery
        - ``agent.env`` — Global environment overrides
        - ``policies/`` — Global policy overrides
    """
    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))


def data_dir() -> Path:
    """Return the XDG data directory for agent-utilities.

    Default: ``~/.local/share/agent-utilities/``

    Override via ``AGENT_UTILITIES_DATA_DIR`` environment variable.

    Contains:
        - ``kg/knowledge_graph.db`` — Global unified Knowledge Graph
        - ``kg/ontologies/`` — User-provided domain ontologies
        - ``runtime/`` — Harness registry, evolution manifests
        - ``research/papers/`` — Downloaded research papers
    """
    override = os.environ.get("AGENT_UTILITIES_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_data_path(APP_NAME, APP_AUTHOR))


def cache_dir() -> Path:
    """Return the XDG cache directory for agent-utilities.

    Default: ``~/.cache/agent-utilities/``

    Override via ``AGENT_UTILITIES_CACHE_DIR`` environment variable.

    Contains:
        - ``embeddings/`` — Vector embedding cache
        - ``similarity_indexes/`` — Precomputed similarity graphs
        - ``skill_graphs/`` — Cached skill graph data
    """
    override = os.environ.get("AGENT_UTILITIES_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_cache_path(APP_NAME, APP_AUTHOR))


def kg_db_path(workspace: Path | str | None = None) -> Path:
    """Resolve the Knowledge Graph database path with priority chain.

    Resolution order:
        1. ``GRAPH_DB_PATH`` environment variable (explicit override)
        2. XDG data directory ``~/.local/share/agent-utilities/kg/knowledge_graph.db``
           (shared KG — canonical location for all agents)
        3. Workspace-local ``knowledge_graph.db`` (deprecated, emits warning)

    Args:
        workspace: Optional workspace root path. If provided and a
            ``knowledge_graph.db`` exists there, a deprecation warning
            is emitted. Set ``GRAPH_DB_PATH`` explicitly to use a
            workspace-local database.

    Returns:
        Resolved path to the KG database file.
    """
    # Priority 1: Explicit environment override
    env_path = os.environ.get("GRAPH_DB_PATH")
    if env_path:
        return Path(env_path).expanduser()

    # Priority 2: XDG data directory (shared KG — canonical location)
    xdg_path = data_dir() / "kg" / "knowledge_graph.db"
    xdg_path.parent.mkdir(parents=True, exist_ok=True)

    # Priority 3: Workspace-local fallback (deprecated, warn if found)
    if workspace:
        ws = Path(workspace)
        local_db = ws / "knowledge_graph.db"
        if local_db.exists():
            logger.warning(
                "DEPRECATED: Found workspace-local knowledge_graph.db at %s. "
                "Set GRAPH_DB_PATH=%s to use it explicitly, or remove it to "
                "use the shared XDG database at %s.",
                local_db,
                local_db,
                xdg_path,
            )

    return xdg_path


def mcp_config_path() -> Path:
    """Return the path to the global MCP config for cross-IDE discovery.

    Default: ``~/.config/agent-utilities/mcp_config.json``
    """
    return config_dir() / "mcp_config.json"


def a2a_config_path() -> Path:
    """Return the path to the global A2A config for agent discovery.

    Default: ``~/.config/agent-utilities/a2a_config.json``
    """
    return config_dir() / "a2a_config.json"


def ontology_dir() -> Path:
    """Return the path for user-provided external ontologies.

    Default: ``~/.local/share/agent-utilities/ontologies/``

    Package-bundled ontologies (``ontology.ttl``, ``ontology_banking.ttl``, etc.)
    are shipped as package data. This directory is for user-provided domain
    extensions that complement the built-in ontologies.
    """
    return data_dir() / "ontologies"


def runtime_dir() -> Path:
    """Return the path for runtime artifacts (not git-tracked).

    Default: ``~/.local/share/agent-utilities/runtime/``

    Contains:
        - ``harness_registry.json`` — Agentic harness state
        - ``manifests/`` — Evolution manifests
    """
    return data_dir() / "runtime"


def research_dir() -> Path:
    """Return the path for downloaded research papers.

    Default: ``~/.local/share/agent-utilities/research/``
    """
    return data_dir() / "research"


def memory_view_dir() -> Path:
    """Return the path for materialized memory views.

    CONCEPT:KG-2.1 — Observational Memory Bridge

    Default: ``~/.local/share/agent-utilities/memory/``

    Override via ``AGENT_UTILITIES_MEMORY_DIR`` environment variable.

    Contains:
        - ``observations.md`` — Recent observation notes with priorities
        - ``reflections.md`` — Long-term condensed memory
        - ``profile.md`` — Stable user identity context
        - ``active.md`` — Current working context
        - ``.memory_cursor.json`` — Materialization state tracking
    """
    override = os.environ.get("AGENT_UTILITIES_MEMORY_DIR")
    if override:
        return Path(override).expanduser()
    return data_dir() / "memory"


def ensure_dirs() -> None:
    """Create all XDG directories on first run.

    Called during server startup or KG initialization to ensure the
    directory structure exists before any file operations.
    """
    dirs = [
        config_dir(),
        data_dir() / "kg",
        cache_dir(),
        ontology_dir(),
        runtime_dir(),
        research_dir(),
        memory_view_dir(),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "XDG directories ensured at: config=%s, data=%s, cache=%s",
        config_dir(),
        data_dir(),
        cache_dir(),
    )
