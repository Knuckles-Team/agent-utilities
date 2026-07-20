#!/usr/bin/python
"""XDG-Compliant Path Resolution Module.

CONCEPT:AU-OS.config.agent-os-infrastructure — Agent OS Infrastructure (Extension)

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

Directory roots can be overridden via environment variables:
    - ``AGENT_UTILITIES_CONFIG_DIR``
    - ``AGENT_UTILITIES_DATA_DIR``
    - ``AGENT_UTILITIES_CACHE_DIR``
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
        - ``config.json`` — typed, reference-only runtime configuration
        - ``runtime-secrets.json`` — optional private ``env://`` value source
        - ``policies/`` — Global policy overrides
    """
    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))


def runtime_secrets_path() -> Path:
    """Return the implicit private source for referenced runtime secret values.

    The filename is fixed beneath :func:`config_dir`; there is intentionally no
    setting for a machine-specific file path.
    """
    return config_dir() / "runtime-secrets.json"


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


def skills_dir() -> Path:
    """Return the XDG data directory for custom/user-installed agent skills.

    Default: ``~/.local/share/agent-utilities/skills/``

    Override via ``AGENT_UTILITIES_SKILLS_DIR`` environment variable.

    This is the standard drop-in location for skills the agent should have at its
    disposal beyond packaged providers. Operator-owned skills use the flat
    ``skills/<skill>/SKILL.md`` form. Nested provider roots contain immutable
    ``skills/<provider>/.generations/<digest>/`` trees selected by a closed v2
    marker. They load only when registration, source, marker, and content digests
    agree. Set ``custom_skills_directory`` / ``CUSTOM_SKILLS_DIRECTORY`` for a
    separate operator-managed tree. Duplicate skill identities fail closed.
    """
    override = os.environ.get("AGENT_UTILITIES_SKILLS_DIR")
    if override:
        return Path(override).expanduser()
    return data_dir() / "skills"


def prompts_dir() -> Path:
    """Return the XDG overlay directory for user-supplied system-prompt blueprints.

    Default: ``~/.config/agent-utilities/prompts/``

    Override via ``AGENT_UTILITIES_PROMPTS_DIR`` environment variable.

    CONCEPT:AU-KG.compute.user-override-prompt-library — the user-override layer of the prompt library, mirroring
    :func:`skills_dir`. ``*.json`` blueprints placed here take precedence over
    the packaged base prompts and fleet-contributed (``agent_utilities.prompt_providers``)
    prompts of the same id when the KG prompt registry is ingested
    (``ingest_prompts_to_graph``). It lives under the *config* dir (not data),
    because these are operator-authored overrides, not runtime artifacts.
    """
    override = os.environ.get("AGENT_UTILITIES_PROMPTS_DIR")
    if override:
        return Path(override).expanduser()
    return config_dir() / "prompts"


def unified_prompts_dir() -> Path:
    """Return the XDG *data* directory for materialized prompt-provider blueprints.

    Default: ``~/.local/share/agent-utilities/prompts/``

    CONCEPT:AU-OS.governance.concept-2 — one leg of the unified install tree. The
    ``agent-utilities install`` command materializes every ``prompt_providers``
    contribution (plus the hub's own base prompts) here as content-addressed
    ``prompts/<provider>/.generations/<digest>/*.json``. The prompt registry uses a
    generation only when it exactly matches the current distribution-owned source;
    otherwise it reads that validated current source directly.

    Distinct from :func:`prompts_dir` (the *config*-dir operator override layer):
    this is the runtime materialization sink under the *data* dir, alongside
    :func:`skills_dir` and :func:`ontology_dir`.
    """
    return data_dir() / "prompts"


def kg_db_path() -> Path:
    """Return the canonical XDG path for the embedded graph database.

    External and custom database locations are supplied through
    ``GRAPH_DB_CONNECTION_PROFILE_REF``.  The embedded default intentionally
    has no independent path override so every non-default connection follows
    the same secret-reference contract.
    """
    xdg_path = data_dir() / "kg" / "knowledge_graph.db"
    xdg_path.parent.mkdir(parents=True, exist_ok=True)
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

    CONCEPT:AU-KG.memory.observational-memory-bridge — Observational Memory Bridge

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


def messaging_sessions_dir() -> Path:
    """Return the path for messaging backend session data.

    CONCEPT:AU-ECO.messaging.native-backend-abstraction — Native Messaging Backend Abstraction

    Default: ``~/.local/share/agent-utilities/messaging/``

    Contains:
        - ``sessions/`` — Backend-specific session/auth state
        - ``history/`` — Local message history cache
    """
    return data_dir() / "messaging"


def messaging_config_path() -> Path:
    """Return the path to the messaging section in the global config.

    CONCEPT:AU-ECO.messaging.native-backend-abstraction — Native Messaging Backend Abstraction

    Default: ``~/.config/agent-utilities/config.json``
    (messaging keys are inside the same config.json, not a separate file)

    See Also:
        The ``messaging_*`` keys in config.json are loaded by
        ``_load_xdg_json_config()`` in ``core/config.py`` and become
        environment variables that the ``MessagingRegistry`` reads.
    """
    return config_dir() / "config.json"


def services_config_path() -> Path:
    """Return the path to the dashboard services YAML configuration.

    CONCEPT:AU-OS.config.gateway-service-dashboard — Gateway Service Dashboard

    Default: ``~/.config/agent-utilities/services.yaml``

    Contains the user-editable service layout for the gateway dashboard.
    Auto-generated from ``mcp_config.json`` on first run if not present.
    """
    return config_dir() / "services.yaml"


def dashboard_layout_path() -> Path:
    """Return the path to persisted dashboard layout state.

    CONCEPT:AU-OS.config.gateway-service-dashboard — Gateway Service Dashboard

    Default: ``~/.local/share/agent-utilities/layout.yaml``

    Stores user customizations (column order, collapsed groups, theme).
    """
    return data_dir() / "layout.yaml"


def log_dir() -> Path:
    """Return the XDG log directory for agent-utilities.

    Default: ``~/.cache/agent-utilities/log/`` or standard platform user_log_path.

    Override via ``AGENT_UTILITIES_LOG_DIR`` environment variable.
    """
    override = os.environ.get("AGENT_UTILITIES_LOG_DIR")
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_log_path(APP_NAME, APP_AUTHOR))


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
        messaging_sessions_dir(),
        messaging_sessions_dir() / "sessions",
        messaging_sessions_dir() / "history",
        skills_dir(),
        prompts_dir(),
        log_dir(),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "XDG directories ensured at: config=%s, data=%s, cache=%s, log=%s",
        config_dir(),
        data_dir(),
        cache_dir(),
        log_dir(),
    )
