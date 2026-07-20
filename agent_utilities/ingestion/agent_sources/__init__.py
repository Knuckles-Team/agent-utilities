"""Pluggable agent-source parser registry (CONCEPT:AU-ECO.connector.agent-source-ingestion).

Auto-detects installed AI coding agents and parses their session logs into the
normalized usage schema. ``ensure_parsers_loaded()`` registers all 36 sources;
``detect_installed()`` returns only those present on this host.
"""

from .registry import (
    AGENT_REGISTRY,
    AgentSource,
    all_sources,
    detect_installed,
    ensure_parsers_loaded,
    get_source,
    register_source,
)

__all__ = [
    "AGENT_REGISTRY",
    "AgentSource",
    "all_sources",
    "detect_installed",
    "ensure_parsers_loaded",
    "get_source",
    "register_source",
]
