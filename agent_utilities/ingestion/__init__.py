"""Agent log ingestion (CONCEPT:ECO-4.38 / ECO-4.42).

Auto-detects installed AI coding agents, parses their session logs into the
normalized usage schema, and sinks them locally or pushes to a central engine.
"""

from .collector import (
    collect_local_sessions,
    collect_paths,
    iter_local_bundles,
    push_local_sessions,
)

__all__ = [
    "collect_local_sessions",
    "collect_paths",
    "iter_local_bundles",
    "push_local_sessions",
]
