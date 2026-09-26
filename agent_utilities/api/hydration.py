"""Public control-plane port for legacy dashboard hydration requests.

The GraphOS transport calls this port instead of importing AU's private
hydration manager. Mutations use the same manifest-gated source-sync path as
the canonical ingestion entry point.
"""

from __future__ import annotations

from typing import Any


def hydrate_source(engine: Any, source: str) -> dict[str, Any]:
    """Run a full, manifest-gated sync for one named source."""
    from agent_utilities.knowledge_graph.core.source_sync import sync_source

    return sync_source(engine, source, mode="full")


def hydrate_all(engine: Any) -> dict[str, Any]:
    """Start the canonical configured-source sweep."""
    from agent_utilities.knowledge_graph.core.source_sync import sync_source

    return sync_source(engine, "all", mode="full")


def hydration_status() -> dict[str, Any]:
    """Report which legacy hydration sources are configured."""
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager

    return HydrationManager().get_status()


__all__ = ["hydrate_all", "hydrate_source", "hydration_status"]
