"""CONCEPT:AU-KG.memory.live-refreshable-artifact-models — KG-backed Live Artifact refresh resolver.

Re-derives an artifact's data by running its bound ``source_query`` against the epistemic KG
(:meth:`KnowledgeGraph.query`). The legacy AU artifact gateway is retired; this
resolver remains available to callers of the in-process refresh service.

Defensive by design: any failure (KG unavailable, bad query, empty store) raises, and the refresh
service then **preserves the prior render** (bi-temporal valid-time, KG-2.11) — exactly the desired
"failed refresh keeps the last good output" behavior.
"""

from __future__ import annotations

from typing import Any

from .models import LiveArtifact


def kg_source_resolver(artifact: LiveArtifact) -> dict[str, Any]:
    """Run ``artifact.source_query`` against the KG and shape the result into the artifact's data dict.

    The shaped data exposes ``rows`` (the raw result list), ``count``, and ``first`` (the first row,
    for scalar templates). Templates interpolate these via ``{{data.count}}`` / ``{{#each data.rows}}``.

    Raises if there is no bound query or the KG query fails — the caller (RefreshService) then keeps
    the prior render.
    """
    query = (artifact.source_query or "").strip()
    if not query:
        raise ValueError(
            f"artifact {artifact.artifact_id} has no source_query to refresh from"
        )

    # Imported lazily so the live_artifacts package has no hard KG-construction cost at import time.
    from agent_utilities.knowledge_graph.facade import KnowledgeGraph

    kg = KnowledgeGraph()
    rows = kg.query(query)
    if not isinstance(rows, list):
        rows = list(rows) if rows is not None else []
    return {
        "rows": rows,
        "count": len(rows),
        "first": rows[0] if rows else {},
        "source_query": query,
    }
