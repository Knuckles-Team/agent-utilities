"""CONCEPT:KG-2.24 — Live Artifact store (in-memory index + optional KG persistence).

``create`` validates the bounded-JSON contract, renders once, and persists. Persistence is via an
injectable ``writer`` callable so the store is unit-testable without a live KG; in production the
writer is wired to the ``graph_write`` MCP tool (``mcp/kg_server.py``) so artifacts become KG nodes
with ``source_node_ids`` edges.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .models import LiveArtifact

logger = logging.getLogger(__name__)

# A writer persists an artifact's serialized form to a durable backend (e.g. the KG).
ArtifactWriter = Callable[[dict[str, Any]], None]


class LiveArtifactStore:
    """Index of :class:`LiveArtifact` objects with optional durable write-through."""

    def __init__(self, *, writer: ArtifactWriter | None = None) -> None:
        self._artifacts: dict[str, LiveArtifact] = {}
        self._writer = writer

    def create(self, artifact: LiveArtifact) -> LiveArtifact:
        """Validate, render once, persist, and index a new artifact."""
        artifact.validate_data()  # raises BoundedJSONError on violation
        artifact.last_rendered = artifact.render()
        self._artifacts[artifact.artifact_id] = artifact
        self._persist(artifact)
        return artifact

    def get(self, artifact_id: str) -> LiveArtifact | None:
        return self._artifacts.get(artifact_id)

    def put(self, artifact: LiveArtifact) -> None:
        """Index an already-existing artifact (used by the refresh service after re-derivation)."""
        self._artifacts[artifact.artifact_id] = artifact
        self._persist(artifact)

    def ids(self) -> list[str]:
        return sorted(self._artifacts)

    def _persist(self, artifact: LiveArtifact) -> None:
        if self._writer is None:
            return
        try:
            self._writer(artifact.model_dump())
        except (
            Exception
        ):  # durable write is best-effort; never lose the in-memory artifact
            logger.warning(
                "artifact persistence failed for %s",
                artifact.artifact_id,
                exc_info=True,
            )


_default_store: LiveArtifactStore | None = None


def get_live_artifact_store() -> LiveArtifactStore:
    """Process-wide default store (lazy singleton)."""
    global _default_store
    if _default_store is None:
        _default_store = LiveArtifactStore()
    return _default_store
