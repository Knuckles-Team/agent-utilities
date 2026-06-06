"""CONCEPT:KG-2.24 — Refresh service: re-derive artifact data from the KG; preserve prior on failure.

``refresh`` re-runs the artifact's bound source (a callable that returns fresh ``data`` — in
production a KG query over ``source_node_ids``), validates it against the bounded-JSON contract, and
re-renders. **If the new derivation fails** (source error, bounded-JSON violation, render error) the
prior ``data``/``last_rendered`` is preserved — the framework-native expression of open-design's
"failed refresh preserves the prior preview", backed by the KG's bi-temporal valid-time (KG-2.11).

Every attempt (success or failure) is appended to an in-memory ``refreshes`` log (the framework
analogue of ``refreshes.jsonl``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import BoundedJSONError, LiveArtifact
from .store import LiveArtifactStore

logger = logging.getLogger(__name__)

# A source returns fresh data for an artifact (given the artifact, e.g. by querying the KG).
DataSource = Callable[[LiveArtifact], dict[str, Any]]


@dataclass(slots=True)
class RefreshResult:
    """Outcome of a single refresh attempt."""

    artifact_id: str
    ok: bool
    reason: str = ""
    rendered: str = ""
    at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class RefreshService:
    """Re-derive Live Artifacts from their bound source, preserving prior output on failure."""

    def __init__(self, store: LiveArtifactStore) -> None:
        self._store = store
        self.refreshes: list[RefreshResult] = []  # append-only attempt log

    def refresh(self, artifact_id: str, source: DataSource) -> RefreshResult:
        """Refresh one artifact from ``source``. Never raises; returns a :class:`RefreshResult`."""
        artifact = self._store.get(artifact_id)
        if artifact is None:
            res = RefreshResult(artifact_id, ok=False, reason="not found")
            self.refreshes.append(res)
            return res

        prior_data = dict(artifact.data)
        prior_render = artifact.last_rendered
        try:
            new_data = source(artifact)
            if not isinstance(new_data, dict):
                raise BoundedJSONError("source did not return an object")
            candidate = artifact.model_copy(update={"data": new_data})
            candidate.validate_data()
            rendered = candidate.render()
        except Exception as exc:  # preserve prior on ANY failure (bi-temporal valid-time)
            artifact.data = prior_data
            artifact.last_rendered = prior_render
            self._store.put(artifact)
            res = RefreshResult(artifact_id, ok=False, reason=str(exc), rendered=prior_render)
            self.refreshes.append(res)
            logger.info("refresh failed for %s (prior preserved): %s", artifact_id, exc)
            return res

        artifact.data = new_data
        artifact.last_rendered = rendered
        artifact.refresh_count += 1
        artifact.provenance.generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._store.put(artifact)
        res = RefreshResult(artifact_id, ok=True, rendered=rendered)
        self.refreshes.append(res)
        return res
