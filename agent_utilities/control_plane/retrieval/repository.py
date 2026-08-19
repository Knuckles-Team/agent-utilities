"""Small deterministic repositories for retrieval generations and cache entries."""

from __future__ import annotations

import threading

from .errors import (
    CleanupIncompleteError,
    GenerationConflictError,
    GenerationNotFoundError,
    RetrievalReplayError,
)
from .models import (
    CleanupCheckpoint,
    GenerationManifest,
    GenerationRecord,
    RetrievalResult,
)

__all__ = ["InMemoryGenerationCatalog", "InMemoryRetrievalCache"]


class InMemoryGenerationCatalog:
    """CAS catalog with immutable manifests and fail-closed lifecycle state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[tuple[str, str, str], GenerationRecord] = {}
        self._current: dict[tuple[str, str], str] = {}

    def current(self, tenant_ref: str, graph_ref: str) -> GenerationRecord | None:
        with self._lock:
            generation_id = self._current.get((tenant_ref, graph_ref))
            if generation_id is None:
                return None
            return self._records.get((tenant_ref, graph_ref, generation_id))

    def get(
        self, tenant_ref: str, graph_ref: str, generation_id: str
    ) -> GenerationRecord | None:
        with self._lock:
            return self._records.get((tenant_ref, graph_ref, generation_id))

    def publish(
        self,
        manifest: GenerationManifest,
        *,
        expected_current: str | None,
    ) -> GenerationRecord:
        key = (manifest.tenant_ref, manifest.graph_ref)
        record_key = (*key, manifest.generation_id)
        with self._lock:
            actual_current = self._current.get(key)
            if actual_current != expected_current:
                raise GenerationConflictError("generation_current_cas_mismatch")
            prior = self._records.get(record_key)
            if prior is not None:
                if prior.manifest != manifest:
                    raise GenerationConflictError("immutable_generation_conflict")
                if actual_current == manifest.generation_id and prior.state == "active":
                    return prior
                raise GenerationConflictError("generation_identity_reuse")
            record = GenerationRecord(manifest=manifest, state="active")
            self._records[record_key] = record
            self._current[key] = manifest.generation_id
            return record

    def retire(self, generation: GenerationManifest) -> GenerationRecord:
        key = (generation.tenant_ref, generation.graph_ref)
        record_key = (*key, generation.generation_id)
        with self._lock:
            record = self._records.get(record_key)
            if record is None:
                raise GenerationNotFoundError("generation_missing")
            if record.manifest != generation:
                raise GenerationConflictError("generation_manifest_drift")
            if self._current.get(key) == generation.generation_id:
                raise GenerationConflictError("current_generation_cannot_retire")
            if record.state == "retiring":
                return record
            if record.state != "active":
                raise GenerationConflictError("generation_state_transition_invalid")
            updated = GenerationRecord(manifest=generation, state="retiring")
            self._records[record_key] = updated
            return updated

    def finish_cleanup(
        self,
        generation: GenerationManifest,
        checkpoint: CleanupCheckpoint,
    ) -> GenerationRecord:
        key = (generation.tenant_ref, generation.graph_ref, generation.generation_id)
        with self._lock:
            self._require_cleanup_target(generation, checkpoint)
            if not checkpoint.complete:
                raise CleanupIncompleteError("cleanup_checkpoint_incomplete")
            updated = GenerationRecord(
                manifest=generation,
                state="deleted",
                cleanup_checkpoint=checkpoint,
            )
            self._records[key] = updated
        return updated

    def fail_cleanup(
        self,
        generation: GenerationManifest,
        checkpoint: CleanupCheckpoint | None,
    ) -> GenerationRecord:
        key = (generation.tenant_ref, generation.graph_ref, generation.generation_id)
        with self._lock:
            if checkpoint is not None and checkpoint.complete:
                raise GenerationConflictError("complete_checkpoint_requires_finish")
            target = self._require_cleanup_target(generation, checkpoint)
            persisted_checkpoint = (
                checkpoint if checkpoint is not None else target.cleanup_checkpoint
            )
            updated = GenerationRecord(
                manifest=generation,
                state="cleanup_failed",
                cleanup_checkpoint=persisted_checkpoint,
            )
            self._records[key] = updated
        return updated

    def _require_cleanup_target(
        self,
        generation: GenerationManifest,
        checkpoint: CleanupCheckpoint | None,
    ) -> GenerationRecord:
        key = (generation.tenant_ref, generation.graph_ref, generation.generation_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                raise GenerationNotFoundError("generation_missing")
            if record.manifest != generation:
                raise GenerationConflictError("generation_manifest_drift")
            if record.state not in {"retiring", "cleanup_failed"}:
                raise GenerationConflictError("generation_not_retiring")
            if checkpoint is not None:
                if checkpoint.generation_id != generation.generation_id:
                    raise GenerationConflictError("cleanup_generation_drift")
                if checkpoint.index_ref != generation.index_ref:
                    raise GenerationConflictError("cleanup_index_drift")
                if checkpoint.expected_vectors != generation.expected_chunks:
                    raise GenerationConflictError("cleanup_expected_count_drift")
            return record


class InMemoryRetrievalCache:
    """Bounded exact-key cache; entries are never fuzzy or cross-tenant."""

    def __init__(self, *, max_entries: int = 4096) -> None:
        if max_entries < 1 or max_entries > 65_536:
            raise ValueError("cache_capacity_invalid")
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._items: dict[str, RetrievalResult] = {}

    def get(self, request_digest: str) -> RetrievalResult | None:
        with self._lock:
            return self._items.get(request_digest)

    def put(self, result: RetrievalResult) -> None:
        key = result.request.request_digest
        with self._lock:
            prior = self._items.get(key)
            if prior is not None and prior != result:
                raise RetrievalReplayError("cache_identity_conflict")
            if prior is None and len(self._items) >= self.max_entries:
                raise RetrievalReplayError("cache_capacity_exceeded")
            self._items[key] = result

    def delete(self, request_digest: str) -> None:
        with self._lock:
            self._items.pop(request_digest, None)
