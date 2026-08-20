"""Governed retrieval orchestration and generation lifecycle."""

from __future__ import annotations

from collections.abc import Callable

from .errors import (
    CleanupIncompleteError,
    GenerationConflictError,
    GenerationNotFoundError,
    RetrievalAuthorizationError,
    RetrievalDomainError,
    StaleGenerationError,
)
from .models import (
    MAX_HITS,
    AuthorizationEvidence,
    CleanupCheckpoint,
    EngineCandidate,
    GenerationManifest,
    GenerationRecord,
    RankedHit,
    RetrievalRequest,
    RetrievalResult,
)
from .protocols import (
    GenerationCatalog,
    RetrievalAuthorizationAdapter,
    RetrievalCache,
    RetrievalEngineAdapter,
)

__all__ = ["GenerationLifecycle", "GovernedRetriever"]


class GovernedRetriever:
    """ACL-first retrieval facade over the engine and exact-key cache.

    The engine may compute vectors and candidate similarity, but this facade
    owns the ordering invariant: authorize first, verify every returned row,
    then rank, then cache.  A denied or drifted row aborts the response before
    it can enter ranking, cache, or citation construction.
    """

    def __init__(
        self,
        *,
        catalog: GenerationCatalog,
        authorization: RetrievalAuthorizationAdapter,
        engine: RetrievalEngineAdapter,
        cache: RetrievalCache | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.catalog = catalog
        self.authorization = authorization
        self.engine = engine
        self.cache = cache
        self.clock = clock or (lambda: 0)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Authorize, search, rank, and optionally cache one exact request."""

        current = self._require_current(request)
        now = self.clock()
        try:
            evidence = self.authorization.authorize(request)
        except Exception as exc:  # noqa: BLE001 — fail closed at the ACL boundary
            raise RetrievalAuthorizationError("acl_authorization_failed") from exc
        self._validate_evidence(request, current, evidence, now)

        if self.cache is not None:
            cached = self.cache.get(request.request_digest)
            if cached is not None:
                try:
                    self._validate_cached(cached, request, evidence, current, now)
                except RetrievalDomainError:
                    self.cache.delete(request.request_digest)
                else:
                    return cached

        if not evidence.allowed_chunks:
            result = RetrievalResult(
                request=request,
                authorization=evidence,
                hits=(),
            )
            self._cache(result)
            return result

        try:
            candidates = self.engine.search(request, evidence)
        except Exception as exc:  # noqa: BLE001 — degraded reads cannot become success
            raise RetrievalDomainError("engine_search_failed") from exc
        verified = self._verify_candidates(request, evidence, candidates)
        ranked = sorted(
            verified,
            key=lambda candidate: (
                -candidate.score,
                candidate.chunk.chunk_key,
                candidate.engine_result_ref,
            ),
        )[: request.limit]
        hits = tuple(
            RankedHit(
                chunk=candidate.chunk,
                index_ref=candidate.index_ref,
                score=candidate.score,
                rank=rank,
                authorization_digest=evidence.evidence_digest,
                engine_result_ref=candidate.engine_result_ref,
            )
            for rank, candidate in enumerate(ranked, start=1)
        )
        result = RetrievalResult(
            request=request,
            authorization=evidence,
            hits=hits,
        )
        self._cache(result)
        return result

    def _require_current(self, request: RetrievalRequest) -> GenerationRecord:
        current = self.catalog.current(request.tenant_ref, request.graph_ref)
        if current is None or current.state != "active":
            raise StaleGenerationError("current_generation_unresolved")
        manifest = current.manifest
        if (
            manifest.tenant_ref != request.tenant_ref
            or manifest.graph_ref != request.graph_ref
            or manifest.generation_id != request.index_ref.generation_id
            or manifest.index_ref != request.index_ref
        ):
            raise StaleGenerationError("request_generation_stale")
        return current

    @staticmethod
    def _validate_evidence(
        request: RetrievalRequest,
        current: GenerationRecord,
        evidence: AuthorizationEvidence,
        now: int,
    ) -> None:
        if now < evidence.evaluated_at or now >= evidence.expires_at:
            raise RetrievalAuthorizationError("acl_evidence_expired_or_future")
        if evidence.request_id != request.request_id:
            raise RetrievalAuthorizationError("acl_request_drift")
        if evidence.tenant_ref != request.tenant_ref:
            raise RetrievalAuthorizationError("acl_tenant_drift")
        if evidence.graph_ref != request.graph_ref:
            raise RetrievalAuthorizationError("acl_graph_drift")
        if evidence.index_ref != request.index_ref:
            raise RetrievalAuthorizationError("acl_index_drift")
        if evidence.generation_id != current.manifest.generation_id:
            raise RetrievalAuthorizationError("acl_generation_stale")
        for chunk in evidence.allowed_chunks:
            if chunk.document.source_version != current.manifest.source_version:
                raise RetrievalAuthorizationError("acl_source_version_drift")

    @staticmethod
    def _verify_candidates(
        request: RetrievalRequest,
        evidence: AuthorizationEvidence,
        candidates: tuple[EngineCandidate, ...],
    ) -> tuple[EngineCandidate, ...]:
        if len(candidates) > MAX_HITS:
            raise RetrievalDomainError("engine_candidate_bound_exceeded")
        verified: list[EngineCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.index_ref != request.index_ref:
                raise RetrievalDomainError("engine_index_drift")
            if not evidence.allows(candidate.chunk):
                raise RetrievalAuthorizationError("engine_returned_denied_candidate")
            if candidate.chunk.chunk_key in seen:
                raise RetrievalDomainError("engine_candidate_duplicate")
            seen.add(candidate.chunk.chunk_key)
            verified.append(candidate)
        return tuple(verified)

    def _validate_cached(
        self,
        cached: RetrievalResult,
        request: RetrievalRequest,
        evidence: AuthorizationEvidence,
        current: GenerationRecord,
        now: int,
    ) -> None:
        if cached.request != request:
            raise RetrievalDomainError("cache_request_drift")
        if cached.authorization.evidence_digest != evidence.evidence_digest:
            raise RetrievalAuthorizationError("cache_acl_evidence_drift")
        self._validate_evidence(request, current, cached.authorization, now)

    def _cache(self, result: RetrievalResult) -> None:
        if self.cache is None:
            return
        try:
            self.cache.put(result)
        except Exception as exc:  # noqa: BLE001 — cache failure is fail closed
            raise RetrievalDomainError("cache_write_failed") from exc


class GenerationLifecycle:
    """Generation swap and cleanup coordinator with fail-closed checkpoints."""

    def __init__(
        self,
        *,
        catalog: GenerationCatalog,
        engine: RetrievalEngineAdapter,
    ) -> None:
        self.catalog = catalog
        self.engine = engine

    def publish(
        self,
        manifest: GenerationManifest,
        *,
        expected_current: str | None,
    ) -> GenerationRecord:
        """CAS-publish a complete active generation manifest."""

        return self.catalog.publish(manifest, expected_current=expected_current)

    def cleanup(self, generation: GenerationManifest) -> GenerationRecord:
        """Delete one retired generation only after complete engine proof."""

        record = self.catalog.get(
            generation.tenant_ref,
            generation.graph_ref,
            generation.generation_id,
        )
        if record is None:
            raise GenerationNotFoundError("generation_missing")
        if record.manifest != generation:
            raise GenerationConflictError("generation_manifest_drift")
        if record.state == "active":
            record = self.catalog.retire(generation)
        elif record.state not in {"retiring", "cleanup_failed"}:
            raise GenerationConflictError("generation_not_cleanup_eligible")

        checkpoint: CleanupCheckpoint | None = None
        try:
            checkpoint = self.engine.cleanup_generation(generation)
            if not isinstance(checkpoint, CleanupCheckpoint):
                self.catalog.fail_cleanup(generation, None)
                raise CleanupIncompleteError("cleanup_checkpoint_invalid")
            if checkpoint.expected_vectors != generation.expected_chunks:
                self.catalog.fail_cleanup(generation, None)
                raise CleanupIncompleteError("cleanup_expected_count_drift")
            if not checkpoint.complete:
                self.catalog.fail_cleanup(generation, checkpoint)
                raise CleanupIncompleteError("cleanup_checkpoint_incomplete")
            return self.catalog.finish_cleanup(generation, checkpoint)
        except CleanupIncompleteError:
            raise
        except Exception as exc:  # noqa: BLE001 — cleanup failure is fail closed
            try:
                self.catalog.fail_cleanup(generation, checkpoint)
            except Exception as state_exc:  # noqa: BLE001 — preserve fail-closed state
                raise CleanupIncompleteError(
                    "cleanup_failure_state_unrecorded"
                ) from state_exc
            raise CleanupIncompleteError("cleanup_failed") from exc

    def reembed(
        self,
        old_generation: GenerationManifest,
        new_generation: GenerationManifest,
    ) -> GenerationRecord:
        """Publish a new generation, retire the old one, and prove cleanup.

        The new generation becomes current only through the catalog CAS.  If
        old-vector cleanup is partial, the new generation remains current but
        this method raises and leaves the old generation explicitly
        ``cleanup_failed``; it never claims a complete replacement.
        """

        if (
            old_generation.tenant_ref != new_generation.tenant_ref
            or old_generation.graph_ref != new_generation.graph_ref
        ):
            raise GenerationConflictError("reembed_tenant_graph_drift")
        current = self.catalog.current(
            old_generation.tenant_ref,
            old_generation.graph_ref,
        )
        if (
            current is None
            or current.manifest != old_generation
            or current.state != "active"
        ):
            raise StaleGenerationError("reembed_old_generation_stale")
        new_record = self.publish(
            new_generation,
            expected_current=old_generation.generation_id,
        )
        self.catalog.retire(old_generation)
        self.cleanup(old_generation)
        return new_record
