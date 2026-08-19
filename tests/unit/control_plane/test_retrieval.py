"""Focused NE-089 acceptance fixtures for governed retrieval."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.retrieval import (
    ArtifactRef,
    AuthorizationEvidence,
    ChunkRef,
    CitationRecord,
    CitationRef,
    CleanupCheckpoint,
    CleanupIncompleteError,
    DocumentRef,
    EmbeddingModelRef,
    EngineCandidate,
    EvaluationMetric,
    EvaluationRecord,
    GenerationLifecycle,
    GenerationManifest,
    GovernedRetriever,
    IndexRef,
    InMemoryGenerationCatalog,
    InMemoryRetrievalCache,
    RetrievalAuthorizationError,
    RetrievalRequest,
    StaleGenerationError,
    VectorAuthorityRef,
    canonical_digest,
)


def _digest(letter: str) -> str:
    digit = format(sum(ord(char) for char in letter) % 16, "x")
    return f"sha256:{digit * 64}"


def _artifact(letter: str, name: str = "artifact:source") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=name,
        digest=_digest(letter),
        media_type="text/markdown",
    )


def _model() -> EmbeddingModelRef:
    return EmbeddingModelRef(
        model_id="model:retrieval",
        version="1.0.0",
        digest=_digest("m"),
        dimension=3,
    )


def _index(generation: str = "generation:one") -> IndexRef:
    return IndexRef(
        tenant_ref="tenant:one",
        graph_ref="graph:knowledge",
        index_id=f"index:{generation}",
        version="1.0.0",
        digest=_digest("i"),
        generation_id=generation,
        model=_model(),
        dimension=3,
    )


def _document(
    number: str = "one",
    generation: str = "generation:one",
    digest_letter: str = "d",
) -> DocumentRef:
    artifact = _artifact(digest_letter, f"artifact:document-{number}")
    return DocumentRef(
        tenant_ref="tenant:one",
        graph_ref="graph:knowledge",
        document_id=f"document:{number}",
        source_ref="source:git",
        source_version="revision:42",
        content_digest=artifact.digest,
        generation_id=generation,
        artifact_ref=artifact,
    )


def _chunk(
    number: str = "one",
    generation: str = "generation:one",
    digest_letter: str = "d",
) -> ChunkRef:
    artifact = _artifact(digest_letter, f"artifact:chunk-{number}")
    document = _document(number, generation, digest_letter)
    return ChunkRef(
        document=document,
        chunk_id=f"chunk:{number}",
        ordinal=0,
        content_digest=artifact.digest,
        artifact_ref=artifact,
    )


def _vector(chunk: ChunkRef, letter: str = "v") -> VectorAuthorityRef:
    return VectorAuthorityRef(
        tenant_ref=chunk.document.tenant_ref,
        graph_ref=chunk.document.graph_ref,
        generation_id=chunk.document.generation_id,
        model=_model(),
        dimension=3,
        vector_ref=f"engine-vector:{chunk.chunk_id}",
        vector_digest=_digest(letter),
        source_content_digest=chunk.content_digest,
    )


def _manifest(generation: str = "generation:one") -> GenerationManifest:
    index = _index(generation)
    return GenerationManifest(
        tenant_ref=index.tenant_ref,
        graph_ref=index.graph_ref,
        generation_id=generation,
        source_version="revision:42",
        source_digest=_digest("s"),
        index_ref=index,
        model=_model(),
        expected_chunks=2,
    )


def _request(generation: str = "generation:one") -> RetrievalRequest:
    index = _index(generation)
    query_artifact = _artifact("q", "artifact:query")
    query_vector = VectorAuthorityRef(
        tenant_ref=index.tenant_ref,
        graph_ref=index.graph_ref,
        generation_id=generation,
        model=_model(),
        dimension=3,
        vector_ref="engine-vector:query",
        vector_digest=_digest("x"),
        source_content_digest=query_artifact.digest,
    )
    return RetrievalRequest(
        request_id="request:one",
        tenant_ref=index.tenant_ref,
        graph_ref=index.graph_ref,
        index_ref=index,
        query_artifact_ref=query_artifact,
        query_vector_ref=query_vector,
        limit=2,
    )


def _evidence(request: RetrievalRequest, chunks: tuple[ChunkRef, ...]) -> AuthorizationEvidence:
    return AuthorizationEvidence(
        authorization_id="acl:request-one",
        request_id=request.request_id,
        tenant_ref=request.tenant_ref,
        graph_ref=request.graph_ref,
        index_ref=request.index_ref,
        generation_id=request.index_ref.generation_id,
        policy_ref="policy:retrieval",
        allowed_chunks=chunks,
        denied_count=0,
        evaluated_at=90,
        expires_at=120,
    )


def _candidate(chunk: ChunkRef, score: float, letter: str) -> EngineCandidate:
    return EngineCandidate(
        chunk=chunk,
        index_ref=_index(chunk.document.generation_id),
        vector_ref=_vector(chunk, letter),
        score=score,
        engine_result_ref=f"engine-result:{chunk.chunk_id}",
    )


class _Authorization:
    def __init__(self, evidence: AuthorizationEvidence) -> None:
        self.evidence = evidence
        self.calls = 0

    def authorize(self, request: RetrievalRequest) -> AuthorizationEvidence:
        self.calls += 1
        assert request.request_id == self.evidence.request_id
        return self.evidence


class _Engine:
    def __init__(self, candidates: tuple[EngineCandidate, ...] = ()) -> None:
        self.candidates = candidates
        self.calls = 0
        self.cleanup_report: CleanupCheckpoint | None = None

    def search(
        self,
        request: RetrievalRequest,
        authorization: AuthorizationEvidence,
    ) -> tuple[EngineCandidate, ...]:
        self.calls += 1
        assert authorization.request_id == request.request_id
        return self.candidates

    def cleanup_generation(self, generation: GenerationManifest) -> CleanupCheckpoint:
        assert self.cleanup_report is not None
        assert self.cleanup_report.generation_id == generation.generation_id
        return self.cleanup_report


def _checkpoint(
    generation: GenerationManifest,
    *,
    deleted: int,
    remaining: int,
    complete: bool,
) -> CleanupCheckpoint:
    values = {
        "tenant_ref": generation.tenant_ref,
        "graph_ref": generation.graph_ref,
        "generation_id": generation.generation_id,
        "index_ref": generation.index_ref.model_dump(mode="json"),
        "checkpoint_id": f"checkpoint:{generation.generation_id}",
        "expected_vectors": 2,
        "deleted_vectors": deleted,
        "remaining_vectors": remaining,
        "complete": complete,
        "recorded_at": 100,
    }
    return CleanupCheckpoint(
        tenant_ref=generation.tenant_ref,
        graph_ref=generation.graph_ref,
        generation_id=generation.generation_id,
        index_ref=generation.index_ref,
        checkpoint_id=values["checkpoint_id"],
        expected_vectors=values["expected_vectors"],
        deleted_vectors=values["deleted_vectors"],
        remaining_vectors=values["remaining_vectors"],
        complete=values["complete"],
        recorded_at=values["recorded_at"],
        checkpoint_digest=canonical_digest(values),
    )


def test_acl_precedes_ranking_and_denied_rows_never_enter_cache() -> None:
    request = _request()
    allowed = _chunk("one", digest_letter="a")
    denied = _chunk("two", digest_letter="b")
    acl = _Authorization(_evidence(request, (allowed,)))
    engine = _Engine((_candidate(denied, 0.99, "v"),))
    catalog = InMemoryGenerationCatalog()
    catalog.publish(_manifest(), expected_current=None)
    cache = InMemoryRetrievalCache()
    retriever = GovernedRetriever(
        catalog=catalog,
        authorization=acl,
        engine=engine,
        cache=cache,
        clock=lambda: 100,
    )

    with pytest.raises(RetrievalAuthorizationError, match="denied_candidate"):
        retriever.retrieve(request)
    assert acl.calls == 1
    assert engine.calls == 1
    assert cache.get(request.request_digest) is None


def test_successful_result_is_exactly_bound_and_cache_reuses_only_authorized_rows() -> None:
    request = _request()
    first = _chunk("one", digest_letter="a")
    second = _chunk("two", digest_letter="b")
    acl = _Authorization(_evidence(request, (first, second)))
    engine = _Engine(
        (
            _candidate(second, 0.4, "w"),
            _candidate(first, 0.9, "v"),
        )
    )
    catalog = InMemoryGenerationCatalog()
    catalog.publish(_manifest(), expected_current=None)
    retriever = GovernedRetriever(
        catalog=catalog,
        authorization=acl,
        engine=engine,
        cache=InMemoryRetrievalCache(),
        clock=lambda: 100,
    )

    result = retriever.retrieve(request)
    cached = retriever.retrieve(request)
    assert [hit.chunk.chunk_key for hit in result.hits] == [
        first.chunk_key,
        second.chunk_key,
    ]
    assert cached == result
    assert engine.calls == 1
    assert acl.calls == 2

    citation = CitationRecord(
        retrieval=result,
        citations=(
            CitationRef(
                citation_id="citation:first",
                request_id=request.request_id,
                chunk=first,
                source_version=first.document.source_version,
                source_artifact_ref=first.artifact_ref,
                generation_id=first.document.generation_id,
            ),
        ),
    )
    assert citation.citations[0].chunk.document.source_version == "revision:42"
    evaluation = EvaluationRecord(
        evaluation_id="evaluation:one",
        retrieval=result,
        citation_record=citation,
        evaluator_ref="evaluator:retrieval",
        dataset_artifact_ref=_artifact("e", "artifact:evaluation"),
        metrics=(EvaluationMetric(name="metric:precision", value=1.0),),
        verdict="pass",
    )
    assert evaluation.evaluation_digest.startswith("sha256:")


def test_cross_tenant_and_embedding_drift_fail_closed() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="request_vector_tenant_graph_drift"):
        RetrievalRequest(
            request_id=request.request_id,
            tenant_ref=request.tenant_ref,
            graph_ref=request.graph_ref,
            index_ref=request.index_ref,
            query_artifact_ref=request.query_artifact_ref,
            query_vector_ref=VectorAuthorityRef(
                tenant_ref="tenant:other",
                graph_ref=request.graph_ref,
                generation_id=request.index_ref.generation_id,
                model=_model(),
                dimension=3,
                vector_ref="engine-vector:query",
                vector_digest=_digest("x"),
                source_content_digest=request.query_artifact_ref.digest,
            ),
            limit=1,
        )
    with pytest.raises(ValidationError, match="index_dimension_drift"):
        IndexRef(
            tenant_ref="tenant:one",
            graph_ref="graph:knowledge",
            index_id="index:bad",
            version="1.0.0",
            digest=_digest("i"),
            generation_id="generation:bad",
            model=_model(),
            dimension=4,
        )
    with pytest.raises(ValidationError, match="inline_material_forbidden"):
        ArtifactRef.model_validate(
            {
                "artifact_id": "artifact:bad",
                "digest": _digest("q"),
                "text": "must-not-be-stored",
            }
        )
    chunk = _chunk("hash-check", digest_letter="a")
    bad_vector = VectorAuthorityRef(
        tenant_ref=chunk.document.tenant_ref,
        graph_ref=chunk.document.graph_ref,
        generation_id=chunk.document.generation_id,
        model=_model(),
        dimension=3,
        vector_ref="engine-vector:hash-check",
        vector_digest=_digest("v"),
        source_content_digest=_digest("f"),
    )
    with pytest.raises(ValidationError, match="candidate_content_hash_drift"):
        EngineCandidate(
            chunk=chunk,
            index_ref=_index(),
            vector_ref=bad_vector,
            score=0.5,
            engine_result_ref="engine-result:hash-check",
        )


def test_partial_cleanup_is_not_success_and_stale_vectors_are_denied() -> None:
    old = _manifest()
    new = _manifest("generation:two")
    catalog = InMemoryGenerationCatalog()
    catalog.publish(old, expected_current=None)
    catalog.publish(new, expected_current=old.generation_id)
    engine = _Engine()
    lifecycle = GenerationLifecycle(catalog=catalog, engine=engine)
    engine.cleanup_report = _checkpoint(old, deleted=1, remaining=1, complete=False)

    with pytest.raises(CleanupIncompleteError, match="incomplete"):
        lifecycle.cleanup(old)
    assert catalog.get("tenant:one", "graph:knowledge", old.generation_id).state == (
        "cleanup_failed"
    )
    with pytest.raises(StaleGenerationError):
        GovernedRetriever(
            catalog=catalog,
            authorization=_Authorization(_evidence(_request(), (_chunk(),))),
            engine=engine,
            clock=lambda: 100,
        ).retrieve(_request())

    engine.cleanup_report = _checkpoint(old, deleted=2, remaining=0, complete=True)
    deleted = lifecycle.cleanup(old)
    assert deleted.state == "deleted"
