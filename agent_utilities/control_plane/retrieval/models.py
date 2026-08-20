"""Immutable references and evidence for governed retrieval.

The retrieval control plane deliberately stores identities, digests, and
authority references only.  Source content is held by an artifact authority;
vectors are held by the epistemic-graph engine.  This module never accepts
inline document text, query text, vectors, prompts, secrets, or results.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MAX_CITATIONS",
    "MAX_EVALUATION_METRICS",
    "MAX_HITS",
    "MAX_INDEX_CHUNKS",
    "ArtifactRef",
    "AuthorizationEvidence",
    "ChunkRef",
    "CleanupCheckpoint",
    "DocumentRef",
    "EmbeddingModelRef",
    "EngineCandidate",
    "EvaluationMetric",
    "EvaluationRecord",
    "GenerationManifest",
    "GenerationRecord",
    "IndexRef",
    "RankedHit",
    "RetrievalRequest",
    "RetrievalResult",
    "CitationRef",
    "CitationRecord",
    "VectorAuthorityRef",
    "canonical_digest",
]


MAX_HITS = 128
MAX_CITATIONS = 64
MAX_EVALUATION_METRICS = 32
MAX_INDEX_CHUNKS = 1_000_000

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$")
_VERSION_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

StableId: TypeAlias = Annotated[str, Field(pattern=_ID_RE.pattern, min_length=1)]
OpaqueRef: TypeAlias = Annotated[str, Field(pattern=_ID_RE.pattern, min_length=1)]
Version: TypeAlias = Annotated[str, Field(pattern=_VERSION_RE.pattern)]
Digest: TypeAlias = Annotated[str, Field(pattern=_DIGEST_RE.pattern)]
Timestamp: TypeAlias = Annotated[int, Field(ge=0)]
Dimension: TypeAlias = Annotated[int, Field(ge=1, le=65_536)]
Score: TypeAlias = Annotated[float, Field(ge=-1.0, le=1.0)]
GenerationState: TypeAlias = Literal["active", "retiring", "deleted", "cleanup_failed"]

_FORBIDDEN_INLINE_KEYS = {
    "body",
    "content",
    "credentials",
    "password",
    "prompt",
    "result",
    "secret",
    "text",
    "token",
    "values",
    "vector",
}


def _canonical(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_exact_ref(value: str, field_name: str) -> None:
    if value.casefold() in {"latest", "current", "default"}:
        raise ValueError(f"{field_name}_alias_forbidden")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        strict=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_inline_material(cls, value: object) -> object:
        if isinstance(value, Mapping):
            forbidden = {
                str(key).casefold().replace("-", "_")
                for key in value
                if str(key).casefold().replace("-", "_") in _FORBIDDEN_INLINE_KEYS
            }
            if forbidden:
                raise ValueError("inline_material_forbidden")
        return value


class ArtifactRef(_FrozenModel):
    """Opaque source/artifact identity; the artifact service owns its bytes."""

    artifact_id: StableId
    digest: Digest
    media_type: OpaqueRef = "application/octet-stream"

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self)


class DocumentRef(_FrozenModel):
    """One exact source document revision in one tenant/graph generation."""

    tenant_ref: OpaqueRef
    graph_ref: OpaqueRef
    document_id: StableId
    source_ref: OpaqueRef
    source_version: OpaqueRef
    content_digest: Digest
    generation_id: StableId
    artifact_ref: ArtifactRef

    @model_validator(mode="after")
    def _artifact_hash_matches(self) -> DocumentRef:
        _require_exact_ref(self.source_version, "source_version")
        _require_exact_ref(self.generation_id, "generation")
        if self.content_digest != self.artifact_ref.digest:
            raise ValueError("document_artifact_digest_drift")
        return self

    @property
    def document_digest(self) -> str:
        return canonical_digest(self)


class ChunkRef(_FrozenModel):
    """A source-versioned chunk, represented only by its artifact reference."""

    document: DocumentRef
    chunk_id: StableId
    ordinal: int = Field(ge=0, le=MAX_INDEX_CHUNKS)
    content_digest: Digest
    artifact_ref: ArtifactRef

    @model_validator(mode="after")
    def _artifact_hash_matches(self) -> ChunkRef:
        if self.content_digest != self.artifact_ref.digest:
            raise ValueError("chunk_artifact_digest_drift")
        return self

    @property
    def chunk_key(self) -> str:
        return f"{self.document.document_id}:{self.chunk_id}"

    @property
    def chunk_digest(self) -> str:
        return canonical_digest(self)


class EmbeddingModelRef(_FrozenModel):
    """Exact embedding model/version and dimensionality."""

    model_id: StableId
    version: Version
    digest: Digest
    dimension: Dimension

    @property
    def model_digest(self) -> str:
        return canonical_digest(self)


class VectorAuthorityRef(_FrozenModel):
    """Reference to an engine-owned vector; no vector values cross this seam."""

    tenant_ref: OpaqueRef
    graph_ref: OpaqueRef
    generation_id: StableId
    model: EmbeddingModelRef
    dimension: Dimension
    vector_ref: OpaqueRef
    vector_digest: Digest
    source_content_digest: Digest

    @model_validator(mode="after")
    def _dimension_matches_model(self) -> VectorAuthorityRef:
        _require_exact_ref(self.generation_id, "generation")
        if self.dimension != self.model.dimension:
            raise ValueError("vector_dimension_drift")
        return self

    @property
    def authority_digest(self) -> str:
        return canonical_digest(self)


class IndexRef(_FrozenModel):
    """Exact tenant/graph/index generation and embedding-space identity."""

    tenant_ref: OpaqueRef
    graph_ref: OpaqueRef
    index_id: StableId
    version: Version
    digest: Digest
    generation_id: StableId
    model: EmbeddingModelRef
    dimension: Dimension

    @model_validator(mode="after")
    def _dimension_matches_model(self) -> IndexRef:
        _require_exact_ref(self.index_id, "index")
        _require_exact_ref(self.generation_id, "generation")
        if self.dimension != self.model.dimension:
            raise ValueError("index_dimension_drift")
        return self

    @property
    def index_digest(self) -> str:
        return canonical_digest(self)


class GenerationManifest(_FrozenModel):
    """Immutable manifest for a complete source/index generation."""

    tenant_ref: OpaqueRef
    graph_ref: OpaqueRef
    generation_id: StableId
    source_version: OpaqueRef
    source_digest: Digest
    index_ref: IndexRef
    model: EmbeddingModelRef
    expected_chunks: int = Field(ge=1, le=MAX_INDEX_CHUNKS)

    @model_validator(mode="after")
    def _generation_bindings_match(self) -> GenerationManifest:
        _require_exact_ref(self.generation_id, "generation")
        _require_exact_ref(self.source_version, "source_version")
        index = self.index_ref
        if index.tenant_ref != self.tenant_ref or index.graph_ref != self.graph_ref:
            raise ValueError("generation_tenant_graph_drift")
        if index.generation_id != self.generation_id:
            raise ValueError("generation_index_binding_drift")
        if index.model != self.model or index.dimension != self.model.dimension:
            raise ValueError("generation_model_binding_drift")
        return self

    @property
    def manifest_digest(self) -> str:
        return canonical_digest(self)


class CleanupCheckpoint(_FrozenModel):
    """Engine checkpoint proving whether a generation cleanup fully completed."""

    tenant_ref: OpaqueRef
    graph_ref: OpaqueRef
    generation_id: StableId
    index_ref: IndexRef
    checkpoint_id: StableId
    checkpoint_digest: Digest
    expected_vectors: int = Field(ge=0, le=MAX_INDEX_CHUNKS)
    deleted_vectors: int = Field(ge=0, le=MAX_INDEX_CHUNKS)
    remaining_vectors: int = Field(ge=0, le=MAX_INDEX_CHUNKS)
    complete: bool
    recorded_at: Timestamp

    @model_validator(mode="after")
    def _checkpoint_is_deterministic(self) -> CleanupCheckpoint:
        if self.index_ref.tenant_ref != self.tenant_ref:
            raise ValueError("cleanup_tenant_drift")
        if self.index_ref.graph_ref != self.graph_ref:
            raise ValueError("cleanup_graph_drift")
        if self.index_ref.generation_id != self.generation_id:
            raise ValueError("cleanup_generation_drift")
        if self.deleted_vectors > self.expected_vectors:
            raise ValueError("cleanup_deleted_count_invalid")
        if self.remaining_vectors > self.expected_vectors:
            raise ValueError("cleanup_remaining_count_invalid")
        if self.deleted_vectors + self.remaining_vectors != self.expected_vectors:
            raise ValueError("cleanup_checkpoint_partial_accounting")
        if self.complete != (
            self.deleted_vectors == self.expected_vectors
            and self.remaining_vectors == 0
        ):
            raise ValueError("cleanup_completion_claim_invalid")
        expected_digest = canonical_digest(
            self.model_dump(mode="json", exclude={"checkpoint_digest"})
        )
        if self.checkpoint_digest != expected_digest:
            raise ValueError("cleanup_checkpoint_digest_drift")
        return self


class GenerationRecord(_FrozenModel):
    """Immutable manifest plus the catalog's fail-closed lifecycle state."""

    manifest: GenerationManifest
    state: GenerationState
    cleanup_checkpoint: CleanupCheckpoint | None = None

    @model_validator(mode="after")
    def _state_is_consistent(self) -> GenerationRecord:
        checkpoint = self.cleanup_checkpoint
        if self.state == "active" and checkpoint is not None:
            raise ValueError("active_generation_has_cleanup_checkpoint")
        if self.state == "deleted" and (checkpoint is None or not checkpoint.complete):
            raise ValueError("deleted_generation_cleanup_unproven")
        if checkpoint is not None and (
            checkpoint.generation_id != self.manifest.generation_id
            or checkpoint.index_ref != self.manifest.index_ref
        ):
            raise ValueError("generation_cleanup_binding_drift")
        return self


class AuthorizationEvidence(_FrozenModel):
    """ACL proof produced before the engine may return ranked candidates."""

    authorization_id: StableId
    request_id: StableId
    tenant_ref: OpaqueRef
    graph_ref: OpaqueRef
    index_ref: IndexRef
    generation_id: StableId
    policy_ref: OpaqueRef
    allowed_chunks: tuple[ChunkRef, ...] = Field(max_length=MAX_HITS)
    denied_count: int = Field(ge=0, le=MAX_INDEX_CHUNKS)
    evaluated_at: Timestamp
    expires_at: Timestamp

    @model_validator(mode="after")
    def _authorization_is_exact(self) -> AuthorizationEvidence:
        if self.expires_at <= self.evaluated_at:
            raise ValueError("authorization_expiry_invalid")
        if self.index_ref.tenant_ref != self.tenant_ref:
            raise ValueError("authorization_tenant_drift")
        if self.index_ref.graph_ref != self.graph_ref:
            raise ValueError("authorization_graph_drift")
        if self.index_ref.generation_id != self.generation_id:
            raise ValueError("authorization_generation_drift")
        keys = []
        for chunk in self.allowed_chunks:
            if chunk.document.tenant_ref != self.tenant_ref:
                raise ValueError("authorization_chunk_tenant_drift")
            if chunk.document.graph_ref != self.graph_ref:
                raise ValueError("authorization_chunk_graph_drift")
            if chunk.document.generation_id != self.generation_id:
                raise ValueError("authorization_chunk_generation_drift")
            keys.append(chunk.chunk_key)
        if len(keys) != len(set(keys)):
            raise ValueError("authorization_chunk_duplicate")
        return self

    @property
    def evidence_digest(self) -> str:
        return canonical_digest(self)

    def allows(self, chunk: ChunkRef) -> bool:
        return chunk.chunk_key in {item.chunk_key for item in self.allowed_chunks}


class RetrievalRequest(_FrozenModel):
    """Bounded query request using artifact and engine-vector references only."""

    request_id: StableId
    tenant_ref: OpaqueRef
    graph_ref: OpaqueRef
    index_ref: IndexRef
    query_artifact_ref: ArtifactRef
    query_vector_ref: VectorAuthorityRef
    limit: int = Field(ge=1, le=MAX_HITS)

    @model_validator(mode="after")
    def _request_bindings_match(self) -> RetrievalRequest:
        index = self.index_ref
        vector = self.query_vector_ref
        if index.tenant_ref != self.tenant_ref or index.graph_ref != self.graph_ref:
            raise ValueError("request_tenant_graph_drift")
        if vector.tenant_ref != self.tenant_ref or vector.graph_ref != self.graph_ref:
            raise ValueError("request_vector_tenant_graph_drift")
        if vector.generation_id != index.generation_id:
            raise ValueError("request_generation_drift")
        if vector.model != index.model or vector.dimension != index.dimension:
            raise ValueError("request_embedding_space_drift")
        if vector.source_content_digest != self.query_artifact_ref.digest:
            raise ValueError("request_content_hash_drift")
        return self

    @property
    def request_digest(self) -> str:
        return canonical_digest(self)


class EngineCandidate(_FrozenModel):
    """Raw engine candidate, still unranked and uncacheable until ACL proof."""

    chunk: ChunkRef
    index_ref: IndexRef
    vector_ref: VectorAuthorityRef
    score: Score
    engine_result_ref: OpaqueRef

    @model_validator(mode="after")
    def _candidate_bindings_match(self) -> EngineCandidate:
        if not math.isfinite(self.score):
            raise ValueError("candidate_score_non_finite")
        if self.chunk.document.tenant_ref != self.index_ref.tenant_ref:
            raise ValueError("candidate_tenant_drift")
        if self.chunk.document.graph_ref != self.index_ref.graph_ref:
            raise ValueError("candidate_graph_drift")
        if self.chunk.document.generation_id != self.index_ref.generation_id:
            raise ValueError("candidate_generation_drift")
        if self.vector_ref.tenant_ref != self.index_ref.tenant_ref:
            raise ValueError("candidate_vector_tenant_drift")
        if self.vector_ref.graph_ref != self.index_ref.graph_ref:
            raise ValueError("candidate_vector_graph_drift")
        if self.vector_ref.generation_id != self.index_ref.generation_id:
            raise ValueError("candidate_vector_generation_drift")
        if self.vector_ref.model != self.index_ref.model:
            raise ValueError("candidate_embedding_model_drift")
        if self.vector_ref.dimension != self.index_ref.dimension:
            raise ValueError("candidate_dimension_drift")
        if self.vector_ref.source_content_digest != self.chunk.content_digest:
            raise ValueError("candidate_content_hash_drift")
        return self

    @property
    def candidate_digest(self) -> str:
        return canonical_digest(self)


class RankedHit(_FrozenModel):
    """A candidate ranked only after ACL authorization evidence exists."""

    chunk: ChunkRef
    index_ref: IndexRef
    score: Score
    rank: int = Field(ge=1, le=MAX_HITS)
    authorization_digest: Digest
    engine_result_ref: OpaqueRef

    @model_validator(mode="after")
    def _score_is_finite(self) -> RankedHit:
        if not math.isfinite(self.score):
            raise ValueError("ranked_score_non_finite")
        return self

    @property
    def hit_digest(self) -> str:
        return canonical_digest(self)


class RetrievalResult(_FrozenModel):
    """Bounded ranked result carrying the ACL proof used to produce it."""

    request: RetrievalRequest
    authorization: AuthorizationEvidence
    hits: tuple[RankedHit, ...] = Field(max_length=MAX_HITS)

    @model_validator(mode="after")
    def _result_is_authorized(self) -> RetrievalResult:
        if self.authorization.request_id != self.request.request_id:
            raise ValueError("result_request_identity_drift")
        if self.authorization.tenant_ref != self.request.tenant_ref:
            raise ValueError("result_tenant_drift")
        if self.authorization.graph_ref != self.request.graph_ref:
            raise ValueError("result_graph_drift")
        if self.authorization.index_ref != self.request.index_ref:
            raise ValueError("result_index_drift")
        if self.authorization.generation_id != self.request.index_ref.generation_id:
            raise ValueError("result_generation_drift")
        if len(self.hits) > self.request.limit:
            raise ValueError("result_limit_exceeded")
        seen = set()
        for expected_rank, hit in enumerate(self.hits, start=1):
            if hit.rank != expected_rank:
                raise ValueError("result_rank_not_contiguous")
            if hit.authorization_digest != self.authorization.evidence_digest:
                raise ValueError("result_authorization_drift")
            if hit.index_ref != self.request.index_ref:
                raise ValueError("result_hit_index_drift")
            if not self.authorization.allows(hit.chunk):
                raise ValueError("result_contains_denied_chunk")
            if hit.chunk.chunk_key in seen:
                raise ValueError("result_duplicate_chunk")
            seen.add(hit.chunk.chunk_key)
        return self

    @property
    def retrieval_digest(self) -> str:
        return canonical_digest(self)


class CitationRef(_FrozenModel):
    """Citation continuity from a ranked hit to its exact source revision."""

    citation_id: StableId
    request_id: StableId
    chunk: ChunkRef
    source_version: OpaqueRef
    source_artifact_ref: ArtifactRef
    generation_id: StableId

    @model_validator(mode="after")
    def _source_continuity(self) -> CitationRef:
        if self.request_id == "":
            raise ValueError("citation_request_missing")
        if self.source_version != self.chunk.document.source_version:
            raise ValueError("citation_source_version_drift")
        if self.source_artifact_ref != self.chunk.artifact_ref:
            raise ValueError("citation_source_artifact_drift")
        if self.generation_id != self.chunk.document.generation_id:
            raise ValueError("citation_generation_drift")
        return self

    @property
    def citation_digest(self) -> str:
        return canonical_digest(self)


class CitationRecord(_FrozenModel):
    """Bounded citations that can only reference this authorized result."""

    retrieval: RetrievalResult
    citations: tuple[CitationRef, ...] = Field(max_length=MAX_CITATIONS)

    @model_validator(mode="after")
    def _citations_are_continuous(self) -> CitationRecord:
        hit_keys = {hit.chunk.chunk_key for hit in self.retrieval.hits}
        citation_keys = []
        for citation in self.citations:
            if citation.request_id != self.retrieval.request.request_id:
                raise ValueError("citation_request_drift")
            if citation.chunk.chunk_key not in hit_keys:
                raise ValueError("citation_not_in_retrieval")
            if not self.retrieval.authorization.allows(citation.chunk):
                raise ValueError("citation_contains_denied_chunk")
            citation_keys.append(citation.chunk.chunk_key)
        if len(citation_keys) != len(set(citation_keys)):
            raise ValueError("citation_duplicate_chunk")
        return self

    @property
    def citation_digest(self) -> str:
        return canonical_digest(self)


class EvaluationMetric(_FrozenModel):
    name: OpaqueRef
    value: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _finite(self) -> EvaluationMetric:
        if not math.isfinite(self.value):
            raise ValueError("evaluation_metric_non_finite")
        return self


class EvaluationRecord(_FrozenModel):
    """Bounded evaluation evidence without retaining prompt/output content."""

    evaluation_id: StableId
    retrieval: RetrievalResult
    citation_record: CitationRecord | None = None
    evaluator_ref: OpaqueRef
    dataset_artifact_ref: ArtifactRef
    metrics: tuple[EvaluationMetric, ...] = Field(
        min_length=1, max_length=MAX_EVALUATION_METRICS
    )
    verdict: Literal["pass", "fail", "inconclusive"]

    @model_validator(mode="after")
    def _evaluation_is_bound(self) -> EvaluationRecord:
        if (
            self.citation_record is not None
            and self.citation_record.retrieval != self.retrieval
        ):
            raise ValueError("evaluation_citation_retrieval_drift")
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("evaluation_metric_duplicate")
        return self

    @property
    def evaluation_digest(self) -> str:
        return canonical_digest(self)
