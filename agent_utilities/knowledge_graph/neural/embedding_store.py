from __future__ import annotations

"""Versioned, tenant-scoped, content-hash-keyed embedding production (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6).

Populates :class:`~.models.TenantScopedEmbedding` from a REAL embedding call —
the piece the repo survey confirmed was 100% missing for the schema-only
neural-graph models (nothing anywhere constructed one from a live encoder
output). Reuses the EXISTING embedder factory (``core/embedding_utilities``),
the EXISTING engine-native HNSW index (``GraphComputeEngine.add_embedding`` /
``semantic_search`` — no new vector-index library), and the EXISTING
content-hash cache-skip pattern from the pipeline embedding phase — this module
adds no new vector-search or embedding-provider code, only the governed
KG-record wrapper around what already exists.

Gated, not a core dependency: the embedding client (LlamaIndex ``BaseEmbedding``)
is imported lazily inside :func:`_embed_text`, so importing this module — or the
rest of agent-utilities — never requires it; a lean install without an embedder
configured simply cannot call :func:`build_tenant_embedding` (a clear
``RuntimeError``, not an import-time failure).
"""

import hashlib
import logging
from typing import Any

from pydantic import ValidationError

from agent_utilities.core.resource_priority import PriorityClass, priority_scope

from .models import GraphNodeRef, TenantScopedEmbedding

logger = logging.getLogger(__name__)

__all__ = ["build_tenant_embedding", "content_hash"]


def content_hash(text: str) -> str:
    """Stable sha256 over the embedded text — the re-embed cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_text(text: str) -> list[float]:
    """Generate one vector via the shared, provider-agnostic embedder factory.

    Runs under :data:`PriorityClass.BACKGROUND_INGESTION` (CONCEPT:AU-ORCH.scheduling.resource-priority-edict) — embedding
    generation is exactly the "deep extraction" the priority edict reserves
    generator/embedder capacity away from under a saturating ingestion fan-out.
    """
    from agent_utilities.core.embedding_utilities import create_embedding_model

    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        model = create_embedding_model()
        return list(model.get_text_embedding(text))


def _representation_id(tenant: str, node_id: str, encoder_id: str, digest: str) -> str:
    key = f"{tenant}|{node_id}|{encoder_id}|{digest}"
    return f"nrep:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}"


def _existing_record(
    engine: Any, node_id: str, encoder_id: str, encoder_version: str
) -> dict[str, Any] | None:
    """Best-effort read of a prior embedding record (the cache-skip source).

    Returns the WHOLE stored record, not just its ``content_hash``: a cache hit
    has to be able to hand back the representation it already has, otherwise the
    "skip" would still pay for a fresh embedding call and a re-commit — which is
    exactly the cost the content-hash key exists to avoid.
    """
    graph = getattr(engine, "graph", None)
    reader = getattr(graph, "nodes", None)
    if reader is None:
        return None
    try:
        node = reader[f"nrep:{node_id}:{encoder_id}:{encoder_version}"]
    except (KeyError, TypeError):
        return None
    return node if isinstance(node, dict) else None


def _representation_from_record(record: dict[str, Any]) -> TenantScopedEmbedding | None:
    """Rebuild the stored representation, or ``None`` if the record is unusable.

    A record written by an older//newer schema must degrade to a cache MISS (a
    correct, merely slower answer) rather than raise out of a cache lookup.
    """
    fields = {
        key: value
        for key, value in record.items()
        if key in TenantScopedEmbedding.model_fields
    }
    try:
        return TenantScopedEmbedding.model_validate(fields)
    except ValidationError as exc:
        logger.debug(
            "stored NeuralRepresentation %s is not loadable — re-embedding: %s",
            record.get("id"),
            exc,
            exc_info=exc,
        )
        return None


def build_tenant_embedding(
    engine: Any,
    *,
    tenant: str,
    node_id: str,
    node_type: str,
    text: str,
    encoder_id: str = "bge-m3",
    encoder_version: str = "1",
    graph_epoch: int = 0,
    calibration_ref: str = "",
) -> TenantScopedEmbedding:
    """Embed ``text``, index it in the engine's native HNSW, and commit a
    versioned :class:`TenantScopedEmbedding` pointer record.

    Content-hash-gated (skips re-embedding when nothing changed since the last
    call for this exact ``(node_id, encoder_id, encoder_version)``) — the same
    discipline the pipeline embedding phase already applies, so calling this
    repeatedly (e.g. from a delta connector sync) is cheap on the steady state.

    Args:
        tenant: Routes the committed record onto the tenant's named graph
            convention (``tenant_graph_name``) via ``ChangeEnvelope.tenant`` —
            the SAME physical-isolation boundary the rest of the KG uses (empty
            = the default/single-tenant graph).
        node_id: The target KG node's id (any OWL class — NOT restricted to
            OCEL event/object/object_state).
        graph_epoch: Caller-supplied versioning axis (e.g. a KG commit/version
            counter) — this module does not invent one; 0 is a valid "unversioned
            deployment" default.

    Raises:
        RuntimeError: no embedder is configured (surfaced by the lazy
            LlamaIndex factory import/call — never at module import time).
    """
    digest = content_hash(text)
    existing = _existing_record(engine, node_id, encoder_id, encoder_version)
    if existing is not None and existing.get("content_hash") == digest:
        cached = _representation_from_record(existing)
        if cached is not None:
            logger.debug(
                "content_hash unchanged for %s/%s@%s — skipping re-embed",
                node_id,
                encoder_id,
                encoder_version,
            )
            return cached

    vector = _embed_text(text)
    add_embedding = getattr(engine, "add_embedding", None)
    if callable(add_embedding):
        add_embedding(node_id, vector)

    representation = TenantScopedEmbedding(
        representation_id=_representation_id(tenant, node_id, encoder_id, digest),
        tenant=tenant,
        target=GraphNodeRef(node_id=node_id, node_type=node_type),
        encoder_id=encoder_id,
        encoder_version=encoder_version,
        dimension=len(vector),
        artifact_ref=node_id,
        graph_epoch=graph_epoch,
        content_hash=digest,
        calibration_ref=calibration_ref,
    )
    _commit_representation(engine, representation)
    return representation


def _commit_representation(engine: Any, representation: TenantScopedEmbedding) -> None:
    from ...protocols.source_connectors.base import ExternalAccess
    from ..ingestion.change_envelope import ChangeEnvelope
    from ..ingestion.envelope_ingest import ingest_envelope

    node_key = (
        f"nrep:{representation.target.node_id}:"
        f"{representation.encoder_id}:{representation.encoder_version}"
    )
    record = {
        "id": node_key,
        "type": "NeuralRepresentation",
        **representation.model_dump(mode="json"),
        "updatedAt": representation.content_hash,
    }
    env = ChangeEnvelope.from_connector_record(
        record,
        connector="neural-layer",
        tenant=representation.tenant,
        id_field="id",
        version_field="updatedAt",
        # Unreviewed/derived governance artifacts stay restricted by default —
        # the fail-closed default (AU-P0-4) also applies to neural pointers.
        source_acl=ExternalAccess.quarantined(),
    )
    applied = ingest_envelope(engine, env)
    if applied.get("status") not in {"success", "skipped"}:
        raise RuntimeError("NeuralRepresentation ChangeEnvelope failed")
