"""Embedding + semantic cross-linking (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 3).

Embeds entities into one space and uses the **engine's vector search** (HNSW/
cosine — the compute layer) to discover cross-category relationships: a paper
``Concept`` that a code symbol ``REALIZES``, or anything ``RELATES_TO`` a topic/
goal. ``embed_fn``/``search_fn`` are injectable so the logic is testable without a
live model or daemon.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from .models import Concept, EnrichmentEdge

logger = logging.getLogger(__name__)

# texts -> embeddings (batched)
EmbedFn = Callable[[list[str]], list[list[float]]]
# (query_vec, k) -> list of {id, type, _similarity, ...}
SearchFn = Callable[[list[float], int], list[dict[str, Any]]]

# bge-m3 (the deployed embedder) handles large batches per request, so we send a
# big LIST of inputs in ONE ``/v1/embeddings`` POST rather than re-chunking it into
# tiny sub-requests. This caps a single POST's payload (and is also the value we
# pin on the llama-index model's ``embed_batch_size`` so it stops splitting our
# chunk into ``DEFAULT_EMBED_BATCH_SIZE``-sized POSTs). (CONCEPT:AU-KG.ingest.applying-agents-md-batch)
_EMBED_MAX_BATCH = 256

# Durable maintenance state shared by the bounded legacy backfill and the
# ingest-time upsert chokepoint. A real source upsert always clears this marker
# so a formerly textless entity becomes eligible when its content evolves.
EMBEDDING_BACKFILL_STATE_FIELD = "_embedding_backfill_state"
EMBEDDING_BACKFILL_NO_TEXT = "no_text"
# Served-read publication fence for cross-modal embedding replacements.  A
# literal ``False`` means the durable vector property has not yet been projected
# into the engine ANN.  Missing remains readable for legacy records; writers set
# False before replacement and flip it to True only after ANN commit returns.
EMBEDDING_INDEX_READY_FIELD = "_embedding_index_ready"

# D-HYD-4 / D-GS27-6 / D-ORC-7 (2026-08-06) -- the node types
# `find_tools` / `find_relevant_callable_resources`
# and delegation actually search over. `backfill_entity_embeddings`'s default
# candidate order is a blind `ORDER BY n.id` scan across the WHOLE graph with no
# type awareness, so these -- a few thousand nodes -- sort behind ~36k unrelated
# nodes (RuntimeSignal, WorkItem, IngestManifest, mandatory_marking, raw
# Entity/Blob extraction rows with unprefixed ids that sort first of all) and are
# never reached at the 5-minute/64-node governance cadence. Kept as ONE list
# (mirrors `embedding_backfill_eligibility_clause`'s rationale) so the
# governance-agent's steady-state discovery pass, the operator backfill script,
# and any future caller scope to the IDENTICAL set instead of drifting copies.
DISCOVERY_NODE_TYPES: tuple[str, ...] = (
    "Tool",
    "WorkflowDefinition",
    "Skill",
    "CallableResource",
    "Concept",
    "Prompt",
    "MCPServer",
    "NativeTool",
)


def embedding_backfill_eligibility_clause(
    *, alias: str = "n"
) -> tuple[str, dict[str, Any]]:
    """Cypher ``AND``-clause + params excluding secret-bearing nodes from
    embedding-backfill candidacy (D-CDX-102).

    A semantic vector index is queried by SIMILARITY and its hits are handed
    back to agents — embedding a node is a disclosure surface, not a neutral
    read. This is the ONE place the legacy entity-embedding backfill decides
    what is eligible, so every caller (the real backfill, the operator dry
    run, the population/eligible-count snapshot) shares an IDENTICAL,
    construction-time exclusion instead of three copies that can silently
    drift apart:

    * ``graph_name`` — excludes :data:`~agent_utilities.security.
      secrets_client.SECRETS_GRAPH` (``__secrets__``), the dedicated
      engine-encrypted graph secret VALUES live in. Bookkeeping records
      (``IngestManifest`` rows) that merely reference it still carry
      ``graph_name="__secrets__"`` and are excluded too — a manifest of
      *which* prompt/credential file was ingested is itself sensitive
      metadata a similarity search should never surface.
    * ``node_type`` — excludes :data:`~agent_utilities.security.
      secrets_client.SECRET_LABEL` (``Secret``) node labels directly, as
      defense in depth if a future unified-read path ever makes an actual
      ``:Secret`` node reachable from a cross-graph query.

    This is intentionally NOT a maintained denylist of node ids/patterns —
    it is sourced from the SAME constants ``secrets_client`` uses to define
    the secrets store, so it can only drift if that store's own identity
    changes (in which case this clause changes with it, by construction).

    Deliberately ``IS NULL OR <>`` rather than ``coalesce(...) <> $x``:
    ``coalesce()`` (like ``properties(n)`` elsewhere in this module's
    callers) is NOT in the native engine's supported Cypher subset and is
    REJECTED outright (``CypherEngineError(..., error_type=RuntimeError)``,
    empirically confirmed against the live engine) rather than merely
    behaving unexpectedly — so this uses only primitives already proven live
    (``IS NULL``, ``<>``, parameterized literals, ``AND``/``OR``).
    """
    from agent_utilities.security.secrets_client import SECRET_LABEL, SECRETS_GRAPH

    clause = (
        f"AND ({alias}.graph_name IS NULL OR {alias}.graph_name <> "
        "$embedding_backfill_excluded_graph) "
        f"AND ({alias}.node_type IS NULL OR {alias}.node_type <> "
        "$embedding_backfill_excluded_label)"
    )
    params: dict[str, Any] = {
        "embedding_backfill_excluded_graph": SECRETS_GRAPH,
        "embedding_backfill_excluded_label": SECRET_LABEL,
    }
    return clause, params


def embedding_backfill_type_scope_clause(
    node_types: tuple[str, ...] | list[str], *, alias: str = "n"
) -> str:
    """Cypher ``AND``-clause restricting embedding-backfill candidates to
    ``node_types`` (D-HYD-4 addendum, 2026-08-06).

    The default backfill candidate query is a blind ``ORDER BY n.id`` scan with
    no type awareness at all, which is why the small discovery-relevant corpus
    (:data:`DISCOVERY_NODE_TYPES`) was measured at exactly zero embeddings
    despite the governance loop running continuously: those ids sort behind
    tens of thousands of unrelated nodes. This clause lets a caller (the
    governance agent's discovery pass, the operator script's ``--node-types``)
    scope a bounded run to just the types it cares about, so it stops competing
    with the general sweep for the same per-cycle budget.

    Values are inlined as literals rather than bound as a ``$`` list parameter
    — the same choice ``research/loop_controller.py``'s watermark query and
    ``retrieval/governance_rules.py``'s active-rule query already make, because
    this backend does not reliably bind list params. Safe here because callers
    pass a fixed, code-defined enum (:data:`DISCOVERY_NODE_TYPES` or a literal
    list at a call site), never raw external/user input; single quotes are
    stripped defensively regardless.
    """
    literal_list = ", ".join(
        "'" + str(t).replace("'", "") + "'" for t in node_types if str(t).strip()
    )
    if not literal_list:
        return ""
    return f"AND {alias}.node_type IN [{literal_list}] "


def configured_embedding_dimension() -> int:
    """Return the positive vector dimension declared for the active KG schema."""
    from agent_utilities.core.config import config

    try:
        dimension = int(config.kg_embedding_dim or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("KG embedding dimension is not a valid integer") from exc
    if dimension <= 0:
        raise RuntimeError("KG embedding dimension must be positive")
    return dimension


def validate_embedding_vectors(
    vectors: Any,
    *,
    expected_count: int,
    expected_dimension: int | None = None,
) -> list[list[float]]:
    """Validate and normalize an embed response before any vector write."""
    try:
        raw_vectors = list(vectors)
    except TypeError as exc:
        raise RuntimeError(
            "embedding endpoint returned a non-iterable vector response"
        ) from exc
    if len(raw_vectors) != expected_count:
        raise RuntimeError(
            "embedding endpoint returned a vector count that does not match "
            f"the request ({len(raw_vectors)} != {expected_count})"
        )

    dimension = expected_dimension or configured_embedding_dimension()
    if dimension <= 0:
        raise RuntimeError("expected embedding dimension must be positive")
    normalized: list[list[float]] = []
    for index, vector in enumerate(raw_vectors):
        if isinstance(vector, str | bytes | bytearray):
            raise RuntimeError(
                f"embedding endpoint returned an invalid vector at index {index}"
            )
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"embedding endpoint returned an invalid vector at index {index}"
            ) from exc
        if not values or any(not math.isfinite(value) for value in values):
            raise RuntimeError(
                f"embedding endpoint returned an empty or non-finite vector "
                f"at index {index}"
            )
        if len(values) != dimension:
            raise RuntimeError(
                "embedding endpoint returned the wrong vector dimension "
                f"at index {index} ({len(values)} != {dimension})"
            )
        normalized.append(values)
    return normalized


def _embed_concurrency() -> int:
    """Auto-sized number of embed requests to keep in flight concurrently.

    Reuses the *shared* cpu/memory/load sizing anchor (``compute_ingest_worker_count``
    — the same ~36%-of-cores, Pi-OOM-capped budget the ingest pools use) rather than
    inventing a new knob. Embedding is network/GPU-bound (the bge-m3 vLLM endpoint
    services many requests at once), not local-cpu-bound, so we allow ~2× that anchor,
    capped at 16. The model's *declared* parallel capacity (CONCEPT:AU-KG.compute.concurrency-controller-sizing) is a
    hard per-model limit, so this local fan-out may never exceed it. Never below 1.
    (CONCEPT:AU-KG.ingest.applying-agents-md-batch)
    """
    ceiling = 16
    try:
        from agent_utilities.core.model_concurrency import (
            resolve_capacity,
            server_ceiling,
        )

        declared = max(1, resolve_capacity("embedding"))
        # CONCEPT:AU-ORCH.dispatch.embedding-fanout — the embed fan-out is local-cpu-derived (~2× the ingest
        # anchor); never let it exceed the embedder SERVER's real capacity ceiling.
        ceiling = max(1, server_ceiling("embedding"))
    except Exception:  # noqa: BLE001 — capacity is best-effort
        declared = 1
    try:
        from agent_utilities.knowledge_graph.core.engine_tasks import (
            compute_ingest_worker_count,
        )

        anchor = max(1, compute_ingest_worker_count())
    except Exception:  # noqa: BLE001 — sizing is best-effort
        anchor = 4
    # Each call resolves current capacity: config reloads, adaptive controller state,
    # and endpoint failover can change the safe width. The declared model capacity and
    # remote-server ceiling are both hard limits; the local ingest anchor only reduces
    # that safe maximum.
    return min(declared, min(anchor * 2, 16), ceiling)


def _joint_budget_cap(model_key: str, concurrency: int) -> int:
    """Bound the embed fan-out by the ACTIVE endpoint's shared-GPU joint budget.

    CONCEPT:AU-KG.ingest.keys-off (the fallback's capacity-guard inheritance / OOM-safety) over
    CONCEPT:AU-KG.enrichment.each-call-resolves-active (failover) / KG-2.146 (the budget). The fan-out passes an explicit ``capacity`` to
    ``map_concurrent_sync`` (the cpu/load-derived :func:`_embed_concurrency` anchor),
    which bypasses ``resolve_capacity`` — and therefore the per-GPU joint budget.
    That is exactly right for the PRIMARY embedder on its own dedicated endpoint
    (no contention). But while **failed-over** to a GPU shared with the generator
    a shared accelerator, the joint budget MUST govern so bulk embeds cannot OOM the host.

    ``resolve_capacity(model_key)`` seeds the group's priority peers (the generator)
    and applies the budget; a non-``None`` ``group_allowed`` means a budget actually
    governs this endpoint's group → clamp the fan-out to that joint-capped value.
    No GPU budget configured (the primary on its own endpoint) ⇒ ``group_allowed``
    is ``None`` ⇒ no clamp ⇒ zero regression. Fail-safe: any error returns the input
    concurrency unchanged.
    """
    try:
        from agent_utilities.core.config import config
        from agent_utilities.core.gpu_group_budget import group_allowed
        from agent_utilities.core.model_concurrency import resolve_capacity

        group = config.gpu_group(model_key)
        if not group:
            return concurrency
        # Seeds peers + applies the joint budget; returns the budget-capped target.
        guard = resolve_capacity(model_key)
        if group_allowed(group, model_key) is None:
            return concurrency  # no budget governs this group → no regression
        return max(1, min(int(concurrency), int(guard)))
    except Exception:  # noqa: BLE001 — budget clamp is best-effort, never break embeds
        return concurrency


def _auto_batch(n_texts: int, concurrency: int) -> int:
    """Batch size that makes POSTs big BUT leaves enough chunks to fill the lanes.

    A single huge batch would serialize the whole job on one POST; tiny batches
    waste round-trips. Aim for ~``concurrency`` chunks, each a big LIST, clamped to
    ``[32, _EMBED_MAX_BATCH]``. (CONCEPT:AU-KG.ingest.applying-agents-md-batch)
    """
    if n_texts <= 0:
        return 1
    import math

    per = math.ceil(n_texts / max(1, concurrency))
    return max(32, min(per, _EMBED_MAX_BATCH))


def make_embed_fn(batch_size: int | None = None) -> EmbedFn:
    """Batched + concurrent embedding fn backed by the configured embedder (bge-m3).

    Two compounding throughput wins over the historical one-text-per-request loop
    (CONCEPT:AU-KG.ingest.applying-agents-md-batch, applying the AGENTS.md *batch-never-per-element* rule to
    embeddings):

    * **BATCH** — every request carries a big LIST of inputs (auto-sized up to
      :data:`_EMBED_MAX_BATCH`), and the underlying llama-index model's
      ``embed_batch_size`` is pinned so it issues ONE POST per chunk instead of
      re-splitting it into ``DEFAULT_EMBED_BATCH_SIZE`` (=10) sub-POSTs.
    * **CONCURRENCY** — chunks are fanned out CONCURRENTLY up to
      :func:`_embed_concurrency` (cpu/load-derived, ≥ the model's declared
      capacity) via the shared controller (CONCEPT:AU-KG.compute.concurrency-controller-sizing), so the ENRICH stage
      is never one-request-in-flight.

    ``batch_size`` pins the per-request batch explicitly (mainly for tests);
    ``None`` (the default) auto-sizes it per call from the input length and the
    resolved concurrency. Batch boundaries and output order are preserved, so the
    same vectors come out in the same order. The fail-loud KG-2.3 contract below is
    unchanged: a missing/unreachable embedder raises rather than returning a stub.

    **Automatic failover (CONCEPT:AU-KG.enrichment.each-call-resolves-active).** Each call resolves the ACTIVE
    embedder endpoint (:func:`active_embedding_endpoint`): the PRIMARY normally, or
    the configured FALLBACK while the primary's circuit breaker is OPEN. The client
    is rebuilt for that endpoint (the cache swaps it, no stale primary client) and
    the fan-out gates on the endpoint's model KEY, so the capacity guard resolves
    the ACTIVE endpoint's config — including its ``gpu_group``. While failed-over to
    a shared accelerator, the group's JOINT budget bounds the embed fan-out
    so it shares the ceiling with the generator and cannot OOM the box.
    """
    try:
        from agent_utilities.core.embedding_failover import active_embedding_endpoint
        from agent_utilities.core.embedding_utilities import create_embedding_model
        from agent_utilities.core.model_concurrency import map_concurrent_sync
        from agent_utilities.core.model_runtime_auth import resolve_model_api_key

        def _resolve_active() -> tuple[Any, Any]:
            """Resolve the ACTIVE endpoint + its cached client (CONCEPT:AU-KG.enrichment.each-call-resolves-active)."""
            endpoint = active_embedding_endpoint()
            # Build for the resolved endpoint explicitly so the client matches the
            # capacity-guard key we gate with. The cache keys on the base_url, so
            # this returns the fallback's client on failover and the primary's back
            # on recovery — never a stale primary client.
            mdl = create_embedding_model(
                provider=endpoint.provider,
                model=endpoint.model_id,
                base_url=endpoint.base_url,
                api_key=resolve_model_api_key(reference=endpoint.api_key_ref),
            )
            # Pin the model's internal batch so a chunk we hand it is ONE POST, not a
            # fan of DEFAULT_EMBED_BATCH_SIZE-sized sub-POSTs (the serial-POST symptom).
            try:
                current = int(getattr(mdl, "embed_batch_size", 0) or 0)
                mdl.embed_batch_size = max(current, _EMBED_MAX_BATCH)
            except Exception:  # noqa: BLE001 — model may not expose the attr; harmless
                pass
            return endpoint, mdl

        # Probe-build now so a missing embedder dep / endpoint fails LOUD at
        # construction time (KG-2.3), not silently at first use.
        _resolve_active()

        def _fn(texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            # Resolve the ACTIVE endpoint per call so a primary outage fails over —
            # and recovery routes back — transparently mid-run (CONCEPT:AU-KG.enrichment.each-call-resolves-active).
            endpoint, model = _resolve_active()
            concurrency = _embed_concurrency()
            concurrency = _joint_budget_cap(endpoint.model_key, concurrency)
            bs = batch_size or _auto_batch(len(texts), concurrency)
            chunks = [texts[i : i + bs] for i in range(0, len(texts), bs)]
            # Cap concurrency at the chunk count — never spin idle workers.
            capacity = max(1, min(concurrency, len(chunks)))
            # Fan out per-batch embedding up to capacity; order preserved, so
            # flattening the per-chunk results reproduces the input order. Gate on
            # the ACTIVE endpoint's key so the breaker/ceiling/budget track failover.
            chunk_results = map_concurrent_sync(
                chunks,
                model.get_text_embedding_batch,
                model=endpoint.model_key,
                capacity=capacity,
            )
            out: list[list[float]] = []
            for vecs in chunk_results:
                out.extend(vecs)
            # Record embed usage into the active ingest profile (OS-5.69) — embedding
            # endpoints rarely return token counts, so estimate from text length.
            from ..core.ingest_profile import record_embed_usage

            record_embed_usage(texts=texts)
            return out

        return _fn
    except Exception as e:
        # Zero-fabrication compliance (AGENTS.md): NEVER return a degenerate fallback
        # that silently yields 1-dim ``[0.0]`` vectors. That fallback previously masked a
        # missing-embedder deployment (the serving plane shipped bare ``embeddings``
        # without ``embeddings-openai`` → ``No module named 'llama_index.embeddings'``):
        # enrichment "succeeded" while writing garbage vectors into a 1024-dim store,
        # so the failure was invisible (embed_calls=0, no real embeddings) instead of
        # loud. Fail loud here; every production caller wraps embedding as best-effort
        # in try/except, so this degrades to "no enrichment edges" — observable and
        # safe — rather than silent vector-store corruption. (CONCEPT:AU-KG.memory.auto-similarity-memory-graph)
        logger.error("make_embed_fn unavailable (%s)", e)
        raise RuntimeError(
            f"embedding model unavailable: {e}. The KG embedding plane requires the "
            "'embeddings-openai' extra (llama-index-embeddings-openai) and a reachable "
            "bge-m3 vLLM endpoint."
        ) from e


def make_search_fn(backend: Any) -> SearchFn:
    """Vector search over the backend's embedding store (engine HNSW/cosine)."""

    def _fn(query_vec: list[float], k: int) -> list[dict[str, Any]]:
        try:
            return backend.semantic_search(query_vec, k)
        except Exception:
            return []

    return _fn


def entity_text(node_type: str, name: str, summary: str = "", extra: str = "") -> str:
    """Compose the text used to embed an entity."""
    parts = [name]
    if summary:
        parts.append(summary)
    if extra:
        parts.append(extra)
    return " — ".join(p for p in parts if p)


# Field-name priority for the generic (connector-agnostic) text extractor below.
# Every typed-entity connector (ServiceNow, LeanIX, GitHub, Twenty, Jellyfin, ...)
# builds its own ``{"id", "type", **props}`` record shape (see
# ``ChangeEnvelope.from_connector_record``) with its own field names, so there is
# no single canonical "the title field" — these are the common ones observed
# across the fleet, checked in priority order. NAME_FIELDS come first (identity),
# then SUMMARY_FIELDS (free text), mirroring ``entity_text``'s (name, summary)
# shape.
_ENTITY_NAME_FIELDS: tuple[str, ...] = (
    "name",
    "title",
    "displayName",
    "display_name",
    "label",
    "subject",
    "short_description",
)
_ENTITY_SUMMARY_FIELDS: tuple[str, ...] = (
    "description",
    "summary",
    "body",
    "content",
    "comment",
    "message",
    "text",
    "notes",
    # Appended, not inserted, so a real description always wins where one
    # exists. Added for D-HYD-4's discovery-eligibility pass (2026-08-06):
    # measured live, `Prompt` nodes carry NO description/summary/content field
    # at all but DO carry `system_prompt` (8/8 sampled) — without this, every
    # Prompt embeds from its bare name alone. `synonyms` (a JSON-list-shaped
    # string of alternate phrasings, e.g. MCPServer nodes) is genuine
    # query-matching signal the priority list otherwise never reaches.
    "system_prompt",
    "synonyms",
)
# Never embed identifiers, timestamps, urls, or other low-signal/high-churn
# fields even when they happen to be strings — keeps ``derive_entity_text``
# deterministic and avoids polluting the embedding with noise.
_ENTITY_TEXT_SKIP_KEYS = frozenset(
    {
        "id",
        "embedding",
        "text",
        "tenant_id",
        "tenant",
        "source_instance",
        "source_system",
        "domain",
        "classification",
        "retention",
        "legal_hold",
        "external_access",
        "_links",
        "_features",
        "_evidence",
        "_nodes",
    }
)
# Property-value length cap and overall text cap keep one pathological field
# (e.g. an inlined document body) from producing an oversized embedding input.
_ENTITY_FIELD_VALUE_CAP = 2000
_ENTITY_TEXT_CAP = 4000


def derive_entity_text_snapshot(
    props: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return entity text plus the exact property values that selected it.

    CONCEPT:AU-KG.ingest.entity-embedding-at-write — every typed-entity
    connector builds a differently-shaped ``dict`` (see
    ``ChangeEnvelope.from_connector_record``'s docstring), so there is no
    single field name to embed. This checks a priority list of common
    name/title fields, then common description/body fields, then — only if
    neither produced anything — falls back to concatenating every short
    string-valued leaf property (skipping ids/timestamps/governance fields)
    so an entity with unusual field names still gets *something* embedded
    rather than silently landing with no vector.

    The snapshot is suitable for an atomic compare-and-set fence around a slow
    embedding call: if any field that selected or contributed text changes, the
    CAS fails instead of persisting a vector derived from stale content. Missing
    well-known fields are included as ``None`` because adding a higher-priority
    name or summary while the embedder is running also changes the derived text.

    Returns ``("", conditions)`` when no usable text was found. Callers must
    treat that as "defer this record", not as an embedding value.
    """
    if not props:
        return "", {}
    conditions = {
        key: props.get(key)
        for key in (
            "type",
            "node_type",
            *_ENTITY_NAME_FIELDS,
            *_ENTITY_SUMMARY_FIELDS,
        )
    }
    node_type = str(props.get("type") or props.get("node_type") or "")
    name = ""
    for key in _ENTITY_NAME_FIELDS:
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()[:_ENTITY_FIELD_VALUE_CAP]
            break
    summary = ""
    for key in _ENTITY_SUMMARY_FIELDS:
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            summary = value.strip()[:_ENTITY_FIELD_VALUE_CAP]
            break
    if name or summary:
        return (
            entity_text(node_type, name or node_type, summary)[:_ENTITY_TEXT_CAP],
            conditions,
        )

    # Fallback: no recognized field matched — concatenate short string leaves so
    # an unusual connector shape still yields embeddable text.
    fallback_parts: list[str] = [node_type] if node_type else []
    for key, value in props.items():
        if key in _ENTITY_TEXT_SKIP_KEYS or key in _ENTITY_NAME_FIELDS:
            continue
        # The fallback considers every current non-skipped value. Fence even
        # non-string values so a concurrent change from (say) numeric to text
        # cannot make this snapshot stale without failing the CAS.
        conditions[key] = value
        if isinstance(value, str) and value.strip() and len(value) < 500:
            fallback_parts.append(value.strip())
        if sum(len(p) for p in fallback_parts) > _ENTITY_TEXT_CAP:
            break
    return " — ".join(fallback_parts)[:_ENTITY_TEXT_CAP], conditions


def derive_entity_text(props: dict[str, Any]) -> str:
    """Best-effort connector-agnostic text extraction for embedding an entity.

    Returns "" when no usable text was found (callers must treat that as
    "skip embedding this record", not as an error). See
    :func:`derive_entity_text_snapshot` when a caller needs a concurrency fence.
    """
    return derive_entity_text_snapshot(props)[0]


def embed_and_store(
    backend: Any, items: list[tuple[str, str]], embed_fn: EmbedFn
) -> int:
    """Embed (id, text) pairs and store vectors on the backend. Returns count."""
    if not items:
        return 0
    vecs = embed_fn([t for _, t in items])
    n = 0
    for (nid, _), vec in zip(items, vecs, strict=False):
        try:
            backend.add_embedding(nid, vec)
            n += 1
        except Exception as exc:  # noqa: BLE001 — n is only incremented on the line above the try, before the embedding write; a failed write for one node simply isn't counted, matching the return value's meaning ('how many were actually stored')
            logger.debug("Embedding write failed for %s: %s", nid, exc)
    return n


def _result_type(r: dict[str, Any]) -> str:
    # ``_table_label`` is set by PostgreSQL vector search (per-label node
    # tables); ``type``/``node_type`` by other backends.
    return str(r.get("type") or r.get("node_type") or r.get("_table_label") or "")


def link_concepts_to_code(
    concepts: list[Concept],
    embed_fn: EmbedFn,
    search_fn: SearchFn,
    top_k: int = 5,
    relates_threshold: float = 0.55,
    realizes_threshold: float = 0.78,
) -> list[EnrichmentEdge]:
    """Link concepts to the code/features that relate to or realize them.

    Uses vector similarity: above ``relates_threshold`` → ``RELATES_TO``; above
    ``realizes_threshold`` → ``REALIZES`` (the code implements the concept).
    """
    if not concepts:
        return []
    vecs = embed_fn([entity_text("concept", c.name, c.summary) for c in concepts])
    edges: list[EnrichmentEdge] = []
    seen: set[tuple[str, str, str]] = set()
    # Search a WIDER pool than top_k: vector search mixes all node labels, so
    # Code/Feature candidates can be crowded out of a small top_k by other
    # densely-embedded labels (Skill/Agent/Message). Fetch more, then keep the
    # best ``top_k`` Code/Feature matches per concept. (CONCEPT:EG-KG.storage.nonblocking-checkpoint)
    search_k = max(top_k * 6, 30)
    for c, vec in zip(concepts, vecs, strict=False):
        kept = 0
        for r in search_fn(vec, search_k):
            if kept >= top_k:
                break
            if _result_type(r) not in ("Code", "Feature"):
                continue
            score = float(r.get("_similarity", 0.0) or 0.0)
            tgt = r.get("id")
            if not tgt or score < relates_threshold:
                continue
            rel = "REALIZES" if score >= realizes_threshold else "RELATES_TO"
            key = (c.id, tgt, rel)
            if key not in seen:
                seen.add(key)
                edges.append(EnrichmentEdge(source=c.id, target=tgt, rel_type=rel))
                kept += 1
    return edges


def find_related(
    topic: str,
    embed_fn: EmbedFn,
    search_fn: SearchFn,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Goal/topic-driven cross-ingestion discovery: nearest entities to a topic."""
    vec = embed_fn([topic])[0]
    results = search_fn(vec, top_k)
    out = []
    for r in results:
        out.append(
            {
                "id": r.get("id"),
                "type": _result_type(r),
                "name": r.get("name", ""),
                "summary": r.get("summary", ""),
                "similarity": round(float(r.get("_similarity", 0.0) or 0.0), 3),
            }
        )
    return out
