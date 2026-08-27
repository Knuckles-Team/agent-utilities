#!/usr/bin/python
from __future__ import annotations

"""ETL + connector-sync data lineage — record + query system-to-system data flows
(CONCEPT:AU-KG.ontology.kg-3, CONCEPT:AU-KG.ingest.ambient-connector-provenance).

Every ``graph_etl`` run records a lineage trail in the KG itself so an operator can
answer impact-analysis questions ("what flows from ServiceNow to LeanIX?", "where did
this Stardog graph's data originate?"). Reuses the existing provenance ontology — NO
new node/edge types:

* a run is a :class:`RegistryNodeType.PROVENANCE_AGENT` node (``kind="etl_run"``) with
  ``source`` / ``sink`` / ``direction`` / ``nodes`` / ``edges`` / ``status`` / ``at`` props;
* ``source`` and ``sink`` systems are PROVENANCE_AGENT marker nodes
  (``urn:source:<s>`` / ``urn:sink:<s>``, ``kind="system"``) — the same ``urn:source:``
  scheme the Stardog named-graph partitioning and ``sparql_ingestor`` already use;
* :class:`RegistryEdgeType.WAS_DERIVED_FROM` edges chain ``sink → run → source`` so a
  graph walk reconstructs the flow.

Lineage is best-effort: a failure to record never fails the ETL run itself.

**Connector-sync activity + summary claim (W3.4).** :func:`record_connector_sync_activity`
is the SAME idea for ``source_sync``'s per-connector handlers: ONE PROV-O Activity node
per sync run (``kind="connector_sync"``, :class:`RegistryNodeType.PROVENANCE_ACTIVITY` —
the sibling of ``PROVENANCE_AGENT`` this module already uses for a run, wired here for
its first live use), never one per ingested row. Each row synced during that run is
linked to it (``_links``/``"derived_from"`` on the connector's own entity dict, so the
edge commits atomically with the row's own write — see ``source_sync.
_ingest_entities_via_envelope``). :func:`record_connector_sync_claim` persists exactly
ONE ``:Claim`` per run summarizing it ("source X said N records as of T") through the
same lightweight, directly-verified claim-persistence path
``orchestration.agent_dispatch_worker`` already uses (:class:`ClaimNode` +
``engine.add_node(id, "Claim", ...)``) — never the governed mining-flywheel lifecycle,
which is reserved for INFERRED findings needing a confidence floor and review. Same
best-effort, engine-guarded, never-raises contract as the ETL functions above.

**Media-sidecar delegation provenance (W4.6, CONCEPT:AU-KG.ingest.media-sidecar-delegation).**
:func:`record_media_sidecar_activity` / :func:`record_media_sidecar_claim` are
the SAME pair, one call-site family over: ONE PROV-O Activity node per
delegated fleet extraction call (``kind="media_sidecar"``) plus ONE
directly-verified ``:Claim`` summarizing what it produced, used by
``agent_utilities/media/sidecar_delegate.py``'s ``delegate_extract`` (the
activity) and its ``pdf_sidecar.py``/``image_sidecar.py`` callers (the
claim, whose id every per-locus ``store_<locus>_evidence`` write-back links
via ``claim_id`` so ``evidence_citations``'s SUPPORTS-walk resolves every
sidecar-produced locus through this one claim,
CONCEPT:AU-KG.identity.evidence-spine-convergence).

**OpenLineage RunEvent consumption (CA-25, DEC-CA-05, CONCEPT:AU-KG.ingest.openlineage-consumer).**
:func:`record_openlineage_run_event` is the graph-write half of the
``knowledge_graph.etl.openlineage_consumer`` Kafka consumer: ONE
``PROVENANCE_ACTIVITY`` node per logical run (``kind="openlineage_run"``,
keyed deterministically on ``run.runId`` so redelivery merge-upserts rather
than duplicates), with ``prov:used``/``prov:wasGeneratedBy`` edges to
``PROVENANCE_ENTITY`` dataset marker nodes (:func:`_dataset_marker`, the
dataset twin of :func:`_system_marker`) for every OpenLineage
``inputs[]``/``outputs[]`` entry, and a best-effort link back to an existing
``:RunTrace`` (``observability.trace_ontology.correlate_lineage_run_trace``)
when the run correlates to a tool-originated one. Same best-effort,
engine-guarded, never-raises contract as every function above; a RunEvent
that fails to map is quarantined (logged), never guessed.
"""

import hashlib
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType

logger = logging.getLogger(__name__)

_RUN_KIND = "etl_run"
_CONNECTOR_SYNC_KIND = "connector_sync"
_CONNECTOR_SYNC_CLAIM_TYPE = "observation"
_SYSTEM_KIND = "system"
_OPENLINEAGE_ACTIVITY_KIND = "openlineage_run"
_DATASET_MARKER_KIND = "dataset"


def _system_marker(engine: Any, system: str, *, role: str) -> str:
    """Ensure a system marker node exists; return its id (``urn:<role>:<system>``)."""
    node_id = f"urn:{role}:{system}"
    try:
        engine.add_node(
            node_id,
            RegistryNodeType.PROVENANCE_AGENT,
            {"kind": _SYSTEM_KIND, "name": system, "role": role},
        )
    except Exception:  # noqa: BLE001 - marker creation is best-effort
        logger.debug("lineage: marker %s failed", node_id, exc_info=True)
    return node_id


def _dataset_marker(engine: Any, dataset_id: str) -> str:
    """Ensure a ``prov:Entity`` dataset marker node exists; return its id.

    CA-25/DEC-CA-05 — the dataset-entity twin of :func:`_system_marker`,
    reusing its exact create-if-absent pattern rather than a new lookup
    mechanism. ``dataset_id`` is the caller-supplied
    ``iceberg://<catalog>/<ns>/<table>@<snapshot>`` string itself (per
    ``openlineage_consumer.dataset_entity_id``) — used directly as the node
    id, matching DEC-CA-05's "this string IS the prov:Entity id" rule; no
    additional hashing (an infra identifier, not user content, exactly like
    the existing ``urn:source:<system>``/``urn:sink:<system>`` markers).
    """
    try:
        engine.add_node(
            dataset_id,
            RegistryNodeType.PROVENANCE_ENTITY,
            {"kind": _DATASET_MARKER_KIND, "name": dataset_id},
        )
    except Exception:  # noqa: BLE001 - marker creation is best-effort
        logger.debug("lineage: dataset marker %s failed", dataset_id, exc_info=True)
    return dataset_id


def record_etl_run(
    engine: Any,
    *,
    source: str | None,
    sink: str | None,
    direction: str,
    counts: dict[str, Any] | None = None,
    status: str = "ok",
    at: float | None = None,
) -> str | None:
    """Record one ETL run + its source→sink lineage edges. Returns the run id.

    ``direction`` is ``inbound`` (source→KG), ``outbound`` (KG→sink), or ``through``
    (source→KG→sink). Best-effort: returns ``None`` and logs on failure rather than
    raising into the ETL run.
    """
    if engine is None:
        return None
    ts = at if at is not None else time.time()
    counts = counts or {}
    run_id = f"etl-run:{(source or '_')}:{(sink or '_')}:{int(ts * 1000)}"
    try:
        engine.add_node(
            run_id,
            RegistryNodeType.PROVENANCE_AGENT,
            {
                "kind": _RUN_KIND,
                "source": source or "",
                "sink": sink or "",
                "direction": direction,
                "nodes": int(counts.get("nodes", 0) or 0),
                "edges": int(counts.get("edges", 0) or 0),
                "status": status,
                "at": ts,
            },
        )
        if source:
            src_marker = _system_marker(engine, source, role="source")
            engine.link_nodes(run_id, src_marker, RegistryEdgeType.WAS_DERIVED_FROM)
        if sink:
            sink_marker = _system_marker(engine, sink, role="sink")
            engine.link_nodes(sink_marker, run_id, RegistryEdgeType.WAS_DERIVED_FROM)
    except Exception:  # noqa: BLE001 - lineage must never break the ETL run
        logger.debug("lineage: record_etl_run failed", exc_info=True)
        return None
    return run_id


def query_lineage(
    engine: Any,
    *,
    source: str | None = None,
    sink: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return recorded ETL runs (most-recent first), optionally filtered by source
    and/or sink. Property-based query (label-agnostic) so it works across backends.
    """
    backend = getattr(engine, "backend", None)
    if backend is None or not hasattr(backend, "execute"):
        return []
    where = ["n.kind = $kind"]
    params: dict[str, Any] = {"kind": _RUN_KIND}
    if source:
        where.append("n.source = $source")
        params["source"] = source.strip().lower()
    if sink:
        where.append("n.sink = $sink")
        params["sink"] = sink.strip().lower()
    query = (
        f"MATCH (n) WHERE {' AND '.join(where)} "
        f"RETURN n.id AS id, n.source AS source, n.sink AS sink, "
        f"n.direction AS direction, n.nodes AS nodes, n.edges AS edges, "
        f"n.status AS status, n.at AS at "
        f"ORDER BY n.at DESC LIMIT {int(limit)}"
    )
    try:
        rows = backend.execute(query, params)
    except Exception:  # noqa: BLE001 - read is best-effort
        logger.debug("lineage: query failed", exc_info=True)
        return []
    return [dict(r) for r in (rows or []) if isinstance(r, dict)]


def record_connector_sync_activity(
    engine: Any,
    *,
    connector: str,
    source_instance: str = "",
    status: str = "running",
    record_count: int | None = None,
    failed_count: int | None = None,
    activity_id: str | None = None,
    at: float | None = None,
) -> str | None:
    """Record (or update) one connector sync run as a PROV-O Activity node.

    CONCEPT:AU-KG.ingest.ambient-connector-provenance (W3.4) — the batch-level
    provenance twin of :func:`record_etl_run`, scoped to ``source_sync``'s
    connector handlers: ONE :class:`RegistryNodeType.PROVENANCE_ACTIVITY` node
    per sync run (``kind="connector_sync"``), never one per ingested row (that
    would be ingestion-hot-path-prohibitive at connector-sync volumes).

    Call once BEFORE the batch (omit ``activity_id``/counts — mints a fresh
    id and a ``status="running"`` node) and once AFTER it (pass back the SAME
    ``activity_id`` plus the final ``record_count``/``failed_count``/
    ``status``) — ``engine.add_node`` is a merge-upsert, so the second call
    only adds the final counts onto the same node.

    Best-effort and engine-guarded, exactly like :func:`record_etl_run`: an
    ``engine`` with no callable ``add_node`` (e.g. a lightweight test double
    that only implements ``ingest_external_batch``) or any write failure is
    tolerated — a failure to record provenance never breaks the sync it is
    observing. Returns the activity id, or ``None`` when nothing was recorded.
    """
    if engine is None:
        return None
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        return None
    ts = at if at is not None else time.time()
    # The volatile part (source_instance/timestamp/uuid) is folded into ONE
    # 64-hex sha256 digest rather than kept as separate colon segments: the
    # persistence privacy gate's opaque-identity exemption
    # (``envelope_ingest._OPAQUE_NAMESPACED_ID_RE``) only skips its
    # PII/pattern scan for a ``namespace:<opaque-digest>`` id whose FINAL
    # segment is a bare 24/32/40/64-hex digest — a raw millisecond timestamp
    # (starts with a digit) or the ``source_instance or '_'`` fallback value
    # (starts with ``_``) breaks that match, so the trailing random
    # ``uuid4().hex`` fell through to the full-text scan instead of being
    # treated as opaque and reproducibly tripped the generic IBAN pattern
    # (~1 in 20 hex strings, same class as D-GM-3/D-GM-4) — a legitimate,
    # non-sensitive internal id rejected as an "unsafe identity" with no PII
    # involved. Keeping only ``activity:<connector>`` ahead of the digest
    # keeps the connector name scanned (as intended) while the digest itself
    # is exempted.
    digest = hashlib.sha256(
        f"{source_instance}:{ts}:{uuid.uuid4().hex}".encode()
    ).hexdigest()
    activity_id = activity_id or f"activity:{connector}:{digest}"
    props: dict[str, Any] = {
        "kind": _CONNECTOR_SYNC_KIND,
        "connector": connector,
        "sourceInstance": source_instance,
        "status": status,
        "at": ts,
    }
    if record_count is not None:
        props["recordCount"] = int(record_count)
    if failed_count is not None:
        props["failedCount"] = int(failed_count)
    try:
        add_node(activity_id, RegistryNodeType.PROVENANCE_ACTIVITY, props)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        logger.debug(
            "lineage: connector sync activity %s failed", activity_id, exc_info=True
        )
        return None
    return activity_id


def record_connector_sync_claim(
    engine: Any,
    *,
    connector: str,
    source_instance: str = "",
    record_count: int,
    activity_id: str | None = None,
    at: float | None = None,
) -> str | None:
    """Persist ONE ``:Claim`` summarizing a connector sync run, e.g. "dockerhub
    reported 42 record(s) as of 2026-07-23T12:00:00+00:00".

    CONCEPT:AU-KG.ingest.ambient-connector-provenance (W3.4) — the run-level
    summary twin of :func:`record_connector_sync_activity`. Deliberately ONE
    claim per RUN, never per ingested row (a ``:Claim``-per-row would be too
    heavy at connector-sync volumes): reuses the same direct
    :class:`~agent_utilities.models.knowledge_graph.ClaimNode` +
    ``engine.add_node(id, "Claim", ...)`` persistence path already used for
    lightweight, self-verifying system observations (e.g.
    ``orchestration.agent_dispatch_worker``'s policy-decision claim) rather
    than the governed mining-flywheel lifecycle (``claim_flywheel.
    ClaimFlywheel`` / ``candidate_insight.CandidateInsight``), which is
    reserved for INFERRED findings that need a confidence floor and human/
    governance review — a routine sync-run count is a directly observed fact
    about this run, not a mined inference, so it persists with
    ``confidence=1.0``/``is_verified=True`` from the start, never a
    ``"proposal"``.

    Best-effort, same contract as :func:`record_connector_sync_activity`.
    """
    if engine is None:
        return None
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        return None
    from agent_utilities.models.knowledge_graph import ClaimNode

    ts = at if at is not None else time.time()
    when = datetime.fromtimestamp(ts, tz=UTC).isoformat()
    scope = f"{connector}/{source_instance}" if source_instance else connector
    claim_id = f"claim:sync:{connector}:{source_instance or '_'}:{int(ts * 1000)}"
    claim = ClaimNode(
        id=claim_id,
        name=f"Sync summary: {scope}",
        claim_text=f"{scope} reported {int(record_count)} record(s) as of {when}",
        claim_type=_CONNECTOR_SYNC_CLAIM_TYPE,
        confidence=1.0,
        is_verified=True,
        source_ids=[activity_id] if activity_id else [],
        extracted_from=activity_id,
        domain=connector,
    )
    props = claim.to_graph_properties(exclude={"id"})
    try:
        add_node(claim_id, "Claim", props)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        logger.debug("lineage: connector sync claim %s failed", claim_id, exc_info=True)
        return None
    return claim_id


_MEDIA_SIDECAR_KIND = "media_sidecar"
_MEDIA_SIDECAR_CLAIM_TYPE = "observation"


def record_media_sidecar_activity(
    engine: Any,
    *,
    sidecar: str,
    tool: str,
    modality: str,
    action: str = "",
    status: str = "ok",
    locus_count: int | None = None,
    activity_id: str | None = None,
    at: float | None = None,
) -> str | None:
    """Record one media-sidecar delegation call as a PROV-O Activity node.

    CONCEPT:AU-KG.ingest.media-sidecar-delegation (W4.6) — the sidecar-delegate twin of
    :func:`record_connector_sync_activity`: ONE
    :class:`RegistryNodeType.PROVENANCE_ACTIVITY` node per delegated
    extraction call (``kind="media_sidecar"``), carrying the
    sidecar/tool/modality/action that produced it — the generator identity
    :func:`record_media_sidecar_claim`'s ``was_generated_by`` metadata
    mirrors. Called once, up front, by
    ``agent_utilities/media/sidecar_delegate.py``'s ``delegate_extract`` —
    unlike :func:`record_connector_sync_activity`'s call-once-before/
    call-once-after pair, a single delegated tool call has no separate
    "running" phase worth recording, so this writes the completed activity
    in one shot.

    Best-effort and engine-guarded, exactly like
    :func:`record_connector_sync_activity`: an ``engine`` with no callable
    ``add_node`` or any write failure is tolerated — a failure to record
    provenance never breaks the delegation it is observing. Returns the
    activity id, or ``None`` when nothing was recorded.
    """
    if engine is None:
        return None
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        return None
    ts = at if at is not None else time.time()
    # Same fix as record_connector_sync_activity above (see its comment): fold
    # the volatile parts into one opaque sha256 digest instead of raw colon
    # segments, so the privacy gate's opaque-identity exemption actually
    # matches and the random uuid suffix doesn't undergo the full-text scan.
    digest = hashlib.sha256(f"{modality}:{ts}:{uuid.uuid4().hex}".encode()).hexdigest()
    activity_id = activity_id or f"activity:{_MEDIA_SIDECAR_KIND}:{sidecar}:{digest}"
    props: dict[str, Any] = {
        "kind": _MEDIA_SIDECAR_KIND,
        "sidecar": sidecar,
        "tool": tool,
        "modality": modality,
        "action": action,
        "status": status,
        "at": ts,
    }
    if locus_count is not None:
        props["locusCount"] = int(locus_count)
    try:
        add_node(activity_id, RegistryNodeType.PROVENANCE_ACTIVITY, props)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        logger.debug(
            "lineage: media sidecar activity %s failed", activity_id, exc_info=True
        )
        return None
    return activity_id


def record_media_sidecar_claim(
    engine: Any,
    *,
    sidecar: str,
    modality: str,
    artifact_id: str,
    summary: str,
    activity_id: str | None = None,
    at: float | None = None,
) -> str | None:
    """Persist ONE ``:Claim`` summarizing a media-sidecar delegation call,
    e.g. "stirlingpdf-mcp extracted 4 page(s) from doc-quarterly-report as of
    2026-07-24T00:00:00+00:00".

    CONCEPT:AU-KG.ingest.media-sidecar-delegation (W4.6) — the summary twin of
    :func:`record_media_sidecar_activity`, following
    :func:`record_connector_sync_claim`'s EXACT pattern: a directly-verified
    system observation (``confidence=1.0``/``is_verified=True`` from the
    start — the sidecar genuinely produced this many loci, it is not an
    inferred finding needing review), carrying the SAME ``was_generated_by``/
    ``generated_at_time`` PROV-O metadata convention
    ``mcp/tools/ops_causal_tools.py``'s claim materialization and
    ``owlready2_backend``'s PROV-O edge-alias table already recognize. The
    caller links each ``store_<locus>_evidence`` write-back's ``claim_id``
    to the returned id so ``evidence_citations``'s SUPPORTS-walk resolves
    every sidecar-produced locus through this ONE claim
    (CONCEPT:AU-KG.identity.evidence-spine-convergence).

    Best-effort, same contract as :func:`record_media_sidecar_activity`.
    """
    if engine is None:
        return None
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        return None
    from agent_utilities.models.knowledge_graph import ClaimNode

    ts = at if at is not None else time.time()
    when = datetime.fromtimestamp(ts, tz=UTC).isoformat()
    claim_id = f"claim:media_sidecar:{sidecar}:{modality}:{int(ts * 1000)}"
    claim = ClaimNode(
        id=claim_id,
        name=f"Media sidecar: {sidecar}/{modality}",
        claim_text=f"{summary} (via {sidecar}, as of {when})",
        claim_type=_MEDIA_SIDECAR_CLAIM_TYPE,
        confidence=1.0,
        is_verified=True,
        source_ids=[activity_id] if activity_id else [],
        extracted_from=activity_id,
        domain=modality,
        metadata={
            "was_generated_by": f"agent:{sidecar}",
            "generated_at_time": when,
            "artifact_id": artifact_id,
        },
    )
    props = claim.to_graph_properties(exclude={"id"})
    try:
        add_node(claim_id, "Claim", props)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        logger.debug("lineage: media sidecar claim %s failed", claim_id, exc_info=True)
        return None
    return claim_id


def record_openlineage_run_event(engine: Any, event: dict[str, Any]) -> str | None:
    """Map one OpenLineage RunEvent into ``prov:Activity``/``prov:Entity``/
    ``prov:used``/``prov:wasGeneratedBy`` and return the Activity node id.

    CA-25/DEC-CA-05 (CONCEPT:AU-KG.ingest.openlineage-consumer) — the
    OpenLineage-fed twin of :func:`record_connector_sync_activity`: ONE
    :class:`RegistryNodeType.PROVENANCE_ACTIVITY` node per logical run
    (``kind="openlineage_run"``), keyed deterministically on ``run.runId`` so
    a redelivered event (same ``runId``) merge-upserts the SAME node rather
    than creating a duplicate — ``engine.add_node`` is upsert, and a later
    lifecycle event (e.g. ``START`` then ``COMPLETE`` for the same run)
    naturally advances the SAME node's ``status`` rather than needing a
    separate dedupe table.

    Validation/classification is :func:`~.openlineage_consumer.map_openlineage_event`'s
    job (imported locally to avoid a module-load cycle with
    ``openlineage_consumer``, which itself calls this function). A
    :class:`~.openlineage_consumer.QuarantinedLineageEvent` result is logged
    and this returns ``None`` — no ``prov:Entity``/``prov:Activity`` is ever
    fabricated for an event that failed to map (DEC-CA-05's Authority
    section). Otherwise best-effort and engine-guarded, exactly like
    :func:`record_connector_sync_activity`: any write failure is tolerated —
    a failure to record lineage must never break the run it describes.
    """
    if engine is None:
        return None
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        return None

    from .openlineage_consumer import QuarantinedLineageEvent, map_openlineage_event

    mapped = map_openlineage_event(event)
    if isinstance(mapped, QuarantinedLineageEvent):
        logger.warning(
            "lineage: quarantined openlineage event run=%s job=%s (%s)",
            mapped.run_id,
            mapped.job_name,
            mapped.reason,
        )
        return None

    digest = hashlib.sha256(mapped.run_id.encode()).hexdigest()
    activity_id = f"activity:{_OPENLINEAGE_ACTIVITY_KIND}:{digest}"
    try:
        add_node(
            activity_id,
            RegistryNodeType.PROVENANCE_ACTIVITY,
            {
                "kind": _OPENLINEAGE_ACTIVITY_KIND,
                "job": mapped.job_name,
                "jobNamespace": mapped.job_namespace,
                "eventType": mapped.event_type,
                "status": mapped.activity_status,
                "at": time.time(),
            },
        )
        for dataset_id in mapped.input_dataset_ids:
            marker = _dataset_marker(engine, dataset_id)
            engine.link_nodes(activity_id, marker, RegistryEdgeType.USED)
        for dataset_id in mapped.output_dataset_ids:
            marker = _dataset_marker(engine, dataset_id)
            engine.link_nodes(marker, activity_id, RegistryEdgeType.WAS_GENERATED_BY)
    except Exception:  # noqa: BLE001 - lineage must never break the run it describes
        logger.debug(
            "lineage: record_openlineage_run_event failed for run=%s",
            mapped.run_id,
            exc_info=True,
        )
        return None

    # RunTrace correlation is a nice-to-have, isolated from the write above:
    # a correlation failure (or the run simply not being tool-originated)
    # must never void an otherwise-successful Activity/Entity write.
    try:
        from agent_utilities.observability.trace_ontology import (
            TRACE_LINEAGE_ACTIVITY_EDGE,
            correlate_lineage_run_trace,
        )

        trace_node_id = correlate_lineage_run_trace(engine, mapped.run_id) or (
            correlate_lineage_run_trace(engine, mapped.parent_run_id)
            if mapped.parent_run_id
            else None
        )
        if trace_node_id:
            engine.link_nodes(trace_node_id, activity_id, TRACE_LINEAGE_ACTIVITY_EDGE)
    except Exception:  # noqa: BLE001 - correlation is best-effort
        logger.debug(
            "lineage: RunTrace correlation failed for run=%s",
            mapped.run_id,
            exc_info=True,
        )

    return activity_id
