#!/usr/bin/python
from __future__ import annotations

"""The unified `:DurableExecutionUnit` KG mirror (CONCEPT:AU-KG.storage.durable-execution-unit).

DE1 of `plans/au-eg-program/program/durable-execution-native.md`: DE0 (lane
`w6-de0-contracts`) reserved the concept block and wrote the mirrored-property/
edge schema into the canonical ontology
(`agent_utilities/knowledge_graph/ontology_orchestration.ttl`) — one abstract
`:DurableExecutionUnit` class plus four subclasses (`:StatechartInstance`,
`:AnalyticsJob`, `:SagaCoordination`, `:DurableRun`), each a PROVENANCE MIRROR of
an already-real, already-durable backend row, never a second store. This module
is DE1's mirror-on-write implementation for the one backend fully reachable from
Python today: `agent_utilities.orchestration.durable_execution.DurableRun` (the
`eg-durable` routing crate's own doc names this backend "reached only over the
engine RPC boundary DE2 wires, never a Rust dependency edge" — for the KG mirror
specifically, no RPC hop is needed, since `DurableRun` already runs in-process
here and this module writes through the same `engine.add_node`/`engine.link_nodes`
surface `KgAuditSink`/`RunTrace` use).

Mirroring `eg-statechart::MachineInstance` / `eg-jobs::AnalyticsJob` /
`eg-mutation-store::SagaBegin` has no writer in this module: those rows
are authoritative Rust state with no equivalent Python-callable read/write path
today (the generated wire protocol, `protocols/epistemic_operations/_generated.py`,
carries `AnalyticsJob` as a DTO but no live query/mutate method for it, and no DTO
at all yet for a statechart instance or a saga). Wiring those three requires
either a new engine wire-protocol surface or Rust-side dispatch-handler wiring
(the same "projection function exists, no live caller yet" posture
`eg-statechart::kg::project` — the statechart DEFINITION projector — already
carries, unwired since it shipped). That gap is real and is not silently
narrowed away here; see `docs/durable-execution.md`.

Fail-soft, exactly like `KgAuditSink`: every write takes an optional ``engine``
and is a silent no-op (logged at DEBUG) when one is not supplied, so a
provenance mirror write can never fail (or slow down) the durable run it
describes.
"""

import logging
from typing import Any

from agent_utilities.security.identifiers import (
    InvalidIdentifierError,
    validate_identifier,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DURABLE_EXECUTION_UNIT_LABEL",
    "DURABLE_RUN_LABEL",
    "ANALYTICS_JOB_LABEL",
    "STATECHART_INSTANCE_LABEL",
    "SAGA_COORDINATION_LABEL",
    "DURABLE_EXECUTION_UNIT_LABELS",
    "AWAITS_EDGE",
    "COORDINATED_BY_EDGE",
    "PRODUCED_EDGE",
    "durable_run_unit_id",
    "mirror_durable_run",
    "link_durable_run_produced",
    "list_durable_execution_units",
]

# Concrete subclass labels — CONCEPT:AU-KG.storage.durable-execution-unit /
# AU-KG.storage.statechart-instance-mirror / AU-KG.txn.saga-coordination-mirror /
# AU-ORCH.scheduling.analytics-job-mirror / AU-ORCH.runvcs.durable-run-mirror.
# `add_node`'s `node_type` is a single label (matching how `RunTrace` itself is
# mirrored, `observability.trace_ontology.TRACE_NODE_LABEL`) — a node is written
# under its concrete subclass label; the OWL `rdfs:subClassOf :DurableExecutionUnit`
# relationship in the ontology, not a second stamped label, is what makes it a
# `:DurableExecutionUnit` for ontology-aware reasoning. `durable_unit_kind` (see
# `mirror_durable_run`) additionally makes that superclass identity a plain,
# ontology-agnostic Cypher-filterable property for the cross-backend query below.
DURABLE_EXECUTION_UNIT_LABEL = "DurableExecutionUnit"
DURABLE_RUN_LABEL = "DurableRun"
ANALYTICS_JOB_LABEL = "AnalyticsJob"
STATECHART_INSTANCE_LABEL = "StatechartInstance"
SAGA_COORDINATION_LABEL = "SagaCoordination"

# The four concrete backend labels DE0's ontology reserves. Only DURABLE_RUN_LABEL
# has a live writer today (this module); the other three are named here so
# `list_durable_execution_units` already queries the family DE1 defines, not just
# the one subset mirrored so far — a future writer for one of them needs no
# change to the read side.
DURABLE_EXECUTION_UNIT_LABELS = (
    DURABLE_RUN_LABEL,
    ANALYTICS_JOB_LABEL,
    STATECHART_INSTANCE_LABEL,
    SAGA_COORDINATION_LABEL,
)

AWAITS_EDGE = "awaits"
COORDINATED_BY_EDGE = "coordinatedBy"
PRODUCED_EDGE = "produced"


def durable_run_unit_id(session_id: str, run_id: str) -> str:
    """Stable, opaque `:DurableRun` node id for one `DurableRun` (`:backendRef`)."""

    return f"durable-run:{session_id}:{run_id}"


def mirror_durable_run(
    engine: Any | None,
    *,
    session_id: str,
    run_id: str,
    durable_status: str,
    checkpoint_ref: str | None = None,
    definition_version: str | None = None,
    idempotency_key: str | None = None,
) -> str | None:
    """Upsert one `:DurableRun` mirror node; return its id, or ``None`` if no-op.

    Mirrors `DurableExecutionManager`'s own checkpoint fields field-for-field
    (`durable_status` <- the checkpoint `status`, `checkpoint_ref` <- the most
    recently completed step name, `idempotency_key` <- the run-scoped step key) —
    a PROVENANCE MIRROR, never a second persistence layer: the SQLite/Postgres
    checkpoint row this call describes remains the sole durability authority.
    ``engine`` is optional and the write is a no-op (logged, never raised) when
    absent — the same posture `KgAuditSink(engine=None)` already uses, so this
    call is safe to make unconditionally from `DurableRun`.
    """

    if engine is None:
        return None
    node_id = durable_run_unit_id(session_id, run_id)
    properties: dict[str, Any] = {
        "backend_ref": f"{session_id}:{run_id}",
        "durable_status": str(durable_status),
        "durable_unit_kind": DURABLE_EXECUTION_UNIT_LABEL,
        "session_id": session_id,
        "run_id": run_id,
    }
    if checkpoint_ref:
        properties["checkpoint_ref"] = str(checkpoint_ref)
    if definition_version:
        properties["definition_version"] = str(definition_version)
    if idempotency_key:
        properties["idempotency_key"] = str(idempotency_key)
    try:
        engine.add_node(node_id, DURABLE_RUN_LABEL, properties=properties)
    except Exception:  # noqa: BLE001 — a provenance mirror must never fail the run
        logger.warning(
            "durable_execution_kg: failed to mirror DurableRun %r",
            node_id,
            exc_info=True,
        )
        return None
    return node_id


def link_durable_run_produced(
    engine: Any | None, *, session_id: str, run_id: str, run_trace_id: str
) -> None:
    """Link `:DurableRun -[:PRODUCED]-> :RunTrace` (DE0's `:produced`/`:producedBy`).

    Completes, in the reverse direction, the linkage `RunTrace` already carries
    one-directionally (`graph_checkpoint_ids`/`graph_resume_supported` —
    `observability.trace_ontology.trace_properties`): a caller that knows which
    `RunTrace` a durable run produced can now also ask, from the `:DurableRun`
    side, "which trace did this in-flight unit produce".
    """

    if engine is None or not run_trace_id:
        return
    node_id = durable_run_unit_id(session_id, run_id)
    try:
        engine.link_nodes(node_id, run_trace_id, PRODUCED_EDGE)
    except Exception:  # noqa: BLE001 — a provenance mirror must never fail the run
        logger.warning(
            "durable_execution_kg: failed to link %r -[:PRODUCED]-> %r",
            node_id,
            run_trace_id,
            exc_info=True,
        )


def list_durable_execution_units(
    engine: Any,
    *,
    kinds: tuple[str, ...] = DURABLE_EXECUTION_UNIT_LABELS,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Answer "what is durably in flight, waiting on what" across the whole family.

    Queries every concrete `:DurableExecutionUnit` subclass label in ``kinds``
    (default: all four DE0 reserves) via the engine's own Cypher surface — one
    `MATCH` per label, merged, so a future writer for `:AnalyticsJob` /
    `:StatechartInstance` / `:SagaCoordination` needs no change here. Each row
    also carries whatever `:AWAITS`/`:COORDINATED_BY` edges the node has, so a
    caller can answer "waiting on what" without a second round trip.
    """

    rows: list[dict[str, Any]] = []
    for label in kinds:
        try:
            safe_label = validate_identifier(label, kind="label")
        except InvalidIdentifierError:
            logger.warning(
                "durable_execution_kg: rejected malformed label kind", exc_info=True
            )
            continue
        where = "" if status is None else "WHERE u.durable_status = $status"
        query = (
            f"MATCH (u:{safe_label}) {where} "
            "OPTIONAL MATCH (u)-[:awaits]->(waiting_on) "
            "OPTIONAL MATCH (u)-[:coordinatedBy]->(saga) "
            "RETURN u, collect(DISTINCT waiting_on.id) AS awaits, "
            "collect(DISTINCT saga.id) AS coordinated_by "
            f"LIMIT {int(limit)}"
        )
        params = {} if status is None else {"status": status}
        try:
            result = engine.query_cypher(query, params)
        except Exception:  # noqa: BLE001 — a read helper degrades, never raises
            logger.warning(
                "durable_execution_kg: query failed for label %r",
                safe_label,
                exc_info=True,
            )
            continue
        for row in result or []:
            unit = dict(row.get("u") or {})
            unit["durable_unit_label"] = label
            unit["awaits"] = [x for x in (row.get("awaits") or []) if x]
            unit["coordinated_by"] = [x for x in (row.get("coordinated_by") or []) if x]
            rows.append(unit)
    return rows
