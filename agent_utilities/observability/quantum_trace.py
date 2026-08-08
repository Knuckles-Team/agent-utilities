"""CONCEPT:AU-KG.temporal.quantum-run-provenance -- Q9: wire the quantum surface into
the EXISTING `:ToolCall`/`:RunTrace` provenance model.

Per the lane brief: "the repo already has a `:ToolCall`/`RunTrace` model ... Q9 is
'wire quantum into it,' not build it from scratch." This module adds exactly ONE new
typed node, `:QuantumJob`, grounded to the calling run the SAME way
`agent_utilities/runtime/provenance.py`'s `:WorkspaceObservation` is grounded to its
`:WorkspaceAction` (a `HAS_*`-shaped edge off the run/action, not a parallel
provenance system) -- see `trace_ontology.py`'s own module doc for the shared schema
this reuses (`TRACE_SCHEMA_VERSION`, `content_digest`, `trace_id`).

Graph shape::

    (:RunTrace {id: trace:<opaque-ref>}) -[:HAS_QUANTUM_RUN]-> (:QuantumJob)

A `:QuantumJob` carries the FULL Q0/Q9 metadata the engine's `Method::Quantum`
response already returns verbatim (backend id, formalism, seed, shots, circuit
hash, exact, noise model id, fidelity hint, wall time, peak memory) PLUS the
planner's R0-R5 audit trail (`eg_quantum_core::planner::PlannerDecision.audit`,
called "what Q9 observability persists" by that crate's own doc) so an explicit R5
`backend_id` override is durably auditable from the KG side, per the program's
"always honoured but MUST be audited" requirement -- satisfied here, not in the
engine's own graph-tamper-evidence chain (see `eg-capabilities`' `Method::Quantum`
policy doc for why).

Best-effort, like every other provenance write in this codebase: a cold or absent
KG must never break a `graph_quantum` call.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from agent_utilities.observability.trace_ontology import (
    TRACE_SCHEMA_VERSION,
    content_digest,
    next_event_sequence,
)
from agent_utilities.observability.trace_ontology import trace_id as canonical_trace_id
from agent_utilities.security.persistence_privacy import persistence_reference

logger = logging.getLogger(__name__)

QUANTUM_JOB_NODE_LABEL = "QuantumJob"
HAS_QUANTUM_RUN_EDGE = "HAS_QUANTUM_RUN"


def _quantum_job_id(run_ref: str, event_sequence: int) -> str:
    # Sequence-addressed like a ToolCall id (`toolcall:<run_ref>:<i>`), but keyed by
    # the monotonic event sequence rather than a batch index -- this node is written
    # from INSIDE the live call, not reconstructed post-hoc from a collected list.
    return f"quantumjob:{run_ref}:{event_sequence}"


def persist_quantum_job(
    engine: Any,
    *,
    run_id: str,
    actor: str,
    tenant: str,
    operation: str,
    result: dict[str, Any],
    backend_override_requested: str | None,
) -> None:
    """Persist one `:QuantumJob` node grounded to the caller's `:RunTrace`.

    `result` is the exact JSON dict `Method::Quantum`'s response decoded to (see
    `src/server/handlers/quantum.rs`'s `result_json`/`planner_json`) -- this function
    reshapes it into durable, privacy-appropriate KG properties; it never re-derives
    the metadata.
    """

    if engine is None or not isinstance(result, dict):
        return
    trace_ref = canonical_trace_id(run_id)
    run_ref = trace_ref.removeprefix("trace:")
    seq = next_event_sequence()
    node_id = _quantum_job_id(run_ref, seq)
    planner = result.get("planner") if isinstance(result.get("planner"), dict) else {}
    audit_trail = planner.get("audit_trail") if isinstance(planner, dict) else None

    props: dict[str, Any] = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": trace_ref,
        "operation": str(operation or ""),
        "actor_ref": persistence_reference("actor", actor or "", namespace="quantum-run"),
        "tenant_ref": persistence_reference("tenant", tenant or "", namespace="quantum-run"),
        "backend_id": str(result.get("backend_id") or ""),
        "formalism": str(result.get("formalism") or ""),
        "seed": result.get("seed"),
        "shots": result.get("shots"),
        "circuit_hash": str(result.get("circuit_hash") or ""),
        "exact": bool(result.get("exact", False)),
        "proposal": bool(result.get("proposal", True)),
        "noise_model_id": result.get("noise_model_id"),
        "fidelity_hint": result.get("fidelity_hint"),
        "wall_time_ms": result.get("wall_time_ms"),
        "peak_memory_bytes": result.get("peak_memory_bytes"),
        # R5 escape hatch, audited (Q9): non-null exactly when the caller explicitly
        # overrode the planner. `planner_rule` is the rule that actually fired
        # (`"r5_override"` when it did); the full trail is digested (it can carry
        # caller-influenced free text in its `note` fields).
        "backend_override_requested": backend_override_requested,
        "planner_chosen_backend": planner.get("chosen_backend") if isinstance(planner, dict) else None,
        "planner_rule": planner.get("rule") if isinstance(planner, dict) else None,
        "planner_audit_count": len(audit_trail) if isinstance(audit_trail, list) else 0,
        "planner_audit_digest": content_digest(audit_trail),
        "payload_digest": content_digest(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "backend_id",
                    "formalism",
                    "seed",
                    "shots",
                    "circuit_hash",
                    "exact",
                    "proposal",
                    "noise_model_id",
                    "fidelity_hint",
                    "wall_time_ms",
                    "peak_memory_bytes",
                    "planner",
                    "operation",
                }
            }
        ),
        "event_sequence": seq,
        "event_cursor": seq,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with contextlib.suppress(Exception):
        engine.add_node(node_id, QUANTUM_JOB_NODE_LABEL, properties=props)
        # Best-effort: ground the run's RunTrace even if it has not been separately
        # materialized yet (a `graph_quantum` call can be the FIRST tool call of a
        # turn) -- `link_nodes` is a no-op/soft-fail against a missing endpoint on
        # every other provenance writer in this codebase (see `provenance.py`), never
        # a hard dependency.
        engine.link_nodes(trace_ref, node_id, HAS_QUANTUM_RUN_EDGE)
