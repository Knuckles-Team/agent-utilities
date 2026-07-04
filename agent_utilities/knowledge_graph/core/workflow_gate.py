"""Execution-time workflow ontology gate.

CONCEPT:AU-ORCH.execution.ontology-validation-execution-path — Ontology Validation on the Execution Path

Until now the ontology (SHACL shapes, permission ACLs) governed *ingestion*
(pipeline ``shacl_gate``) and *reads* (OS-5.14 secured reads), but workflow
*execution* dispatched whatever was stored. This module is the missing gate
``execute_workflow`` runs **before dispatch**:

1. **Shape gate** (``KG_WORKFLOW_SHAPE_GATE``, default ON — cheap, LLM-free):
   the stored ``WorkflowDefinition`` + its ``WorkflowStep`` nodes are
   materialized into a focused RDF graph (``http://knuckles.team/kg#``
   namespace, matching the ``sh:targetClass`` IRIs) and validated against the
   bundled governance shapes (``WorkflowDefinitionShape`` /
   ``WorkflowStepShape``). Violations refuse execution with a structured
   report — a malformed definition never burns an agent run.

2. **Permission gate** (active only when ``KG_BRAIN_ENFORCE`` is on,
   OS-5.14 semantics): the ontology permissioning row gate
   (:func:`~agent_utilities.knowledge_graph.ontology.permissioning.enforce`,
   mandatory markings + discretionary ACLs, fail-closed) is applied to the
   workflow node for the current :class:`ActorContext`. A denied actor gets
   ``PermissionError``; enforcement off skips the check entirely (legacy
   behavior).

A workflow name with **no stored definition** passes through untouched —
dynamic/ad-hoc execution paths (completion-state loops) are not stored
workflows and keep their existing behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# kg# namespace — matches the ontology + shapes files and the pipeline gate.
KG_NS = "http://knuckles.team/kg#"

_GOVERNANCE_SHAPES = Path(__file__).parent.parent / "shapes" / "governance.shapes.ttl"


def workflow_shape_gate_enabled() -> bool:
    """Whether the execution-time SHACL gate is active (``KG_WORKFLOW_SHAPE_GATE``)."""
    try:
        from agent_utilities.core.config import config

        return bool(getattr(config, "kg_workflow_shape_gate", True))
    except Exception:  # noqa: BLE001 — config unavailable → gate on (safe default)
        return True


# ---------------------------------------------------------------------------
# Stored-definition lookup
# ---------------------------------------------------------------------------


def _find_workflow(
    engine: Any, name: str
) -> tuple[str | None, dict[str, Any], list[dict[str, Any]]]:
    """Locate a stored WorkflowDefinition by name.

    Returns ``(workflow_id, definition_props, step_props_list)``;
    ``workflow_id`` is ``None`` when nothing is stored under that name.
    Mirrors WorkflowStore's backend-then-compute resolution.
    """
    backend = getattr(engine, "backend", None)
    if backend is not None:
        try:
            rows = backend.execute(
                "MATCH (w:WorkflowDefinition) WHERE w.name = $name "
                "RETURN w.id AS wid, w.name AS name, w.step_count AS step_count "
                "ORDER BY w.version DESC LIMIT 1",
                {"name": name},
            )
            if rows:
                wid = rows[0].get("wid")
                props = {
                    "name": rows[0].get("name"),
                    "step_count": rows[0].get("step_count"),
                }
                step_rows = backend.execute(
                    "MATCH (w:WorkflowDefinition {id: $wid})-[:HAS_STEP]->"
                    "(s:WorkflowStep) "
                    "RETURN s.id AS sid, s.node_id AS node_id, "
                    "s.step_order AS step_order",
                    {"wid": wid},
                )
                steps = [dict(r) for r in step_rows or []]
                return wid, props, steps
        except Exception as exc:  # noqa: BLE001 — fall through to compute graph
            logger.debug("[ORCH-1.42] backend workflow lookup failed: %s", exc)

    graph = getattr(engine, "graph", None)
    if graph is not None:
        try:
            wid = None
            nx_props: dict[str, Any] = {}
            for nid, data in graph.nodes(data=True):
                if (
                    data.get("type") == "WorkflowDefinition"
                    and data.get("name") == name
                ):
                    wid = nid
                    nx_props = {
                        "name": data.get("name"),
                        "step_count": data.get("step_count"),
                    }
                    break
            if wid is None:
                return None, {}, []
            steps = []
            for _src, tgt, edata in graph.out_edges(wid, data=True):
                rel = str(
                    (edata or {}).get("type") or (edata or {}).get("rel_type") or ""
                ).upper()
                if rel != "HAS_STEP":
                    continue
                sdata = dict(graph.nodes[tgt])
                steps.append(
                    {
                        "sid": tgt,
                        "node_id": sdata.get("node_id"),
                        "step_order": sdata.get("step_order"),
                    }
                )
            return wid, nx_props, steps
        except Exception as exc:  # noqa: BLE001 — absent mirror → not stored
            logger.debug("[ORCH-1.42] compute-graph workflow lookup failed: %s", exc)

    return None, {}, []


# ---------------------------------------------------------------------------
# SHACL shape validation
# ---------------------------------------------------------------------------


def _build_workflow_rdf(
    workflow_id: str, props: dict[str, Any], steps: list[dict[str, Any]]
) -> Any:
    """Materialize the stored definition into a focused rdflib graph (kg# ns)."""
    import rdflib

    g = rdflib.Graph()
    kg = rdflib.Namespace(KG_NS)
    g.bind("", kg)

    wf_uri = kg[str(workflow_id).replace(" ", "_")]
    g.add((wf_uri, rdflib.RDF.type, kg.WorkflowDefinition))
    name = props.get("name")
    if isinstance(name, str) and name:
        g.add((wf_uri, kg.name, rdflib.Literal(name)))
    step_count = props.get("step_count")
    if isinstance(step_count, int | float):
        g.add((wf_uri, kg.step_count, rdflib.Literal(int(step_count))))

    for step in steps:
        sid = str(step.get("sid") or f"{workflow_id}:step").replace(" ", "_")
        step_uri = kg[sid]
        g.add((step_uri, rdflib.RDF.type, kg.WorkflowStep))
        node_id = step.get("node_id")
        if isinstance(node_id, str) and node_id:
            g.add((step_uri, kg.node_id, rdflib.Literal(node_id)))
        order = step.get("step_order")
        if isinstance(order, int | float):
            g.add((step_uri, kg.step_order, rdflib.Literal(int(order))))
    return g


def _validate_workflow_shape(
    workflow_id: str, props: dict[str, Any], steps: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run the SHACLValidator over the focused workflow graph."""
    try:
        import pyshacl  # noqa: F401
        import rdflib  # noqa: F401
    except ImportError:
        # No SHACL stack installed — the gate cannot run; pass through (the
        # pipeline ingestion gate degrades identically).
        return {"conforms": True, "violations": []}

    from .shacl_validator import SHACLValidator

    if not _GOVERNANCE_SHAPES.exists():  # pragma: no cover - packaged install
        return {"conforms": True, "violations": []}
    data_graph = _build_workflow_rdf(workflow_id, props, steps)
    report = SHACLValidator().validate(data_graph, _GOVERNANCE_SHAPES)
    return {
        "conforms": bool(report.get("conforms", True)),
        "violations": report.get("violations", []),
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def gate_workflow_execution(
    engine: Any, workflow_name: str, actor: Any = None
) -> dict[str, Any]:
    """Validate a stored workflow before dispatch (CONCEPT:AU-ORCH.execution.ontology-validation-execution-path).

    Args:
        engine: The IntelligenceGraphEngine.
        workflow_name: The workflow name the caller asked to execute.
        actor: Optional ActorContext override; defaults to the ambient actor.

    Returns:
        ``{"allowed": bool, "workflow_id": ..., "violations": [...]}``.
        ``allowed=False`` carries the structured SHACL violation report.
        A name with no stored definition returns allowed (legacy dynamic
        execution paths are not gated).

    Raises:
        PermissionError: when ``KG_BRAIN_ENFORCE`` is on and the ontology
            permissioning row gate denies the actor on the workflow node
            (fail-closed, OS-5.14 semantics).
    """
    wid, props, steps = _find_workflow(engine, workflow_name)
    if wid is None:
        return {"allowed": True, "workflow_id": None, "violations": []}

    if workflow_shape_gate_enabled():
        report = _validate_workflow_shape(wid, props, steps)
        if not report["conforms"]:
            logger.warning(
                "[ORCH-1.42] workflow %r failed shape validation: %d violation(s)",
                workflow_name,
                len(report["violations"]),
            )
            return {
                "allowed": False,
                "workflow_id": wid,
                "violations": report["violations"],
            }

    from .company_brain_runtime import brain_enforcement_enabled

    if brain_enforcement_enabled():
        from ...security.brain_context import current_actor
        from ..ontology.permissioning import enforce

        effective = actor or current_actor()
        view = enforce([{"id": wid, "type": "WorkflowDefinition", **props}], effective)
        if not view:
            raise PermissionError(
                f"Actor {getattr(effective, 'actor_id', 'unknown')!r} is not "
                f"permitted to execute workflow {workflow_name!r} ({wid}) — "
                "denied by ontology permissioning (markings/ACL, fail-closed)."
            )

    return {"allowed": True, "workflow_id": wid, "violations": []}
