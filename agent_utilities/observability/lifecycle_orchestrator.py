#!/usr/bin/python
from __future__ import annotations

"""Enter-anywhere SDLC lifecycle orchestrator — the keystone that ties the
(now-ingested) incident / ticket / spec / MR / CI / image / deploy nodes into ONE
ontology-driven, gap-filling remediation lifecycle
(``reports/autonomous-sdlc-loop-design.md`` §3).

CONCEPT:AU-OS.host.report-only-remediation-proposal. Given ANY entry node on the
lifecycle spine (an ``:Incident``, a ``:Ticket``, a ``:SpecProposal``, a
``:MergeRequest`` — anything with an ``rdf:type`` in :data:`STAGE_BY_TYPE`), the
orchestrator:

1. **Resolves the entry node's stage** in the lifecycle ontology
   (``knowledge_graph/ontology_sdlc_lifecycle.ttl``).
2. **Diffs against the actual graph state** — walks the loop spine forward
   (missing downstream stages) AND backward (an entry with a missing upstream link,
   e.g. an MR with no linked :SpecProposal), using the REQUIRED shape
   (:data:`FORWARD_CHAIN`, the operational projection of
   ``shapes/sdlc_lifecycle.shapes.ttl``).
3. **Emits each missing transition as a REPORT-ONLY** ``:LifecycleStep`` proposal
   bound (``:boundCapability``) to the fleet capability that WOULD execute it —
   **without executing anything**. Idempotent/deduped on a stable signature.

**Report-only by design.** This module NEVER invokes a bound capability. Emitting a
``:LifecycleStep`` records the INTENDED transition + which capability would run it;
an operator turns proposals into action by driving them through the existing
Loop-engine execution model — ``graph_loops(action="submit", kind="develop"|...)``
and ``WorkflowRunner`` — once they enable autonomy for that transition class (the
same seam ``incidents.propose_remediation`` documents for the alert-bridge path).
The design's §3.4 escalation policy decides, per transition, ``auto`` vs
``escalate`` at that point; this keystone only PLANS the gap-fill workflow.

Enablement path (design §3.1–3.4), per transition ``name``:

* ``mint_ticket``  → ``incident_router.route_incident`` (a real SoR once
  ``INCIDENT_TICKET_ENABLE``).
* ``generate_spec`` → ``graph_orchestrate action=execute_agent agent=spec-generator``
  → ``persist_spec_proposal``.
* ``develop_spec``  → ``knowledge_graph.research.spec_proposals.develop_spec`` /
  ``graph_loops(submit, kind="develop")``.
* ``open_change_request`` → ``gitlab_merge_requests(create)`` / ``github_pulls(create)``.
* ``await_ci`` / ``build_image`` → ``gitlab_pipelines`` / ``github_actions`` (+ the
  shared ``container_pipeline.yml``).
* ``deploy`` → ``portainer_stack(redeploy_stack_git)`` / ``cm_k8s(patch_resource)``.
* ``validate_deploy`` → time-windowed ``systems-manager`` ``:HealthTrend`` query.
* ``resolve_incident`` → ``incident_router.close_ticket`` + set ``:Incident.status``.

Best-effort + engine-guarded: with no reachable engine every entry point degrades to
an empty/no-op result rather than raising. Run over one node via ``run_lifecycle`` or
as a scheduled sweep across all open spine nodes via
``python -m agent_utilities.observability.lifecycle_orchestrator``.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from agent_utilities.observability import health_ingest

logger = logging.getLogger("agent_utilities.observability.lifecycle_orchestrator")

_SOURCE = "agent-utilities-lifecycle"

# The ordered lifecycle spine (design §1.1). A node's stage is its position here.
STAGE_ORDER: list[str] = [
    "incident",
    "ticket",
    "spec",
    "code_change",
    "merge_request",
    "pipeline_run",
    "container_image",
    "deployment",
    "validation",
]

# Node label / rdf:type -> lifecycle stage. Covers every per-package producer's
# label so an entry node from ANY connector resolves to a stage.
STAGE_BY_TYPE: dict[str, str] = {
    "Incident": "incident",
    "Ticket": "ticket",
    "Issue": "ticket",
    "Spec": "spec",
    "SpecProposal": "spec",
    "CodeChange": "code_change",
    "MergeRequest": "merge_request",
    "PullRequest": "merge_request",
    "CodeChangeProposal": "merge_request",
    "PipelineRun": "pipeline_run",
    "CheckRun": "pipeline_run",
    "Pipeline": "pipeline_run",
    "ContainerImage": "container_image",
    "Deployment": "deployment",
    "DeploymentEvent": "deployment",
    "Stack": "deployment",
    "Validation": "validation",
    "HealthTrend": "validation",
    "HealthAnomaly": "validation",
}


@dataclass(frozen=True)
class Transition:
    """One directed hop on the lifecycle spine + the fleet capability that would
    execute the gap-fill for it.

    ``direction`` is ``"out"`` when the spine edge points from the earlier stage's
    node to the later one (``incident -triggers-> ticket``) or ``"in"`` when it
    points the other way (``deployment -deploys-> container_image``, so walking
    forward from an image to its deployment follows the image's IN edge).
    """

    from_stage: str
    to_stage: str
    edge: str
    to_type: str
    name: str
    capability: str
    direction: str = "out"


# The loop spine as ordered transitions (design §1.2, task predicate list). All
# edges point forward ("out") except container_image->deployment (:Deployment
# -deploys-> :ContainerImage). The terminal resolves transition closes the loop.
FORWARD_CHAIN: list[Transition] = [
    Transition(
        "incident",
        "ticket",
        "triggers",
        "Ticket",
        "mint_ticket",
        "agent_utilities.observability.incident_router:route_incident",
    ),
    Transition(
        "ticket",
        "spec",
        "specifies",
        "SpecProposal",
        "generate_spec",
        "graph_orchestrate:execute_agent:spec-generator",
    ),
    Transition(
        "spec",
        "code_change",
        "implements",
        "CodeChange",
        "develop_spec",
        "agent_utilities.knowledge_graph.research.spec_proposals:develop_spec",
    ),
    Transition(
        "code_change",
        "merge_request",
        "proposes",
        "MergeRequest",
        "open_change_request",
        "gitlab_merge_requests|github_pulls:create",
    ),
    Transition(
        "merge_request",
        "pipeline_run",
        "triggersPipeline",
        "PipelineRun",
        "await_ci",
        "gitlab_pipelines|github_actions:get",
    ),
    Transition(
        "pipeline_run",
        "container_image",
        "builds",
        "ContainerImage",
        "build_image",
        "pipelines:container_pipeline.yml",
    ),
    Transition(
        "container_image",
        "deployment",
        "deploys",
        "Deployment",
        "deploy",
        "portainer_stack:redeploy_stack_git|cm_k8s:patch_resource",
        direction="in",
    ),
    Transition(
        "deployment",
        "validation",
        "validates",
        "Validation",
        "validate_deploy",
        "systems-manager:os_health(deploy-window)",
    ),
    Transition(
        "validation",
        "incident",
        "resolves",
        "Incident",
        "resolve_incident",
        "agent_utilities.observability.incident_router:close_ticket",
    ),
]

# The linear-predecessor map for BACKWARD gap-fill (the resolves closure is not a
# natural predecessor — an :Incident has no upstream stage).
_LINEAR: list[Transition] = FORWARD_CHAIN[:-1]
_PRED_BY_STAGE: dict[str, Transition] = {t.to_stage: t for t in _LINEAR}
_FWD_BY_STAGE: dict[str, Transition] = {t.from_stage: t for t in FORWARD_CHAIN}


@dataclass
class _GraphReader:
    """Minimal edge/label reader over a live :class:`GraphComputeEngine` (or any
    injected fake exposing the same three methods). Kept tiny so a fake in tests
    mirrors ``test_observability_incidents.py``'s ``_FakeEngine``."""

    engine: Any

    def out_edges(self, node_id: str) -> list[tuple[str, str]]:
        try:
            raw = self.engine.out_edges(node_id, data=True) or []
        except Exception as e:  # noqa: BLE001 — read is best-effort
            logger.debug("lifecycle: out_edges(%s) failed: %s", node_id, e)
            return []
        return [(_rel(props), tgt) for _src, tgt, props in raw]

    def in_edges(self, node_id: str) -> list[tuple[str, str]]:
        try:
            raw = self.engine.in_edges(node_id, data=True) or []
        except Exception as e:  # noqa: BLE001 — read is best-effort
            logger.debug("lifecycle: in_edges(%s) failed: %s", node_id, e)
            return []
        return [(_rel(props), src) for src, _tgt, props in raw]

    def nodes_by_label(self, label: str) -> list[tuple[str, dict[str, Any]]]:
        try:
            return self.engine.get_nodes_by_label(label, 0) or []
        except Exception as e:  # noqa: BLE001
            logger.debug("lifecycle: get_nodes_by_label(%s) failed: %s", label, e)
            return []


def _rel(props: Any) -> str:
    """The relationship type off an edge-property dict (engine keys edges by
    ``rel_type``; tolerate ``type`` too)."""
    if isinstance(props, dict):
        return str(props.get("rel_type") or props.get("type") or "")
    return ""


def _follow(reader: _GraphReader, node_id: str, transition: Transition) -> str | None:
    """The target of ``transition``'s edge from ``node_id`` if it already exists in
    the graph, else ``None`` (a gap)."""
    edges = (
        reader.out_edges(node_id)
        if transition.direction == "out"
        else reader.in_edges(node_id)
    )
    for rel_type, other in edges:
        if rel_type == transition.edge:
            return other
    return None


def _signature(entry_id: str, transition: Transition, kind: str) -> str:
    raw = f"{entry_id}|{kind}|{transition.from_stage}->{transition.to_stage}|{transition.name}"
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]


def _proposal(entry_id: str, transition: Transition, kind: str) -> dict[str, Any]:
    """One report-only gap-fill transition as a plain dict (materialized as a
    ``:LifecycleStep`` by :func:`_write_lifecycle_steps`)."""
    sig = _signature(entry_id, transition, kind)
    return {
        "id": f"lifecycle:step:{entry_id}:{sig}",
        "type": "LifecycleStep",
        "entry": entry_id,
        "kind": kind,  # "forward" (missing downstream) | "backfill" (missing upstream)
        "fromStage": transition.from_stage,
        "toStage": transition.to_stage,
        "edge": transition.edge,
        "transition": transition.name,
        "targetType": transition.to_type,
        "boundCapability": transition.capability,
        "signature": sig,
        "status": "proposed",  # report-only — nothing executed
    }


def _existing_step_signatures(reader: _GraphReader) -> set[str]:
    sigs: set[str] = set()
    for _id, props in reader.nodes_by_label("LifecycleStep"):
        if isinstance(props, dict) and props.get("signature"):
            sigs.add(str(props["signature"]))
    return sigs


def diff_lifecycle(
    entry: dict[str, Any], *, reader: _GraphReader
) -> list[dict[str, Any]]:
    """Diff ``entry``'s stage against the required lifecycle shape and return the
    ordered list of missing forward + backward transitions as proposal dicts.

    Forward: from the entry stage, walk the spine; the first missing edge (and
    every stage after it, since the chain is then broken) becomes a proposal.
    Backward: from the entry stage, walk linear predecessors; a missing upstream
    link becomes a ``backfill`` proposal (e.g. an MR with no linked :SpecProposal).
    """
    entry_id = str(entry.get("id") or "")
    entry_type = str(entry.get("type") or "")
    stage = STAGE_BY_TYPE.get(entry_type)
    if not entry_id or stage is None:
        return []

    proposals: list[dict[str, Any]] = []

    # --- forward: fill missing downstream stages -------------------------- #
    current: str | None = entry_id
    stage_idx = STAGE_ORDER.index(stage)
    for transition in FORWARD_CHAIN:
        if STAGE_ORDER.index(transition.from_stage) < stage_idx:
            continue
        if current is not None:
            nxt = _follow(reader, current, transition)
            if nxt is not None:
                current = nxt  # this hop already exists — advance
                continue
            current = None  # chain broken here
        proposals.append(_proposal(entry_id, transition, "forward"))

    # --- backward: backfill missing upstream links ------------------------ #
    up_node: str | None = entry_id
    up_stage: str | None = stage
    while up_stage is not None and up_stage in _PRED_BY_STAGE:
        transition = _PRED_BY_STAGE[up_stage]
        # walk the predecessor edge in reverse: for an "out" edge (pred -edge-> me)
        # the predecessor is one of MY in-edges of that type; for an "in" edge
        # (me -edge-> pred) it is one of my out-edges.
        if up_node is not None:
            edges = (
                reader.in_edges(up_node)
                if transition.direction == "out"
                else reader.out_edges(up_node)
            )
            pred = next((o for rel, o in edges if rel == transition.edge), None)
            if pred is not None:
                up_node = pred
                up_stage = transition.from_stage
                continue
            up_node = None
        proposals.insert(0, _proposal(entry_id, transition, "backfill"))
        up_stage = transition.from_stage

    return proposals


def _write_lifecycle_steps(
    entry_id: str, proposals: list[dict[str, Any]]
) -> dict[str, int] | None:
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    if not proposals:
        return None
    relationships = [
        {"source": entry_id, "target": p["id"], "type": "hasLifecycleStep"}
        for p in proposals
    ]
    return ingest_entities(proposals, relationships, source=_SOURCE, domain="sdlc")


def _annotate_escalation(
    proposals: list[dict[str, Any]],
    entry: dict[str, Any],
    *,
    engine: Any,
    context: dict[str, Any] | None = None,
) -> None:
    """Consult the escalation policy for each proposed transition (design §3.4).

    The escalation-decision policy is a **consultable gate** the orchestrator's
    proposals carry: for each transition, :func:`evaluate_escalation` decides
    ``auto`` vs ``escalate`` from existing graph evidence and (when it escalates)
    emits a queryable ``:EscalationRequest``; the proposal is stamped with the
    resulting ``autonomy`` + the escalation id so a downstream executor can inject
    a human-review gate step (§7.1) at exactly that transition. Report-only:
    stamping never actuates. Best-effort — a policy failure leaves the proposal at
    the autonomous default. Extra per-transition evidence (diff size, run status,
    …) may be threaded via ``context``.
    """
    from agent_utilities.observability import escalation_policy

    entry_id = str(entry.get("id") or "")
    base = dict(context or {})
    for prop in proposals:
        ctx = {
            **base,
            "entry": entry_id,
            "transition": prop.get("transition"),
        }
        try:
            request = escalation_policy.evaluate_escalation(ctx, engine=engine)
        except Exception as e:  # noqa: BLE001 — policy must not break the sweep
            logger.debug("lifecycle: escalation consult failed: %s", e)
            request = None
        if request is not None:
            prop["autonomy"] = "escalate"
            prop["escalation"] = request["id"]
            prop["escalationSignals"] = request["signals"]
        else:
            prop["autonomy"] = "auto"


def run_lifecycle(
    entry: dict[str, Any],
    *,
    engine: Any | None = None,
    write: bool = True,
    consult_escalation: bool = False,
    escalation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enter-anywhere gap-fill for one spine node ``entry`` (``{"id", "type"}``).

    Diffs the node's stage against the required lifecycle shape, dedupes against
    already-proposed ``:LifecycleStep`` nodes, and (when ``write``) MERGEs the new
    report-only proposals. NEVER executes a transition — each proposal only records
    the intended transition and its ``:boundCapability``. Best-effort: an
    unreachable engine yields an all-zero summary.

    When ``consult_escalation`` is set, each proposal is additionally run through
    the escalation-decision policy (design §3.4) and stamped ``autonomy``
    (``auto``/``escalate``) + an ``:EscalationRequest`` id — the consultable gate
    the fuller-autonomy-with-escalation model carries. ``escalation_context``
    threads extra per-transition evidence (diff size, run status, service, …).
    """
    eng = engine or health_ingest._engine()
    if eng is None:
        return {"entry": entry.get("id"), "stage": None, "proposed": 0, "steps": []}
    reader = _GraphReader(engine=eng)

    all_props = diff_lifecycle(entry, reader=reader)
    seen = _existing_step_signatures(reader)
    fresh = [p for p in all_props if p["signature"] not in seen]

    if consult_escalation and fresh:
        _annotate_escalation(fresh, entry, engine=eng, context=escalation_context)

    result = (
        _write_lifecycle_steps(str(entry.get("id") or ""), fresh) if write else None
    )
    return {
        "entry": entry.get("id"),
        "stage": STAGE_BY_TYPE.get(str(entry.get("type") or "")),
        "proposed": len(fresh),
        "deduped": len(all_props) - len(fresh),
        "written": (result or {}).get("nodes", 0),
        "steps": fresh,
    }


def sweep_open_spine(
    *, engine: Any | None = None, write: bool = True
) -> dict[str, Any]:
    """Scheduled sweep (design §3.3): run :func:`run_lifecycle` over every open
    spine node (``:Incident``/``:Ticket``/``:SpecProposal``/``:MergeRequest``/
    ``:PullRequest``) — catches anything that entered the graph without going
    through the orchestrator (e.g. a hand-filed ticket). Best-effort per node.
    """
    eng = engine or health_ingest._engine()
    if eng is None:
        return {"scanned": 0, "proposed": 0}
    reader = _GraphReader(engine=eng)
    entry_labels = ["Incident", "Ticket", "SpecProposal", "MergeRequest", "PullRequest"]
    scanned = 0
    proposed = 0
    for label in entry_labels:
        for node_id, props in reader.nodes_by_label(label):
            if not isinstance(props, dict):
                continue
            if str(props.get("status") or "open").lower() in (
                "resolved",
                "closed",
                "done",
            ):
                continue
            scanned += 1
            try:
                out = run_lifecycle(
                    {"id": node_id, "type": label}, engine=eng, write=write
                )
                proposed += out.get("proposed", 0)
            except Exception as e:  # noqa: BLE001 — one node must not break the sweep
                logger.debug("lifecycle sweep failed for %s: %s", node_id, e)
    return {"scanned": scanned, "proposed": proposed}


def main() -> None:
    """CLI (``python -m agent_utilities.observability.lifecycle_orchestrator``):
    one gap-fill sweep across all open spine nodes; prints a JSON summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    print(json.dumps(sweep_open_spine(), default=str, indent=2))


if __name__ == "__main__":
    main()
