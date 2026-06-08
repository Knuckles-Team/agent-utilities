#!/usr/bin/python
from __future__ import annotations

"""Plan synthesis from a feature's KG neighborhood (CONCEPT:KG-2.7 / KG-2.10).

The last graph-native stage: turn the top-ranked **open** gaps into grounded SDD
plan proposals. Instead of re-reading raw papers per plan (the first attempt's
approach), each plan is synthesized from the feature's *hydrated KG neighborhood* —
the feature, its sources, its synergy partners, and its target pillar — so the plan
is grounded and deduped **by construction**.

* :func:`hydrate_feature` — pull a feature's neighborhood (sources, synergies, pillar).
* :func:`synthesize_plan_for_feature` — neighborhood → SDD plan (injectable
  ``synth_fn``; default tries the ORCH-1.27 ``planner`` role, falls back to a
  deterministic grounded template so it never hard-fails), persisted as a
  proposal (`SDDPlan` node + ``feature -[ADDRESSED_BY{proposed}]-> plan``) and the
  feature flipped to ``proposed`` so it is not re-proposed next cycle (idempotent).
* :func:`synthesize_plans` — rank → take top-N open gaps → synthesize each.

Propose-only: plans are written as proposals; promotion (proposed→active) reuses
the existing AHE-3.14 governance gate at apply time.

Concept: plan-synthesis
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType
from .ledger import _get_node, set_status
from .synergy import _pillar_of, rank_features

SynthFn = Callable[[dict[str, Any]], dict[str, str]]
# Features already in-flight are not re-proposed.
_INFLIGHT_STATUS = {"proposed", "in_progress"}


@dataclass
class PlanProposal:
    feature_id: str
    plan_id: str
    title: str
    body: str
    sources: list[str] = field(default_factory=list)
    synergies: list[str] = field(default_factory=list)
    status: str = "proposed"


def _neighbors_by_rel(engine: Any, fid: str, rel: str) -> list[str]:
    graph = getattr(engine, "graph", None)
    if graph is None:
        return []
    out: list[str] = []
    try:
        for _s, dst, props in graph.out_edges(fid, data=True):
            if isinstance(props, dict) and str(props.get("_rel", "")) == rel:
                out.append(dst)
        for src, _t, props in graph.in_edges(fid, data=True):
            if isinstance(props, dict) and str(props.get("_rel", "")) == rel:
                out.append(src)
    except (TypeError, AttributeError):  # pragma: no cover
        return out
    return list(dict.fromkeys(out))


def hydrate_feature(engine: Any, feature_id: str) -> dict[str, Any]:
    """Pull a feature's grounded neighborhood for plan synthesis."""
    node = _get_node(engine, feature_id) or {}
    sources = list(node.get("research_sources", []) or [])
    sources += [
        s
        for s in _neighbors_by_rel(engine, feature_id, "DERIVED_FROM_RESEARCH")
        if s not in sources
    ]
    return {
        "feature_id": feature_id,
        "name": str(node.get("name", feature_id)),
        "concept_ids": list(node.get("concept_ids", []) or []),
        "pillar": _pillar_of(node),
        "sources": sources,
        "synergies": _neighbors_by_rel(engine, feature_id, "HAS_SYNERGY_WITH"),
        "status": str(node.get("status", "open")),
    }


def _default_synth(neighborhood: dict[str, Any]) -> dict[str, str]:
    """Deterministic grounded SDD plan (no LLM) from the hydrated neighborhood."""
    n = neighborhood
    syn = f" Synergizes with: {', '.join(n['synergies'])}." if n["synergies"] else ""
    src = ", ".join(n["sources"]) or "(no linked sources)"
    title = f"Assimilate: {n['name']}"
    body = (
        f"# SDD Plan: {n['name']}\n\n"
        f"> Pillar: {n['pillar'] or 'n/a'} · Concepts: {', '.join(n['concept_ids']) or 'n/a'}\n\n"
        f"## Overview\nAssimilate the **{n['name']}** capability into the "
        f"agent-utilities ecosystem.{syn}\n\n"
        f"## Evidence / Sources\n{src}\n\n"
        f"## Implementation\nWire the mechanism on a live path with a concept-tagged "
        f"test; on completion, close out the source(s) via the assimilation ledger "
        f"(`ASSIMILATED_INTO`).\n"
    )
    return {"title": title, "body": body}


def _llm_synth(neighborhood: dict[str, Any]) -> dict[str, str] | None:
    """Best-effort plan synthesis via the ORCH-1.27 ``planner`` role.

    Bounded by a hard wall-clock timeout (env ``ASSIMILATION_SYNTH_TIMEOUT_S``,
    default 30s): an unreachable/slow planner endpoint must NOT hang the caller —
    a blocking ``run_sync`` cannot be caught by ``try/except``, so we run it on a
    worker thread and fall back to ``_default_synth`` on timeout. This keeps the
    golden-loop assimilation tick non-blocking when no model is reachable.
    """
    import concurrent.futures
    import os

    try:
        from agent_utilities.agent.factory import Agent
        from agent_utilities.core.model_factory import create_model

        model = create_model(role="planner")
        prompt = (
            "Synthesize a concise SDD implementation plan (markdown: Overview, "
            "Functional Requirements, Implementation, Tests) to assimilate this "
            f"capability into agent-utilities. Neighborhood: {neighborhood}. "
            "Return markdown only."
        )
        agent = Agent(model=model, system_prompt="You are an SDD plan synthesizer.")
        try:
            timeout_s = float(os.environ.get("ASSIMILATION_SYNTH_TIMEOUT_S", "30"))
        except ValueError:
            timeout_s = 30.0
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            result: Any = ex.submit(agent.run_sync, prompt).result(timeout=timeout_s)
        finally:
            ex.shutdown(wait=False)  # don't block on a wedged inference thread
        body = str(getattr(result, "output", None) or getattr(result, "data", "") or "")
        if body.strip():
            return {"title": f"Assimilate: {neighborhood['name']}", "body": body}
    except Exception:  # pragma: no cover - planner is best-effort (incl. timeout)
        return None
    return None


def synthesize_plan_for_feature(
    engine: Any,
    feature_id: str,
    *,
    synth_fn: SynthFn | None = None,
    write: bool = True,
) -> PlanProposal:
    """Synthesize + persist a grounded SDD plan proposal for one feature."""
    nb = hydrate_feature(engine, feature_id)
    plan = None
    if synth_fn is not None:
        plan = synth_fn(nb)
    if plan is None:
        plan = _llm_synth(nb) or _default_synth(nb)
    plan_id = f"plan:{feature_id}"
    proposal = PlanProposal(
        feature_id=feature_id,
        plan_id=plan_id,
        title=plan.get("title", f"Assimilate: {nb['name']}"),
        body=plan.get("body", ""),
        sources=nb["sources"],
        synergies=nb["synergies"],
    )
    if write:
        engine.add_node(
            plan_id,
            "sdd_plan",
            properties={
                "name": proposal.title,
                "body": proposal.body,
                "status": "proposed",
                "feature_id": feature_id,
                "sources": nb["sources"],
            },
        )
        engine.link_nodes(
            feature_id,
            plan_id,
            RegistryEdgeType.ADDRESSED_BY,
            properties={"_rel": "ADDRESSED_BY", "status": "proposed"},
        )
        set_status(engine, feature_id, "proposed")  # in-flight → not re-proposed
    return proposal


def synthesize_plans(
    engine: Any,
    *,
    top_n: int = 5,
    synth_fn: SynthFn | None = None,
    write: bool = True,
) -> list[PlanProposal]:
    """Synthesize plans for the top-N **open** (not in-flight) ranked gaps."""
    ranked = rank_features(engine)
    proposals: list[PlanProposal] = []
    for r in ranked:
        if len(proposals) >= top_n:
            break
        node = _get_node(engine, r.feature_id) or {}
        if str(node.get("status", "open")) in _INFLIGHT_STATUS:
            continue  # already proposed / in progress — idempotent
        proposals.append(
            synthesize_plan_for_feature(
                engine, r.feature_id, synth_fn=synth_fn, write=write
            )
        )
    return proposals


__all__ = [
    "PlanProposal",
    "hydrate_feature",
    "synthesize_plan_for_feature",
    "synthesize_plans",
]
