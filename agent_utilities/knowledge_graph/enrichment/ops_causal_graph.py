#!/usr/bin/python
from __future__ import annotations

"""Enterprise Operations Causal Graph — join + analysis layer (Codex X-2).

CONCEPT:AU-KG.enrichment.ops-causal-graph

Joins entities ALREADY ingested by the connector fleet (langfuse-agent,
container-manager-mcp, gitlab-api/repository-manager, servicenow-api/
atlassian-agent, leanix-agent) into ONE operations causal chain::

    Langfuse Trace/Generation
        -> Agent / Tool / Model
        -> System (service/application)
        -> Container / ContainerStack (deployment)
        -> Commit / MergeRequest (the change)
        -> Incident / ChangeRequest (the ticket)
        -> Capability + Owner
        -> Policy / ComplianceControl + Evidence

This is a **join + analysis** layer, not a new ingestion path and not a new
traversal engine:

* The crosswalk resolving each connector resource onto that chain lives in
  :mod:`agent_utilities.knowledge_graph.ontology.ops_causal_crosswalk`.
* :func:`materialize_ops_causal_links` persists the join edges through the
  SAME generic extractor writer every enrichment source uses
  (:func:`.registry.write_batch`) — it creates zero new node entities, only
  edges between ids that already exist in the graph.
* :func:`build_causal_model` turns a resolved link set into a
  :class:`~agent_utilities.knowledge_graph.core.formal_reasoning_core.StructuralCausalModel`
  — the causal-reasoning engine ALREADY shipped (Pearl-style SCM with
  do-calculus, d-separation, ancestor/descendant traversal, a
  ``CausalVerifier`` and a ``SpuriousnessDetector``). Every analysis below is a
  thin composition over that engine's existing methods:

  - :func:`root_cause_rank`      -> ``StructuralCausalModel.get_causal_ancestors``
  - :func:`blast_radius_analysis` -> ``get_causal_descendants`` (or, given a live
    engine, ``IntelligenceGraphEngine.get_blast_radius`` — the already-shipped
    generic Cypher BFS, KG-2.134's "causal dependency mapping")
  - :func:`change_risk_score`    -> composes :func:`blast_radius_analysis` with a
    deterministic historical-evidence weighting (net-new aggregation; no
    traversal is reinvented)
  - :func:`control_evidence_chain` -> ``get_causal_descendants`` (what the control
    governs) + ``get_causal_ancestors`` (the operational history feeding it) +
    ``shortest_path`` + ``CausalVerifier.verify_chain`` + ``SpuriousnessDetector``
    (all reused)

No new graph-mining algorithm is introduced anywhere in this module.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.formal_reasoning_core import (
    CausalEdge,
    CausalFactor,
    CausalVerifier,
    SpuriousnessDetector,
    StructuralCausalModel,
)
from ..ontology.ops_causal_crosswalk import stage_of
from .models import EnrichmentEdge, ExtractionBatch
from .registry import write_batch

logger = logging.getLogger(__name__)

#: The self-registering extractor category for the ops-causal join layer
#: (mirrors every other extractor's category string, e.g. "archer", "camunda").
CATEGORY = "ops_causal"

__all__ = [
    "CATEGORY",
    "OpsCausalLink",
    "OpsCausalModel",
    "materialize_ops_causal_links",
    "build_causal_model",
    "load_ops_causal_neighborhood",
    "root_cause_rank",
    "blast_radius_analysis",
    "change_risk_score",
    "control_evidence_chain",
]


@dataclass(frozen=True)
class OpsCausalLink:
    """One directed edge in the operations causal chain, already resolved to
    real (already-ingested) node ids — the join layer's unit of work.

    ``strength`` is the causal-edge weight the SCM ranks with (default 1.0 —
    an unweighted, fully-confident edge); ``observed_at`` (unix seconds) is an
    optional recency signal for :func:`root_cause_rank`.
    """

    source: str
    target: str
    rel_type: str
    strength: float = 1.0
    observed_at: float | None = None
    mechanism: str = ""
    props: dict[str, Any] = field(default_factory=dict)

    def as_enrichment_edge(self) -> EnrichmentEdge:
        """This link as the uniform :class:`EnrichmentEdge` every source
        extractor emits — the shape :func:`materialize_ops_causal_links`
        persists via the shared writer."""
        props = dict(self.props)
        if self.observed_at is not None:
            props.setdefault("observed_at", self.observed_at)
        if self.strength != 1.0:
            props.setdefault("strength", self.strength)
        return EnrichmentEdge(
            source=self.source, target=self.target, rel_type=self.rel_type, props=props
        )


def materialize_ops_causal_links(
    backend: Any, links: Sequence[OpsCausalLink]
) -> tuple[int, int]:
    """Persist the join edges linking already-ingested entities, through the
    SAME generic writer every enrichment extractor uses
    (:func:`registry.write_batch`). Creates zero new nodes — ``batch.nodes``
    is always empty; this only adds edges between ids that already exist.

    Returns ``(nodes_written, edges_written)`` (nodes will always be 0).
    """
    batch = ExtractionBatch(
        category=CATEGORY, edges=[link.as_enrichment_edge() for link in links]
    )
    return write_batch(backend, batch, source=CATEGORY)


def load_ops_causal_neighborhood(
    engine: Any, seed_node_id: str, *, depth: int = 4
) -> list[OpsCausalLink]:
    """Best-effort production loader: pull the ops-causal neighborhood of an
    already-ingested seed node (a failing Trace, a candidate Commit, …)
    straight off the live KG backend.

    Reuses the EXACT Cypher traversal shape
    ``IntelligenceGraphEngine.get_blast_radius`` already runs (bounded-depth
    variable-length ``MATCH``), just undirected (root-cause analysis needs the
    reverse arrows too) and unwound to relationships instead of nodes — no
    second traversal engine, one more read over the same backend.

    Degrades to an empty list (never raises) when there is no backend or the
    query fails, matching every other engine surface's "degraded, not broken"
    contract.
    """
    backend = getattr(engine, "backend", None)
    if backend is None:
        return []
    query = f"""
    MATCH p = (s {{id: $node_id}})-[*1..{depth}]-(t)
    UNWIND relationships(p) AS r
    RETURN DISTINCT startNode(r).id AS source, endNode(r).id AS target,
           type(r) AS rel_type
    """
    try:
        rows = backend.execute(query, {"node_id": seed_node_id})
    except Exception as exc:  # noqa: BLE001 — degrade, don't raise
        logger.warning("ops-causal neighborhood query failed: %s", exc)
        return []
    return [
        OpsCausalLink(
            source=row["source"], target=row["target"], rel_type=row["rel_type"]
        )
        for row in rows
        if row.get("source") and row.get("target")
    ]


@dataclass
class OpsCausalModel:
    """The built causal graph plus the auxiliary metadata the analyses need
    but the bare :class:`StructuralCausalModel` doesn't carry (per-edge
    strength/recency lookups) — kept alongside rather than reached into via
    the SCM's private state.
    """

    scm: StructuralCausalModel
    stage_by_node: dict[str, str] = field(default_factory=dict)
    edge_strength: dict[tuple[str, str], float] = field(default_factory=dict)
    edge_observed_at: dict[tuple[str, str], float] = field(default_factory=dict)

    def has_path(self, source: str, target: str) -> bool:
        try:
            self.scm.shortest_path(source, target)
            return True
        except ValueError:
            return False

    def path_strength(self, path: Sequence[str]) -> float:
        """Product of each hop's edge strength along ``path`` (1.0 default per
        hop when unrecorded) — composes the strengths recorded when the model
        was built, never recomputes them."""
        strength = 1.0
        for a, b in zip(path, path[1:], strict=False):
            strength *= self.edge_strength.get((a, b), 1.0)
        return strength

    def path_recency_weight(
        self, path: Sequence[str], *, now: float, half_life_s: float = 86400.0
    ) -> float:
        """Recency weight in ``(0, 1]``: 1.0 if no hop on ``path`` carries an
        ``observed_at`` timestamp (no recency signal available — score purely
        on structure); otherwise an exponential decay from the MOST RECENT
        observed hop (``half_life_s`` default 24h — a cause observed an hour
        ago outranks an otherwise-identical one observed a week ago)."""
        timestamps = [
            self.edge_observed_at[(a, b)]
            for a, b in zip(path, path[1:], strict=False)
            if (a, b) in self.edge_observed_at
        ]
        if not timestamps:
            return 1.0
        age = max(0.0, now - max(timestamps))
        return 0.5 ** (age / half_life_s)


def build_causal_model(
    links: Iterable[OpsCausalLink], *, node_labels: dict[str, str] | None = None
) -> OpsCausalModel:
    """Build a :class:`StructuralCausalModel` (CONCEPT:AU-KG.research.research-pipeline-runner
    — the causal-reasoning engine already shipped) over a resolved ops-causal
    link set. Pure/offline: no engine or backend required, so this is the same
    path unit tests and a live production run both go through.

    ``node_labels`` (optional ``node_id -> hub graph label``, e.g. from
    :mod:`ontology.ops_causal_crosswalk`) lets each factor's ``domain`` be
    tagged with its causal stage via :func:`ops_causal_crosswalk.stage_of`;
    omitted labels just default the factor's domain to ``"general"``.
    """
    scm = StructuralCausalModel()
    labels = dict(node_labels or {})
    stage_by_node: dict[str, str] = {}
    edge_strength: dict[tuple[str, str], float] = {}
    edge_observed_at: dict[tuple[str, str], float] = {}

    def _ensure_factor(node_id: str) -> None:
        if scm.has_node(node_id):
            return
        domain = stage_of(labels[node_id]) if node_id in labels else None
        stage_by_node[node_id] = domain or "general"
        scm.add_factor(
            CausalFactor(id=node_id, name=node_id, domain=domain or "general")
        )

    for link in links:
        _ensure_factor(link.source)
        _ensure_factor(link.target)
        key = (link.source, link.target)
        edge_strength[key] = link.strength
        if link.observed_at is not None:
            edge_observed_at[key] = link.observed_at
        if scm.has_edge(link.source, link.target):
            continue
        scm.add_edge(
            CausalEdge(
                source_id=link.source,
                target_id=link.target,
                mechanism=link.mechanism or link.rel_type,
                strength=link.strength,
            )
        )
    return OpsCausalModel(
        scm=scm,
        stage_by_node=stage_by_node,
        edge_strength=edge_strength,
        edge_observed_at=edge_observed_at,
    )


def root_cause_rank(
    model: OpsCausalModel,
    failure_node: str,
    *,
    max_results: int = 10,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Rank probable root-cause changes/services for a ``failure_node``.

    Reuses :meth:`StructuralCausalModel.get_causal_ancestors` (upstream causal
    traversal — no second BFS written here) then scores each ancestor by::

        score = path_strength(ancestor -> failure) * recency_weight / max(1, hops)

    ``path_strength``/``recency_weight`` come straight from :class:`OpsCausalModel`
    (built from the SAME edges the SCM traversed, never recomputed).

    Ranking is PRIMARILY by ``is_root`` — whether :meth:`StructuralCausalModel.get_predecessors`
    is empty for that ancestor, i.e. it is a topological SOURCE of the causal
    DAG (reused directly, not reimplemented). That is the graph-theoretic
    definition of "root cause": a candidate with nothing further upstream
    outranks an intermediate ticket/observation that itself has a recorded
    cause, even if the intermediate is structurally closer to the failure —
    a 1-hop *symptom* (e.g. the incident that flagged the failure) must not
    outrank the 2-hop *change* that actually caused it. ``score`` is only the
    tie-breaker among nodes at the same root-ness.
    """
    scm = model.scm
    if not scm.has_node(failure_node):
        return []
    ranked: list[dict[str, Any]] = []
    for ancestor in scm.get_causal_ancestors(failure_node):
        if not model.has_path(ancestor, failure_node):
            continue
        path = scm.shortest_path(ancestor, failure_node)
        hops = max(1, len(path) - 1)
        strength = model.path_strength(path)
        recency = model.path_recency_weight(path, now=now) if now is not None else 1.0
        score = (strength * recency) / hops
        is_root = len(scm.get_predecessors(ancestor)) == 0
        ranked.append(
            {
                "node_id": ancestor,
                "is_root": is_root,
                "score": round(score, 6),
                "hops": hops,
                "path": path,
                "path_strength": round(strength, 6),
                "recency_weight": round(recency, 6),
                "stage": model.stage_by_node.get(ancestor, "general"),
            }
        )
    ranked.sort(key=lambda r: (not r["is_root"], -r["score"], r["hops"], r["node_id"]))
    return ranked[:max_results]


def blast_radius_analysis(
    model: OpsCausalModel | Any,
    change_node: str,
    *,
    depth: int = 6,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """Downstream impact of a proposed/applied ``change_node``.

    Given an :class:`OpsCausalModel`, reuses
    :meth:`StructuralCausalModel.get_causal_descendants` (the SAME family as
    :func:`root_cause_rank`, just the forward direction). Given a live
    ``IntelligenceGraphEngine`` instead, delegates straight to its
    already-shipped generic ``get_blast_radius`` Cypher BFS (KG-2.134's
    "causal dependency mapping") rather than re-implementing traversal for
    the production path.
    """
    if isinstance(model, OpsCausalModel):
        scm = model.scm
        if not scm.has_node(change_node):
            return []
        results: list[dict[str, Any]] = []
        for node in scm.get_causal_descendants(change_node):
            if not model.has_path(change_node, node):
                continue
            path = scm.shortest_path(change_node, node)
            results.append(
                {
                    "node_id": node,
                    "depth": len(path) - 1,
                    "path": path,
                    "stage": model.stage_by_node.get(node, "general"),
                }
            )
        results.sort(key=lambda r: (r["depth"], r["node_id"]))
        return results[:max_results] if max_results else results

    engine = model
    get_blast_radius = getattr(engine, "get_blast_radius", None)
    if not callable(get_blast_radius):
        raise TypeError(
            "blast_radius_analysis requires an OpsCausalModel or an engine "
            "exposing get_blast_radius()"
        )
    radius = get_blast_radius(change_node, depth)
    normalized = [
        {"node_id": r.get("id"), "depth": r.get("depth"), "type": r.get("type")}
        for r in radius
    ]
    normalized.sort(key=lambda r: (r["depth"], r["node_id"]))
    return normalized[:max_results] if max_results else normalized


def change_risk_score(
    model: OpsCausalModel,
    change_node: str,
    *,
    incident_history: Iterable[dict[str, Any]] | None = None,
    exposure_saturation: int = 10,
) -> dict[str, Any]:
    """Predict the risk of a proposed change from its blast radius + history.

    Composes :func:`blast_radius_analysis` (which itself reuses
    ``get_causal_descendants`` — no traversal is reinvented here) with a
    deterministic weighted-evidence aggregation over ``incident_history``
    (an iterable of ``{"node_id":..., "severity": 0..1 (default 0.5)}`` dicts —
    historical incidents observed on nodes downstream of this change):

        structural_exposure = min(1, len(blast_radius) / exposure_saturation)
        historical_severity = mean(severity) over incidents whose node_id
                               falls in the blast-radius set (0 if none)
        risk_score = 0.5 * structural_exposure + 0.5 * historical_severity

    Both terms and the score are in ``[0, 1]``.
    """
    downstream = blast_radius_analysis(model, change_node)
    downstream_ids = {d["node_id"] for d in downstream} | {change_node}
    structural = min(1.0, len(downstream) / exposure_saturation)
    contributing = [
        h for h in (incident_history or []) if h.get("node_id") in downstream_ids
    ]
    historical = (
        sum(float(h.get("severity", 0.5)) for h in contributing) / len(contributing)
        if contributing
        else 0.0
    )
    score = 0.5 * structural + 0.5 * historical
    return {
        "node_id": change_node,
        "risk_score": round(score, 6),
        "structural_exposure": round(structural, 6),
        "historical_severity": round(historical, 6),
        "blast_radius_size": len(downstream),
        "blast_radius": downstream,
        "contributing_incidents": contributing,
    }


def control_evidence_chain(model: OpsCausalModel, control_node: str) -> dict[str, Any]:
    """Gather + verify the evidence chain that backs a governance ``control_node``.

    A control sits BETWEEN what it governs (downstream — the capability it
    governs, the evidence it produces) and the operational history that fed
    into what it governs (upstream of the GOVERNED node, not of the control
    itself — a policy has no causal ancestors of its own in this chain, see
    :mod:`ontology.ops_causal_crosswalk`). So this:

    1. Reuses :meth:`StructuralCausalModel.get_causal_descendants` (control ->
       governed capability/system, control -> evidence — CONCEPT reused
       wholesale, not reimplemented).
    2. For each governed descendant, reuses
       :meth:`StructuralCausalModel.get_causal_ancestors` again to pull in the
       ticket/change/deployment/service history that feeds it — the
       operational evidence the control needs to account for.
    3. Verifies the combined (cause, effect) step set via
       :class:`~agent_utilities.knowledge_graph.core.formal_reasoning_core.CausalVerifier`
       and flags spurious hops via
       :class:`~agent_utilities.knowledge_graph.core.formal_reasoning_core.SpuriousnessDetector`
       — both reused wholesale, no new verification logic.
    """
    scm = model.scm
    empty = {
        "control": control_node,
        "evidence_chain": [],
        "governs": [],
        "upstream_operational_history": [],
        "is_consistent": True,
        "consistency_score": 1.0,
        "violations": [],
        "spurious_edges": [],
        "spuriousness_detail": [],
    }
    if not scm.has_node(control_node):
        return empty

    governed = scm.get_causal_descendants(control_node)
    if not governed:
        return empty

    steps: list[dict[str, str]] = []
    chain_nodes: list[str] = [control_node]
    for target in sorted(governed):
        if not model.has_path(control_node, target):
            continue
        path = scm.shortest_path(control_node, target)
        for a, b in zip(path, path[1:], strict=False):
            steps.append({"cause": a, "effect": b})
        chain_nodes.append(target)

    upstream: set[str] = set()
    for target in governed:
        upstream |= scm.get_causal_ancestors(target)
    upstream -= governed
    upstream.discard(control_node)

    upstream_chain_nodes: list[str] = []
    for target in sorted(governed):
        candidates = [a for a in upstream if model.has_path(a, target)]
        if not candidates:
            continue
        best = max((scm.shortest_path(a, target) for a in candidates), key=len)
        for a, b in zip(best, best[1:], strict=False):
            steps.append({"cause": a, "effect": b})
        upstream_chain_nodes = best[:-1] + upstream_chain_nodes

    seen: set[tuple[str, str]] = set()
    deduped_steps: list[dict[str, str]] = []
    for step in steps:
        key = (step["cause"], step["effect"])
        if key in seen:
            continue
        seen.add(key)
        deduped_steps.append(step)

    verification = CausalVerifier(scm).verify_chain(deduped_steps)
    candidate_edges = [(s["cause"], s["effect"]) for s in deduped_steps]
    spuriousness = (
        SpuriousnessDetector(scm).detect_spurious_edges(candidate_edges)
        if candidate_edges
        else []
    )
    evidence_chain = list(dict.fromkeys(upstream_chain_nodes + chain_nodes))
    return {
        "control": control_node,
        "evidence_chain": evidence_chain,
        "governs": sorted(governed),
        "upstream_operational_history": sorted(upstream),
        "is_consistent": verification.is_consistent,
        "consistency_score": verification.consistency_score,
        "violations": verification.violations,
        "spurious_edges": verification.spurious_edges,
        "spuriousness_detail": spuriousness,
    }
