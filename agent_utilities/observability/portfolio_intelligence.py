#!/usr/bin/python
from __future__ import annotations

"""Portfolio comparative-intelligence engine (CONCEPT:AU-KG.enrichment.portfolio-intelligence).

``reports/enterprise-comparative-intelligence-design.md``: when a ServiceNow TRM
request or an ERPNext new-software request names a candidate product, this
module assesses it against the existing portfolio and recommends
**adopt / reject / consolidate / migrate** — weighing functionality (incl.
unique/niche capability), cost, licensing, ecosystem-integration fit, and
alignment to financial/strategic objectives, with compliance as a hard gate.

**Composes existing primitives — no new kernel.** Mirrors
:mod:`agent_utilities.knowledge_graph.enrichment.ops_causal_graph`'s
composition style:

* ``GraphComputeEngine.personalized_pagerank`` (weight-seeded ranking) for the
  ecosystem-integration-fit criterion — seeded on the deployed
  ``:AssetInstance`` estate, read at the candidate's node.
* A ``change_risk_score``-style deterministic weighted sum
  (``Sigma(w_i * score_i)``) over the ``:ComparisonCriterion`` vector.
* :class:`~agent_utilities.knowledge_graph.core.shacl_validator.SHACLValidator`
  (``shapes/portfolio_intelligence.shapes.ttl``) as an ADDITIVE audit of the
  verdict shape, mirroring :mod:`.research.promotion_governance`'s multi-check
  pattern — never the decision authority itself.
* The inferred ``:RedundantCapability``/``:swappableWith`` capability topology
  (``ontology_capability.ttl``) for peer-group discovery and the
  consolidation-benefit criterion — a pure graph read, no new mining algorithm.

**Two-tier evaluation (design Sec 2c).** :func:`assess_candidate` runs the GATE
tier first — every REQUIRED compliance unit for the candidate's declared
sector/data-class, resolved generically off the ``legal-peripherals-mcp``
regulatory ontology (``compliance.ttl`` + its per-regulation modules — this
module never models regulations itself, only walks the shape): the candidate's
``sector``/``dataClass`` resolve the applicable ``:Regulation``(s) via
``:appliesToSector``/``:appliesToDataClass``; each applicable Regulation's
``:ComplianceRequirement``(s) (``:derivedFromRegulation``), optionally
evaluated by a named ``:ComplianceGate`` (``:evaluatesRequirement``), are the
REQUIRED units; a candidate satisfies one via a declared
``certifications``/``attestations`` entry or an ``:attestsTo``-style edge
naming the gate/requirement/regulation. Plus EOL and gov-ATO gates. A failed
REQUIRED gate is an immediate ``reject`` — no score can buy it back. Only
gate-passing candidates reach the WEIGHTED SCORE tier. With NO ``:Regulation``
nodes in the graph at all (substrate not ingested/federated), the gate
degrades to a pass but logs a warning — distinct from "evaluated and passed."

Report-only + engine-guarded throughout: with no reachable engine every entry
point degrades to a safe no-op/empty result rather than raising, matching
:mod:`.lifecycle_orchestrator` and :mod:`.incident_router`.

Two workflows:

* :func:`assess_candidate` — pure compute (gate + score a candidate against its
  peer group); no graph writes.
* :func:`run_trm_assessment` — the full request -> assessment -> recommendation
  -> writeback flow: mints/updates a ``:TRMRequest``, runs
  :func:`assess_candidate`, persists ``:Assessment``/``:ComparisonCriterion``/
  ``:Recommendation`` nodes, and backfeeds the verdict to the source ticket via
  the existing fail-closed, dry-run-first ``kg_writeback`` ServiceNow sink
  (``work_notes`` op-kind).
* :func:`rationalize_portfolio` — the periodic sweep (design Sec 5b): finds
  redundant/overlapping products via shared ``:providesCapability`` targets and
  recommends consolidating, with no incoming request.
"""

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.observability import health_ingest

logger = logging.getLogger("agent_utilities.observability.portfolio_intelligence")

_SOURCE = "agent-utilities-portfolio"

# ── weighted criteria (design Sec 2a) ───────────────────────────────────────

#: Default weight vector over the seven MCDA criteria. Renormalized to sum to
#: 1.0 by :func:`resolve_weights` after any goal-type policy multiplier is
#: applied — these are relative starting weights, not final values.
DEFAULT_WEIGHTS: dict[str, float] = {
    "functionality": 0.15,
    "unique-capability": 0.20,
    "cost": 0.15,
    "licensing": 0.05,
    "integration-fit": 0.15,
    "objective-alignment": 0.10,
    "consolidation-benefit": 0.10,
    "compliance-ato": 0.10,
}

#: ``:goalType`` (``ontology_company.ttl``) -> per-criterion weight multiplier
#: (design Sec 5a). A ``profit``/``efficiency`` quarter raises cost +
#: consolidation-benefit; ``growth``/``innovation`` raises functionality +
#: unique-capability; ``compliance`` raises compliance-ato. Weight-policy
#: OWNERSHIP (who tunes this, how often) is a human decision the design flags
#: as still-open config — this is the mechanism, not the policy authority.
GOAL_TYPE_WEIGHT_POLICY: dict[str, dict[str, float]] = {
    "profit": {"cost": 1.5, "consolidation-benefit": 1.4},
    "efficiency": {"cost": 1.4, "consolidation-benefit": 1.5},
    "growth": {"functionality": 1.3, "unique-capability": 1.4},
    "innovation": {"unique-capability": 1.5, "functionality": 1.2},
    "compliance": {"compliance-ato": 2.0},
}

_LICENSE_MODEL_SCORE: dict[str, float] = {
    "opensource": 1.0,
    "perpetual": 0.8,
    "subscription": 0.5,
    "usage": 0.4,
}

_SHAPES_PATH = (
    Path(__file__).parent.parent
    / "knowledge_graph"
    / "shapes"
    / "portfolio_intelligence.shapes.ttl"
)

__all__ = [
    "GateCheck",
    "CriterionScore",
    "DEFAULT_WEIGHTS",
    "GOAL_TYPE_WEIGHT_POLICY",
    "resolve_weights",
    "resolve_peer_group",
    "score_criteria",
    "assess_candidate",
    "validate_verdict_shape",
    "run_trm_assessment",
    "rationalize_portfolio",
]


@dataclass(frozen=True)
class GateCheck:
    """One pass/fail rule in the gate tier (design Sec 2c) — mirrors
    :class:`~agent_utilities.knowledge_graph.research.promotion_governance.GovernanceCheck`."""

    name: str
    passed: bool
    required: bool = True
    reason: str = ""


@dataclass(frozen=True)
class CriterionScore:
    """One weighted MCDA criterion — a :ComparisonCriterion in the graph."""

    kind: str
    score: float
    weight: float
    weighted: float


# ── graph-read helpers (engine-guarded, degrade to empty rather than raise) ─


def _rel(props: Any) -> str:
    if isinstance(props, dict):
        return str(props.get("rel_type") or props.get("type") or "")
    return ""


def _node_props(engine: Any, node_id: str) -> dict[str, Any]:
    """Best-effort per-id property fetch — mirrors
    :func:`agent_utilities.knowledge_graph.assimilation.gap_analysis._node_data_by_id`."""
    for meth in ("get_node_properties", "_get_node_properties"):
        fn = getattr(engine, meth, None)
        if callable(fn):
            try:
                data = fn(node_id)
            except Exception:  # noqa: BLE001 — try the next access path
                continue
            if isinstance(data, dict):
                return data
    nodes = getattr(engine, "nodes", None)
    getter = getattr(nodes, "get", None) if nodes is not None else None
    if callable(getter):
        try:
            data = getter(node_id, {})
        except Exception:  # noqa: BLE001
            data = {}
        if isinstance(data, dict):
            return data
    return {}


def _out(engine: Any, node_id: str) -> list[tuple[str, str, dict[str, Any]]]:
    try:
        return engine.out_edges(node_id, data=True) or []
    except Exception as e:  # noqa: BLE001 — read is best-effort
        logger.debug("portfolio: out_edges(%s) failed: %s", node_id, e)
        return []


def _in(engine: Any, node_id: str) -> list[tuple[str, str, dict[str, Any]]]:
    try:
        return engine.in_edges(node_id, data=True) or []
    except Exception as e:  # noqa: BLE001
        logger.debug("portfolio: in_edges(%s) failed: %s", node_id, e)
        return []


def _by_label(
    engine: Any, label: str, limit: int = 0
) -> list[tuple[str, dict[str, Any]]]:
    fn = getattr(engine, "get_nodes_by_label", None)
    if not callable(fn):
        return []
    try:
        return fn(label, limit) or []
    except Exception as e:  # noqa: BLE001
        logger.debug("portfolio: get_nodes_by_label(%s) failed: %s", label, e)
        return []


def _capability_set(product_id: str, engine: Any) -> set[str]:
    return {
        tgt
        for _s, tgt, p in _out(engine, product_id)
        if _rel(p) == "providesCapability"
    }


def _annual_cost(product_id: str, engine: Any) -> float | None:
    for _s, tgt, p in _out(engine, product_id):
        if _rel(p) != "incursCost":
            continue
        val = _node_props(engine, tgt).get("annualCost")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _swappable_peers(product_id: str, engine: Any) -> set[str]:
    swappable = {
        tgt for _s, tgt, p in _out(engine, product_id) if _rel(p) == "swappableWith"
    }
    swappable |= {
        src for src, _t, p in _in(engine, product_id) if _rel(p) == "swappableWith"
    }
    return swappable


# ── peer-group resolution ───────────────────────────────────────────────────


def resolve_peer_group(
    candidate_id: str, engine: Any, *, explicit: list[str] | None = None
) -> list[str]:
    """The candidate's best-in-category cohort (design Sec 1b ``:PeerGroup``).

    ``explicit`` (caller-supplied peer ids) wins when given. Otherwise auto-
    resolves via shared ``:providesCapability`` targets (any other product that
    provides one of the candidate's capabilities — the inferred
    ``:RedundantCapability`` seed) plus any directly-asserted ``:swappableWith``
    peers. Pure graph read; never writes.
    """
    if explicit:
        return sorted({p for p in explicit if p and p != candidate_id})
    peers: set[str] = set(_swappable_peers(candidate_id, engine))
    for cap in _capability_set(candidate_id, engine):
        for src, _tgt, p in _in(engine, cap):
            if _rel(p) == "providesCapability" and src != candidate_id:
                peers.add(src)
    peers.discard(candidate_id)
    return sorted(peers)


# ── weight resolution (design Sec 5a) ───────────────────────────────────────


def _active_goal_types(engine: Any, *, limit: int = 200) -> list[str]:
    types: set[str] = set()
    for label in ("Objective", "StrategicGoal"):
        for _nid, props in _by_label(engine, label, limit):
            if isinstance(props, dict) and props.get("goalType"):
                types.add(str(props["goalType"]).strip().lower())
    return sorted(types)


def resolve_weights(
    engine: Any | None = None,
    *,
    goal_types: list[str] | None = None,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """The weight vector for the current run: :data:`DEFAULT_WEIGHTS`, adjusted
    by any active ``:goalType`` policy multipliers (:data:`GOAL_TYPE_WEIGHT_POLICY`)
    and caller ``overrides``, renormalized to sum to 1.0.

    ``goal_types`` bypasses the graph read (test-friendly); omitted with an
    engine present, the active goal types are read off ``:Objective``/
    ``:StrategicGoal`` nodes' ``goalType`` property (best-effort — an
    unreachable/empty engine leaves the default weights unchanged).
    """
    weights = dict(DEFAULT_WEIGHTS)
    active = (
        list(goal_types)
        if goal_types is not None
        else (_active_goal_types(engine) if engine is not None else [])
    )
    for goal_type in active:
        policy = GOAL_TYPE_WEIGHT_POLICY.get(str(goal_type).strip().lower())
        if not policy:
            continue
        for kind, multiplier in policy.items():
            if kind in weights:
                weights[kind] *= multiplier
    if overrides:
        weights.update({k: v for k, v in overrides.items() if k in weights})
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


# ── gate tier (design Sec 2c — runs FIRST, a failure is a hard reject) ──────


def _norm(value: Any) -> str:
    """Lowercase, punctuation-stripped comparison key — tolerant of an ontology
    individual's local name (``MedicalSector``) vs. a caller's plain label
    (``medical``)."""
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _label_for(node_id: str, props: dict[str, Any] | None) -> str:
    """The human-readable name for a graph node: its ``rdfs:label``/``name``
    property, falling back to the raw id."""
    props = props or {}
    return str(props.get("label") or props.get("name") or node_id)


def _matches(value: str, node_id: str, props: dict[str, Any]) -> bool:
    """Whether caller-supplied ``value`` (e.g. request ``sector``/``dataClass``)
    identifies the graph node ``node_id`` — compares (normalized) against the
    node id and its label/name, either direction, so ``"medical"`` matches a
    ``MedicalSector`` node and ``"phi"`` matches a ``PHI`` node."""
    v = _norm(value)
    if not v:
        return False
    for candidate in (node_id, props.get("label"), props.get("name")):
        c = _norm(candidate)
        if c and (v == c or v in c or c in v):
            return True
    return False


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(v) for v in value if v]
    return [str(value)] if value else []


def _applicable_regulations(
    request: dict[str, Any], regulations: list[tuple[str, dict[str, Any]]], engine: Any
) -> list[tuple[str, dict[str, Any]]]:
    """``:Regulation`` nodes applicable to ``request``'s declared sector/data
    class(es), resolved via the real ``:appliesToSector``/``:appliesToDataClass``
    edges (``compliance.ttl``). A Regulation applies only when EVERY scope
    dimension it actually declares matches the candidate's declared value(s) —
    e.g. HIPAA (declares both ``:MedicalSector`` and ``:PHI``) applies only to a
    medical-sector candidate handling PHI, not to every medical-sector candidate."""
    sector = str(request.get("sector") or "").strip()
    data_classes = _as_list(
        request.get("dataClass")
        or request.get("data_class")
        or request.get("dataClasses")
    )
    applicable: list[tuple[str, dict[str, Any]]] = []
    for reg_id, reg_props in regulations:
        declared_sectors = [
            tgt for _s, tgt, p in _out(engine, reg_id) if _rel(p) == "appliesToSector"
        ]
        declared_data = [
            tgt
            for _s, tgt, p in _out(engine, reg_id)
            if _rel(p) == "appliesToDataClass"
        ]
        if not declared_sectors and not declared_data:
            continue  # no declared scope — not resolvable, skip rather than over-apply
        sector_ok = not declared_sectors or any(
            _matches(sector, tgt, _node_props(engine, tgt)) for tgt in declared_sectors
        )
        data_ok = not declared_data or any(
            _matches(dc, tgt, _node_props(engine, tgt))
            for tgt in declared_data
            for dc in data_classes
        )
        if sector_ok and data_ok:
            applicable.append((reg_id, reg_props))
    return applicable


def _gate_for_requirement(
    requirement_id: str, gate_ids: set[str], engine: Any
) -> tuple[str, dict[str, Any]]:
    """The ``:ComplianceGate`` (if any) that ``:evaluatesRequirement`` this
    requirement — a Regulation's ComplianceRequirement may be evaluated by an
    ``:Assessment`` instead (e.g. BSA/AML's SAR requirement), which is not a hard
    gate, so this only matches nodes that are actually labeled ``ComplianceGate``."""
    for src, _tgt, p in _in(engine, requirement_id):
        if _rel(p) == "evaluatesRequirement" and src in gate_ids:
            return src, _node_props(engine, src)
    return "", {}


def _required_compliance_units(
    request: dict[str, Any], engine: Any
) -> tuple[list[dict[str, Any]], bool]:
    """Every REQUIRED compliance unit applicable to ``request``, walking the real
    ``legal-peripherals-mcp`` ontology shape (design Sec 1b/9b, ``compliance.ttl``):
    ``:Regulation`` --``:appliesToSector``/``:appliesToDataClass``--> resolves the
    applicable Regulation(s) (:func:`_applicable_regulations`); each Regulation's
    ``:ComplianceRequirement``(s) (``:derivedFromRegulation``) are the REQUIRED
    units, each optionally named by a ``:ComplianceGate`` that
    ``:evaluatesRequirement`` it (:func:`_gate_for_requirement`).

    Returns ``(units, substrate_present)``. ``substrate_present`` is ``False``
    ONLY when the graph has no ``:Regulation`` nodes at all — the compliance
    substrate genuinely isn't ingested/federated — distinguishing "evaluated,
    nothing applies" from "couldn't evaluate."
    """
    regulations = _by_label(engine, "Regulation")
    if not regulations:
        return [], False
    gate_ids = {gid for gid, _p in _by_label(engine, "ComplianceGate")}
    units: list[dict[str, Any]] = []
    for reg_id, reg_props in _applicable_regulations(request, regulations, engine):
        reg_name = _label_for(reg_id, reg_props)
        for req_id, _tgt, p in _in(engine, reg_id):
            if _rel(p) != "derivedFromRegulation":
                continue
            req_props = _node_props(engine, req_id)
            gate_id, gate_props = _gate_for_requirement(req_id, gate_ids, engine)
            units.append(
                {
                    "regulationId": reg_id,
                    "regulationName": reg_name,
                    "requirementId": req_id,
                    "requirementName": _label_for(req_id, req_props),
                    "gateId": gate_id,
                    "gateName": _label_for(gate_id, gate_props) if gate_id else "",
                    "required": bool(req_props.get("required", True)),
                }
            )
    return units, True


def _unit_satisfied(
    candidate_id: str, unit: dict[str, Any], request: dict[str, Any], engine: Any
) -> bool:
    """Whether the candidate has compliance evidence for ``unit`` — a declared
    ``certifications``/``attestations`` entry (on the request/candidate) naming
    the gate/requirement/regulation, or a graph edge from the candidate
    (``:attestsTo`` and the legacy ``governedBy``/``satisfiesCompliance``/
    ``conformsToStandard`` synonyms) targeting one of them. No evidence at all
    ⇒ an applicable REQUIRED unit is UNMET."""
    declared = {
        _norm(e)
        for e in (
            *_as_list(request.get("certifications")),
            *_as_list(request.get("attestations")),
        )
        if e
    }
    if declared:
        targets = {
            _norm(unit["regulationName"]),
            _norm(unit["regulationId"]),
            _norm(unit["requirementName"]),
            _norm(unit["requirementId"]),
            _norm(unit["gateName"]),
            _norm(unit["gateId"]),
        }
        targets.discard("")
        if declared & targets:
            return True
    evidence_ids = {unit["gateId"], unit["requirementId"], unit["regulationId"]}
    evidence_ids.discard("")
    for _s, tgt, p in _out(engine, candidate_id):
        if (
            _rel(p)
            in ("attestsTo", "governedBy", "satisfiesCompliance", "conformsToStandard")
            and tgt in evidence_ids
        ):
            return True
    return False


def _check_compliance_gates(
    candidate_id: str, request: dict[str, Any], engine: Any
) -> GateCheck:
    units, substrate_present = _required_compliance_units(request, engine)
    if not substrate_present:
        logger.warning(
            "portfolio: compliance substrate unavailable (no :Regulation nodes in "
            "the graph) — compliance_gate for candidate %s degraded to pass; "
            "confirm the legal-peripherals-mcp ontology is ingested/federated",
            candidate_id,
        )
        return GateCheck(
            "compliance_gate",
            True,
            True,
            "compliance substrate unavailable (no :Regulation nodes) — gate not evaluated",
        )
    required = [u for u in units if u["required"]]
    if not required:
        return GateCheck(
            "compliance_gate", True, True, "no REQUIRED compliance gate applies"
        )
    unmet = [
        u for u in required if not _unit_satisfied(candidate_id, u, request, engine)
    ]
    if unmet:
        names = ", ".join(
            f"{u['gateName'] or u['requirementName']} ({u['regulationName']})"
            for u in unmet
        )
        return GateCheck(
            "compliance_gate",
            False,
            True,
            f"required compliance gate(s) not satisfied: {names}",
        )
    return GateCheck(
        "compliance_gate", True, True, f"{len(required)} required gate(s) satisfied"
    )


def _check_eol(candidate_id: str, engine: Any, *, grace_days: int = 180) -> GateCheck:
    eol = _node_props(engine, candidate_id).get("endOfLifeDate")
    if not eol:
        return GateCheck("eol", True, True, "no endOfLifeDate recorded")
    try:
        eol_date = datetime.fromisoformat(str(eol).replace("Z", "+00:00"))
    except ValueError:
        return GateCheck("eol", True, True, "endOfLifeDate unparsable — not applicable")
    if eol_date.tzinfo is None:
        eol_date = eol_date.replace(tzinfo=UTC)
    if eol_date <= datetime.now(UTC) + timedelta(days=grace_days):
        return GateCheck(
            "eol",
            False,
            True,
            f"end-of-life {eol} within the {grace_days}-day grace window",
        )
    return GateCheck("eol", True, True, f"end-of-life {eol} beyond the grace window")


def _check_ato(candidate_id: str, request: dict[str, Any], engine: Any) -> GateCheck:
    if str(request.get("profile") or "commercial").strip().lower() != "gov":
        return GateCheck(
            "ato", True, True, "commercial profile — ATO gate not applicable"
        )
    for _s, tgt, p in _out(engine, candidate_id):
        if _rel(p) != "authorizedBy":
            continue
        if (
            str(_node_props(engine, tgt).get("atoStatus") or "").strip().lower()
            == "authorized"
        ):
            return GateCheck("ato", True, True, "authorized ATO on file")
    return GateCheck(
        "ato", False, True, "gov profile requires an authorized ATO; none found"
    )


def evaluate_gates(
    candidate_id: str, request: dict[str, Any], engine: Any
) -> list[GateCheck]:
    """The full gate tier (design Sec 2c): compliance + EOL + gov-ATO. All are
    REQUIRED by default — any failure is a hard reject, checked before a single
    weighted-score point is computed."""
    return [
        _check_compliance_gates(candidate_id, request, engine),
        _check_eol(candidate_id, engine),
        _check_ato(candidate_id, request, engine),
    ]


# ── score tier (design Sec 2a/2b) ───────────────────────────────────────────


def _integration_fit_score(
    candidate_id: str, engine: Any, *, sample_limit: int = 500
) -> float:
    """Ecosystem-integration-fit: ``personalized_pagerank`` (``client.py:670``)
    seeded on the DEPLOYED estate (``:AssetInstance``), read at the candidate —
    a product that wires into more of what's already deployed ranks higher."""
    pagerank = getattr(engine, "personalized_pagerank", None)
    if not callable(pagerank):
        return 0.5
    seeds = {nid: 1.0 for nid, _p in _by_label(engine, "AssetInstance", sample_limit)}
    if not seeds:
        return 0.5
    try:
        ranks = pagerank(seed_nodes=seeds) or {}
    except Exception as e:  # noqa: BLE001 — degrade, don't raise
        logger.debug("portfolio: personalized_pagerank failed: %s", e)
        return 0.5
    if not ranks:
        return 0.5
    peak = max(ranks.values()) or 1.0
    return max(0.0, min(1.0, float(ranks.get(candidate_id, 0.0)) / peak))


def _cost_score(candidate_id: str, peer_ids: list[str], engine: Any) -> float:
    candidate_cost = _annual_cost(candidate_id, engine)
    peer_costs = [
        c for c in (_annual_cost(p, engine) for p in peer_ids) if c is not None
    ]
    if candidate_cost is None or not peer_costs:
        return 0.5  # insufficient cost data — neutral, never penalize on absence
    avg_peer = sum(peer_costs) / len(peer_costs)
    if avg_peer <= 0:
        return 0.5
    ratio = candidate_cost / avg_peer
    return max(0.0, min(1.0, 1.0 - ratio / 2.0))


def _license_score(candidate_id: str, engine: Any) -> float:
    for _s, tgt, p in _out(engine, candidate_id):
        if _rel(p) != "licensedUnder":
            continue
        model = str(_node_props(engine, tgt).get("licenseModel") or "").strip().lower()
        if model in _LICENSE_MODEL_SCORE:
            return _LICENSE_MODEL_SCORE[model]
    return 0.5


def _objective_alignment_score(candidate_id: str, engine: Any) -> float:
    aligned = 0.0
    for _s, tgt, p in _out(engine, candidate_id):
        if _rel(p) != "servesObjective":
            continue
        weight = _node_props(engine, tgt).get("objectiveWeight", 1.0)
        try:
            aligned += float(weight)
        except (TypeError, ValueError):
            aligned += 1.0
    return max(0.0, min(1.0, aligned))


def _consolidation_score(candidate_id: str, peer_ids: list[str], engine: Any) -> float:
    if not peer_ids:
        return 0.0
    retireable = _swappable_peers(candidate_id, engine) & set(peer_ids)
    return max(0.0, min(1.0, len(retireable) / len(peer_ids)))


def _compliance_soft_score(
    candidate_id: str, request: dict[str, Any], engine: Any
) -> float:
    if str(request.get("profile") or "commercial").strip().lower() != "gov":
        return 0.5  # not applicable in the commercial profile — neutral
    required_controls = int(request.get("requiredControlCount") or 0)
    for _s, tgt, p in _out(engine, candidate_id):
        if _rel(p) != "authorizedBy":
            continue
        covered = _node_props(engine, tgt).get("controlsCovered")
        if covered is not None and required_controls > 0:
            try:
                return max(0.0, min(1.0, float(covered) / required_controls))
            except (TypeError, ValueError):
                continue
    return 0.3  # gov profile, no control-coverage data — light penalty, not a gate


def score_criteria(
    candidate_id: str,
    peer_ids: list[str],
    request: dict[str, Any],
    engine: Any,
    weights: dict[str, float],
) -> list[CriterionScore]:
    """The weighted ``:ComparisonCriterion`` vector (design Sec 2a table) —
    composes graph reads + :func:`_integration_fit_score`'s
    ``personalized_pagerank`` call; no criterion re-derives capability
    membership from scratch outside :func:`_capability_set`."""
    candidate_caps = _capability_set(candidate_id, engine)
    peer_union: set[str] = set()
    for p in peer_ids:
        peer_union |= _capability_set(p, engine)

    functionality = (
        len(candidate_caps & peer_union) / len(peer_union) if peer_union else 1.0
    )
    unique = (
        len(candidate_caps - peer_union) / len(candidate_caps)
        if candidate_caps
        else 0.0
    )
    raw: dict[str, float] = {
        "functionality": functionality,
        "unique-capability": unique,
        "cost": _cost_score(candidate_id, peer_ids, engine),
        "licensing": _license_score(candidate_id, engine),
        "integration-fit": _integration_fit_score(candidate_id, engine),
        "objective-alignment": _objective_alignment_score(candidate_id, engine),
        "consolidation-benefit": _consolidation_score(candidate_id, peer_ids, engine),
        "compliance-ato": _compliance_soft_score(candidate_id, request, engine),
    }
    return [
        CriterionScore(
            kind=kind,
            score=round(value, 6),
            weight=weights.get(kind, 0.0),
            weighted=round(value * weights.get(kind, 0.0), 6),
        )
        for kind, value in raw.items()
    ]


def _weighted_total(criteria: list[CriterionScore]) -> float:
    total_weight = sum(c.weight for c in criteria) or 1.0
    return sum(c.weighted for c in criteria) / total_weight


# ── verdict (design Sec 2b — adopt/reject/consolidate/migrate) ─────────────


def _decide_verdict(
    ranking: list[dict[str, Any]], retireable: list[str], peer_ids: list[str]
) -> tuple[str, str]:
    """PromotionGovernanceValidator-style multi-rule verdict for a candidate
    that has ALREADY won its peer group (the caller only reaches this once
    ``ranking[0]`` is the candidate — see :func:`assess_candidate`):
    **consolidate** if a formally-swappable peer can be retired; **migrate**
    if it wins over peers that aren't formally swappable; **adopt** if it wins
    with no existing peers (greenfield)."""
    score = ranking[0]["score"]
    if retireable:
        names = ", ".join(retireable)
        return (
            "consolidate",
            f"wins its peer group (score {score:.3f}) and can retire {len(retireable)} "
            f"redundant peer(s) via consolidation: {names}",
        )
    if peer_ids:
        names = ", ".join(peer_ids)
        return (
            "migrate",
            f"wins its peer group (score {score:.3f}) over incumbent(s) {names}; "
            "recommend migrating off the incumbent(s)",
        )
    return (
        "adopt",
        f"no existing portfolio peers; adopted on its own merit (score {score:.3f})",
    )


def validate_verdict_shape(recommendation: dict[str, Any]) -> dict[str, Any]:
    """ADDITIVE SHACL audit of the computed verdict
    (``shapes/portfolio_intelligence.shapes.ttl``) — confirms the
    ``:Recommendation``/``:Assessment`` the engine would write is well-formed
    (valid verdict enum, non-empty rationale, a recorded score). Never itself
    decides the verdict; mirrors
    :meth:`~agent_utilities.knowledge_graph.research.promotion_governance.PromotionGovernanceValidator._check_shacl`.
    Degrades to ``conforms=True`` when rdflib/pyshacl aren't installed."""
    try:
        import rdflib
    except ImportError:
        return {
            "conforms": True,
            "violations": [],
            "results_text": "rdflib not installed — SHACL check skipped",
        }
    from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

    kg = rdflib.Namespace("http://knuckles.team/kg#")
    graph = rdflib.Graph()
    node = rdflib.URIRef(
        kg[f"{recommendation.get('candidateId', 'candidate')}:recommendation"]
    )
    graph.add((node, rdflib.RDF.type, kg.Recommendation))
    verdict = str(recommendation.get("verdict") or "")
    if verdict:
        graph.add((node, kg.verdict, rdflib.Literal(verdict)))
    rationale = str(recommendation.get("rationale") or "")
    if rationale:
        graph.add((node, kg.rationale, rdflib.Literal(rationale)))
    assessment = rdflib.URIRef(
        kg[f"{recommendation.get('candidateId', 'candidate')}:assessment"]
    )
    graph.add((assessment, rdflib.RDF.type, kg.Assessment))
    graph.add(
        (
            assessment,
            kg.assessmentScore,
            rdflib.Literal(
                float(recommendation.get("assessmentScore") or 0.0),
                datatype=rdflib.XSD.float,
            ),
        )
    )
    return SHACLValidator().validate(graph, _SHAPES_PATH)


# ── the public compute entrypoint ───────────────────────────────────────────


def assess_candidate(
    request: dict[str, Any],
    *,
    engine: Any | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Two-tier gate-then-score assessment of one candidate (design Sec 2c).

    ``request``: ``{"candidateId": <TechnologyProduct node id>, "sector":,
    "dataClass":, "profile": "commercial"|"gov", "peerIds": [...] (optional
    explicit peer group), "goalTypes": [...] (optional weight-policy override),
    "requiredControlCount": (gov)}``.

    Returns a plain dict — the ``:Assessment``/``:Recommendation`` shape
    :func:`run_trm_assessment` persists. Pure compute: never writes to the
    graph. Engine-guarded: an unreachable engine (or a missing ``candidateId``)
    degrades to a safe ``reject`` rather than raising.
    """
    candidate_id = str(
        request.get("candidateId") or request.get("candidate_id") or ""
    ).strip()
    empty = {
        "candidateId": candidate_id,
        "verdict": "reject",
        "rationale": "",
        "confidence": 0.0,
        "gates": [],
        "criteria": [],
        "assessmentScore": 0.0,
        "financialDelta": 0.0,
        "peerRanking": [],
        "consolidates": [],
    }
    if not candidate_id:
        return {**empty, "rationale": "request requires a candidateId"}

    eng = engine if engine is not None else health_ingest._engine()
    if eng is None:
        return {**empty, "rationale": "no engine reachable — cannot assess"}

    gates = evaluate_gates(candidate_id, request, eng)
    failures = [g for g in gates if g.required and not g.passed]
    if failures:
        return {
            **empty,
            "rationale": "; ".join(f"{g.name}: {g.reason}" for g in failures),
            "confidence": 1.0,
            "gates": [asdict(g) for g in gates],
        }

    peer_ids = resolve_peer_group(candidate_id, eng, explicit=request.get("peerIds"))
    w = weights or resolve_weights(eng, goal_types=request.get("goalTypes"))

    criteria = score_criteria(candidate_id, peer_ids, request, eng, w)
    candidate_score = _weighted_total(criteria)

    ranking: list[dict[str, Any]] = [
        {"id": candidate_id, "score": round(candidate_score, 6)}
    ]
    for peer_id in peer_ids:
        peer_peers = [p for p in peer_ids if p != peer_id] + [candidate_id]
        peer_criteria = score_criteria(peer_id, peer_peers, request, eng, w)
        ranking.append(
            {"id": peer_id, "score": round(_weighted_total(peer_criteria), 6)}
        )
    ranking.sort(key=lambda r: (-float(r["score"]), str(r["id"])))

    wins = bool(ranking) and ranking[0]["id"] == candidate_id
    retireable = (
        sorted(_swappable_peers(candidate_id, eng) & set(peer_ids)) if wins else []
    )
    verdict, rationale = (
        _decide_verdict(ranking, retireable, peer_ids)
        if wins
        else (
            "reject",
            f"does not win its peer group (score {candidate_score:.3f}, "
            f"best peer {ranking[0]['id']} scored {ranking[0]['score']:.3f})",
        )
    )

    financial_delta = round(
        sum(_annual_cost(p, eng) or 0.0 for p in retireable)
        - (_annual_cost(candidate_id, eng) or 0.0),
        2,
    )

    return {
        "candidateId": candidate_id,
        "verdict": verdict,
        "rationale": rationale,
        "confidence": round(min(1.0, 0.5 + candidate_score / 2), 4),
        "gates": [asdict(g) for g in gates],
        "criteria": [asdict(c) for c in criteria],
        "assessmentScore": round(candidate_score, 6),
        "financialDelta": financial_delta,
        "peerRanking": ranking,
        "consolidates": retireable,
        "peerIds": peer_ids,
    }


# ── graph persistence + writeback (design Sec 4) ───────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _work_note_text(outcome: dict[str, Any]) -> str:
    lines = [
        f"[portfolio-intelligence] verdict: {outcome['verdict']} "
        f"(confidence {outcome.get('confidence', 0.0):.2f})",
        outcome.get("rationale", ""),
        f"assessment score: {outcome.get('assessmentScore', 0.0):.3f}",
        f"financial delta: {outcome.get('financialDelta', 0.0):+.2f}",
    ]
    if outcome.get("consolidates"):
        lines.append(f"consolidates: {', '.join(outcome['consolidates'])}")
    if outcome.get("peerRanking"):
        ranked = ", ".join(
            f"{r['id']}={r['score']:.3f}" for r in outcome["peerRanking"][:5]
        )
        lines.append(f"peer ranking: {ranked}")
    return "\n".join(line for line in lines if line)


def _route_writeback(
    request: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    """Backfeed the recommendation to the source ticket (design Sec 4 step 5) —
    fail-closed, dry-run-first, config-driven backend selection
    (``TRM_WRITEBACK_BACKEND``, default ``none`` = graph-only), mirroring
    :func:`.incident_router.get_adapter`. A non-``none`` backend only places a
    LIVE call when its own enable flag is set (``SERVICENOW_ENABLE_WRITE`` for
    ``servicenow`` — reused, not a new write gate); otherwise the intended
    work-note is previewed and returned, never applied."""
    backend = str(setting("TRM_WRITEBACK_BACKEND", "none") or "none").strip().lower()
    if backend != "servicenow":
        return {"backend": "none", "status": "graph-only"}

    from agent_utilities.knowledge_graph.enrichment.writeback.core import run_writeback

    dry_run = not bool(setting("SERVICENOW_ENABLE_WRITE", False, cast=bool))
    sys_id = request.get("sysId") or request.get("sys_id")
    table = request.get("table") or "u_trm_request"
    out = run_writeback(
        "servicenow",
        dry_run=dry_run,
        work_notes=[
            {
                "table": table,
                "sys_id": sys_id,
                "node": request.get("id"),
                "note": _work_note_text(outcome),
            }
        ],
    )
    return {"backend": "servicenow", **out}


def _write_assessment(
    request_id: str, candidate_id: str, outcome: dict[str, Any]
) -> dict[str, int] | None:
    """Persist ``:Assessment`` -> ``:ComparisonCriterion``/``:Recommendation``
    (design Sec 1b) through the shared native-ingest writer — the same path
    :mod:`.incident_router`/:mod:`.lifecycle_orchestrator` write typed nodes
    through."""
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    assessment_id = f"{request_id}:assessment"
    recommendation_id = f"{request_id}:recommendation"
    entities: list[dict[str, Any]] = [
        {
            "id": assessment_id,
            "type": "Assessment",
            "assessmentScore": outcome.get("assessmentScore", 0.0),
            "financialDelta": outcome.get("financialDelta", 0.0),
            "runTimestamp": _now_iso(),
        },
        {
            "id": recommendation_id,
            "type": "Recommendation",
            "verdict": outcome.get("verdict"),
            "rationale": outcome.get("rationale"),
            "confidence": outcome.get("confidence", 0.0),
        },
    ]
    relationships: list[dict[str, Any]] = [
        {"source": assessment_id, "target": candidate_id, "type": "assesses"},
        {
            "source": assessment_id,
            "target": recommendation_id,
            "type": "yieldsRecommendation",
        },
        {"source": recommendation_id, "target": candidate_id, "type": "recommends"},
    ]
    for peer_id in outcome.get("peerIds") or []:
        relationships.append(
            {"source": assessment_id, "target": peer_id, "type": "assessedAgainst"}
        )
    for retired_id in outcome.get("consolidates") or []:
        relationships.append(
            {"source": recommendation_id, "target": retired_id, "type": "consolidates"}
        )
    if outcome.get("verdict") == "migrate":
        for retired_id in outcome.get("peerIds") or []:
            relationships.append(
                {
                    "source": recommendation_id,
                    "target": retired_id,
                    "type": "migratesFrom",
                }
            )
        relationships.append(
            {"source": recommendation_id, "target": candidate_id, "type": "migratesTo"}
        )
    for criterion in outcome.get("criteria") or []:
        crit_id = f"{assessment_id}:criterion:{criterion['kind']}"
        entities.append(
            {
                "id": crit_id,
                "type": "ComparisonCriterion",
                "criterionKind": criterion["kind"],
                "criterionWeight": criterion["weight"],
                "normalizedScore": criterion["score"],
            }
        )
        relationships.append(
            {"source": assessment_id, "target": crit_id, "type": "appliesCriterion"}
        )

    if request_id:
        entities.append(
            {"id": request_id, "type": "TRMRequest", "requestState": "assessed"}
        )
        relationships.append(
            {"source": request_id, "target": assessment_id, "type": "hasAssessment"}
        )
    return ingest_entities(entities, relationships, source=_SOURCE, domain="trm")


def run_trm_assessment(
    request: dict[str, Any],
    *,
    engine: Any | None = None,
    weights: dict[str, float] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """The full request -> assessment -> recommendation -> writeback flow
    (design Sec 4).

    ``request``: everything :func:`assess_candidate` reads, plus ``id`` (the
    ``:TRMRequest`` id — defaults to a stable id derived from the candidate),
    ``table``/``sysId`` (the ServiceNow record to backfeed).

    1. Runs :func:`assess_candidate`.
    2. Persists ``:TRMRequest`` -> ``:Assessment`` -> ``:Recommendation`` (+
       ``:ComparisonCriterion`` nodes) when ``write`` (default True).
    3. Backfeeds the verdict to the source ticket via :func:`_route_writeback`
       (fail-closed, dry-run-first — never skipped even when ``write=False``,
       since a dry-run preview is itself safe/report-only).
    """
    candidate_id = str(request.get("candidateId") or request.get("candidate_id") or "")
    request_id = str(request.get("id") or f"trm:request:{candidate_id or 'unknown'}")

    outcome = assess_candidate(request, engine=engine, weights=weights)

    written = _write_assessment(request_id, candidate_id, outcome) if write else None
    writeback = _route_writeback({**request, "id": request_id}, outcome)

    return {
        "requestId": request_id,
        **outcome,
        "written": written,
        "writeback": writeback,
    }


def rationalize_portfolio(
    *,
    engine: Any | None = None,
    weights: dict[str, float] | None = None,
    write: bool = True,
    limit: int = 500,
) -> dict[str, Any]:
    """Periodic sweep (design Sec 5b): find redundant/overlapping products —
    ``:TechnologyProduct``s sharing a ``:providesCapability`` target — and, for
    each cluster, run :func:`assess_candidate` with NO incoming request
    (candidate set = the cluster) to surface consolidate/migrate
    ``:Recommendation``s ranked by financial delta. A pure OWL-reasoner-style
    graph read (capability -> providers, >= 2 members = redundant), zero new
    analytics. Engine-guarded: an unreachable engine yields an all-zero
    summary.
    """
    eng = engine if engine is not None else health_ingest._engine()
    if eng is None:
        return {"clusters": 0, "recommendations": []}

    products = [pid for pid, _p in _by_label(eng, "TechnologyProduct", limit)]
    providers: dict[str, set[str]] = {}
    for pid in products:
        for cap in _capability_set(pid, eng):
            providers.setdefault(cap, set()).add(pid)
    clusters = {cap: members for cap, members in providers.items() if len(members) >= 2}

    seen: set[frozenset[str]] = set()
    recommendations: list[dict[str, Any]] = []
    for cap, members in clusters.items():
        key = frozenset(members)
        if key in seen:
            continue
        seen.add(key)
        member_ids = sorted(members)
        best: dict[str, Any] | None = None
        for candidate in member_ids:
            peers = [m for m in member_ids if m != candidate]
            outcome = assess_candidate(
                {"candidateId": candidate, "peerIds": peers},
                engine=eng,
                weights=weights,
            )
            if best is None or outcome["assessmentScore"] > best["assessmentScore"]:
                best = outcome
        if best is not None and best["verdict"] in ("consolidate", "migrate"):
            recommendations.append({"capability": cap, "members": member_ids, **best})
            if write:
                run_trm_assessment(
                    {
                        "id": f"trm:rationalize:{cap}",
                        "candidateId": best["candidateId"],
                    },
                    engine=eng,
                    weights=weights,
                    write=True,
                )

    recommendations.sort(key=lambda r: -float(r.get("financialDelta") or 0.0))
    return {"clusters": len(clusters), "recommendations": recommendations}


def main() -> None:
    """CLI (``python -m agent_utilities.observability.portfolio_intelligence``):
    one rationalization sweep across the portfolio; prints a JSON summary.
    Report-only — see the ``portfolio-intelligence`` CronJob manifest
    (suspended by default)."""
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    print(json.dumps(rationalize_portfolio(), default=str, indent=2))


if __name__ == "__main__":
    main()
