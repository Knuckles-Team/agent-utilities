"""CONCEPT:KG-2.28 — Persona Decision-Heuristic Enrichment

Fincept Terminal ships a granular persona registry where each investor agent
carries an explicit, scored decision framework (Buffett: ROE>=15%, D/E<0.5,
owner-earnings yield; Graham: P/E<15, P/B<1.5, margin-of-safety). Our existing
``INVESTOR_PERSONAS`` only bind a *prompt voice* to a swarm role. This module
makes those frameworks **structured, queryable, and executable**:

* :data:`PERSONA_HEURISTICS` attaches a list of typed :class:`Heuristic` rules to
  each investor archetype (Graham/Buffett/Burry/Damodaran/Druckenmiller + a
  Lynch-style PEG growth lens).
* :func:`evaluate_persona` runs a persona's heuristics against a ticker's metrics
  and returns each rule's pass/fail + a rationale, plus a verdict.
* :func:`evaluate_all` scores every persona for a screen/debate.
* :func:`persona_heuristics_batch` emits ``:DecisionHeuristic`` KG nodes so the
  rules are facts the graph can reason over.

KG/OWL uniqueness leveraged: heuristics become ``:DecisionHeuristic`` nodes
``HEURISTIC_OF`` a persona ``:Agent`` (with matching OWL classes in
``ontology_quant.ttl``), so a debate can *query* "which personas' value criteria
does ACME pass?" and a Buffett bull can cite the exact failing/ passing rule
rather than hand-waving. The evaluator is deterministic and offline.

This wires into the screen/debate: :func:`evaluate_all` is the structured,
engine-free pre-filter the Bull/Bear ``DebateEngine`` and ``ForensicScreener``
hand to each persona so its argument is grounded in named, numeric rules.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class Heuristic:
    """One named, evaluable decision rule.

    ``metric`` is the key looked up in a ticker's metrics dict; ``op`` + ``value``
    form the threshold test (e.g. ``"pe", "<", 15``). ``weight`` is the rule's
    share of the persona's framework; ``rationale`` is the human "why".
    """

    name: str
    metric: str
    op: str
    value: float
    weight: float
    rationale: str

    def evaluate(self, metrics: dict[str, Any]) -> HeuristicResult:
        """Test the rule against a metrics dict.

        A missing metric yields an ``UNKNOWN`` result (never a silent pass), so a
        persona can't claim conviction on data it doesn't have.
        """
        actual = metrics.get(self.metric)
        if actual is None:
            return HeuristicResult(self.name, status="unknown", actual=None, rule=self)
        try:
            passed = _OPS[self.op](float(actual), float(self.value))
        except (TypeError, ValueError):
            return HeuristicResult(
                self.name, status="unknown", actual=actual, rule=self
            )
        return HeuristicResult(
            self.name,
            status="pass" if passed else "fail",
            actual=float(actual),
            rule=self,
        )


@dataclass
class HeuristicResult:
    """Outcome of one heuristic against a ticker."""

    name: str
    status: str  # pass | fail | unknown
    actual: Any
    rule: Heuristic

    def explain(self) -> str:
        """Citable one-liner: the threshold, the actual, and the verdict."""
        thr = f"{self.rule.metric} {self.rule.op} {self.rule.value:g}"
        act = "n/a" if self.actual is None else f"{self.actual:g}"
        verb = {"pass": "PASS", "fail": "FAIL", "unknown": "UNKNOWN"}[self.status]
        return (
            f"[{verb}] {self.rule.name} ({thr}; actual {act}) — {self.rule.rationale}"
        )


@dataclass
class PersonaEvaluation:
    """A persona's full evaluation of a ticker against its heuristics."""

    persona: str
    archetype: str
    score: float  # weighted pass fraction over *known* rules, in [0, 1]
    verdict: str  # bullish | neutral | bearish | insufficient_data
    results: list[HeuristicResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "archetype": self.archetype,
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "actual": r.actual,
                    "explain": r.explain(),
                }
                for r in self.results
            ],
        }

    def citation(self) -> str:
        """A debate-ready summary naming the decisive rules."""
        passed = [r.rule.name for r in self.results if r.status == "pass"]
        failed = [r.rule.name for r in self.results if r.status == "fail"]
        return (
            f"{self.archetype} verdict on this name: {self.verdict.upper()} "
            f"(score {self.score:.0%}). Passes: {', '.join(passed) or 'none'}. "
            f"Fails: {', '.join(failed) or 'none'}. "
            "(source: persona_heuristics KG-2.28)"
        )


# ── Persona decision frameworks ──────────────────────────────────────────────
# Keyed by the persona prompt stem (matching ``INVESTOR_PERSONAS``). Each rule's
# ``metric`` is a plain key a caller supplies in the metrics dict, e.g.::
#
#   {"pe": 12.0, "pb": 1.1, "margin_of_safety": 0.35, "roe": 0.18, ...}
#
PERSONA_HEURISTICS: dict[str, tuple[Heuristic, ...]] = {
    "graham_investor": (
        Heuristic("low_pe", "pe", "<", 15.0, 0.3, "Graham: earnings multiple under 15"),
        Heuristic("low_pb", "pb", "<", 1.5, 0.25, "Graham: price-to-book under 1.5"),
        Heuristic(
            "margin_of_safety",
            "margin_of_safety",
            ">=",
            0.30,
            0.3,
            "Graham: >=30% discount to intrinsic value",
        ),
        Heuristic(
            "current_ratio",
            "current_ratio",
            ">=",
            2.0,
            0.15,
            "Graham: current ratio >= 2 (solvency)",
        ),
    ),
    "buffett_investor": (
        Heuristic(
            "high_roe",
            "roe",
            ">=",
            0.15,
            0.3,
            "Buffett: ROE >= 15% (moat/returns on capital)",
        ),
        Heuristic(
            "high_roic", "roic", ">=", 0.10, 0.2, "Buffett: ROIC above cost of capital"
        ),
        Heuristic(
            "low_leverage",
            "de_ratio",
            "<",
            0.5,
            0.2,
            "Buffett: debt/equity < 0.5 (fortress)",
        ),
        Heuristic(
            "owner_earnings_yield",
            "owner_earnings_yield",
            ">=",
            0.05,
            0.2,
            "Buffett: owner-earnings yield >= 5%",
        ),
        Heuristic(
            "predictable_earnings",
            "earnings_positive_years",
            ">=",
            8.0,
            0.1,
            "Buffett: positive earnings in >=8 of last 10 years",
        ),
    ),
    "burry_investor": (
        Heuristic(
            "accruals_red_flag",
            "accruals_ratio",
            ">",
            0.10,
            0.3,
            "Burry: high Sloan accruals — earnings not backed by cash (short trigger)",
        ),
        Heuristic(
            "beneish_manipulation",
            "m_score",
            ">",
            -1.78,
            0.3,
            "Burry: Beneish M-score above -1.78 suggests manipulation",
        ),
        Heuristic(
            "distress_z",
            "z_score",
            "<",
            1.81,
            0.25,
            "Burry: Altman Z below 1.81 — distress zone",
        ),
        Heuristic(
            "stretched_valuation",
            "pe",
            ">",
            40.0,
            0.15,
            "Burry: nosebleed P/E with deteriorating quality",
        ),
    ),
    "damodaran_investor": (
        Heuristic(
            "undervalued_dcf",
            "price_to_dcf",
            "<",
            1.0,
            0.4,
            "Damodaran: price below DCF intrinsic value",
        ),
        Heuristic(
            "reinvestment_quality",
            "roic",
            ">=",
            0.08,
            0.3,
            "Damodaran: ROIC supports the growth narrative",
        ),
        Heuristic(
            "revenue_growth",
            "revenue_growth",
            ">=",
            0.05,
            0.3,
            "Damodaran: story needs real top-line growth",
        ),
    ),
    "druckenmiller_investor": (
        Heuristic(
            "momentum_regime",
            "trend_strength",
            ">",
            0.0,
            0.4,
            "Druckenmiller: trade with the macro/price regime",
        ),
        Heuristic(
            "liquidity_tailwind",
            "liquidity_score",
            ">",
            0.0,
            0.3,
            "Druckenmiller: lean in when liquidity is a tailwind",
        ),
        Heuristic(
            "asymmetric_payoff",
            "reward_risk_ratio",
            ">=",
            2.0,
            0.3,
            "Druckenmiller: only size up asymmetric (>=2:1) setups",
        ),
    ),
    # A Lynch-style PEG growth lens (Fincept registry includes Lynch).
    "lynch_investor": (
        Heuristic(
            "cheap_peg",
            "peg",
            "<",
            1.0,
            0.5,
            "Lynch: PEG < 1 (growth at a reasonable price)",
        ),
        Heuristic(
            "earnings_growth",
            "earnings_growth",
            ">=",
            0.15,
            0.3,
            "Lynch: sustained >=15% earnings growth",
        ),
        Heuristic(
            "manageable_debt", "de_ratio", "<", 1.0, 0.2, "Lynch: debt/equity under 1"
        ),
    ),
}

# Personas whose framework being *satisfied* is a SHORT/bearish thesis (forensic
# shorts): a high pass-score means the company looks broken → bearish verdict.
_BEARISH_WHEN_SATISFIED = {"burry_investor"}

_ARCHETYPES = {
    "graham_investor": "GrahamInvestor",
    "buffett_investor": "BuffettInvestor",
    "burry_investor": "BurryInvestor",
    "damodaran_investor": "DamodaranInvestor",
    "druckenmiller_investor": "DruckenmillerInvestor",
    "lynch_investor": "LynchInvestor",
}


def list_personas() -> list[str]:
    """Persona stems that carry a structured heuristic framework."""
    return sorted(PERSONA_HEURISTICS)


def persona_archetype(persona: str) -> str:
    """Archetype label for a persona stem (e.g. ``GrahamInvestor``)."""
    return _ARCHETYPES.get(persona, persona)


def _verdict(persona: str, score: float, known: int) -> str:
    """Map a weighted pass-score → bullish/neutral/bearish.

    For forensic-short personas a high score means *broken* → bearish.
    """
    if known == 0:
        return "insufficient_data"
    bearish_lens = persona in _BEARISH_WHEN_SATISFIED
    if score >= 0.66:
        return "bearish" if bearish_lens else "bullish"
    if score <= 0.33:
        return "bullish" if bearish_lens else "bearish"
    return "neutral"


def evaluate_persona(persona: str, metrics: dict[str, Any]) -> PersonaEvaluation:
    """Run one persona's heuristics against a ticker's metrics.

    The score is the weighted pass-fraction over *known* (non-``unknown``) rules,
    so missing data lowers confidence rather than silently passing.
    """
    rules = PERSONA_HEURISTICS.get(persona)
    if rules is None:
        raise KeyError(f"No heuristic framework for persona '{persona}'")
    results = [r.evaluate(metrics) for r in rules]
    known = [r for r in results if r.status in ("pass", "fail")]
    known_weight = sum(r.rule.weight for r in known)
    pass_weight = sum(r.rule.weight for r in known if r.status == "pass")
    score = (pass_weight / known_weight) if known_weight > 0 else 0.0
    return PersonaEvaluation(
        persona=persona,
        archetype=persona_archetype(persona),
        score=score,
        verdict=_verdict(persona, score, len(known)),
        results=results,
    )


def evaluate_all(metrics: dict[str, Any]) -> dict[str, PersonaEvaluation]:
    """Evaluate every heuristic-bearing persona against a ticker.

    This is the structured pre-filter the screen/debate hands each persona so its
    bull/bear argument is grounded in named, numeric rules.
    """
    return {p: evaluate_persona(p, metrics) for p in PERSONA_HEURISTICS}


def persona_heuristics_batch() -> Any:
    """Build an ``ExtractionBatch`` of ``:DecisionHeuristic`` nodes (KG-2.28).

    Each heuristic is a node ``HEURISTIC_OF`` its persona's prompt id, so the KG
    can answer "what rules does Graham screen on?" and link them to evaluations.
    """
    from agent_utilities.knowledge_graph.enrichment.models import (
        EnrichmentEdge,
        ExtractionBatch,
        GraphNode,
    )

    nodes = []
    edges = []
    for persona, rules in PERSONA_HEURISTICS.items():
        persona_node = f"prompt:{persona}"
        for r in rules:
            hid = f"decision_heuristic:{persona}:{r.name}"
            nodes.append(
                GraphNode(
                    id=hid,
                    type="DecisionHeuristic",
                    props={
                        "name": r.name,
                        "metric": r.metric,
                        "op": r.op,
                        "threshold": r.value,
                        "weight": r.weight,
                        "rationale": r.rationale,
                        "persona": persona,
                        "concept": "KG-2.28",
                    },
                )
            )
            edges.append(
                EnrichmentEdge(source=hid, target=persona_node, rel_type="HEURISTIC_OF")
            )
    return ExtractionBatch(category="persona_heuristics", nodes=nodes, edges=edges)


def seed_persona_heuristics(backend: Any) -> tuple[int, int]:
    """Persist persona decision heuristics into the KG via ``write_batch``.

    ``None`` backend (offline) is a no-op returning ``(0, 0)``.
    """
    if backend is None:
        return (0, 0)
    from agent_utilities.knowledge_graph.enrichment.registry import write_batch

    n, e = write_batch(backend, persona_heuristics_batch())
    logger.info("Seeded persona decision heuristics: %d nodes, %d edges", n, e)
    return n, e
