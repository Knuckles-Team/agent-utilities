#!/usr/bin/python
"""Data model for shortcut-resistant search-task synthesis.

CONCEPT:KG-2.70 — evidence-graph workspace

Distilled from FORT-Searcher (arXiv:2606.12087). An :class:`EvidenceGraph` is the
construction workspace FORT calls ``G``: real-world entities (nodes) connected by
*verified facts* (edges) that can later be verbalized as clues. It is **not** the
object the final agent solves — it organizes facts, their evidence sources, and
their dependencies before a subgraph is rendered as a natural-language question.

These dataclasses are engine-agnostic and pure so the shortcut detectors
(:mod:`.shortcut_risks`) and the formulation/refinement loop
(:mod:`.question_formulation`) are fully unit-testable from hand-built graphs;
:mod:`.evidence_subgraph` is the live adapter that populates them from the
epistemic graph.

Concept: evidence-graph
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A clue whose standalone candidate pool ``s({c}) = |Ans({c})|`` is at or below
# this size already (nearly) identifies the answer on its own — a single-clue
# selectivity shortcut (FORT §2.3). Generic clues sit well above it.
SINGLE_CLUE_POOL_FLOOR = 1


@dataclass
class EvidenceFact:
    """One verified fact connecting the answer (or an intermediate) to evidence.

    Carries exactly the structural signals the four shortcut detectors read:
    its standalone selectivity, its provenance source, whether it is a derived
    (multi-source) fact, the intermediate entity names it would expose if
    verbalized naively, and whether it is *required* to jointly identify the
    answer (vs a prunable redundant clue).
    """

    id: str
    clue: str
    """Verbalized clue text as it would appear in the question."""
    source_document_id: str
    """Provenance — the evidence item this fact was verified against (co-coverage)."""
    standalone_pool: int = 1_000_000
    """``|Ans({this clue})|`` — candidate pool this clue alone leaves (selectivity)."""
    derived: bool = False
    """True when constructed from multiple atomic facts (resists verbatim recovery)."""
    required: bool = True
    """True when this clue is necessary for the clue set to identify the answer."""
    referenced_names: tuple[str, ...] = ()
    """Intermediate entity names this clue exposes (candidates for withholding)."""
    referring_expr: str = "the related entity"
    """Generic stand-in used when a referenced name is withheld."""

    @property
    def generic(self) -> bool:
        """A clue is generic (weak alone, FORT Table 4) when it is not over-selective."""
        return self.standalone_pool > SINGLE_CLUE_POOL_FLOOR


@dataclass
class EvidenceGraph:
    """The construction workspace: answer node + selected clue facts (FORT §3.1)."""

    answer_id: str
    answer_aliases: tuple[str, ...]
    facts: list[EvidenceFact] = field(default_factory=list)
    root_popularity: float = 0.0
    """0..1 prior-familiarity of the seed/answer (high → prior-knowledge binding risk)."""

    def fact(self, fact_id: str) -> EvidenceFact:
        for f in self.facts:
            if f.id == fact_id:
                return f
        raise KeyError(fact_id)

    def required_ids(self) -> set[str]:
        return {f.id for f in self.facts if f.required}

    def identifies(self, fact_ids: set[str]) -> bool:
        """A clue subset identifies the answer iff it covers all required clues.

        The well-posedness condition ``Ans(C_q) = {y*}`` (FORT §2.1): the full
        required set pins a unique gold answer, so any superset of the required
        clues is identifying and redundant (non-required) clues may be pruned.
        """
        return self.required_ids().issubset(fact_ids)


@dataclass
class RiskFinding:
    """One shortcut detector's verdict over an evidence graph / draft question."""

    risk: str
    score: float
    """0..1 severity (0 = no risk, 1 = certain shortcut)."""
    tripped: bool
    offenders: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class RiskReport:
    findings: list[RiskFinding] = field(default_factory=list)

    @property
    def clear(self) -> bool:
        return not any(f.tripped for f in self.findings)

    def by_risk(self, risk: str) -> RiskFinding | None:
        for f in self.findings:
            if f.risk == risk:
                return f
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "clear": self.clear,
            "findings": [
                {
                    "risk": f.risk,
                    "score": f.score,
                    "tripped": f.tripped,
                    "offenders": f.offenders,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


@dataclass
class SearchTask:
    """A synthesized, shortcut-resistant search task ready for training/eval."""

    question: str
    answer_id: str
    answer_aliases: tuple[str, ...]
    evidence_fact_ids: list[str]
    difficulty: int
    """Number of required clues (≈ hops the solver must chain)."""
    risk_report: RiskReport

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer_id": self.answer_id,
            "answer_aliases": list(self.answer_aliases),
            "evidence_fact_ids": list(self.evidence_fact_ids),
            "difficulty": self.difficulty,
            "risk_report": self.risk_report.to_dict(),
        }
