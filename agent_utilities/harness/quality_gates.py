"""CONCEPT:AHE-3.13 — Layered Pre-Emit Gate Pipeline.

Assimilated from open-design's quality culture (prompts/discovery.ts): a blocking pre-emit pipeline of
**preflight checklist (P0/P1/P2)** → **multi-dimensional self-critique (score 1-5, fix any <3)** that
reads a KG-stored **anti-slop antipattern registry**. Unlike the source (prompt-only), these are
executable gates whose scores are returned for persistence and feed the evolution engine (AHE-3.1/3.2).

The critique scorer is pluggable: the default is a deterministic heuristic (no LLM, fully testable); a
real deployment can inject an LLM-backed scorer. Gates run in ``warn`` (default) or ``block`` mode.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

# ── Anti-slop antipattern registry (KG-stored in production; in-module default here) ──


@dataclass(frozen=True, slots=True)
class Antipattern:
    """A named output antipattern with a detector and a preferred alternative."""

    name: str
    pattern: str  # regex (case-insensitive)
    alternative: str


DEFAULT_ANTIPATTERNS: tuple[Antipattern, ...] = (
    Antipattern(
        "invented-metric",
        r"\b\d+(\.\d+)?\s*(x|%)\s+(faster|better|more|uptime)\b",
        "cite a source or use an honest placeholder (—)",
    ),
    Antipattern(
        "filler-copy",
        r"\b(lorem ipsum|feature one|feature two|your text here)\b",
        "write specific copy or leave a clearly-marked stub",
    ),
    Antipattern(
        "emoji-feature-icon",
        r"[✨\U0001F680\U0001F3AF]",  # ✨ 🚀 🎯
        "use a real icon set or none",
    ),
    Antipattern(
        "gradient-spam",
        r"linear-gradient\([^)]*\).*linear-gradient\(",
        "limit decorative gradients; prefer a flat palette",
    ),
)


class AntipatternRegistry:
    """Detects anti-slop patterns in output text."""

    def __init__(
        self, patterns: tuple[Antipattern, ...] = DEFAULT_ANTIPATTERNS
    ) -> None:
        self._patterns = patterns
        self._compiled = [
            (p, re.compile(p.pattern, re.IGNORECASE | re.DOTALL)) for p in patterns
        ]

    def detect(self, text: str) -> list[Antipattern]:
        """Return the antipatterns found in ``text``."""
        return [p for p, rx in self._compiled if rx.search(text or "")]


# ── Preflight checklist (P0/P1/P2) ───────────────────────────────────


class Priority(StrEnum):
    P0 = "P0"  # blocking
    P1 = "P1"
    P2 = "P2"


@dataclass(slots=True)
class ChecklistRule:
    priority: Priority
    text: str


@dataclass(slots=True)
class PreflightResult:
    passed: bool
    failed_p0: list[str] = field(default_factory=list)
    failed_other: list[str] = field(default_factory=list)


def parse_checklist(markdown: str) -> list[ChecklistRule]:
    """Parse a ``references/checklist.md`` of ``- [P0] rule`` lines into rules."""
    rules: list[ChecklistRule] = []
    for line in markdown.splitlines():
        m = re.match(r"\s*[-*]\s*\[?(P[012])\]?\s+(.*\S)", line)
        if m:
            rules.append(ChecklistRule(Priority(m.group(1)), m.group(2).strip()))
    return rules


# A predicate evaluates one rule against the output → True if the rule passes.
RulePredicate = Callable[[ChecklistRule, str], bool]


class PreflightGate:
    """Evaluate output against a P0/P1/P2 checklist. A failing P0 blocks emission in ``block`` mode."""

    def __init__(
        self, rules: list[ChecklistRule], *, predicate: RulePredicate | None = None
    ) -> None:
        self._rules = rules
        # Default predicate: a rule passes unless its keyword(s) are absent — overridable per skill.
        self._predicate = predicate or (lambda rule, out: True)

    def check(self, output: str) -> PreflightResult:
        failed_p0: list[str] = []
        failed_other: list[str] = []
        for rule in self._rules:
            if not self._predicate(rule, output):
                (failed_p0 if rule.priority is Priority.P0 else failed_other).append(
                    rule.text
                )
        return PreflightResult(
            passed=not failed_p0, failed_p0=failed_p0, failed_other=failed_other
        )


# ── Multi-dimensional self-critique ──────────────────────────────────

DIMENSIONS = ("coverage", "coherence", "evidence", "safety", "specificity")


@dataclass(slots=True)
class CritiqueResult:
    scores: dict[str, int]
    weakest: str
    passed: bool  # all dimensions >= 3
    antipatterns: list[str] = field(default_factory=list)


# A scorer returns a dict {dimension: 1..5} for the given output.
CritiqueScorer = Callable[[str], dict[str, int]]


def _heuristic_scorer(text: str) -> dict[str, int]:
    """Deterministic, LLM-free scorer used by default and in tests.

    Rewards length/structure/specific tokens; penalizes emptiness. Bounded to 1..5.
    """
    t = text or ""
    n = len(t)
    coverage = 5 if n > 400 else 4 if n > 150 else 3 if n > 40 else 1
    coherence = 4 if ("\n" in t or ". " in t) else 3 if n > 20 else 2
    evidence = 4 if re.search(r"https?://|\[\d+\]|source:", t, re.IGNORECASE) else 3
    safety = (
        1
        if re.search(r"\b(password|api[_-]?key|secret)\b\s*[:=]", t, re.IGNORECASE)
        else 5
    )
    specificity = (
        2
        if re.search(r"lorem ipsum|feature one|your text here", t, re.IGNORECASE)
        else 4
    )
    return {
        "coverage": coverage,
        "coherence": coherence,
        "evidence": evidence,
        "safety": safety,
        "specificity": specificity,
    }


class MultiDimensionalCritique:
    """Score output on named dimensions (1-5); any dimension < 3 is a regression to fix-and-re-score."""

    def __init__(
        self,
        *,
        scorer: CritiqueScorer | None = None,
        antipatterns: AntipatternRegistry | None = None,
    ) -> None:
        self._scorer = scorer or _heuristic_scorer
        self._antipatterns = antipatterns or AntipatternRegistry()

    def critique(self, output: str) -> CritiqueResult:
        scores = {d: max(1, min(5, int(v))) for d, v in self._scorer(output).items()}
        weakest = min(scores, key=lambda d: scores[d]) if scores else ""
        passed = all(v >= 3 for v in scores.values())
        aps = [a.name for a in self._antipatterns.detect(output)]
        return CritiqueResult(
            scores=scores, weakest=weakest, passed=passed and not aps, antipatterns=aps
        )


# ── Composed pre-emit gate ───────────────────────────────────────────


@dataclass(slots=True)
class GateResult:
    """Outcome of the layered pre-emit gate; ``blocked`` is only True in ``block`` mode on a failure."""

    ok: bool
    blocked: bool
    preflight: PreflightResult | None
    critique: CritiqueResult
    mode: str


class PreEmitGate:
    """Compose preflight + critique into one pre-emit decision (CONCEPT:AHE-3.13)."""

    def __init__(
        self,
        *,
        preflight: PreflightGate | None = None,
        critique: MultiDimensionalCritique | None = None,
        mode: str = "warn",
    ) -> None:
        self._preflight = preflight
        self._critique = critique or MultiDimensionalCritique()
        self.mode = mode

    def evaluate(self, output: str) -> GateResult:
        pf = self._preflight.check(output) if self._preflight else None
        crit = self._critique.critique(output)
        ok = (pf is None or pf.passed) and crit.passed
        blocked = (self.mode == "block") and not ok
        return GateResult(
            ok=ok, blocked=blocked, preflight=pf, critique=crit, mode=self.mode
        )
