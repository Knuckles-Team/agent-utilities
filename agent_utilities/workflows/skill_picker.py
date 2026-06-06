"""CONCEPT:ORCH-1.2 (extension) — Scenario taxonomy + eval-scored skill picker.

Assimilated from open-design's scenario-grouped skills + picker, made **self-improving**: candidates
are scored by keyword overlap + scenario fit + **prior success rate from the eval engine (AHE-3.1)**,
so skills that historically work rank higher than keyword-equal rivals. Extension-only — no new
concept; this is a new routing strategy over the existing skill registry.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

SCENARIOS = (
    "design",
    "marketing",
    "operation",
    "engineering",
    "finance",
    "hr",
    "sales",
    "personal",
)

# Keyword hints used to infer a scenario for skills that don't declare one (open-design parity).
_SCENARIO_HINTS: dict[str, tuple[str, ...]] = {
    "finance": ("invoice", "finance", "budget", "ledger", "revenue", "accounting"),
    "hr": ("onboard", "payroll", "hr", "employee", "recruit"),
    "sales": ("sales", "lead", "crm", "pipeline", "deal"),
    "marketing": ("campaign", "seo", "landing", "social", "brand"),
    "engineering": ("code", "deploy", "build", "test", "runbook", "api", "debug"),
    "operation": ("ops", "incident", "monitor", "dashboard", "alert"),
    "design": ("design", "ui", "poster", "deck", "wireframe", "theme"),
    "personal": ("personal", "todo", "note", "journal"),
}


def infer_scenario(text: str) -> str:
    """Infer a scenario from free text (description/body) when none is declared."""
    low = (text or "").lower()
    best, best_hits = "engineering", 0
    for scenario, hints in _SCENARIO_HINTS.items():
        hits = sum(1 for h in hints if h in low)
        if hits > best_hits:
            best, best_hits = scenario, hits
    return best


@dataclass(slots=True)
class SkillCandidate:
    """A pickable skill with taxonomy + a per-skill critique policy override."""

    name: str
    description: str = ""
    scenario: str = ""
    category: str = ""
    source: str = "built-in"  # "built-in" | "user"
    tags: tuple[str, ...] = ()
    critique_policy: str | None = None  # required | opt-out | opt-in | None
    tier: str = "normal"

    def resolved_scenario(self) -> str:
        return self.scenario or infer_scenario(f"{self.name} {self.description}")


# A success-rate provider returns a skill's historical win-rate in [0,1] (from AHE-3.1 eval traces).
SuccessRateProvider = Callable[[str], float]


@dataclass(slots=True)
class ScoredSkill:
    candidate: SkillCandidate
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class SkillPicker:
    """Rank skills for a query by keyword overlap + scenario fit + prior success rate.

    Weights: keyword 0.4, success-rate 0.4, scenario/tier fit 0.2 (matches the design doc). Missing
    success-rate defaults to a neutral 0.5 prior so cold skills are neither punished nor favored.
    """

    def __init__(
        self,
        *,
        success_rate: SuccessRateProvider | None = None,
        w_keyword: float = 0.4,
        w_success: float = 0.4,
        w_fit: float = 0.2,
    ) -> None:
        self._success_rate = success_rate or (lambda _name: 0.5)
        self._w_keyword, self._w_success, self._w_fit = w_keyword, w_success, w_fit

    def _score(
        self,
        c: SkillCandidate,
        q_tokens: set[str],
        scenario: str | None,
        tier: str | None,
    ) -> ScoredSkill:
        c_tokens = _tokenize(f"{c.name} {c.description} {' '.join(c.tags)}")
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
        success = max(0.0, min(1.0, self._success_rate(c.name)))
        fit = 0.0
        if scenario and c.resolved_scenario() == scenario:
            fit += 0.7
        if tier and c.tier == tier:
            fit += 0.3
        score = (
            self._w_keyword * overlap + self._w_success * success + self._w_fit * fit
        )
        return ScoredSkill(
            c, score, {"keyword": overlap, "success": success, "fit": fit}
        )

    def rank(
        self,
        query: str,
        candidates: list[SkillCandidate],
        *,
        scenario: str | None = None,
        tier: str | None = None,
    ) -> list[ScoredSkill]:
        """Return candidates ranked best-first. If ``scenario`` is given, filter to it first."""
        pool = [
            c
            for c in candidates
            if (scenario is None or c.resolved_scenario() == scenario)
        ]
        q_tokens = _tokenize(query)
        scored = [self._score(c, q_tokens, scenario, tier) for c in pool]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def pick(
        self, query: str, candidates: list[SkillCandidate], **kw
    ) -> SkillCandidate | None:
        """Return the single best skill for ``query`` (or None if no candidates)."""
        ranked = self.rank(query, candidates, **kw)
        return ranked[0].candidate if ranked else None
