"""CONCEPT:AU-AHE.harness.when-task-is-scope — Self-Guided Self-Play (SGS).

Assimilated from "Scaling Self-Play with Self-Guidance" (SGS), arXiv:2604.20209
(Bailey, Wen, Dong, Hashimoto, Ma). The paper observes that asymmetric self-play
plateaus because the Conjecturer learns to *hack its reward*: over long runs it
collapses to artificially complex, superficially-related problems (disjunction
spam, over-long conclusions, redundant premises) that do not help the Solver
improve. SGS adds a third LLM role — the **Guide** — that scores each generated
problem on its (1) relevance to the unsolved target, (2) clean formulation (a
simple conclusion, not padded with complexity), and (3) natural/elegant logic;
low-scoring conjectures are rejected so they never train the Solver. This is the
supervision against Conjecturer collapse.

This module instantiates the three roles as a deterministic, dependency-injected
loop so it is unit-testable with no LLM and no network: the caller injects a
``conjecture_fn`` (the Conjecturer), a ``solve_fn`` (the Solver), and optionally a
Guide ``scorer``. The :class:`Guide` ships a deterministic heuristic scorer that
mirrors the paper's rubric (relevance = lexical overlap with the target;
conciseness = penalise over-long / redundantly-repeated conclusions; naturalness =
penalise the paper's documented collapse markers — connective/disjunction spam,
runaway sentence length, contradictory cue words), so the gate is real even
without a model. A real deployment injects an LLM-backed ``scorer``.

Beyond the paper, this implementation carries an AHE-3.x curriculum + plateau
breaker: accepted-and-solved conjectures raise the target difficulty (curriculum),
rejected conjectures advance nothing (this is exactly what denies the Conjecturer
its reward hack), and a stalled rolling solve-rate trips a plateau breaker that
perturbs the difficulty downward to escape the plateau.

Style mirrors CONCEPT:AU-AHE.harness.pre-emit-quality-gate (``quality_gates.py``) dataclass/pluggable-scorer
gates and CONCEPT:AU-AHE.harness.evolutionary-aggregation (``variant_pool.py``) population-collapse signalling.
It is opt-in (off the hot path until wired) and intended to be driven by the
evolution engine as a verifier-free task-quality gate on generated subproblems.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "ConjectureFn",
    "SolveFn",
    "GuideScore",
    "Guide",
    "PlayRound",
    "PlayReport",
    "SelfGuidedSelfPlay",
]

# A Conjecturer: (target_task, target_difficulty in 0..1) -> a generated task.
ConjectureFn = Callable[[str, float], str]
# A Solver: (task) -> (solution_text, solved_ok).
SolveFn = Callable[[str], tuple[str, bool]]

# A Guide scorer: (target_task, generated_task) -> GuideScore. Injected for LLM use.
GuideScorer = Callable[[str, str], "GuideScore"]

# Connective/disjunction spam markers — the paper's documented collapse mode where
# the Conjecturer pads conclusions with disjunctions (OR) and chained clauses.
_CONNECTIVE_MARKERS: tuple[str, ...] = (
    " or ",
    " and ",
    " then ",
    " moreover ",
    " furthermore ",
    " additionally ",
    " ∨ ",
    " ∧ ",
    " => ",
    " -> ",
)
# Cue words that signal contradictory / illogical padding (gamed naturalness).
_ILLOGICAL_MARKERS: tuple[str, ...] = (
    "but not",
    "except when",
    "if and only if not",
    "contradiction",
    "vacuously",
    "trivially false",
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Lowercase word/number tokens used for lexical overlap (deterministic)."""
    return _WORD_RE.findall((text or "").lower())


@dataclass(slots=True)
class GuideScore:
    """The Guide's three-dimensional verdict on one generated task (CONCEPT:AU-AHE.harness.when-task-is-scope).

    Each dimension is in ``0..1``. A *low* score on **any** dimension should tank
    the conjecture — a problem that is superficially relevant but messy or gamed
    must be rejected — so :attr:`overall` is the minimum dimension, matching the
    paper's "superficially related but inelegant is down-weighted" behaviour.
    """

    relevance: float
    conciseness: float
    naturalness: float

    @property
    def overall(self) -> float:
        """Overall quality = the weakest dimension (a single low score tanks it)."""
        return min(self.relevance, self.conciseness, self.naturalness)

    def accepted(self, threshold: float = 0.5) -> bool:
        """True if the conjecture clears ``threshold`` on its weakest dimension."""
        return self.overall >= threshold


def _heuristic_guide_score(target_task: str, generated_task: str) -> GuideScore:
    """Deterministic, LLM-free Guide rubric mirroring the SGS paper (CONCEPT:AU-AHE.harness.when-task-is-scope).

    - **relevance**: Jaccard lexical overlap between target and generated tokens —
      how on-target the generated problem is.
    - **conciseness**: penalise an over-long conclusion and redundant clause
      repetition (the paper's "average conclusion length increasing to ~10x" and
      "redundant premises" collapse signals).
    - **naturalness**: penalise complexity-padding — runaway sentence length,
      connective/disjunction spam, and contradictory/illogical cue words (the
      paper's disjunction-heavy, convoluted-statement collapse mode).
    """
    target_tokens = set(_tokens(target_task))
    gen_token_list = _tokens(generated_task)
    gen_tokens = set(gen_token_list)

    # relevance — Jaccard overlap (symmetric, bounded 0..1, deterministic).
    if not target_tokens and not gen_tokens:
        relevance = 1.0
    else:
        union = target_tokens | gen_tokens
        relevance = len(target_tokens & gen_tokens) / len(union) if union else 0.0

    # conciseness — short, non-repetitive conclusions score high.
    n_words = len(gen_token_list)
    length_penalty = max(0.0, (n_words - 12) / 48.0)  # free up to ~12 words
    distinct_ratio = (len(gen_tokens) / n_words) if n_words else 1.0
    redundancy_penalty = max(0.0, 0.7 - distinct_ratio)  # repeated clauses -> low
    conciseness = max(0.0, 1.0 - length_penalty - redundancy_penalty)

    # naturalness — penalise complexity-padding markers.
    lowered = f" {(generated_task or '').lower()} "
    connective_hits = sum(lowered.count(m) for m in _CONNECTIVE_MARKERS)
    illogical_hits = sum(lowered.count(m) for m in _ILLOGICAL_MARKERS)
    connective_penalty = max(0.0, (connective_hits - 1) * 0.2)  # 1 connective is fine
    illogical_penalty = illogical_hits * 0.5
    sentence_penalty = 0.3 if n_words > 30 else 0.0  # one runaway sentence
    naturalness = max(
        0.0, 1.0 - connective_penalty - illogical_penalty - sentence_penalty
    )

    return GuideScore(
        relevance=round(relevance, 6),
        conciseness=round(conciseness, 6),
        naturalness=round(naturalness, 6),
    )


class Guide:
    """SGS quality gatekeeper — rejects gamed/illogical conjectures (CONCEPT:AU-AHE.harness.when-task-is-scope).

    The Guide is the paper's third role: it scores a generated task against its
    target on relevance / conciseness / naturalness and rejects low-quality
    conjectures so they never reward the Conjecturer or train the Solver. The
    scorer is pluggable: the default is the deterministic heuristic above; a real
    deployment injects an LLM-backed scorer with the same signature.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        scorer: GuideScorer | None = None,
    ) -> None:
        self.threshold = threshold
        self._scorer = scorer or _heuristic_guide_score

    def evaluate(self, target_task: str, generated_task: str) -> GuideScore:
        """Score ``generated_task`` against ``target_task`` (injected scorer if given)."""
        return self._scorer(target_task, generated_task)

    def gate(self, target_task: str, generated_task: str) -> bool:
        """True if the conjecture is accepted at this Guide's threshold."""
        return self.evaluate(target_task, generated_task).accepted(self.threshold)


@dataclass(slots=True)
class PlayRound:
    """A single Conjecturer -> Guide -> Solver round.

    ``solved`` is ``None`` when the conjecture was rejected by the Guide and thus
    never handed to the Solver — the mechanism that denies the Conjecturer its
    reward hack and stops degenerate problems from advancing the curriculum.
    """

    round_index: int
    generated_task: str
    guide_score: GuideScore
    accepted: bool
    solved: bool | None
    difficulty: float = 0.0


@dataclass(slots=True)
class PlayReport:
    """Outcome of a self-play run."""

    rounds: list[PlayRound] = field(default_factory=list)
    solve_rate: float = 0.0
    accept_rate: float = 0.0
    plateaued: bool = False


class SelfGuidedSelfPlay:
    """Conjecturer -> Guide -> Solver loop with plateau breaking (CONCEPT:AU-AHE.harness.when-task-is-scope).

    Each round, the Conjecturer proposes a task at the current difficulty, the
    Guide gates it, and only **accepted** tasks are handed to the Solver. An
    accepted-and-solved task raises the difficulty by ``difficulty_step`` (the
    curriculum); rejected conjectures advance nothing — exactly the paper's defence
    against the Conjecturer gaming difficulty. A rolling solve-rate that fails to
    improve for ``plateau_patience`` rounds trips the plateau breaker, which
    perturbs the difficulty downward to widen the solvable band and escape the
    plateau (recorded as ``plateaued=True`` in the report).
    """

    def __init__(
        self,
        conjecture_fn: ConjectureFn,
        solve_fn: SolveFn,
        *,
        guide: Guide | None = None,
        plateau_patience: int = 3,
        difficulty_step: float = 0.1,
    ) -> None:
        if plateau_patience < 1:
            raise ValueError("plateau_patience must be >= 1")
        if not 0.0 < difficulty_step <= 1.0:
            raise ValueError("difficulty_step must be in (0, 1]")
        self._conjecture = conjecture_fn
        self._solve = solve_fn
        self._guide = guide or Guide()
        self._plateau_patience = plateau_patience
        self._difficulty_step = difficulty_step

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def run(
        self,
        target_task: str,
        *,
        rounds: int = 10,
        start_difficulty: float = 0.3,
    ) -> PlayReport:
        """Run ``rounds`` of self-play against ``target_task``.

        Returns a :class:`PlayReport` with per-round detail and the aggregate
        solve-rate (over accepted rounds), accept-rate (over all rounds), and
        whether the plateau breaker fired.
        """
        if rounds < 0:
            raise ValueError("rounds must be >= 0")

        difficulty = self._clamp(start_difficulty)
        played: list[PlayRound] = []
        accepted_count = 0
        solved_count = 0

        # Rolling solve-rate plateau tracking (AHE-3.x curriculum breaker).
        best_solve_rate = -1.0
        stall_streak = 0
        plateaued = False

        for i in range(rounds):
            generated = self._conjecture(target_task, difficulty)
            score = self._guide.evaluate(target_task, generated)
            accepted = score.accepted(self._guide.threshold)

            solved: bool | None
            if accepted:
                accepted_count += 1
                _solution, solved_ok = self._solve(generated)
                solved = bool(solved_ok)
                if solved:
                    solved_count += 1
                    # Curriculum: a solved, on-target conjecture raises difficulty.
                    difficulty = self._clamp(difficulty + self._difficulty_step)
            else:
                # Rejected conjectures never reach the Solver and never advance
                # the curriculum — the SGS defence against difficulty gaming.
                solved = None

            played.append(
                PlayRound(
                    round_index=i,
                    generated_task=generated,
                    guide_score=score,
                    accepted=accepted,
                    solved=solved,
                    difficulty=difficulty,
                )
            )

            # Plateau detection over the cumulative solve-rate (accepted rounds).
            current_rate = (solved_count / accepted_count) if accepted_count else 0.0
            if current_rate > best_solve_rate + 1e-9:
                best_solve_rate = current_rate
                stall_streak = 0
            else:
                stall_streak += 1
                if stall_streak >= self._plateau_patience:
                    # Break the plateau: perturb difficulty downward to widen the
                    # solvable band, and reset the stall counter to give it room.
                    plateaued = True
                    difficulty = self._clamp(difficulty - self._difficulty_step)
                    stall_streak = 0

        solve_rate = (solved_count / accepted_count) if accepted_count else 0.0
        accept_rate = (accepted_count / len(played)) if played else 0.0
        return PlayReport(
            rounds=played,
            solve_rate=round(solve_rate, 6),
            accept_rate=round(accept_rate, 6),
            plateaued=plateaued,
        )
