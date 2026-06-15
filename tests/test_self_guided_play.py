"""Tests for CONCEPT:AHE-3.37 Self-Guided Self-Play (SGS, arXiv:2604.20209).

All callables are deterministic stubs — no LLM, no network.
"""

from __future__ import annotations

from agent_utilities.harness.self_guided_play import (
    Guide,
    GuideScore,
    PlayReport,
    SelfGuidedSelfPlay,
)

TARGET = "prove the triangle inequality for side lengths a b c"

# An on-target, concise, natural conjecture — a clean simpler stepping stone.
GOOD_TASK = "prove the inequality for side lengths a and b"

# The paper's collapse failure mode: superficially related but messy — irrelevant
# padding, an over-long conclusion, and disjunction/connective spam (illogical
# complexity padding that games the difficulty reward).
GAMED_TASK = (
    "prove zebra orbit quantum or widget or gadget or sprocket or flange and "
    "moreover the lemma holds or fails and furthermore it is trivially false "
    "or vacuously true except when contradiction arises then it holds or not"
)


# ── Guide.evaluate heuristic ─────────────────────────────────────────


def test_good_task_scores_high_and_accepted() -> None:
    guide = Guide()
    score = guide.evaluate(TARGET, GOOD_TASK)
    assert isinstance(score, GuideScore)
    assert score.relevance > 0.4
    assert score.conciseness > 0.8
    assert score.naturalness > 0.8
    assert score.accepted() is True
    assert guide.gate(TARGET, GOOD_TASK) is True


def test_gamed_task_scores_low_and_rejected() -> None:
    guide = Guide()
    score = guide.evaluate(TARGET, GAMED_TASK)
    # Naturalness must collapse under connective spam + illogical cue words.
    assert score.naturalness < 0.3
    # overall = weakest dimension -> a single low dim tanks the whole conjecture.
    assert score.overall == min(
        score.relevance, score.conciseness, score.naturalness
    )
    assert score.accepted() is False
    assert guide.gate(TARGET, GAMED_TASK) is False


def test_overall_is_min_dimension() -> None:
    score = GuideScore(relevance=0.9, conciseness=0.2, naturalness=0.95)
    assert score.overall == 0.2
    assert score.accepted(threshold=0.5) is False
    assert score.accepted(threshold=0.1) is True


# ── Loop mechanics: rejection denies the curriculum advance ──────────


def _solver_always(ok: bool) -> object:
    def _solve(task: str) -> tuple[str, bool]:
        return (f"solution::{task}", ok)

    return _solve


def test_rejected_conjectures_not_solved_and_dont_advance_difficulty() -> None:
    """A Conjecturer that only emits gamed tasks: nothing is solved, no curriculum."""
    solve_calls: list[str] = []

    def solve(task: str) -> tuple[str, bool]:
        solve_calls.append(task)
        return ("sol", True)

    sgs = SelfGuidedSelfPlay(
        conjecture_fn=lambda target, diff: GAMED_TASK,
        solve_fn=solve,
        difficulty_step=0.1,
        plateau_patience=100,  # isolate the rejection property from the breaker
    )
    report = sgs.run(TARGET, rounds=5, start_difficulty=0.3)

    # The Guide rejected every conjecture -> Solver was never called.
    assert solve_calls == []
    assert all(r.accepted is False for r in report.rounds)
    assert all(r.solved is None for r in report.rounds)
    # No accepted+solved conjecture -> the curriculum never advanced upward.
    assert all(r.difficulty == 0.3 for r in report.rounds)
    assert report.accept_rate == 0.0
    assert report.solve_rate == 0.0


def test_accepted_and_solved_raises_difficulty_curriculum() -> None:
    """Good, solved conjectures climb the curriculum by difficulty_step each round."""
    sgs = SelfGuidedSelfPlay(
        conjecture_fn=lambda target, diff: GOOD_TASK,
        solve_fn=_solver_always(True),  # type: ignore[arg-type]
        difficulty_step=0.1,
        plateau_patience=100,  # don't let the breaker interfere
    )
    report = sgs.run(TARGET, rounds=4, start_difficulty=0.3)

    assert all(r.accepted for r in report.rounds)
    assert all(r.solved is True for r in report.rounds)
    difficulties = [round(r.difficulty, 4) for r in report.rounds]
    assert difficulties == [0.4, 0.5, 0.6, 0.7]
    assert report.accept_rate == 1.0
    assert report.solve_rate == 1.0


def test_difficulty_clamped_at_one() -> None:
    sgs = SelfGuidedSelfPlay(
        conjecture_fn=lambda target, diff: GOOD_TASK,
        solve_fn=_solver_always(True),  # type: ignore[arg-type]
        difficulty_step=0.5,
        plateau_patience=100,
    )
    report = sgs.run(TARGET, rounds=5, start_difficulty=0.5)
    assert all(0.0 <= r.difficulty <= 1.0 for r in report.rounds)
    assert report.rounds[-1].difficulty == 1.0


# ── Plateau breaking ─────────────────────────────────────────────────


def test_plateau_break_fires_when_solve_rate_stalls() -> None:
    """Accepted but never solved -> solve-rate stuck at 0 -> breaker fires."""
    sgs = SelfGuidedSelfPlay(
        conjecture_fn=lambda target, diff: GOOD_TASK,
        solve_fn=_solver_always(False),  # type: ignore[arg-type]
        plateau_patience=3,
        difficulty_step=0.1,
    )
    report = sgs.run(TARGET, rounds=6, start_difficulty=0.5)
    assert all(r.accepted for r in report.rounds)
    assert all(r.solved is False for r in report.rounds)
    assert report.plateaued is True
    # The breaker perturbed difficulty downward below the start.
    assert report.rounds[-1].difficulty < 0.5


def test_no_plateau_on_steady_improvement() -> None:
    """A solve-rate that keeps improving must NOT trip the breaker.

    Pattern: first round fails (rate 0), then all succeed (rate climbs each
    accepted round toward 1) so best_solve_rate strictly improves and never
    stalls for plateau_patience rounds.
    """
    seq = iter([False, True, True, True, True, True, True])

    def solve(task: str) -> tuple[str, bool]:
        return ("sol", next(seq))

    sgs = SelfGuidedSelfPlay(
        conjecture_fn=lambda target, diff: GOOD_TASK,
        solve_fn=solve,
        plateau_patience=3,
        difficulty_step=0.05,
    )
    report = sgs.run(TARGET, rounds=7, start_difficulty=0.3)
    assert report.plateaued is False
    # 6 of 7 solved among 7 accepted.
    assert report.accept_rate == 1.0
    assert report.solve_rate == round(6 / 7, 6)


# ── Injected scorer ──────────────────────────────────────────────────


def test_injected_scorer_is_used() -> None:
    seen: list[tuple[str, str]] = []

    def scorer(target: str, generated: str) -> GuideScore:
        seen.append((target, generated))
        return GuideScore(relevance=0.95, conciseness=0.95, naturalness=0.95)

    guide = Guide(threshold=0.5, scorer=scorer)
    # Even the gamed task is accepted because the injected scorer overrides heuristics.
    assert guide.gate(TARGET, GAMED_TASK) is True
    assert seen == [(TARGET, GAMED_TASK)]


# ── Report rates ─────────────────────────────────────────────────────


def test_play_report_rates_computed_correctly() -> None:
    """Mixed accept/solve: rates are over the right denominators."""
    # Conjecturer alternates good (accepted) and gamed (rejected) tasks.
    tasks = iter([GOOD_TASK, GAMED_TASK, GOOD_TASK, GAMED_TASK])
    solved = iter([True, False])  # only good tasks reach the solver

    def conjecture(target: str, diff: float) -> str:
        return next(tasks)

    def solve(task: str) -> tuple[str, bool]:
        return ("sol", next(solved))

    sgs = SelfGuidedSelfPlay(
        conjecture_fn=conjecture,
        solve_fn=solve,
        plateau_patience=100,
    )
    report = sgs.run(TARGET, rounds=4, start_difficulty=0.3)

    assert isinstance(report, PlayReport)
    accepted = [r for r in report.rounds if r.accepted]
    assert len(accepted) == 2  # the two GOOD tasks
    assert report.accept_rate == 0.5  # 2 of 4 rounds accepted
    assert report.solve_rate == 0.5  # 1 of 2 accepted rounds solved
    rejected = [r for r in report.rounds if not r.accepted]
    assert all(r.solved is None for r in rejected)


# ── Determinism ──────────────────────────────────────────────────────


def test_deterministic_runs_are_identical() -> None:
    def build() -> SelfGuidedSelfPlay:
        tasks = iter([GOOD_TASK, GAMED_TASK] * 5)
        results = iter([True, False, True, False, True])

        def conjecture(target: str, diff: float) -> str:
            return next(tasks)

        def solve(task: str) -> tuple[str, bool]:
            return ("sol", next(results))

        return SelfGuidedSelfPlay(conjecture_fn=conjecture, solve_fn=solve)

    r1 = build().run(TARGET, rounds=6, start_difficulty=0.3)
    r2 = build().run(TARGET, rounds=6, start_difficulty=0.3)
    assert r1 == r2


def test_guide_heuristic_is_deterministic() -> None:
    guide = Guide()
    a = guide.evaluate(TARGET, GAMED_TASK)
    b = guide.evaluate(TARGET, GAMED_TASK)
    assert a == b
