"""CONCEPT:AU-AHE.rlm.long-context-benchmark — RLM long-context benchmark harness (CPU, no live LLM)."""

from __future__ import annotations

import pytest

from agent_utilities.rlm.benchmarks import (
    list_tasks,
    render_scoreboard,
    run_benchmark,
)
from agent_utilities.rlm.benchmarks.base import BenchResult, get_task
from agent_utilities.rlm.benchmarks.baselines import (
    Completion,
    CompactionSystem,
    System,
    SystemOutput,
    VanillaSystem,
    chunk_text,
    head_tail_truncate,
)
from agent_utilities.rlm.benchmarks.cost import estimate_cost_usd, normalize_model

ALL_TASKS = ["s_niah", "oolong", "oolong_pairs", "browsecomp_plus", "longbench_codeqa"]


def test_all_tasks_registered():
    assert set(ALL_TASKS).issubset(set(list_tasks()))


@pytest.mark.parametrize("name", ALL_TASKS)
def test_task_builds_gradeable_case(name):
    task = get_task(name)
    case = task.build(20_000, seed=1)
    # Context is sizeable and the reference answer self-grades to 1.0.
    assert len(case.context) >= 10_000
    assert case.grade(case.answer) == 1.0
    # A clearly wrong answer does not score.
    assert case.grade("definitely not the answer zzz") == 0.0


@pytest.mark.parametrize("name", ALL_TASKS)
def test_task_is_deterministic(name):
    task = get_task(name)
    a = task.build(15_000, seed=7)
    b = task.build(15_000, seed=7)
    assert a.context == b.context and a.answer == b.answer


def test_s_niah_needle_present_and_recoverable():
    case = get_task("s_niah").build(40_000, seed=3)
    assert case.answer in case.context  # needle truly embedded
    assert case.grade(f"the passphrase is {case.answer}") == 1.0


def test_oolong_pairs_numeric_grading():
    case = get_task("oolong_pairs").build(12_000, seed=2)
    # numeric grader tolerates surrounding prose
    assert case.grade(f"The count is {case.answer}.") == 1.0
    assert case.grade("The count is 0.") == (1.0 if case.answer == "0" else 0.0)


def test_truncate_and_chunk():
    assert head_tail_truncate("abcdef", 100) == "abcdef"
    out = head_tail_truncate("x" * 1000, 100)
    assert "TRUNCATED" in out and len(out) < 1000
    assert chunk_text("abcdef", 2) == ["ab", "cd", "ef"]
    assert chunk_text("", 10) == [""]


def test_cost_model():
    assert normalize_model("openai:gpt-4o-mini") == "gpt-4o-mini"
    assert estimate_cost_usd(1000, "openai:gpt-4o-mini") == pytest.approx(0.0004)
    # unknown model falls back to a non-zero default
    assert estimate_cost_usd(1000, "mystery:model-x") > 0


class _CheatCompleter:
    """Completer that returns the case answer — exercises the system plumbing + grading."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    async def complete(self, system: str, user: str, *, model_id: str) -> Completion:
        self.calls += 1
        return Completion(text=f"The answer is {self.answer}.", tokens=120)


async def test_vanilla_system_plumbing():
    case = get_task("s_niah").build(8_000, seed=0)
    sys = VanillaSystem("openai:gpt-4o-mini", completer=_CheatCompleter(case.answer))
    out = await sys.answer(case)
    assert case.grade(out.prediction) == 1.0
    assert out.tokens == 120 and out.cost_usd > 0


async def test_compaction_system_summarizes_then_answers():
    case = get_task("oolong").build(20_000, seed=0)
    completer = _CheatCompleter(case.answer)
    sys = CompactionSystem(chunk_chars=4_000, completer=completer)
    out = await sys.answer(case)
    # one summarize call per chunk + one final answer call
    assert completer.calls >= 2
    assert out.cost_usd > 0


class _FakeSystem(System):
    def __init__(self, name: str, correct: bool):
        self.name = name
        self.correct = correct

    async def answer(self, case) -> SystemOutput:
        pred = case.answer if self.correct else "wrong"
        return SystemOutput(system=self.name, prediction=pred, tokens=50, cost_usd=0.01)


async def test_run_benchmark_aggregates():
    results = await run_benchmark(
        "browsecomp_plus",
        scales=[6_000],
        systems=[_FakeSystem("rlm", True), _FakeSystem("vanilla", False)],
        cases_per_scale=4,
    )
    assert len(results) == 2
    by = {r.system: r for r in results}
    assert by["rlm"].accuracy == 1.0
    assert by["vanilla"].accuracy == 0.0
    assert by["rlm"].n == 4 and by["rlm"].cost_usd == pytest.approx(0.01)


async def test_run_benchmark_survives_system_errors():
    class _BoomSystem(System):
        name = "boom"

        async def answer(self, case):
            raise RuntimeError("kaboom")

    results = await run_benchmark(
        "s_niah", scales=[5_000], systems=[_BoomSystem()], cases_per_scale=2
    )
    assert results[0].accuracy == 0.0 and "error" in results[0].notes


def test_scoreboard_renders_paper_comparison():
    results = [
        BenchResult(
            task="oolong",
            complexity="O(n)",
            system="rlm",
            scale=50_000,
            accuracy=0.9,
            n=3,
            cost_usd=0.5,
            mode="synthetic",
        )
    ]
    md = render_scoreboard(results)
    assert "oolong" in md
    assert "Paper RLM" in md and "56.0%" in md  # paper number present
    assert "not run" in md  # external baselines flagged
    assert "synthetic" in md
