"""EH-463 / ST-11: a running plan only continues, narrows or stops; stopping releases.

The committed stop rule is evaluated over L5 observations (execution, not
decision); the continuation question is evaluate-only with narrow-only options.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from agent_utilities.decide.consumers.continuation import (
    PLAN_METADATA_KEY,
    RoundProgress,
    after_wave,
    continuation_options,
    continue_or_stop,
    install_release,
    stop_reason,
)
from agent_utilities.decide.points import POINTS, LogMode
from tests.unit.decide.fakes import FakeTransport, acted

RECORD = "decision:" + "ab" * 32


@pytest.fixture
def released() -> Iterator[list[str]]:
    seen: list[str] = []

    async def release(record_id: str) -> None:
        seen.append(record_id)

    install_release(release)
    yield seen
    install_release(None)


@pytest.mark.parametrize(
    ("rule", "progress", "reason"),
    [
        ({"rule": "max_rounds", "n": 2}, RoundProgress(rounds_done=1), None),
        ({"rule": "max_rounds", "n": 2}, RoundProgress(rounds_done=2), "max_rounds"),
        ({"rule": "quorum", "k": 2, "n": 3}, RoundProgress(1, votes=2), "quorum"),
        (
            {"rule": "verifier_pass", "max_rounds": 4},
            RoundProgress(1, True),
            "verifier_pass",
        ),
        ({"rule": "verifier_pass", "max_rounds": 4}, RoundProgress(4), "verifier_pass"),
        (
            {"rule": "budget"},
            RoundProgress(1, tokens_spent=9000, token_ceiling=8000),
            "budget",
        ),
        (
            {"rule": "max_rounds", "n": 5},
            RoundProgress(1, elapsed_ms=10, deadline_ms=10),
            "deadline",
        ),
        ({"rule": "deadline"}, RoundProgress(3), None),
        ({"rule": "novel"}, RoundProgress(1), "unknown_rule:novel"),
    ],
)
def test_the_committed_stop_rule_is_executed_over_observations(
    rule: dict[str, Any], progress: RoundProgress, reason: str | None
) -> None:
    assert stop_reason(rule, progress) == reason


def test_the_options_only_narrow() -> None:
    assert [o.option_id for o in continuation_options(3)] == [
        "continue",
        "narrow",
        "stop",
    ]
    assert [o.option_id for o in continuation_options(1)] == ["continue", "stop"]
    widths = [o.numbers["width"] for o in continuation_options(3)]
    assert max(widths) == 3.0, "no option widens past the running width"


def test_the_point_is_evaluate_only_and_sampled() -> None:
    point = POINTS["au.swarm.continue"]
    assert point.log_mode is LogMode.SAMPLED and point.row == "EH-463"
    assert POINTS["au.swarm.topology"].log_mode is LogMode.NEVER


def test_a_met_rule_stops_and_releases_without_asking(
    eg: FakeTransport, released: list[str]
) -> None:
    step = asyncio.run(
        continue_or_stop(RECORD, {"rule": "max_rounds", "n": 1}, RoundProgress(1), 3)
    )
    assert (step.action, step.reason) == ("stop", "max_rounds")
    assert released == [RECORD]
    assert eg.requests == [], "execution, not decision"


def test_eg_may_narrow_and_the_question_cites_the_parent_plan(
    eg: FakeTransport, released: list[str]
) -> None:
    eg.answer = acted("narrow")
    step = asyncio.run(
        continue_or_stop(RECORD, {"rule": "max_rounds", "n": 4}, RoundProgress(1), 3)
    )
    assert step.action == "narrow"
    assert released == [], "narrowing keeps the per-cell leases until the stop"
    [request] = eg.requests
    assert RECORD in str(request), "the parent record is a request parameter"


def test_without_a_runner_the_run_continues(released: list[str]) -> None:
    step = asyncio.run(
        continue_or_stop(RECORD, {"rule": "max_rounds", "n": 4}, RoundProgress(1), 2)
    )
    assert step.action == "continue" and released == []


def test_the_parallel_engine_hook_governs_only_plan_runs(
    eg: FakeTransport, released: list[str]
) -> None:
    waves: list[list[Any]] = [["a"], ["b", "c", "d"], ["e", "f"]]
    assert asyncio.run(after_wave({}, 0, waves)) is True
    plan = {
        PLAN_METADATA_KEY: {"record_id": RECORD, "stop": {"rule": "max_rounds", "n": 9}}
    }
    eg.answer = acted("narrow")
    assert asyncio.run(after_wave(plan, 0, waves)) is True
    assert waves == [["a"], ["b", "c"], ["e"]], "narrow drops one agent per wave"
    eg.answer = acted("stop")
    assert asyncio.run(after_wave(plan, 1, waves)) is False
    assert released == [RECORD]


def test_the_last_wave_of_a_plan_run_releases_its_leases(released: list[str]) -> None:
    plan = {
        PLAN_METADATA_KEY: {"record_id": RECORD, "stop": {"rule": "max_rounds", "n": 9}}
    }
    assert asyncio.run(after_wave(plan, 1, [["a"], ["b"]])) is True
    assert released == [RECORD]
