"""CONCEPT:AHE-3.32 — live RLM benchmark + 10M-token stress (requires an LLM endpoint).

Gated by the ``live`` marker (skipped by the default ``-m "not live"``). Run explicitly with a
configured model endpoint:  ``pytest -m live tests/unit/rlm/test_ahe_3_32_benchmark_live.py``.
"""

from __future__ import annotations

import os

import pytest

from agent_utilities.rlm import RLM
from agent_utilities.rlm.benchmarks import render_scoreboard, run_benchmark
from agent_utilities.rlm.benchmarks.base import get_task
from agent_utilities.rlm.benchmarks.baselines import RLMSystem, VanillaSystem

pytestmark = pytest.mark.live

# These tests hit a real LLM endpoint (and generate up to ~40M chars). The repo's pytest pre-commit
# hook runs with ``-m "not slow"`` (not ``-m "not live"``), so gate them behind an explicit opt-in
# to keep the hook/CI green without an endpoint:  RLM_LIVE_BENCH=1 pytest -m live <this file>.
if not os.environ.get("RLM_LIVE_BENCH"):
    pytest.skip(
        "set RLM_LIVE_BENCH=1 to run live RLM benchmark tests", allow_module_level=True
    )


async def test_rlm_vs_vanilla_oolong_pairs():
    # Quadratic aggregation: the regime where the paper shows base models near 0 and RLM wins big.
    results = await run_benchmark(
        "oolong_pairs",
        scales=[80_000],
        systems=[RLMSystem(), VanillaSystem(window_chars=40_000)],
        cases_per_scale=3,
    )
    board = render_scoreboard(results)
    assert "oolong_pairs" in board
    by = {r.system: r for r in results}
    # RLM (programmatic) should not be worse than vanilla truncation on a quadratic task.
    assert by["rlm"].accuracy >= by["vanilla"].accuracy


async def test_10m_token_stress_needle():
    # ~40M chars ≈ 10M tokens — two orders of magnitude beyond a native context window.
    case = get_task("s_niah").build(40_000_000, seed=11)
    assert case.meta["context_chars"] >= 40_000_000
    out = await RLMSystem().answer(case)
    assert case.grade(out.prediction) == 1.0
    assert out.tokens > 0  # usage was captured for the cost column


async def test_dropin_completion_smoke():
    rlm = RLM(backend="openai", backend_kwargs={"model_name": "gpt-4o-mini"})
    resp = await rlm.acompletion(
        "Find the secret code 778899 buried here: " + ("filler " * 5000) + " code=778899",
    )
    assert resp.ok
