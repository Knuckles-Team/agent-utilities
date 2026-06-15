"""Live-path: the distiller applies research-craft discipline (AHE-3.34/35/36).

Wire-First proof — `TraceDistiller.distill` forecasts+resolves each round's score
(AHE-3.34), baseline-gates round-over-round (AHE-3.35), and triages failures into
piles + records (dis)confirming evidence (AHE-3.36).
"""

import pytest

from agent_utilities.harness.continuous_evaluation_engine import (
    DistillationConfig,
    TraceDistiller,
)


class _StubBackend:
    """Minimal TraceBackend returning canned, scored traces per round."""

    def __init__(self, traces_by_round):
        self._t = traces_by_round

    async def get_traces(self, round_id):
        return self._t.get(round_id, [])

    async def store_evidence(self, *a, **k):
        return None


def _traces(scores):
    return [
        {"id": f"t{i}", "name": f"task{i}", "score": s, "error": "" if s >= 0.6 else "boom"}
        for i, s in enumerate(scores)
    ]


@pytest.mark.asyncio
async def test_distill_applies_research_craft():
    backend = _StubBackend(
        {
            "r1": _traces([0.9, 0.8, 0.2, 0.1]),  # 50% pass
            "r2": _traces([0.2, 0.1, 0.1, 0.1]),  # regression vs r1
        }
    )
    d = TraceDistiller(backend, config=DistillationConfig())

    c1 = await d.distill("r1")
    # AHE-3.34: a forecast for r1 was registered and resolved to its score.
    assert d.forecasts.summary()["resolved"] == 1
    # AHE-3.36: the round outcome was recorded as a belief entry.
    assert d.research_log.balance("round r1 approach is sound")["supports"] >= 1

    c2 = await d.distill("r2")
    # AHE-3.34: two rounds now resolved; calibration is accumulating.
    assert d.forecasts.summary()["resolved"] == 2
    # AHE-3.35: r2's score is below r1's baseline (regression path exercised).
    assert c2.benchmark_score < c1.benchmark_score
    # AHE-3.36: r2 (low pass-rate) recorded as disconfirming evidence.
    assert d.research_log.balance("round r2 approach is sound")["refutes"] >= 1
