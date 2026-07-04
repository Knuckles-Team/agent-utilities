"""Agentic tool-judge over large/complex traces (CONCEPT:AU-AHE.harness.instead-context-stuffing-small)."""

from __future__ import annotations

from agent_utilities.harness import tool_judge as tj
from agent_utilities.harness.online_scoring import AutomationRule, OnlineScoringSampler
from agent_utilities.harness.trace_backend import KGTraceBackend


def _entry(output: str, n_spans: int = 0):
    be = KGTraceBackend()
    for i in range(n_spans):
        be.record_event(trace_id="x", span_id=f"x:s{i}", name=f"step{i}", is_root=False)
    be.record_event(
        trace_id="x",
        span_id="x:r",
        name="run",
        is_root=True,
        input_text="q",
        output_text=output,
    )
    return be, be.get_trace("x")


def test_should_use_threshold():
    _, small = _entry("short", n_spans=2)
    _, big = _entry("ok", n_spans=20)  # deep run → many spans → navigate with tools
    assert tj.should_use(small) is False
    assert tj.should_use(big) is True


def test_sampler_routes_complex_traces_to_tool_judge(monkeypatch):
    used = {"tool": 0, "inline": 0}
    monkeypatch.setattr(
        tj.ToolEnabledJudge,
        "judge",
        lambda self, entry, criteria: (
            used.__setitem__("tool", used["tool"] + 1) or (1.0, "tool")
        ),
    )
    be, _ = _entry("ok", n_spans=20)

    def inline(criteria, q, a):
        used["inline"] += 1
        return (1.0, "inline")

    sampler = OnlineScoringSampler(
        backend=be, rules=[AutomationRule("d", "c")], judge=inline
    )
    sampler.score_trace("x")
    assert used["tool"] == 1 and used["inline"] == 0


def test_sampler_uses_inline_for_small_traces(monkeypatch):
    monkeypatch.setattr(
        tj.ToolEnabledJudge,
        "judge",
        lambda self, e, c: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    be, _ = _entry("short answer", n_spans=1)
    used = {"inline": 0}

    def inline(criteria, q, a):
        used["inline"] += 1
        return (1.0, "inline")

    sampler = OnlineScoringSampler(
        backend=be, rules=[AutomationRule("d", "c")], judge=inline
    )
    sampler.score_trace("x")
    assert used["inline"] == 1
