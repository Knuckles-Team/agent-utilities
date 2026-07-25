"""Regression: context compaction never orphans a tool-call/tool-return pair.

CONCEPT:AU-KG.memory.mementified-context (tool-pair safety). A compacted history that keeps a
``ToolCallPart`` but drops its matching ``ToolReturnPart`` (or the reverse) is rejected
by an OpenAI/vLLM-compatible provider with HTTP 400 mid-run. The live Memento sawtooth
(``MementoCompaction``) evicts semantic blocks whose boundaries can fall between a call
and its return, so it must route every eviction through
``capabilities.compaction._shared.enforce_tool_pair_safety``.
"""

from __future__ import annotations

import sys
import types

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agent_utilities.capabilities.compaction._shared import (
    enforce_tool_pair_safety,
    iter_tool_pairs,
)


def _install_numeric_stub() -> None:
    """Make ``agent_utilities.knowledge_graph.memory`` importable without the compiled
    ``epistemic_graph.numeric`` kernel.

    The kernel ships only in the ``epistemic-graph[full]`` wheel; a lean/CI checkout may
    lack it. The compaction logic under test is pure Python, so a numpy-backed stand-in
    lets the memory package import. It is a NO-OP when the real kernel is present (every
    full install and the pipeline job), so it never shadows the certified kernel there.
    """
    try:  # real kernel present -> nothing to do
        import epistemic_graph.numeric  # noqa: F401

        return
    except ImportError:
        pass
    import epistemic_graph
    import numpy

    stub = types.ModuleType("epistemic_graph.numeric")
    stub.__kernel__ = "eg-numeric"  # type: ignore[attr-defined]
    stub.ndarray = numpy.ndarray  # type: ignore[attr-defined]
    stub.LinAlgError = numpy.linalg.LinAlgError  # type: ignore[attr-defined]
    stub.__getattr__ = lambda name, _np=numpy: getattr(_np, name)  # type: ignore[attr-defined]
    sys.modules["epistemic_graph.numeric"] = stub
    epistemic_graph.numeric = stub  # type: ignore[attr-defined]


def _tool_ids(messages: list[ModelMessage]) -> tuple[set[str], set[str]]:
    calls: set[str] = set()
    returns: set[str] = set()
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart) and part.tool_call_id:
                calls.add(part.tool_call_id)
            elif isinstance(part, ToolReturnPart) and part.tool_call_id:
                returns.add(part.tool_call_id)
    return calls, returns


def _assert_no_orphans(messages: list[ModelMessage]) -> None:
    calls, returns = _tool_ids(messages)
    assert calls == returns, f"orphaned tool ids: calls={calls} returns={returns}"


def _history_with_pair() -> list[ModelMessage]:
    """head system+user, a tool call/return pair (idx 1/2), then more turns."""
    return [
        ModelRequest(
            parts=[SystemPromptPart(content="sys"), UserPromptPart(content="start")]
        ),
        ModelResponse(
            parts=[ToolCallPart(tool_name="t", args={"q": 1}, tool_call_id="c1")]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="t", content="result-1", tool_call_id="c1")]
        ),
        ModelResponse(parts=[TextPart(content="thinking")]),
        ModelRequest(parts=[UserPromptPart(content="next")]),
    ]


class TestToolPairSafetyPrimitive:
    def test_iter_tool_pairs_matches_by_id(self) -> None:
        assert iter_tool_pairs(_history_with_pair()) == [(1, 2)]

    def test_evicting_only_the_call_side_unevicts_both(self) -> None:
        # Dropping the call but keeping the return would orphan the return.
        assert enforce_tool_pair_safety(_history_with_pair(), {1}) == set()

    def test_evicting_only_the_return_side_unevicts_both(self) -> None:
        assert enforce_tool_pair_safety(_history_with_pair(), {2}) == set()

    def test_evicting_both_sides_is_kept(self) -> None:
        assert enforce_tool_pair_safety(_history_with_pair(), {1, 2}) == {1, 2}

    def test_result_is_always_a_subset(self) -> None:
        # An unrelated non-pair message (idx 3, a plain text response) stays evictable.
        assert enforce_tool_pair_safety(_history_with_pair(), {1, 3}) == {3}

    def test_no_pairs_passthrough(self) -> None:
        msgs: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="x")]),
            ModelResponse(parts=[TextPart(content="y")]),
        ]
        assert enforce_tool_pair_safety(msgs, {0, 1}) == {0, 1}

    def test_parallel_calls_resolve_transitively(self) -> None:
        # One response fires two parallel calls; returns land in two later requests.
        # Evicting the shared call-message plus only one return must un-evict the whole
        # component (fixed-point), never leaving c2 half-evicted.
        msgs: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="t", args={}, tool_call_id="c1"),
                    ToolCallPart(tool_name="t", args={}, tool_call_id="c2"),
                ]
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="t", content="r1", tool_call_id="c1")]
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="t", content="r2", tool_call_id="c2")]
            ),
        ]
        assert iter_tool_pairs(msgs) == [(0, 1), (0, 2)]
        # Plan evicts the call message (0) and the first return (1) but not the second (2).
        assert enforce_tool_pair_safety(msgs, {0, 1}) == set()


class TestMementoCompactionToolPairSafety:
    def test_mementoize_never_orphans_when_plan_splits_a_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_numeric_stub()
        from agent_utilities.capabilities.memento import MementoCompaction
        from agent_utilities.knowledge_graph.memory import agent_context
        from agent_utilities.knowledge_graph.memory import memento_compressor as mc

        msgs = _history_with_pair()
        # Force the over-budget trigger, then hand back a plan that evicts ONLY the
        # tool-call message (idx 1). Without the safety filter this replaces the call
        # with a memento while the return at idx 2 survives -> orphaned return -> 400.
        monkeypatch.setattr(agent_context, "estimate_message_tokens", lambda _d: 10_000)
        monkeypatch.setattr(
            mc, "plan_block_eviction", lambda _dicts, **_kw: ([[1]], [0, 2, 3, 4])
        )

        new_msgs, n_evicted = MementoCompaction().mementoize_messages(
            msgs, budget_tokens=100, engine=None
        )

        _assert_no_orphans(new_msgs)
        # The split pair was un-evicted (kept raw); nothing was compacted this pass.
        assert n_evicted == 0
        assert new_msgs == msgs

    def test_mementoize_evicts_when_plan_keeps_the_pair_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_numeric_stub()
        from agent_utilities.capabilities.memento import MementoCompaction
        from agent_utilities.knowledge_graph.memory import agent_context
        from agent_utilities.knowledge_graph.memory import memento_compressor as mc

        msgs = _history_with_pair()
        monkeypatch.setattr(agent_context, "estimate_message_tokens", lambda _d: 10_000)
        # A safe plan evicts BOTH halves of the pair (idx 1 and 2) as one block.
        monkeypatch.setattr(
            mc, "plan_block_eviction", lambda _dicts, **_kw: ([[1, 2]], [0, 3, 4])
        )
        monkeypatch.setattr(
            mc, "compress_to_memento", lambda *_a, **_k: "PRIOR-BLOCK-MEMENTO"
        )

        new_msgs, n_evicted = MementoCompaction().mementoize_messages(
            msgs, budget_tokens=100, engine=None
        )

        _assert_no_orphans(new_msgs)
        assert n_evicted == 2
        # Both halves were compacted into the memento; neither survives raw.
        calls, returns = _tool_ids(new_msgs)
        assert "c1" not in calls and "c1" not in returns
