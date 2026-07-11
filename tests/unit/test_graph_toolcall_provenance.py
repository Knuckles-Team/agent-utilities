"""ToolCall provenance on the multi-agent graph path (F3).

CONCEPT:AU-KG.temporal.message-history-read — the direct single-server loop surfaced per-tool-call
provenance, but a delegation routed through the multi-agent graph wrote ZERO
``:ToolCall`` nodes. These tests pin the wire: the shared extractor reads the
pydantic-ai history, ``GraphState`` accumulates per-node tool calls, and the
``GraphResponse`` carries them through ``model_dump`` — which is exactly what
``run_agent``'s persist gate reads (``result.get("tool_calls")``).
"""

from __future__ import annotations

from agent_utilities.graph.state import GraphState
from agent_utilities.models.graph import GraphResponse
from agent_utilities.orchestration.tool_provenance import (
    extract_tool_calls,
    sanitize_tool_args,
)


class _ToolCallPart:
    part_kind = "tool-call"

    def __init__(self, name, args, tcid):
        self.tool_name = name
        self.args = args
        self.tool_call_id = tcid


class _ToolReturnPart:
    part_kind = "tool-return"

    def __init__(self, tcid, content):
        self.tool_call_id = tcid
        self.content = content


class _Msg:
    def __init__(self, parts):
        self.parts = parts


class _RunResult:
    def __init__(self, messages):
        self._messages = messages

    def all_messages(self):
        return self._messages


def test_extract_tool_calls_pairs_call_with_return():
    res = _RunResult(
        [
            _Msg([_ToolCallPart("cm_docker_ps", {"context": "local"}, "t1")]),
            _Msg([_ToolReturnPart("t1", "web, db, cache")]),
        ]
    )
    tcs = extract_tool_calls(res)
    assert len(tcs) == 1
    assert tcs[0]["tool_name"] == "cm_docker_ps"
    assert tcs[0]["result"] == "web, db, cache"
    assert tcs[0]["error"] == ""


def test_extract_tool_calls_best_effort_on_junk():
    class NoHistory:
        pass

    assert extract_tool_calls(NoHistory()) == []
    assert extract_tool_calls(object()) == []


def test_sanitize_tool_args_redacts_secrets():
    out = sanitize_tool_args({"token": "hunter2", "context": "local"})
    assert "hunter2" not in out
    assert "local" in out


def test_graph_state_accumulates_tool_calls():
    state = GraphState(query="list containers")
    assert state.tool_calls == []
    state.tool_calls.extend(
        extract_tool_calls(
            _RunResult([_Msg([_ToolCallPart("cm_docker_ps", {}, "t1")])])
        )
    )
    assert state.tool_calls and state.tool_calls[0]["tool_name"] == "cm_docker_ps"


def test_graph_response_carries_tool_calls_through_model_dump():
    """run_agent's persist gate reads result['tool_calls'] on the dumped response."""
    tcs = [{"tool_name": "cm_docker_ps", "args": "{}", "result": "ok", "error": ""}]
    dumped = GraphResponse(status="completed", tool_calls=tcs).model_dump()
    assert dumped["tool_calls"] == tcs  # what agent_runner._persist_tool_calls consumes


def test_engine_injection_semantics_fill_empty_from_state():
    """Mirror the engine injection: an empty response inherits the state's tool calls."""
    state = GraphState(query="q")
    state.tool_calls = [
        {"tool_name": "cm_docker_ps", "args": "{}", "result": "ok", "error": ""}
    ]
    result = GraphResponse(status="completed")  # graph built it without tool_calls
    # exact condition from orchestration/engine.py
    if not result.tool_calls and getattr(state, "tool_calls", None):
        result.tool_calls = list(state.tool_calls)
    assert result.model_dump()["tool_calls"] == state.tool_calls
