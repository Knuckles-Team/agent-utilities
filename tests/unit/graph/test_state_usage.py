"""``GraphState._update_usage`` cache/reasoning token accumulation (D-54c-1).

``_update_usage`` is the ONE live call site every specialist node feeds
(``graph/executor.py::specialized_step``: ``ctx.state._update_usage(stream.usage)``),
and ``orchestration/engine.py::AgentOrchestrationEngine.execute_graph`` reads its
accumulated ``state.session_usage`` to build the ``token_usage`` dict that feeds
``usage/recorder.py``, the Langfuse exporter, and the OTel span — all of which were
previously fed a permanently-empty dict because nothing populated
``GraphResponse.metadata["token_usage"]``. This covers the accumulator itself,
including both pydantic-ai's ``RunUsage`` field names and the raw
Anthropic-native fallback names.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_utilities.graph.state import GraphState


def _state() -> GraphState:
    return GraphState(query="q")


def test_update_usage_accumulates_pydantic_ai_cache_and_reasoning_tokens():
    state = _state()
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=40,
        total_tokens=140,
        cache_write_tokens=25,
        cache_read_tokens=10,
        details={"reasoning_tokens": 7},
    )
    state._update_usage(usage)

    assert state.session_usage.input_tokens == 100
    assert state.session_usage.output_tokens == 40
    assert state.session_usage.cache_creation_input_tokens == 25
    assert state.session_usage.cache_read_input_tokens == 10
    assert state.session_usage.reasoning_tokens == 7


def test_update_usage_falls_back_to_anthropic_native_names():
    state = _state()
    usage = SimpleNamespace(
        input_tokens=50,
        output_tokens=20,
        total_tokens=70,
        cache_creation_input_tokens=12,
        cache_read_input_tokens=6,
    )
    state._update_usage(usage)

    assert state.session_usage.cache_creation_input_tokens == 12
    assert state.session_usage.cache_read_input_tokens == 6
    assert state.session_usage.reasoning_tokens == 0


def test_update_usage_reads_gemini_style_thoughts_tokens():
    state = _state()
    usage = SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        details={"thoughts_tokens": 9},
    )
    state._update_usage(usage)

    assert state.session_usage.reasoning_tokens == 9


def test_update_usage_accumulates_across_multiple_specialist_calls():
    """Two nodes in the same graph run both call ``_update_usage`` — the session-level
    totals must sum, not overwrite, matching the existing input/output-token behavior."""
    state = _state()
    first = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cache_write_tokens=3,
        cache_read_tokens=1,
        details={"reasoning_tokens": 2},
    )
    second = SimpleNamespace(
        input_tokens=20,
        output_tokens=8,
        total_tokens=28,
        cache_write_tokens=0,
        cache_read_tokens=4,
        details={"reasoning_tokens": 1},
    )
    state._update_usage(first)
    state._update_usage(second)

    assert state.session_usage.input_tokens == 30
    assert state.session_usage.output_tokens == 13
    assert state.session_usage.cache_creation_input_tokens == 3
    assert state.session_usage.cache_read_input_tokens == 5
    assert state.session_usage.reasoning_tokens == 3
