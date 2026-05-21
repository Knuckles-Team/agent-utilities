"""Tests for CONCEPT:OS-5.1 — Langfuse Tracing Decorators.

Validates the @trace decorator behavior including:
- Sync/async function decoration
- Trace emission with proper nesting
- No-op behavior when Langfuse is unconfigured
- Session ID propagation
- Error tracing
"""

import pytest
from unittest.mock import patch, MagicMock
from agent_utilities.harness.tracing import trace, set_session_id, get_trace_id
from agent_utilities.core.config import config


@pytest.fixture
def mock_langfuse_config(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key", "sk-lf-test")
    monkeypatch.setattr(config, "langfuse_public_key", "pk-lf-test")


@patch("agent_utilities.harness.tracing._emit_trace")
def test_sync_trace_decorator(mock_emit, mock_langfuse_config):
    """CONCEPT:OS-5.1 — Sync function tracing emits properly."""

    @trace(name="test_sync_function")
    def my_sync_func(x):
        return x * 2

    result = my_sync_func(5)
    assert result == 10

    mock_emit.assert_called_once()
    kwargs = mock_emit.call_args.kwargs
    assert kwargs["name"] == "test_sync_function"
    assert kwargs["is_root"] is True  # First trace should be root


@pytest.mark.asyncio
@patch("agent_utilities.harness.tracing._emit_trace")
async def test_async_trace_decorator(mock_emit, mock_langfuse_config):
    """CONCEPT:OS-5.1 — Async function tracing emits properly."""

    @trace(name="test_async_function")
    async def my_async_func(x):
        return x * 3

    result = await my_async_func(4)
    assert result == 12

    mock_emit.assert_called_once()
    kwargs = mock_emit.call_args.kwargs
    assert kwargs["name"] == "test_async_function"
    assert kwargs["is_root"] is True


@patch("agent_utilities.harness.tracing._emit_trace")
def test_trace_decorator_disabled(mock_emit, monkeypatch):
    """CONCEPT:OS-5.1 — Tracing is no-op without Langfuse keys."""
    monkeypatch.setattr(config, "langfuse_secret_key", None)

    @trace(name="test_disabled")
    def my_disabled_func(x):
        return x + 1

    result = my_disabled_func(10)
    assert result == 11

    # emit_trace should not be called if disabled
    mock_emit.assert_not_called()


@patch("agent_utilities.harness.tracing._emit_trace")
def test_trace_nesting_propagation(mock_emit, mock_langfuse_config):
    """CONCEPT:OS-5.1 — Nested traces share parent trace_id."""

    @trace(name="outer_trace")
    def outer():
        @trace(name="inner_trace")
        def inner():
            return "nested"

        return inner()

    result = outer()
    assert result == "nested"

    # Should emit twice: once for outer (root), once for inner (child)
    assert mock_emit.call_count == 2

    outer_call = mock_emit.call_args_list[1]  # outer completes after inner
    inner_call = mock_emit.call_args_list[0]

    # Both should share the same trace_id
    assert outer_call.kwargs["trace_id"] == inner_call.kwargs["trace_id"]
    # Only the outer should be root
    assert outer_call.kwargs["is_root"] is True
    assert inner_call.kwargs["is_root"] is False


@patch("agent_utilities.harness.tracing._emit_trace")
def test_trace_error_handling(mock_emit, mock_langfuse_config):
    """CONCEPT:OS-5.1 — Errors are captured in trace output."""

    @trace(name="error_function")
    def failing_func():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing_func()

    mock_emit.assert_called_once()
    kwargs = mock_emit.call_args.kwargs
    assert kwargs["level"] == "ERROR"
    assert kwargs["status_message"] == "test error"


@patch("agent_utilities.harness.tracing._emit_trace")
def test_trace_with_tags(mock_emit, mock_langfuse_config):
    """CONCEPT:OS-5.1 — Tags are passed through to trace emission."""

    @trace(name="tagged_function", tags=["test", "unit"])
    def tagged_func():
        return True

    tagged_func()

    mock_emit.assert_called_once()
    kwargs = mock_emit.call_args.kwargs
    assert kwargs["tags"] == ["test", "unit"]


@patch("agent_utilities.harness.tracing._emit_trace")
def test_session_id_in_trace(mock_emit, mock_langfuse_config):
    """CONCEPT:OS-5.1 — Session ID flows into trace events."""
    set_session_id("test-session-abc")

    @trace(name="session_func")
    def session_func():
        return True

    session_func()

    mock_emit.assert_called_once()
    kwargs = mock_emit.call_args.kwargs
    assert kwargs["session_id"] == "test-session-abc"
