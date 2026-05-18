import pytest
from unittest.mock import patch, MagicMock
from agent_utilities.harness.tracing import trace
from agent_utilities.core.config import config


@pytest.fixture
def mock_langfuse_config(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key", "sk-lf-test")
    monkeypatch.setattr(config, "langfuse_public_key", "pk-lf-test")


@patch("agent_utilities.harness.tracing._emit_trace")
def test_sync_trace_decorator(mock_emit, mock_langfuse_config):
    @trace(name="test_sync_function")
    def my_sync_func(x):
        return x * 2

    result = my_sync_func(5)
    assert result == 10

    mock_emit.assert_called_once()
    kwargs = mock_emit.call_args.kwargs
    assert kwargs["name"] == "test_sync_function"
    assert kwargs["output_data"] == 10
    assert kwargs["input_data"] == {"args": (5,), "kwargs": {}}


@pytest.mark.asyncio
@patch("agent_utilities.harness.tracing._emit_trace")
async def test_async_trace_decorator(mock_emit, mock_langfuse_config):
    @trace(name="test_async_function")
    async def my_async_func(x):
        return x * 3

    result = await my_async_func(4)
    assert result == 12

    mock_emit.assert_called_once()
    kwargs = mock_emit.call_args.kwargs
    assert kwargs["name"] == "test_async_function"
    assert kwargs["output_data"] == 12
    assert kwargs["input_data"] == {"args": (4,), "kwargs": {}}


@patch("agent_utilities.harness.tracing._emit_trace")
def test_trace_decorator_disabled(mock_emit, monkeypatch):
    # Ensure config has no langfuse_secret_key
    monkeypatch.setattr(config, "langfuse_secret_key", None)

    @trace(name="test_disabled")
    def my_disabled_func(x):
        return x + 1

    result = my_disabled_func(10)
    assert result == 11

    # emit_trace should not be called if disabled
    mock_emit.assert_not_called()
