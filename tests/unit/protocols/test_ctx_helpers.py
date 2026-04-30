from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities import mcp_utilities


@pytest.mark.asyncio
async def test_ctx_progress():
    # Test with ctx
    mock_ctx = AsyncMock()
    await mcp_utilities.ctx_progress(mock_ctx, 50, 100)
    mock_ctx.report_progress.assert_called_once_with(progress=50, total=100)

    # Test with ctx=None (no error)
    await mcp_utilities.ctx_progress(None, 50, 100)

@pytest.mark.asyncio
async def test_ctx_confirm_destructive():
    # Test with ctx - Accepted
    mock_ctx = AsyncMock()
    mock_result = MagicMock()
    mock_result.action = "accept"
    mock_result.data = True
    mock_ctx.elicit.return_value = mock_result

    res = await mcp_utilities.ctx_confirm_destructive(mock_ctx, "delete something")
    assert res is True
    mock_ctx.elicit.assert_called_once()

    # Test with ctx - Cancelled
    mock_ctx.elicit.reset_mock()
    mock_result.action = "cancelled" # or anything not "accept"
    res = await mcp_utilities.ctx_confirm_destructive(mock_ctx, "delete something")
    assert res is False

    # Test with ctx=None (backward compatibility - should allow)
    res = await mcp_utilities.ctx_confirm_destructive(None, "delete something")
    assert res is True

    # Test with elicitation failure (fallback to True)
    mock_ctx.elicit.side_effect = Exception("MCP client error")
    res = await mcp_utilities.ctx_confirm_destructive(mock_ctx, "delete something")
    assert res is True

def test_ctx_log():
    mock_logger = MagicMock()
    mock_ctx = MagicMock()

    # Test info level
    mcp_utilities.ctx_log(mock_ctx, mock_logger, "info", "test message")
    mock_logger.info.assert_called_once_with("test message")
    mock_ctx.info.assert_called_once_with("test message")

    # Test error level
    mock_logger.reset_mock()
    mock_ctx.reset_mock()
    mcp_utilities.ctx_log(mock_ctx, mock_logger, "error", "error message")
    mock_logger.error.assert_called_once_with("error message")
    mock_ctx.error.assert_called_once_with("error message")

    # Test with ctx=None
    mock_logger.reset_mock()
    mcp_utilities.ctx_log(None, mock_logger, "info", "no ctx message")
    mock_logger.info.assert_called_once_with("no ctx message")

    # Test with missing level on ctx (fallback to info if possible, though ctx.log is preferred in real MCP, FastMCP uses levels)
    mock_ctx = MagicMock()
    del mock_ctx.debug
    mcp_utilities.ctx_log(mock_ctx, mock_logger, "debug", "debug message")
    mock_ctx.info.assert_called() # Should fallback to info on ctx if debug missing

@pytest.mark.asyncio
async def test_ctx_state_management():
    mock_ctx = AsyncMock()
    mock_ctx.session = AsyncMock()

    # Test set_state
    await mcp_utilities.ctx_set_state(mock_ctx, "testproj", "mykey", "myval")
    mock_ctx.session.set_state.assert_called_once_with("testproj_mykey", "myval")

    # Test get_state
    mock_ctx.session.get_state.return_value = "cached_val"
    val = await mcp_utilities.ctx_get_state(mock_ctx, "testproj", "mykey")
    assert val == "cached_val"
    mock_ctx.session.get_state.assert_called_once_with("testproj_mykey")

    # Test get_state with default
    mock_ctx.session.get_state.return_value = None
    val = await mcp_utilities.ctx_get_state(mock_ctx, "testproj", "missing", default="fallback")
    assert val == "fallback"

    # Test with ctx=None
    await mcp_utilities.ctx_set_state(None, "p", "k", "v")
    val = await mcp_utilities.ctx_get_state(None, "p", "k", default="def")
    assert val == "def"

@pytest.mark.asyncio
async def test_ctx_sample():
    # Test successful sampling
    mock_ctx = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = MagicMock()
    mock_result.content.text = "Sampled response"
    mock_ctx.sample.return_value = mock_result

    with patch("mcp.types.CreateMessageRequestParams", MagicMock()):
        res = await mcp_utilities.ctx_sample(mock_ctx, "Tell me a joke")
        assert res == "Sampled response"
        mock_ctx.sample.assert_called_once()

    # Test sampling with ctx=None
    res = await mcp_utilities.ctx_sample(None, "Prompt")
    assert res is None

    # Test sampling failure
    mock_ctx.sample.side_effect = Exception("Sampling not supported")
    res = await mcp_utilities.ctx_sample(mock_ctx, "Prompt")
    assert res is None
