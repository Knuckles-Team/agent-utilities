import pytest
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from agent_utilities import mcp_utilities

@pytest.mark.asyncio
async def test_destructive_guard_scenario():
    """Scenario: User attempts a delete operation, helper should elicit confirmation."""
    mock_ctx = AsyncMock()
    # Mock user accepting the operation
    mock_ctx.elicit.return_value = MagicMock(action="accept", data=True)

    # Simulate a tool function
    async def delete_tool(ctx):
        if not await mcp_utilities.ctx_confirm_destructive(ctx, "delete database"):
            return "cancelled"
        return "deleted"

    result = await delete_tool(mock_ctx)
    assert result == "deleted"
    mock_ctx.elicit.assert_called_once()

    # Mock user cancelling the operation
    mock_ctx.elicit.reset_mock()
    mock_ctx.elicit.return_value = MagicMock(action="cancelled", data=False)
    result = await delete_tool(mock_ctx)
    assert result == "cancelled"

@pytest.mark.asyncio
async def test_progress_scenario():
    """Scenario: Long running operation should report 0% and 100%."""
    mock_ctx = AsyncMock()

    async def long_op_tool(ctx):
        await mcp_utilities.ctx_progress(ctx, 0, 100)
        # do work
        await mcp_utilities.ctx_progress(ctx, 100, 100)
        return "done"

    await long_op_tool(mock_ctx)
    assert mock_ctx.report_progress.call_count == 2
    mock_ctx.report_progress.assert_any_call(progress=0, total=100)
    mock_ctx.report_progress.assert_any_call(progress=100, total=100)

@pytest.mark.asyncio
async def test_dual_logging_scenario():
    """Scenario: Logging should go to both the python logger and the MCP context."""
    mock_ctx = MagicMock()
    mock_logger = MagicMock()

    def tool_with_logs(ctx):
        mcp_utilities.ctx_log(ctx, mock_logger, "info", "Operation started")

    tool_with_logs(mock_ctx)
    mock_logger.info.assert_called_with("Operation started")
    mock_ctx.info.assert_called_with("Operation started")

@pytest.mark.asyncio
async def test_auth_state_scenario():
    """Scenario: Auth token should be saved to state."""
    mock_ctx = AsyncMock()
    mock_ctx.session = AsyncMock()

    async def login_tool(ctx):
        token = "secret_jwt"
        await mcp_utilities.ctx_set_state(ctx, "myproj", "token", token)
        return "logged_in"

    await login_tool(mock_ctx)
    mock_ctx.session.set_state.assert_called_with("myproj_token", "secret_jwt")

@pytest.mark.asyncio
async def test_sampling_scenario():
    """Scenario: Tool uses LLM sampling for enrichment."""
    mock_ctx = AsyncMock()
    mock_ctx.sample.return_value = MagicMock(content=MagicMock(text="AI summary"))

    async def search_tool(ctx):
        results = {"data": "raw data"}
        summary = await mcp_utilities.ctx_sample(ctx, f"Summarize: {results}")
        results["summary"] = summary or ""
        return results

    with patch("mcp.types.CreateMessageRequestParams", MagicMock()):
        res = await search_tool(mock_ctx)
        assert res["summary"] == "AI summary"
        mock_ctx.sample.assert_called_once()
