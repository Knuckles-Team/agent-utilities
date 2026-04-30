from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.models import AgentDeps, Spec
from agent_utilities.patterns.first_run_tests import TestResult, run_first_tests
from agent_utilities.patterns.interactive_explanations import (
    generate_interactive_explanation,
)
from agent_utilities.patterns.manager import PatternManager
from agent_utilities.patterns.manual_testing import run_manual_test_cycle
from agent_utilities.patterns.tdd import (
    tdd_green_phase,
    tdd_red_phase,
    tdd_refactor_phase,
)
from agent_utilities.patterns.walkthroughs import generate_linear_walkthrough


@pytest.fixture
def mock_deps():
    deps = MagicMock(spec=AgentDeps)
    deps.workspace_path = Path("/tmp/fake_workspace")
    deps.graph_event_queue = MagicMock()
    deps.knowledge_engine = MagicMock()
    return deps


@pytest.mark.asyncio
async def test_pattern_manager_initialization(mock_deps):
    manager = PatternManager(mock_deps)
    assert manager.deps == mock_deps


@pytest.mark.asyncio
@patch("agent_utilities.patterns.first_run_tests.asyncio.create_subprocess_shell")
async def test_run_first_tests(mock_shell, mock_deps):
    # Mock workspace exists
    with patch.object(Path, "exists", return_value=True):
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"All tests passed", b"")
        mock_process.returncode = 0
        mock_shell.return_value = mock_process

        result = await run_first_tests(mock_deps.workspace_path)
        assert isinstance(result, TestResult)
        assert result.success is True
        assert "All tests passed" in result.output


@pytest.mark.asyncio
@patch("agent_utilities.patterns.tdd.dispatch_subagent")
async def test_tdd_red_phase(mock_dispatch, mock_deps):
    mock_dispatch.return_value = "Failing tests content"
    spec = Spec(feature_id="test_feat", title="Test Feature", user_stories=[])
    result = await tdd_red_phase(spec, mock_deps)
    assert result == "Failing tests content"
    mock_dispatch.assert_called_once()


@pytest.mark.asyncio
@patch("agent_utilities.patterns.tdd.dispatch_subagent")
async def test_tdd_green_phase(mock_dispatch, mock_deps):
    mock_dispatch.return_value = "Passing implementation"
    spec = Spec(feature_id="test_feat", title="Test Feature", user_stories=[])
    result = await tdd_green_phase(spec, "red_tests", mock_deps)
    assert result == "Passing implementation"


@pytest.mark.asyncio
@patch("agent_utilities.patterns.tdd.dispatch_subagent")
async def test_tdd_refactor_phase(mock_dispatch, mock_deps):
    mock_dispatch.return_value = "Refactored implementation"
    spec = Spec(feature_id="test_feat", title="Test Feature", user_stories=[])
    result = await tdd_refactor_phase(spec, "green_result", "red_result", mock_deps)
    assert result == "Refactored implementation"


@pytest.mark.asyncio
@patch("agent_utilities.patterns.manual_testing.dispatch_subagent")
async def test_run_manual_test_cycle(mock_dispatch, mock_deps):
    mock_dispatch.return_value = "Verification successful"
    result = await run_manual_test_cycle("Test goal", mock_deps)
    # run_manual_test_cycle wraps the dispatch result in an ExecutionNotes markdown log
    assert "Verification successful" in result
    assert "Execution Log: Test goal" in result


@pytest.mark.asyncio
@patch("agent_utilities.patterns.walkthroughs.dispatch_subagent")
async def test_generate_linear_walkthrough(mock_dispatch, mock_deps):
    mock_dispatch.return_value = "# Walkthrough\nStep 1..."
    result = await generate_linear_walkthrough("/tmp/path", mock_deps)
    assert "# Walkthrough" in result


@pytest.mark.asyncio
@patch("agent_utilities.patterns.interactive_explanations.dispatch_subagent")
async def test_generate_interactive_explanation(mock_dispatch, mock_deps):
    mock_dispatch.return_value = "<html><body>Explanation</body></html>"
    result = await generate_interactive_explanation("Explain this", "content", mock_deps)
    assert "<html>" in result


@pytest.mark.asyncio
@patch("agent_utilities.patterns.manager.run_first_tests")
async def test_pattern_manager_calls(mock_first_run, mock_deps):
    manager = PatternManager(mock_deps)
    mock_first_run.return_value = TestResult(True, "Success", 0, "cmd")
    await manager.first_run_tests()
    mock_first_run.assert_called_once()
