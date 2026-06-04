"""Tests for the dynamic workflow adversarial loop.

CONCEPT:ORCH-1.10 — Dynamic Workflows.
"""

import pytest
from unittest.mock import AsyncMock, patch
from agent_utilities.orchestration.engine import AgentOrchestrationEngine

@pytest.mark.asyncio
async def test_execute_workflow_dynamic_convergence():
    """Test that execute_workflow correctly loops until adversarial verification passes."""
    engine = AgentOrchestrationEngine()

    # Mock the components that `execute_workflow` calls
    with patch("agent_utilities.orchestration.agent_runner.run_agent", new_callable=AsyncMock) as mock_run_agent, \
         patch("agent_utilities.capabilities.adversarial_verifier.run_adversarial_pass", new_callable=AsyncMock) as mock_adv_pass, \
         patch("agent_utilities.agent.factory.create_agent") as mock_create_agent:

        # Let the agent runner return a dummy output
        mock_run_agent.return_value = "Subagent output"

        # The adversarial verifier returns failure on first pass, success on second
        class MockAdvRes:
            def __init__(self, fails):
                self.vulnerabilities_found = fails
                self.findings = "Missing criteria" if fails else "All good"

        mock_adv_pass.side_effect = [MockAdvRes(True), MockAdvRes(False)]

        # Mock the PR submitter agent
        mock_pr_agent = AsyncMock()
        mock_pr_agent.run.return_value.output = "https://github.com/test/pr/1"
        mock_create_agent.return_value = (mock_pr_agent, None)

        # Run the workflow
        result = await engine.execute_workflow(
            workflow_id="test_wf",
            task="Build a web app",
            completion_state="Must have a PR ready",
            max_fan_out=2,
            max_iterations=5
        )

        # Verify the outcome
        assert result["status"] == "converged"
        assert result["iterations"] == 2
        assert result["pr_result"] == "https://github.com/test/pr/1"
        assert result["final_output"] == "Subagent output"

        # Verify calls
        assert mock_run_agent.call_count == 4  # 2 fan_out * 2 iterations
        assert mock_adv_pass.call_count == 2
        assert mock_pr_agent.run.call_count == 1
