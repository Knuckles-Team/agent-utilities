"""
Tests for Graph-Native Durable Execution (CONCEPT:ECO-4.0).
"""

from agent_utilities.orchestration.durable_execution import DurableExecutionManager


def test_durable_execution_flow():
    manager = DurableExecutionManager(session_id="test_session_1")

    # Test saving a checkpoint
    node_id = manager.save_checkpoint("trade_step_1", {"asset": "BTC", "qty": 1.5})
    assert node_id == "trade_step_1"

    # Test resuming the checkpoint
    resumed = manager.resume_session()
    assert resumed is not None
    assert resumed["node_id"] == "trade_step_1"
    assert "BTC" in resumed["state"]

    # Test marking as completed
    manager.mark_completed("trade_step_1")

    # Completed node should not be resumed
    resumed_after = manager.resume_session()
    assert resumed_after is None
