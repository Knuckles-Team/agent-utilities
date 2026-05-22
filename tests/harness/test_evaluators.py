from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.core.config import config
from agent_utilities.harness.evaluators import (
    capture_feedback,
    evaluate_length,
    evaluate_regex,
)
from agent_utilities.harness.trace_backend import LangfuseTraceBackend


@pytest.fixture
def mock_langfuse_config(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key", "sk-lf-test")
    monkeypatch.setattr(config, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(config, "langfuse_dataset_capture_threshold", 0.5)


@pytest.mark.asyncio
@patch("agent_utilities.harness.evaluators.create_trace_backend")
async def test_capture_feedback_above_threshold(
    mock_create_backend, mock_langfuse_config
):
    mock_backend = MagicMock(spec=LangfuseTraceBackend)
    mock_backend.submit_score.return_value = True
    mock_create_backend.return_value = mock_backend

    # Score above threshold (0.8 > 0.5), should not add to dataset
    success = await capture_feedback("trace_123", "Test Score", 0.8)
    assert success is True

    mock_backend.submit_score.assert_called_once_with(
        trace_id="trace_123", name="Test Score", value=0.8, comment=None
    )
    mock_backend.add_to_dataset.assert_not_called()


@pytest.mark.asyncio
@patch("agent_utilities.harness.evaluators.create_trace_backend")
async def test_capture_feedback_below_threshold(
    mock_create_backend, mock_langfuse_config
):
    mock_backend = MagicMock(spec=LangfuseTraceBackend)
    mock_backend.submit_score.return_value = True
    mock_create_backend.return_value = mock_backend

    # Score below threshold (0.3 < 0.5), should add to dataset
    success = await capture_feedback(
        "trace_123", "Test Score", 0.3, "Failed condition", "input_1", "expected_1"
    )
    assert success is True

    mock_backend.submit_score.assert_called_once_with(
        trace_id="trace_123", name="Test Score", value=0.3, comment="Failed condition"
    )
    mock_backend.add_to_dataset.assert_called_once_with(
        dataset_name="continuous_learning_test_score",
        trace_id="trace_123",
        input_data="input_1",
        expected_output="expected_1",
    )


@pytest.mark.asyncio
@patch("agent_utilities.harness.evaluators.capture_feedback")
async def test_evaluate_regex(mock_capture_feedback):
    # This just tests the regex logic, capture_feedback runs async via create_task
    # so we don't await it here, but we can verify the sync return value.
    match = evaluate_regex("trace_123", "The quick brown fox", r"quick.*fox")
    assert match is True

    no_match = evaluate_regex("trace_123", "The lazy dog", r"quick.*fox")
    assert no_match is False


@pytest.mark.asyncio
@patch("agent_utilities.harness.evaluators.capture_feedback")
async def test_evaluate_length(mock_capture_feedback):
    valid = evaluate_length("trace_123", "short", 10)
    assert valid is True

    invalid = evaluate_length("trace_123", "this is way too long", 10)
    assert invalid is False
