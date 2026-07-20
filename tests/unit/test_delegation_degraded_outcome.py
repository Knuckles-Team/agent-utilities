"""Fail-loud + self-healing on degraded delegations.

CONCEPT:AU-ORCH.execution.degraded-no-data-outcome / AU-AHE.evaluation.action-outcome-feedback —
a delegation that falls through to the graph's "no data" sentinel (or returns an
empty answer) must NOT be recorded as a successful ``"completed"`` run. These tests
pin the detector that distinguishes a real answer from a non-answer, and that a
non-answer is fed back as a negative action-outcome so routing self-corrects.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.orchestration.agent_runner import (
    _delegation_degraded,
    _record_degraded_feedback,
)

# The exact sentinel the graph synthesizer emits when zero results were gathered
# (agent_utilities/graph/verification.py). This is what every toolless delegation
# returned before the fix, stamped as status="completed".
SENTINEL = (
    "The agent completed its analysis but was unable to find specific data "
    "matching your request. Please verify the query or target system status."
)


@pytest.mark.parametrize(
    "result,expected,label",
    [
        (
            {"results": {"output": SENTINEL}, "metadata": {"degraded": True}},
            True,
            "graph no-data via structured metadata flag",
        ),
        (
            {"results": {"output": SENTINEL}},
            True,
            "sentinel text fallback when metadata is absent",
        ),
        ({"results": {"output": ""}}, True, "empty graph output"),
        (
            {
                "output": "The query was executed, but a final synthesis could not be generated."
            },
            True,
            "partial-synthesis sentinel",
        ),
        ("", True, "bare empty string (single-server path)"),
        (
            {
                "results": {"output": "Found 3 running containers: web, db, cache."},
                "metadata": {"degraded": False},
            },
            False,
            "real answer with metadata",
        ),
        (
            "Found 3 running containers: web, db, cache.",
            False,
            "bare-string real answer (single-server path)",
        ),
    ],
)
def test_delegation_degraded_detection(result, expected, label):
    assert _delegation_degraded(result) is expected, label


def test_delegation_degraded_never_raises_on_junk():
    """The detector runs on the success path and must never break the run."""
    for junk in (None, 123, object(), {"results": 5}, {"metadata": "x"}):
        # Must return a bool, never raise.
        assert isinstance(_delegation_degraded(junk), bool)


def test_degraded_run_records_negative_feedback():
    """A degraded delegation feeds back success=False on the agent's reward-EMA."""
    feedback = MagicMock()
    engine = MagicMock()
    with patch(
        "agent_utilities.knowledge_graph.adaptation.feedback.FeedbackService.from_engine",
        return_value=feedback,
    ):
        _record_degraded_feedback(
            engine,
            "container-manager-mcp",
            "List running containers",
            {"results": {"output": SENTINEL}, "metadata": {"degraded": True}},
        )
    feedback.record_action_outcome.assert_called_once()
    _, kwargs = feedback.record_action_outcome.call_args
    args, _ = feedback.record_action_outcome.call_args
    assert args and args[0] == "agent:container-manager-mcp"
    assert kwargs["success"] is False
    assert kwargs["reason"] == "delegation_degraded_no_data"


def test_degraded_feedback_is_best_effort_without_engine():
    """No engine -> no-op, never raises (provenance must never fail the run)."""
    _record_degraded_feedback(None, "x", "task", {"results": {"output": SENTINEL}})
