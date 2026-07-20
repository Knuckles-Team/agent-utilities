"""CONCEPT:AU-ORCH.execution.inject-signal-board-observations"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_utilities.models.knowledge_graph import (
    OutcomeEvaluationNode,
    RegistryNodeType,
)

# These are pure-Pydantic property tests — they never touch the engine. But the
# suite now runs the REAL ephemeral epistemic-graph engine (CONCEPT:AU-KG.memory.provides-real-ephemeral-one)
# alongside them, so the box is under heavy load and Hypothesis's *timing*-based
# health checks (``too_slow`` example generation, the per-example ``deadline``)
# fire non-deterministically — a flake that has nothing to do with the property
# under test. Disable only the load-sensitive timing guards (the property's
# correctness checks are untouched) so these pass deterministically regardless of
# concurrent engine/suite load. Inputs are already small/bounded for cheap,
# deterministic generation; any string/float value round-trips through the model.
_LOAD_TOLERANT = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_LOAD_TOLERANT
@given(
    st.text(min_size=1, max_size=32),
    st.floats(min_value=0.0, max_value=1.0),
    st.lists(st.text(max_size=16), max_size=8),
    st.text(max_size=32),
)
def test_outcome_evaluation_node_properties(node_id, reward, criteria, feedback):
    """Test that OutcomeEvaluationNode always maintains valid properties."""
    node = OutcomeEvaluationNode(
        id=node_id,
        type=RegistryNodeType.OUTCOME_EVALUATION,
        name=f"Outcome: {node_id}",
        reward=reward,
        success_criteria_met=criteria,
        feedback_text=feedback,
    )

    assert node.id == node_id
    assert node.reward == reward
    assert node.success_criteria_met == criteria
    assert node.feedback_text == feedback
    assert node.type == RegistryNodeType.OUTCOME_EVALUATION


@_LOAD_TOLERANT
@given(
    st.floats(
        min_value=0.0,
        max_value=1.0,
        exclude_min=True,
        exclude_max=True,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_reward_scaling(reward):
    """Test that reward scaling logic (if any) handles various floats."""
    # Example logic: if reward > 0.5, it's considered 'good'

    node = OutcomeEvaluationNode(
        id="test",
        type=RegistryNodeType.OUTCOME_EVALUATION,
        name="test",
        reward=reward,
        feedback_text="test",
    )

    assert node.reward == reward
