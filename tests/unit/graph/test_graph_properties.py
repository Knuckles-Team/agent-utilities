"""CONCEPT:ORCH-1.0"""

from hypothesis import given
from hypothesis import strategies as st

from agent_utilities.models.knowledge_graph import (
    OutcomeEvaluationNode,
    RegistryNodeType,
)


# Bound the text/list sizes so input generation stays cheap and deterministic:
# unbounded ``st.text()`` example generation can trip Hypothesis's ``too_slow``
# health check when the machine is under heavy load (e.g. the full suite running
# real container-sandbox tests concurrently). Small bounded strategies cover the
# same property (any string value round-trips) without the load-sensitive timing.
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
