import pytest
from hypothesis import given, strategies as st
from agent_utilities.models.knowledge_graph import OutcomeEvaluationNode, RegistryNodeType

@given(
    st.text(min_size=1),
    st.floats(min_value=0.0, max_value=1.0),
    st.lists(st.text()),
    st.text()
)
def test_outcome_evaluation_node_properties(node_id, reward, criteria, feedback):
    """Test that OutcomeEvaluationNode always maintains valid properties."""
    node = OutcomeEvaluationNode(
        id=node_id,
        type=RegistryNodeType.OUTCOME_EVALUATION,
        name=f"Outcome: {node_id}",
        reward=reward,
        success_criteria_met=criteria,
        feedback_text=feedback
    )

    assert node.id == node_id
    assert node.reward == reward
    assert node.success_criteria_met == criteria
    assert node.feedback_text == feedback
    assert node.type == RegistryNodeType.OUTCOME_EVALUATION

@given(st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True, allow_nan=False, allow_infinity=False))
def test_reward_scaling(reward):
    """Test that reward scaling logic (if any) handles various floats."""
    # Example logic: if reward > 0.5, it's considered 'good'
    is_good = reward > 0.5

    node = OutcomeEvaluationNode(
        id="test",
        type=RegistryNodeType.OUTCOME_EVALUATION,
        name="test",
        reward=reward,
        feedback_text="test"
    )

    assert node.reward == reward
