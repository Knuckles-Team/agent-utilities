"""``RewardSignal``/``blend`` — the named [0, 1] reward contract (CONCEPT:
AU-AHE.reward.unified-reward-signal). Every producer in the codebase already
converges on this convention informally; these tests pin the NAMED contract
(clamping, provenance, and the N-operand generalization of ``blend_reward``).
"""

from __future__ import annotations

from agent_utilities.harness.reward_signal import RewardSignal, blend


class TestRewardSignal:
    def test_value_and_confidence_clamp_to_unit_interval(self):
        sig = RewardSignal(value=1.4, source="test", confidence=-0.2)
        assert sig.value == 1.0
        assert sig.confidence == 0.0

    def test_negative_value_clamps_to_zero(self):
        assert RewardSignal(value=-0.5, source="test").value == 0.0

    def test_carries_provenance(self):
        sig = RewardSignal(
            value=0.8,
            source="langfuse_score",
            provenance_ref="trace:abc123",
            reason="check_failed",
        )
        assert sig.source == "langfuse_score"
        assert sig.provenance_ref == "trace:abc123"
        assert sig.reason == "check_failed"

    def test_is_frozen(self):
        sig = RewardSignal(value=0.5, source="test")
        try:
            sig.value = 0.9  # type: ignore[misc]
            raised = False
        except Exception:
            raised = True
        assert raised, "RewardSignal must be immutable"


class TestBlend:
    def test_empty_signals_returns_neutral_prior(self):
        result = blend()
        assert result.value == 0.5
        assert result.source == "blend:empty"

    def test_even_split_default_weight(self):
        a = RewardSignal(value=1.0, source="a")
        b = RewardSignal(value=0.0, source="b")
        result = blend(a, b)
        assert result.value == 0.5

    def test_explicit_weights_match_2operand_blend_reward(self):
        # Mirrors langfuse_signal.blend_reward(internal=0.4, langfuse=0.8, weight=0.3):
        # blended = (1 - 0.3) * 0.4 + 0.3 * 0.8 = 0.52
        internal = RewardSignal(value=0.4, source="internal_corpus")
        langfuse = RewardSignal(value=0.8, source="langfuse_score")
        result = blend(internal, langfuse, weights=[0.7, 0.3])
        assert round(result.value, 6) == 0.52

    def test_zero_confidence_signal_contributes_nothing(self):
        # Mirrors blend_reward's "no Langfuse signal yet -> return internal
        # reward unchanged" — a confidence=0 signal is fully ignored.
        internal = RewardSignal(value=0.4, source="internal_corpus", confidence=1.0)
        unset = RewardSignal(value=0.9, source="langfuse_score", confidence=0.0)
        result = blend(internal, unset, weights=[0.5, 0.5])
        assert result.value == 0.4

    def test_all_zero_confidence_degrades_to_neutral_prior(self):
        a = RewardSignal(value=0.9, source="a", confidence=0.0)
        b = RewardSignal(value=0.1, source="b", confidence=0.0)
        result = blend(a, b)
        assert result.value == 0.5
        assert result.source == "blend:no_confidence"

    def test_mismatched_weights_length_raises(self):
        a = RewardSignal(value=0.5, source="a")
        try:
            blend(a, weights=[0.5, 0.5])
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_never_produces_out_of_range_value(self):
        a = RewardSignal(value=1.0, source="a")
        b = RewardSignal(value=1.0, source="b")
        result = blend(a, b)
        assert 0.0 <= result.value <= 1.0
