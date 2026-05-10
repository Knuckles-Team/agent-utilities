#!/usr/bin/python
from __future__ import annotations
"""Tests for Tool Repetition Guard (CONCEPT:OS-5.5)."""


import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def guard():
    from agent_utilities.security.execution_stability_engine import RepetitionGuard

    return RepetitionGuard(max_consecutive_repeats=3, max_calls_per_session=10)


@pytest.fixture
def strict_guard():
    from agent_utilities.security.execution_stability_engine import RepetitionGuard

    return RepetitionGuard(max_consecutive_repeats=2, max_calls_per_session=5)


# ---------------------------------------------------------------------------
# Consecutive repetition detection
# ---------------------------------------------------------------------------


class TestConsecutiveRepetition:
    """Tests for consecutive identical call detection."""

    def test_first_call_allowed(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        result = guard.check_tool_call("shell", {"command": "ls"})
        assert result.verdict == RepetitionVerdict.ALLOW
        assert result.consecutive_count == 1
        assert result.total_count == 1

    def test_two_same_calls_allowed(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        guard.check_tool_call("shell", {"command": "ls"})
        result = guard.check_tool_call("shell", {"command": "ls"})
        assert result.verdict == RepetitionVerdict.WARN  # max=3, at 2 it warns
        assert result.consecutive_count == 2

    def test_three_same_calls_denied(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        guard.check_tool_call("shell", {"command": "ls"})
        guard.check_tool_call("shell", {"command": "ls"})
        result = guard.check_tool_call("shell", {"command": "ls"})
        assert result.verdict == RepetitionVerdict.DENY
        assert result.consecutive_count == 3
        assert "consecutively" in result.explanation

    def test_different_args_resets_consecutive(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        guard.check_tool_call("shell", {"command": "ls"})
        guard.check_tool_call("shell", {"command": "ls"})
        result = guard.check_tool_call("shell", {"command": "pwd"})
        assert result.verdict == RepetitionVerdict.ALLOW
        assert result.consecutive_count == 1

    def test_different_tool_resets_consecutive(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        guard.check_tool_call("shell", {"command": "ls"})
        guard.check_tool_call("shell", {"command": "ls"})
        result = guard.check_tool_call("read_file", {"path": "/foo"})
        assert result.verdict == RepetitionVerdict.ALLOW
        assert result.consecutive_count == 1

    def test_strict_guard_two_repeats(self, strict_guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        strict_guard.check_tool_call("shell", {"command": "ls"})
        result = strict_guard.check_tool_call("shell", {"command": "ls"})
        assert result.verdict == RepetitionVerdict.DENY
        assert result.consecutive_count == 2


# ---------------------------------------------------------------------------
# Per-session budget
# ---------------------------------------------------------------------------


class TestSessionBudget:
    """Tests for per-tool session budget limits."""

    def test_within_budget(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        for i in range(10):
            # Vary args to avoid consecutive limit
            result = guard.check_tool_call("shell", {"command": f"cmd-{i}"})
            assert result.verdict == RepetitionVerdict.ALLOW

    def test_exceed_budget(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        for i in range(10):
            guard.check_tool_call("shell", {"command": f"cmd-{i}"})
        result = guard.check_tool_call("shell", {"command": "cmd-extra"})
        assert result.verdict == RepetitionVerdict.DENY
        assert "session budget" in result.explanation

    def test_budget_per_tool(self, strict_guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        for i in range(5):
            strict_guard.check_tool_call("shell", {"command": f"cmd-{i}"})
        # Different tool should have its own budget — not denied by shell budget
        result = strict_guard.check_tool_call("read_file", {"path": "/foo"})
        assert result.verdict != RepetitionVerdict.DENY
        assert result.total_count == 1  # New tool starts fresh


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for state reset."""

    def test_reset_clears_state(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        guard.check_tool_call("shell", {"command": "ls"})
        guard.check_tool_call("shell", {"command": "ls"})
        guard.reset()
        result = guard.check_tool_call("shell", {"command": "ls"})
        assert result.verdict == RepetitionVerdict.ALLOW
        assert result.consecutive_count == 1
        assert result.total_count == 1

    def test_reset_clears_statistics(self, guard):
        guard.check_tool_call("shell", {"command": "ls"})
        guard.check_tool_call("read_file", {"path": "/foo"})
        guard.reset()
        stats = guard.get_statistics()
        assert stats["tool_counts"] == {}
        assert stats["consecutive_count"] == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Tests for get_statistics."""

    def test_initial_statistics(self, guard):
        stats = guard.get_statistics()
        assert stats["tool_counts"] == {}
        assert stats["current_tool"] == ""
        assert stats["consecutive_count"] == 0
        assert stats["max_consecutive_repeats"] == 3
        assert stats["max_calls_per_session"] == 10

    def test_statistics_after_calls(self, guard):
        guard.check_tool_call("shell", {"command": "ls"})
        guard.check_tool_call("read_file", {"path": "/foo"})
        guard.check_tool_call("shell", {"command": "pwd"})

        stats = guard.get_statistics()
        assert stats["tool_counts"]["shell"] == 2
        assert stats["tool_counts"]["read_file"] == 1
        assert stats["current_tool"] == "shell"


# ---------------------------------------------------------------------------
# ExperienceNode distillation
# ---------------------------------------------------------------------------


class TestExperienceNode:
    """Tests for ExperienceNode creation from denied results."""

    def test_create_experience_on_deny(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        guard.check_tool_call("shell", {"command": "ls"})
        guard.check_tool_call("shell", {"command": "ls"})
        result = guard.check_tool_call("shell", {"command": "ls"})
        assert result.verdict == RepetitionVerdict.DENY

        exp = guard.create_experience_node(result, session_id="test-session")
        assert exp is not None
        assert exp["type"] == "experience"
        assert "shell" in exp["condition"]
        assert "alternative" in exp["action"].lower()
        assert exp["session_id"] == "test-session"
        assert exp["confidence"] == 0.95

    def test_no_experience_on_allow(self, guard):
        result = guard.check_tool_call("shell", {"command": "ls"})
        exp = guard.create_experience_node(result)
        assert exp is None

    def test_no_experience_on_warn(self, guard):
        guard.check_tool_call("shell", {"command": "ls"})
        result = guard.check_tool_call("shell", {"command": "ls"})
        exp = guard.create_experience_node(result)
        assert exp is None


# ---------------------------------------------------------------------------
# Argument hashing
# ---------------------------------------------------------------------------


class TestArgumentHashing:
    """Tests for deterministic argument hashing."""

    def test_same_args_same_hash(self, guard):
        """Identical arguments produce the same hash."""
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        guard.check_tool_call("shell", {"command": "ls", "cwd": "/home"})
        result = guard.check_tool_call("shell", {"cwd": "/home", "command": "ls"})
        # Key order shouldn't matter - should be same hash
        assert result.consecutive_count == 2

    def test_none_args(self, guard):
        from agent_utilities.security.execution_stability_engine import RepetitionVerdict

        result = guard.check_tool_call("shell", None)
        assert result.verdict == RepetitionVerdict.ALLOW


# ---------------------------------------------------------------------------
# PolicyEngine integration
# ---------------------------------------------------------------------------


class TestPolicyEngineIntegration:
    """Tests for RepetitionPolicy integration."""

    def test_policy_blocks_on_deny(self):
        from agent_utilities.security.guardrails import PolicyEngine
        from agent_utilities.security.execution_stability_engine import RepetitionPolicy

        policy = RepetitionPolicy()
        engine = PolicyEngine()
        engine.register(policy)

        # Call 3 times with same tool/args
        for _ in range(3):
            results = engine.evaluate(
                context={"tool_name": "shell", "tool_arguments": {"command": "ls"}}
            )

        blocked = [r for r in results if r.policy_name == "execution_stability_engine"]
        assert len(blocked) == 1
        assert not blocked[0].allowed

    def test_policy_allows_normal(self):
        from agent_utilities.security.guardrails import PolicyEngine
        from agent_utilities.security.execution_stability_engine import RepetitionPolicy

        engine = PolicyEngine()
        engine.register(RepetitionPolicy())
        results = engine.evaluate(
            context={"tool_name": "shell", "tool_arguments": {"command": "ls"}}
        )
        rep_result = [r for r in results if r.policy_name == "execution_stability_engine"]
        assert len(rep_result) == 1
        assert rep_result[0].allowed

    def test_policy_no_context(self):
        from agent_utilities.security.guardrails import PolicyEngine
        from agent_utilities.security.execution_stability_engine import RepetitionPolicy

        engine = PolicyEngine()
        engine.register(RepetitionPolicy())
        results = engine.evaluate(input_text="hello")
        rep_result = [r for r in results if r.policy_name == "execution_stability_engine"]
        assert len(rep_result) == 1
        assert rep_result[0].allowed
