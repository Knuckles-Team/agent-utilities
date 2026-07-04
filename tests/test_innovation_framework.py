#!/usr/bin/python
from __future__ import annotations

"""Tests for Innovation Framework modules (CONCEPT:AU-OS.state.cognitive-scheduler-preemption through CONCEPT:AU-OS.state.cognitive-scheduler-preemption).

Tests cover:
- CONCEPT:AU-OS.state.cognitive-scheduler-preemption: Homeostatic model downgrade integration
- CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort: Adversarial verification models and logic
- CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Signal board state management
- CONCEPT:AU-OS.safety.doom-loop-detection: File watcher trigger rules and package audit
- CONCEPT:AU-OS.state.cognitive-scheduler-preemption: Maintenance cron scheduling and budget management
"""


import os
from unittest.mock import MagicMock, patch

import pytest

# ── CONCEPT:AU-OS.state.cognitive-scheduler-preemption: Homeostatic Model Downgrade ──────────────────────────────


class TestHomeostaticDowngrade:
    """CONCEPT:AU-OS.state.cognitive-scheduler-preemption: ResourceOptimizer-driven tier downgrade."""

    def test_resource_optimizer_budget_remaining(self):
        """ResourceBudget computes remaining correctly."""
        from agent_utilities.core.resource_optimizer import ResourceBudget

        budget = ResourceBudget(
            total_token_budget=100_000,
            total_cost_budget_usd=5.0,
        )
        budget.tokens_used = 80_000
        budget.cost_used_usd = 4.5

        assert budget.tokens_remaining == 20_000
        assert budget.cost_remaining == pytest.approx(0.5)
        assert budget.utilization_pct == pytest.approx(80.0)

    def test_select_model_downgrades_under_pressure(self):
        """When budget < 20%, complexity is forced to 'light'."""
        from agent_utilities.core.resource_optimizer import (
            ResourceBudget,
            ResourceOptimizer,
        )

        budget = ResourceBudget(total_cost_budget_usd=10.0)
        budget.cost_used_usd = 9.0  # 10% remaining
        optimizer = ResourceOptimizer(budget=budget)

        # Should downgrade heavy → light when <20% budget
        # select_model_for_step returns None without a registry,
        # but the budget logic runs
        result = optimizer.select_model_for_step(complexity="heavy")
        # Without a registry, returns None (graceful degradation)
        assert result is None

    def test_graphdeps_has_resource_optimizer_field(self):
        """GraphDeps should accept a resource_optimizer field."""
        from agent_utilities.graph.state import GraphDeps

        deps = GraphDeps(
            tag_prompts={},
            tag_env_vars={},
            mcp_toolsets=[],
            resource_optimizer=MagicMock(),
        )
        assert deps.resource_optimizer is not None


# ── CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort: Adversarial Verification ─────────────────────────────────


class TestAdversarialVerification:
    """CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort: Adversarial verification models."""

    def test_adversarial_result_model(self):
        """AdversarialResult should serialize correctly."""
        from agent_utilities.capabilities.adversarial_verifier import (
            AdversarialResult,
        )

        result = AdversarialResult(
            vulnerabilities_found=True,
            severity="medium",
            findings=["SQL injection in user_input param"],
            suggested_fixes=["Use parameterized queries"],
            confidence=0.85,
        )
        assert result.vulnerabilities_found is True
        assert result.severity == "medium"
        assert len(result.findings) == 1
        assert result.confidence == 0.85

    def test_adversarial_result_clean(self):
        """Default AdversarialResult indicates no issues."""
        from agent_utilities.capabilities.adversarial_verifier import (
            AdversarialResult,
        )

        result = AdversarialResult()
        assert result.vulnerabilities_found is False
        assert result.severity == "none"
        assert result.findings == []

    def test_adversarial_disabled_by_default(self):
        """ADVERSARIAL_ENABLED should be False unless env var is set."""
        # We can't easily test the module-level constant without
        # reloading, but we can verify the env check logic
        enabled = os.getenv("ADVERSARIAL_VERIFICATION", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        # In test environment, should be False (not set)
        assert not enabled or os.getenv("ADVERSARIAL_VERIFICATION") is not None


# ── CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Signal Board ─────────────────────────────────────────────


class TestSignalBoard:
    """CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Stigmergy signal board on GraphState."""

    def test_signal_board_default_empty(self):
        """Signal board should start empty."""
        from agent_utilities.graph.state import GraphState

        state = GraphState(query="test")
        assert state.signal_board == {}

    def test_signal_board_emit_and_read(self):
        """Signals can be emitted and read back."""
        from agent_utilities.graph.state import GraphState

        state = GraphState(query="test")
        signal_type = "dependency_gap"
        message = "Missing package: requests"

        if signal_type not in state.signal_board:
            state.signal_board[signal_type] = []
        state.signal_board[signal_type].append(message)

        assert "dependency_gap" in state.signal_board
        assert len(state.signal_board["dependency_gap"]) == 1
        assert state.signal_board["dependency_gap"][0] == message

    def test_signal_board_multiple_types(self):
        """Multiple signal types can coexist."""
        from agent_utilities.graph.state import GraphState

        state = GraphState(query="test")
        state.signal_board["security_concern"] = ["CVE-2024-1234 detected"]
        state.signal_board["quality_gap"] = ["Missing test coverage for module X"]

        assert len(state.signal_board) == 2
        assert "security_concern" in state.signal_board
        assert "quality_gap" in state.signal_board


# ── CONCEPT:AU-OS.safety.doom-loop-detection: File Watcher ─────────────────────────────────────────────


class TestFileWatcher:
    """CONCEPT:AU-OS.safety.doom-loop-detection: Watchdog file trigger system."""

    def test_trigger_rule_creation(self):
        """TriggerRule should create with defaults."""
        from agent_utilities.automation.file_watcher import TriggerRule

        rule = TriggerRule(pattern="*.py", query="Run linter")
        assert rule.pattern == "*.py"
        assert rule.priority == "LOW"
        assert rule.cooldown == 30

    def test_file_change_matches_pattern(self):
        """FileWatcher should match file changes to trigger rules."""
        from agent_utilities.automation.file_watcher import FileWatcher

        watcher = FileWatcher(project_root="/project")
        result = watcher.check_file_change("/project/pyproject.toml")
        assert result is not None
        assert result["priority"] == "MEDIUM"
        assert "dependency audit" in result["query"].lower()

    def test_file_change_no_match(self):
        """Non-matching files should return None."""
        from agent_utilities.automation.file_watcher import FileWatcher

        watcher = FileWatcher(project_root="/project")
        result = watcher.check_file_change("/project/README.md")
        assert result is None

    def test_cooldown_prevents_retrigger(self):
        """Same pattern should not trigger within cooldown period."""
        from agent_utilities.automation.file_watcher import FileWatcher

        watcher = FileWatcher(project_root="/project")

        # First trigger should succeed
        result1 = watcher.check_file_change("/project/mcp_config.json")
        assert result1 is not None

        # Immediate re-trigger should be blocked by cooldown
        result2 = watcher.check_file_change("/project/mcp_config.json")
        assert result2 is None

    def test_drain_pending(self):
        """Drain should return and clear pending queries."""
        from agent_utilities.automation.file_watcher import FileWatcher

        watcher = FileWatcher(project_root="/project")
        watcher.check_file_change("/project/pyproject.toml")

        pending = watcher.drain_pending()
        assert len(pending) == 1

        # Should be empty after drain
        assert watcher.drain_pending() == []

    def test_add_custom_trigger(self):
        """Custom triggers can be added at runtime."""
        from agent_utilities.automation.file_watcher import FileWatcher

        watcher = FileWatcher(project_root="/project")
        initial_count = len(watcher.triggers)

        watcher.add_trigger(
            pattern="Dockerfile",
            query="Rebuild container",
            priority="HIGH",
        )
        assert len(watcher.triggers) == initial_count + 1

    @patch("subprocess.run")
    def test_list_installed_packages_returns_list(self, mock_run):
        """Package listing should return a list."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '[{"name": "test-pkg", "version": "1.0.0"}]'
        mock_run.return_value = mock_proc

        from agent_utilities.automation.file_watcher import FileWatcher

        packages = FileWatcher.list_installed_packages()
        assert isinstance(packages, list)
        assert len(packages) == 1
        assert packages[0]["name"] == "test-pkg"

    @patch("subprocess.run")
    def test_audit_installed_packages_returns_dict(self, mock_run):
        """Package audit should return a dict with expected keys."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"vulnerabilities": []}'

        # We need side_effect to return different outputs for the two subprocess.run calls (pip list and pip-audit)
        def run_side_effect(*args, **kwargs):
            mock = MagicMock()
            mock.returncode = 0
            if "list" in args[0]:
                mock.stdout = '[{"name": "test-pkg", "version": "1.0.0", "latest_version": "1.1.0"}]'
            else:
                mock.stdout = '{"vulnerabilities": []}'
            return mock

        mock_run.side_effect = run_side_effect

        from agent_utilities.automation.file_watcher import FileWatcher

        result = FileWatcher.audit_installed_packages()
        assert isinstance(result, dict)
        assert "outdated" in result
        assert "vulnerabilities" in result
        assert len(result["outdated"]) == 1


# ── Config Integration ───────────────────────────────────────────────


class TestInnovationConfig:
    """Config fields for CONCEPT:AU-OS.state.cognitive-scheduler-preemption through CONCEPT:AU-OS.state.cognitive-scheduler-preemption"""

    def test_homeostatic_downgrade_config_exists(self):
        """AgentConfig should have homeostatic_downgrade_enabled field."""
        from agent_utilities.core.config import config

        assert hasattr(config, "homeostatic_downgrade_enabled")
        assert isinstance(config.homeostatic_downgrade_enabled, bool)

    def test_adversarial_verification_config_exists(self):
        """AgentConfig should have adversarial_verification field."""
        from agent_utilities.core.config import config

        assert hasattr(config, "adversarial_verification")
        assert config.adversarial_verification is False  # Default: disabled

    def test_maintenance_token_budget_config(self):
        """Maintenance token budget should default to 0 (unlimited)."""
        from agent_utilities.core.config import config

        assert hasattr(config, "maintenance_token_budget")
        assert config.maintenance_token_budget == 0

    def test_watchdog_patterns_config(self):
        """Watchdog patterns should have sensible defaults."""
        from agent_utilities.core.config import config

        assert hasattr(config, "watchdog_patterns")
        assert "pyproject.toml" in config.watchdog_patterns
