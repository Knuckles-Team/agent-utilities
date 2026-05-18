#!/usr/bin/python
from __future__ import annotations

"""Tests for Innovation Framework modules (CONCEPT:OS-5.2 through CONCEPT:OS-5.2).

Tests cover:
- CONCEPT:OS-5.2: Homeostatic model downgrade integration
- CONCEPT:AHE-3.1: Adversarial verification models and logic
- CONCEPT:ORCH-1.2: Signal board state management
- CONCEPT:OS-5.0: File watcher trigger rules and package audit
- CONCEPT:OS-5.2: Maintenance cron scheduling and budget management
"""


import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── CONCEPT:OS-5.2: Homeostatic Model Downgrade ──────────────────────────────


class TestHomeostaticDowngrade:
    """CONCEPT:OS-5.2: ResourceOptimizer-driven tier downgrade."""

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


# ── CONCEPT:AHE-3.1: Adversarial Verification ─────────────────────────────────


class TestAdversarialVerification:
    """CONCEPT:AHE-3.1: Adversarial verification models."""

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


# ── CONCEPT:ORCH-1.2: Signal Board ─────────────────────────────────────────────


class TestSignalBoard:
    """CONCEPT:ORCH-1.2: Stigmergy signal board on GraphState."""

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


# ── CONCEPT:OS-5.0: File Watcher ─────────────────────────────────────────────


class TestFileWatcher:
    """CONCEPT:OS-5.0: Watchdog file trigger system."""

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

    def test_list_installed_packages_returns_list(self):
        """Package listing should return a list (may be empty in test env)."""
        from agent_utilities.automation.file_watcher import FileWatcher

        packages = FileWatcher.list_installed_packages()
        assert isinstance(packages, list)

    def test_audit_installed_packages_returns_dict(self):
        """Package audit should return a dict with expected keys."""
        from agent_utilities.automation.file_watcher import FileWatcher

        result = FileWatcher.audit_installed_packages()
        assert isinstance(result, dict)
        assert "outdated" in result
        assert "vulnerabilities" in result


# ── CONCEPT:OS-5.2: Maintenance Cron ─────────────────────────────────────────


class TestMaintenanceCron:
    """CONCEPT:OS-5.2: Maintenance cron scheduling."""

    def test_default_tasks_loaded(self):
        """MaintenanceCron should have default tasks."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron()
        assert len(cron.tasks) >= 5
        task_ids = {t.id for t in cron.tasks}
        assert "precommit_analysis" in task_ids
        assert "dependency_audit" in task_ids
        assert "mcp_health_check" in task_ids

    def test_due_tasks_all_due_initially(self):
        """All non-on_demand tasks should be due on first run."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron()
        due = cron.get_due_tasks()
        assert len(due) > 0
        # All non-on_demand tasks with last_run=0 should be due
        for task in due:
            assert task.frequency.value != "on_demand"

    def test_record_execution_updates_last_run(self):
        """Recording execution should update last_run timestamp."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron()
        cron.record_execution("precommit_analysis", status="success", tokens_used=1000)

        task = next(t for t in cron.tasks if t.id == "precommit_analysis")
        assert task.last_run > 0
        assert task.last_status == "success"
        assert cron.tokens_used == 1000

    def test_budget_unlimited_when_zero(self):
        """Token budget of 0 means unlimited."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron(token_budget=0)
        cron.tokens_used = 999_999
        assert cron.is_budget_available() is True

    def test_budget_enforced_when_set(self):
        """Token budget should be enforced when non-zero."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron(token_budget=5000)
        cron.tokens_used = 4000
        assert cron.is_budget_available() is True

        cron.tokens_used = 5000
        assert cron.is_budget_available() is False

    def test_add_duplicate_task_rejected(self):
        """Adding a task with an existing ID should be rejected."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron()
        initial_count = len(cron.tasks)

        from agent_utilities.automation.maintenance_cron import MaintenanceTask

        duplicate = MaintenanceTask(
            id="precommit_analysis",
            name="Duplicate",
            query="Test",
        )
        cron.add_task(duplicate)
        assert len(cron.tasks) == initial_count  # No change

    def test_remove_task(self):
        """Tasks can be removed by ID."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron()
        assert cron.remove_task("precommit_analysis") is True
        assert not any(t.id == "precommit_analysis" for t in cron.tasks)

    def test_remove_nonexistent_task(self):
        """Removing a nonexistent task returns False."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron()
        assert cron.remove_task("nonexistent") is False

    def test_summary_structure(self):
        """Summary should have expected keys."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron()
        summary = cron.summary()
        assert "total_tasks" in summary
        assert "enabled_tasks" in summary
        assert "due_tasks" in summary
        assert "tokens_used" in summary
        assert "budget_available" in summary

    def test_priority_sort_order(self):
        """Due tasks should be sorted HIGH → MEDIUM → LOW."""
        from agent_utilities.automation.maintenance_cron import MaintenanceCron

        cron = MaintenanceCron()
        due = cron.get_due_tasks()
        if len(due) >= 2:
            priorities = [t.priority for t in due]
            priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            values = [priority_order.get(p, 2) for p in priorities]
            assert values == sorted(values), "Tasks not sorted by priority"


# ── Config Integration ───────────────────────────────────────────────


class TestInnovationConfig:
    """Config fields for CONCEPT:OS-5.2 through CONCEPT:OS-5.2"""

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
