#!/usr/bin/python
from __future__ import annotations
"""Tests for Structured Retry Manager (CONCEPT:ORCH-1.3)."""


import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    from agent_utilities.security.execution_stability_engine import RetryManager

    return RetryManager()


@pytest.fixture
def success_config():
    from agent_utilities.security.execution_stability_engine import RetryConfig, SuccessCheck

    return RetryConfig(
        max_retries=3,
        checks=[SuccessCheck(command="echo 'test'")],
        timeout_seconds=10,
    )


@pytest.fixture
def fail_config():
    from agent_utilities.security.execution_stability_engine import RetryConfig, SuccessCheck

    return RetryConfig(
        max_retries=2,
        checks=[SuccessCheck(command="false")],
        on_failure="echo 'cleanup'",
        timeout_seconds=10,
        on_failure_timeout_seconds=5,
    )


# ---------------------------------------------------------------------------
# Shell command execution
# ---------------------------------------------------------------------------


class TestShellExecution:
    """Tests for shell command execution."""

    @pytest.mark.asyncio
    async def test_successful_command(self):
        from agent_utilities.security.execution_stability_engine import execute_shell_command

        result = await execute_shell_command("echo 'hello world'", timeout_seconds=10)
        assert result.success
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert not result.timed_out

    @pytest.mark.asyncio
    async def test_failing_command(self):
        from agent_utilities.security.execution_stability_engine import execute_shell_command

        result = await execute_shell_command("false", timeout_seconds=10)
        assert not result.success
        assert result.exit_code != 0
        assert not result.timed_out

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        from agent_utilities.security.execution_stability_engine import execute_shell_command

        result = await execute_shell_command("sleep 5", timeout_seconds=1)
        assert not result.success
        assert result.timed_out
        assert "timed out" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_command_with_stderr(self):
        from agent_utilities.security.execution_stability_engine import execute_shell_command

        result = await execute_shell_command(
            "echo 'error' >&2 && exit 1", timeout_seconds=10
        )
        assert not result.success
        assert "error" in result.stderr


# ---------------------------------------------------------------------------
# Success checks
# ---------------------------------------------------------------------------


class TestSuccessChecks:
    """Tests for success check execution."""

    @pytest.mark.asyncio
    async def test_all_checks_pass(self, manager, success_config):
        all_passed, results = await manager.execute_success_checks(success_config)
        assert all_passed
        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_check_fails(self, manager, fail_config):
        all_passed, results = await manager.execute_success_checks(fail_config)
        assert not all_passed
        assert len(results) == 1
        assert not results[0].success

    @pytest.mark.asyncio
    async def test_empty_checks(self, manager):
        from agent_utilities.security.execution_stability_engine import RetryConfig

        config = RetryConfig(checks=[])
        all_passed, results = await manager.execute_success_checks(config)
        assert all_passed
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_fail_fast(self, manager):
        from agent_utilities.security.execution_stability_engine import RetryConfig, SuccessCheck

        config = RetryConfig(
            checks=[
                SuccessCheck(command="false"),  # Fails
                SuccessCheck(command="echo 'never'"),  # Should not run
            ],
            timeout_seconds=10,
        )
        all_passed, results = await manager.execute_success_checks(config)
        assert not all_passed
        assert len(results) == 1  # Only one was run


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for handle_retry flow."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self, manager, success_config):
        from agent_utilities.security.execution_stability_engine import RetryResult

        outcome = await manager.handle_retry(success_config)
        assert outcome.result == RetryResult.SUCCESS
        assert outcome.attempts == 0

    @pytest.mark.asyncio
    async def test_failure_triggers_retry(self, manager, fail_config):
        from agent_utilities.security.execution_stability_engine import RetryResult

        outcome = await manager.handle_retry(fail_config)
        assert outcome.result == RetryResult.RETRIED
        assert outcome.attempts == 1

    @pytest.mark.asyncio
    async def test_max_retries_reached(self, manager, fail_config):
        from agent_utilities.security.execution_stability_engine import RetryResult

        # Exhaust retries
        for _ in range(fail_config.max_retries):
            outcome = await manager.handle_retry(fail_config)

        outcome = await manager.handle_retry(fail_config)
        assert outcome.result == RetryResult.MAX_ATTEMPTS_REACHED

    @pytest.mark.asyncio
    async def test_skip_when_no_checks(self, manager):
        from agent_utilities.security.execution_stability_engine import RetryConfig, RetryResult

        config = RetryConfig(checks=[])
        outcome = await manager.handle_retry(config)
        assert outcome.result == RetryResult.SKIPPED

    @pytest.mark.asyncio
    async def test_on_failure_hook_runs(self, manager, fail_config):
        outcome = await manager.handle_retry(fail_config)
        assert outcome.on_failure_result is not None
        assert outcome.on_failure_result.success  # "echo 'cleanup'" should succeed

    @pytest.mark.asyncio
    async def test_on_failure_hook_absent(self, manager):
        from agent_utilities.security.execution_stability_engine import RetryConfig, SuccessCheck

        config = RetryConfig(
            checks=[SuccessCheck(command="false")],
            on_failure=None,
            timeout_seconds=10,
        )
        outcome = await manager.handle_retry(config)
        assert outcome.on_failure_result is None


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for state reset."""

    @pytest.mark.asyncio
    async def test_reset_attempts(self, manager, fail_config):
        await manager.handle_retry(fail_config)
        assert manager.attempts == 1
        manager.reset()
        assert manager.attempts == 0


# ---------------------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------------------


class TestTimeoutConfig:
    """Tests for timeout resolution."""

    def test_config_timeout(self, manager, success_config):
        assert manager._get_check_timeout(success_config) == 10

    def test_default_timeout(self, manager):
        from agent_utilities.security.execution_stability_engine import (
            DEFAULT_RETRY_TIMEOUT_SECONDS,
            RetryConfig,
        )

        config = RetryConfig(timeout_seconds=None)
        assert manager._get_check_timeout(config) == DEFAULT_RETRY_TIMEOUT_SECONDS

    def test_env_timeout(self, manager, monkeypatch):
        from agent_utilities.security.execution_stability_engine import RetryConfig

        monkeypatch.setenv("AGENT_RETRY_TIMEOUT_SECONDS", "42")
        config = RetryConfig(timeout_seconds=None)
        assert manager._get_check_timeout(config) == 42

    def test_on_failure_timeout_config(self, manager, fail_config):
        assert manager._get_on_failure_timeout(fail_config) == 5


# ---------------------------------------------------------------------------
# Reward signal
# ---------------------------------------------------------------------------


class TestRewardSignal:
    """Tests for TeamConfig reward signal creation."""

    def test_success_reward(self, manager):
        from agent_utilities.security.execution_stability_engine import RetryOutcome, RetryResult

        outcome = RetryOutcome(result=RetryResult.SUCCESS, attempts=0)
        signal = manager.create_reward_signal(outcome, session_id="s1")
        assert signal["reward"] == 1.0
        assert signal["source"] == "execution_stability_engine"
        assert signal["session_id"] == "s1"

    def test_retried_reward(self, manager):
        from agent_utilities.security.execution_stability_engine import RetryOutcome, RetryResult

        outcome = RetryOutcome(result=RetryResult.RETRIED, attempts=1)
        signal = manager.create_reward_signal(outcome)
        assert signal["reward"] == 0.5

    def test_max_attempts_reward(self, manager):
        from agent_utilities.security.execution_stability_engine import RetryOutcome, RetryResult

        outcome = RetryOutcome(result=RetryResult.MAX_ATTEMPTS_REACHED, attempts=3)
        signal = manager.create_reward_signal(outcome)
        assert signal["reward"] == -0.5

    def test_skipped_reward(self, manager):
        from agent_utilities.security.execution_stability_engine import RetryOutcome, RetryResult

        outcome = RetryOutcome(result=RetryResult.SKIPPED, attempts=0)
        signal = manager.create_reward_signal(outcome)
        assert signal["reward"] == 0.0


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for Pydantic model validation."""

    def test_retry_config_defaults(self):
        from agent_utilities.security.execution_stability_engine import RetryConfig

        config = RetryConfig()
        assert config.max_retries == 3
        assert config.checks == []
        assert config.on_failure is None

    def test_success_check_model(self):
        from agent_utilities.security.execution_stability_engine import SuccessCheck

        check = SuccessCheck(command="pytest tests/ -q")
        assert check.type == "shell"
        assert check.command == "pytest tests/ -q"

    def test_shell_check_result(self):
        from agent_utilities.security.execution_stability_engine import ShellCheckResult

        result = ShellCheckResult(
            command="echo hi", success=True, exit_code=0, stdout="hi\n"
        )
        assert result.success
        assert not result.timed_out
