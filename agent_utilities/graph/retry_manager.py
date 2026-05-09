#!/usr/bin/python
"""Structured Retry Manager (CONCEPT:AHE-3.11).

Provides structured retry logic with configurable success checks,
on-failure hooks, and timeout management.  Adapted from Goose's
``retry.rs`` with Python-native subprocess execution and integration
into the graph executor's verification pipeline.

Key features:

* **Shell-based success checks** — configurable commands that must exit
  with code 0 for the run to be considered successful (e.g.,
  ``pytest tests/``, ``mypy src/``).
* **On-failure hooks** — optional cleanup commands that run before
  each retry (e.g., ``git checkout .``).
* **Configurable timeouts** — per-check and per-hook timeout durations
  with environment variable overrides.
* **TeamConfig reward integration** — retry outcomes feed into
  ``TeamConfigNode.record_team_outcome()`` (CONCEPT:AHE-3.3) for
  routing improvement.
"""

from __future__ import annotations

import asyncio
import logging
import os
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_RETRY_TIMEOUT_SECONDS: int = 300
DEFAULT_ON_FAILURE_TIMEOUT_SECONDS: int = 120

ENV_RETRY_TIMEOUT = "AGENT_RETRY_TIMEOUT_SECONDS"
ENV_ON_FAILURE_TIMEOUT = "AGENT_ON_FAILURE_TIMEOUT_SECONDS"


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class SuccessCheck(BaseModel):
    """A shell command that must pass for the run to be considered successful.

    Attributes:
        type: Check type (currently only ``shell`` is supported).
        command: Shell command to execute.
    """

    type: str = "shell"
    command: str


class RetryConfig(BaseModel):
    """Configuration for the retry manager.

    CONCEPT:AHE-3.11 — Structured Retry Manager

    Attributes:
        max_retries: Maximum number of retry attempts (default 3).
        checks: List of success checks that must pass.
        on_failure: Optional shell command to run before each retry.
        timeout_seconds: Timeout for each success check (default 300).
        on_failure_timeout_seconds: Timeout for on_failure hook (default 120).
    """

    max_retries: int = 3
    checks: list[SuccessCheck] = Field(default_factory=list)
    on_failure: str | None = None
    timeout_seconds: int | None = None
    on_failure_timeout_seconds: int | None = None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class RetryResult(StrEnum):
    """Outcome of a retry evaluation."""

    SKIPPED = "skipped"
    SUCCESS = "success"
    RETRIED = "retried"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"


class ShellCheckResult(BaseModel):
    """Result of executing a shell check command.

    Attributes:
        command: The command that was executed.
        success: Whether the command exited with code 0.
        exit_code: The process exit code.
        stdout: Standard output (truncated).
        stderr: Standard error (truncated).
        timed_out: Whether the command timed out.
    """

    command: str = ""
    success: bool = False
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class RetryOutcome(BaseModel):
    """Full outcome of a retry evaluation cycle.

    Attributes:
        result: The retry result enum.
        attempts: Current attempt count.
        check_results: Results of each success check.
        on_failure_result: Result of the on_failure hook, if executed.
    """

    result: RetryResult
    attempts: int = 0
    check_results: list[ShellCheckResult] = Field(default_factory=list)
    on_failure_result: ShellCheckResult | None = None


# ---------------------------------------------------------------------------
# Shell command execution
# ---------------------------------------------------------------------------


async def execute_shell_command(
    command: str,
    timeout_seconds: int = DEFAULT_RETRY_TIMEOUT_SECONDS,
) -> ShellCheckResult:
    """Execute a shell command with timeout.

    Uses ``asyncio.create_subprocess_shell`` for non-blocking execution
    with a hard timeout.

    Args:
        command: Shell command to execute.
        timeout_seconds: Maximum execution time in seconds.

    Returns:
        ShellCheckResult with exit code, stdout, stderr.
    """
    logger.debug(
        "Executing shell command with timeout %ds: %s",
        timeout_seconds,
        command,
    )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "AGENT": "agent-utilities"},
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(
                "Shell command timed out after %ds: %s",
                timeout_seconds,
                command,
            )
            return ShellCheckResult(
                command=command,
                success=False,
                exit_code=-1,
                timed_out=True,
                stderr=f"Command timed out after {timeout_seconds}s",
            )

        exit_code = proc.returncode or 0
        stdout = stdout_bytes.decode("utf-8", errors="replace")[:5000]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:5000]

        logger.debug(
            "Shell command completed: exit_code=%d, stdout=%d chars, stderr=%d chars",
            exit_code,
            len(stdout),
            len(stderr),
        )

        return ShellCheckResult(
            command=command,
            success=exit_code == 0,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

    except Exception as exc:
        logger.error("Shell command execution error: %s", exc)
        return ShellCheckResult(
            command=command,
            success=False,
            exit_code=-1,
            stderr=str(exc),
        )


# ---------------------------------------------------------------------------
# RetryManager
# ---------------------------------------------------------------------------


class RetryManager:
    """Manages retry state and operations for agent execution.

    CONCEPT:AHE-3.11 — Structured Retry Manager

    Adapted from Goose's ``RetryManager`` (Rust) with the following
    design:

    * **Async-native** — uses ``asyncio.create_subprocess_shell``
      for non-blocking check execution.
    * **Configurable timeouts** — per-check and per-hook timeouts
      with env var overrides (``AGENT_RETRY_TIMEOUT_SECONDS``,
      ``AGENT_ON_FAILURE_TIMEOUT_SECONDS``).
    * **TeamConfig integration** — retry outcomes can be fed into
      the ``TeamConfigNode.record_team_outcome()`` reward system.

    Example::

        config = RetryConfig(
            max_retries=3,
            checks=[SuccessCheck(command="pytest tests/ -q")],
            on_failure="git checkout .",
        )
        manager = RetryManager()
        outcome = await manager.handle_retry(config)
        if outcome.result == RetryResult.SUCCESS:
            print("All checks passed!")
    """

    def __init__(self) -> None:
        self._attempts: int = 0

    @property
    def attempts(self) -> int:
        """Current retry attempt count."""
        return self._attempts

    def reset(self) -> None:
        """Reset the attempt counter."""
        self._attempts = 0

    def _get_check_timeout(self, config: RetryConfig) -> int:
        """Resolve check timeout: config → env → default."""
        if config.timeout_seconds is not None:
            return config.timeout_seconds
        try:
            return int(os.environ.get(ENV_RETRY_TIMEOUT, ""))
        except (ValueError, TypeError):
            return DEFAULT_RETRY_TIMEOUT_SECONDS

    def _get_on_failure_timeout(self, config: RetryConfig) -> int:
        """Resolve on_failure timeout: config → env → default."""
        if config.on_failure_timeout_seconds is not None:
            return config.on_failure_timeout_seconds
        try:
            return int(os.environ.get(ENV_ON_FAILURE_TIMEOUT, ""))
        except (ValueError, TypeError):
            return DEFAULT_ON_FAILURE_TIMEOUT_SECONDS

    async def execute_success_checks(
        self, config: RetryConfig
    ) -> tuple[bool, list[ShellCheckResult]]:
        """Execute all success checks and return results.

        Args:
            config: RetryConfig with checks to execute.

        Returns:
            Tuple of (all_passed, check_results).
        """
        if not config.checks:
            return True, []

        timeout = self._get_check_timeout(config)
        results: list[ShellCheckResult] = []
        all_passed = True

        for check in config.checks:
            result = await execute_shell_command(check.command, timeout)
            results.append(result)
            if not result.success:
                logger.warning(
                    "Success check failed: '%s' exited with code %d. stderr: %s",
                    check.command,
                    result.exit_code,
                    result.stderr[:200],
                )
                all_passed = False
                break  # Fail fast — no need to run remaining checks

        return all_passed, results

    async def execute_on_failure(self, config: RetryConfig) -> ShellCheckResult | None:
        """Execute the on_failure hook command.

        Args:
            config: RetryConfig with on_failure command.

        Returns:
            ShellCheckResult, or None if no on_failure command is set.
        """
        if not config.on_failure:
            return None

        timeout = self._get_on_failure_timeout(config)
        logger.info(
            "Executing on_failure hook with timeout %ds: %s",
            timeout,
            config.on_failure,
        )
        result = await execute_shell_command(config.on_failure, timeout)

        if not result.success:
            logger.warning(
                "on_failure hook failed: '%s' exited with code %d",
                config.on_failure,
                result.exit_code,
            )

        return result

    async def handle_retry(
        self,
        config: RetryConfig,
    ) -> RetryOutcome:
        """Execute the full retry logic cycle.

        1. Run all success checks.
        2. If all pass → SUCCESS.
        3. If any fail:
           a. Check if max retries reached → MAX_ATTEMPTS_REACHED.
           b. Run on_failure hook (if configured).
           c. Increment attempts → RETRIED.

        Args:
            config: RetryConfig with checks, limits, and hooks.

        Returns:
            RetryOutcome with result and check details.
        """
        if not config.checks:
            return RetryOutcome(
                result=RetryResult.SKIPPED,
                attempts=self._attempts,
            )

        # Execute checks
        all_passed, check_results = await self.execute_success_checks(config)

        if all_passed:
            logger.info("All %d success checks passed", len(config.checks))
            return RetryOutcome(
                result=RetryResult.SUCCESS,
                attempts=self._attempts,
                check_results=check_results,
            )

        # Checks failed — check retry budget
        if self._attempts >= config.max_retries:
            logger.warning(
                "Maximum retry attempts (%d) reached",
                config.max_retries,
            )
            return RetryOutcome(
                result=RetryResult.MAX_ATTEMPTS_REACHED,
                attempts=self._attempts,
                check_results=check_results,
            )

        # Run on_failure hook
        on_failure_result = await self.execute_on_failure(config)

        # Increment and signal retry
        self._attempts += 1
        logger.info("Retry attempt %d/%d", self._attempts, config.max_retries)

        return RetryOutcome(
            result=RetryResult.RETRIED,
            attempts=self._attempts,
            check_results=check_results,
            on_failure_result=on_failure_result,
        )

    def create_reward_signal(
        self, outcome: RetryOutcome, session_id: str = ""
    ) -> dict[str, Any]:
        """Create a reward signal from a retry outcome for TeamConfig integration.

        CONCEPT:AHE-3.3 — TeamConfig Promotion

        Maps retry outcomes to reward values:
        * SUCCESS → +1.0
        * RETRIED (eventually succeeds) → +0.5
        * MAX_ATTEMPTS_REACHED → -0.5
        * SKIPPED → 0.0

        Returns:
            Dict with reward signal data for ``record_team_outcome()``.
        """
        reward_map = {
            RetryResult.SUCCESS: 1.0,
            RetryResult.RETRIED: 0.5,
            RetryResult.MAX_ATTEMPTS_REACHED: -0.5,
            RetryResult.SKIPPED: 0.0,
        }

        return {
            "reward": reward_map.get(outcome.result, 0.0),
            "source": "retry_manager",
            "attempts": outcome.attempts,
            "result": outcome.result.value,
            "session_id": session_id,
        }
