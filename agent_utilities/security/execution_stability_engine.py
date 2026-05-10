from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agent_utilities.models.knowledge_graph import DoomLoopIncidentNode

# --- Merged from execution_stability_engine.py ---

#!/usr/bin/python
"""Tool Repetition Guard (CONCEPT:OS-5.5).

Detects and prevents infinite tool call loops by tracking consecutive
identical calls and per-tool invocation counts.  Adapted from Goose's
``tool_monitor.rs`` and ``tool_inspection.rs``.

Key features:

* **Consecutive repeat detection** — blocks when the same tool is called
  N times in a row with identical arguments (configurable via
  ``MAX_TOOL_REPEATS``, default 3).
* **Per-tool call budget** — optional absolute cap on how many times
  any single tool can be called in a session (``MAX_TOOL_CALLS_PER_SESSION``).
* **ExperienceNode distillation** — when a repetition is detected,
  an ``ExperienceNode`` (CONCEPT:AHE-3.5) is created with the
  condition/action pair so the agent avoids the same loop pattern
  in future sessions.
* **PolicyEngine adapter** — ``RepetitionPolicy`` plugs into the
  existing guardrails system.
"""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class RepetitionVerdict(StrEnum):
    """Outcome of a repetition check."""

    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"


class RepetitionResult(BaseModel):
    """Result of checking a tool call for repetition.

    Attributes:
        verdict: ALLOW, WARN, or DENY.
        tool_name: The tool that was checked.
        consecutive_count: How many consecutive identical calls have occurred.
        total_count: Total invocations of this tool in the session.
        explanation: Human-readable explanation.
    """

    verdict: RepetitionVerdict = RepetitionVerdict.ALLOW
    tool_name: str = ""
    consecutive_count: int = 0
    total_count: int = 0
    explanation: str = ""


# ---------------------------------------------------------------------------
# RepetitionGuard
# ---------------------------------------------------------------------------


class RepetitionGuard:
    """Tracks tool call patterns and detects repetitive loops.

    CONCEPT:OS-5.5 — Tool Repetition Guard

    Adapted from Goose's ``RepetitionInspector`` (Rust) with the
    following design choices:

    * **Argument hashing** — uses SHA-256 digests of serialized
      arguments so memory stays bounded regardless of payload size.
    * **Configurable thresholds** — consecutive repeat limit via
      ``MAX_TOOL_REPEATS`` (default 3), per-session cap via
      ``MAX_TOOL_CALLS_PER_SESSION`` (default 50).
    * **Warn-before-deny** — at ``max_consecutive - 1`` calls,
      a WARN verdict is returned; at ``max_consecutive``, DENY.
    * **ExperienceNode integration** — on DENY, creates a KG-
      persistable ``ExperienceNode`` capturing the loop pattern
      for future avoidance (CONCEPT:AHE-3.5).

    Example::

        guard = RepetitionGuard()
        for i in range(5):
            result = guard.check_tool_call("shell", {"command": "ls"})
            if result.verdict == RepetitionVerdict.DENY:
                print(f"Blocked: {result.explanation}")
                break
    """

    def __init__(
        self,
        max_consecutive_repeats: int | None = None,
        max_calls_per_session: int | None = None,
    ) -> None:
        self._max_consecutive = max_consecutive_repeats or int(
            os.environ.get("MAX_TOOL_REPEATS", "3")
        )
        self._max_per_session = max_calls_per_session or int(
            os.environ.get("MAX_TOOL_CALLS_PER_SESSION", "50")
        )

        # State tracking
        self._last_tool_name: str = ""
        self._last_arg_hash: str = ""
        self._consecutive_count: int = 0
        self._tool_counts: dict[str, int] = defaultdict(int)
        self._tool_arg_history: dict[str, list[str]] = defaultdict(list)

    @property
    def max_consecutive_repeats(self) -> int:
        """Maximum allowed consecutive identical calls."""
        return self._max_consecutive

    @property
    def max_calls_per_session(self) -> int:
        """Maximum allowed total calls per tool per session."""
        return self._max_per_session

    def _hash_arguments(self, arguments: dict[str, Any] | None) -> str:
        """Create a deterministic hash of tool arguments."""
        if not arguments:
            return "empty"
        import json

        normalized = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def check_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> RepetitionResult:
        """Check a tool call for repetition.

        Args:
            tool_name: Name of the tool being called.
            arguments: Tool call arguments dict.

        Returns:
            RepetitionResult with verdict and context.
        """
        arg_hash = self._hash_arguments(arguments)

        # Update total counts
        self._tool_counts[tool_name] += 1
        self._tool_arg_history[tool_name].append(arg_hash)
        total = self._tool_counts[tool_name]

        # Check consecutive repetition
        if tool_name == self._last_tool_name and arg_hash == self._last_arg_hash:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._last_tool_name = tool_name
            self._last_arg_hash = arg_hash

        consecutive = self._consecutive_count

        # Check per-session budget
        if total > self._max_per_session:
            return RepetitionResult(
                verdict=RepetitionVerdict.DENY,
                tool_name=tool_name,
                consecutive_count=consecutive,
                total_count=total,
                explanation=(
                    f"Tool '{tool_name}' exceeded session budget: "
                    f"{total}/{self._max_per_session} calls"
                ),
            )

        # Check consecutive limit
        if consecutive >= self._max_consecutive:
            return RepetitionResult(
                verdict=RepetitionVerdict.DENY,
                tool_name=tool_name,
                consecutive_count=consecutive,
                total_count=total,
                explanation=(
                    f"Tool '{tool_name}' called {consecutive} times consecutively "
                    f"with identical arguments (limit: {self._max_consecutive})"
                ),
            )

        # Warn one step before deny
        if consecutive == self._max_consecutive - 1 and self._max_consecutive > 1:
            return RepetitionResult(
                verdict=RepetitionVerdict.WARN,
                tool_name=tool_name,
                consecutive_count=consecutive,
                total_count=total,
                explanation=(
                    f"Tool '{tool_name}' called {consecutive} times consecutively "
                    f"— approaching limit ({self._max_consecutive})"
                ),
            )

        return RepetitionResult(
            verdict=RepetitionVerdict.ALLOW,
            tool_name=tool_name,
            consecutive_count=consecutive,
            total_count=total,
        )

    def reset(self) -> None:
        """Reset all tracking state (e.g., on new conversation turn)."""
        self._last_tool_name = ""
        self._last_arg_hash = ""
        self._consecutive_count = 0
        self._tool_counts.clear()
        self._tool_arg_history.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get current per-tool call statistics.

        Returns:
            Dict with per-tool counts and consecutive state.
        """
        return {
            "tool_counts": dict(self._tool_counts),
            "current_tool": self._last_tool_name,
            "consecutive_count": self._consecutive_count,
            "max_consecutive_repeats": self._max_consecutive,
            "max_calls_per_session": self._max_per_session,
        }

    def create_experience_node(
        self,
        result: RepetitionResult,
        session_id: str = "",
    ) -> dict[str, Any] | None:
        """Create a KG-persistable ExperienceNode from a DENY result.

        CONCEPT:AHE-3.5 — Experience Node Architecture

        Converts a detected repetition loop into a tactical rule so the
        agent avoids the same pattern in future sessions.

        Returns:
            Dict with ExperienceNode data, or None if verdict is not DENY.
        """
        if result.verdict != RepetitionVerdict.DENY:
            return None

        return {
            "id": f"exp:{uuid.uuid4().hex[:8]}",
            "type": "experience",
            "condition": (
                f"Tool '{result.tool_name}' called {result.consecutive_count} "
                f"times consecutively with identical arguments"
            ),
            "action": (
                f"Break the loop by trying an alternative approach. "
                f"Do not call '{result.tool_name}' with the same arguments again."
            ),
            "source": "execution_stability_engine",
            "confidence": 0.95,
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# PolicyEngine adapter
# ---------------------------------------------------------------------------


@dataclass
class RepetitionPolicy:
    """PolicyEngine-compatible adapter for the RepetitionGuard.

    CONCEPT:OS-5.5 — Tool Repetition Guard

    Plugs into the existing :class:`PolicyEngine` from ``guardrails.py``.
    Uses the ``context`` dict to extract ``tool_name`` and ``tool_arguments``
    for repetition checking.

    Example::

        from agent_utilities.security.guardrails import PolicyEngine
        from agent_utilities.security.execution_stability_engine import RepetitionPolicy

        engine = PolicyEngine()
        engine.register(RepetitionPolicy())
        results = engine.evaluate(
            context={"tool_name": "shell", "tool_arguments": {"command": "ls"}}
        )
    """

    name: str = "execution_stability_engine"
    guard: RepetitionGuard = field(default_factory=RepetitionGuard)

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Evaluate for tool repetition."""
        from agent_utilities.security.guardrails import PolicyResult

        if not context:
            return PolicyResult(allowed=True, policy_name=self.name)

        tool_name = context.get("tool_name", "")
        tool_args = context.get("tool_arguments")

        if not tool_name:
            return PolicyResult(allowed=True, policy_name=self.name)

        result = self.guard.check_tool_call(tool_name, tool_args)

        if result.verdict == RepetitionVerdict.DENY:
            return PolicyResult(
                allowed=False,
                policy_name=self.name,
                reason=result.explanation,
                severity="block",
                metadata={
                    "consecutive_count": result.consecutive_count,
                    "total_count": result.total_count,
                    "tool_name": result.tool_name,
                },
            )

        if result.verdict == RepetitionVerdict.WARN:
            return PolicyResult(
                allowed=True,
                policy_name=self.name,
                reason=result.explanation,
                severity="warn",
                metadata={
                    "consecutive_count": result.consecutive_count,
                    "total_count": result.total_count,
                    "tool_name": result.tool_name,
                },
            )

        return PolicyResult(allowed=True, policy_name=self.name)


# --- Merged from execution_stability_engine.py ---

#!/usr/bin/python
"""Enhanced Doom-Loop Detector.

CONCEPT:OS-5.18 — Enhanced Doom-Loop Detector

Extends the existing Tool Repetition Guard (CONCEPT:OS-5.5) with
pattern-aware doom-loop detection adapted from ml-intern's doom_loop.py.

Key enhancements over OS-5.5:

* **Result-aware signatures** — includes tool result hashes to distinguish
  legitimate polling (same args, different results) from true loops.
* **Sequence pattern detection** — detects repeating multi-tool sequences
  like [A,B,A,B] in addition to simple consecutive repeats.
* **Corrective prompt generation** — produces context-aware prompts to
  break detected loops.
* **KG integration** — creates ``DoomLoopIncidentNode`` for persistence.
"""


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallSignature:
    """Hashable signature for a single tool call plus its observed result.

    Including the result hash prevents legitimate polling from being
    classified as a doom loop when the arguments stay constant but
    the observed result keeps changing.
    """

    name: str
    args_hash: str
    result_hash: str | None = None


def _normalize_args(args: dict[str, Any] | str | None) -> str:
    """Canonicalize tool-call arguments before hashing.

    LLMs can emit semantically-identical JSON with different key orderings
    or whitespace. We parse-and-redump with ``sort_keys=True`` plus compact
    separators so trivially-different spellings collapse to the same form.
    """
    if not args:
        return ""
    if isinstance(args, dict):
        return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    try:
        return json.dumps(json.loads(str(args)), sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return str(args)


def _hash_string(s: str) -> str:
    """Return a short hash of the given string."""
    return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:12]


class DoomLoopDetector:
    """Pattern-aware doom-loop detector with corrective prompt generation.

    CONCEPT:OS-5.18 — Enhanced Doom-Loop Detector

    Complements the existing ``RepetitionGuard`` (OS-5.5) with:

    * Result-aware tool call signatures
    * Repeating sequence detection (patterns of length 2-5)
    * Corrective prompt injection

    Example::

        detector = DoomLoopDetector()
        for call in tool_calls:
            detector.record_call(call.name, call.args, call.result)
            incident = detector.check()
            if incident:
                print(f"Doom loop: {incident.corrective_prompt}")
    """

    def __init__(
        self,
        consecutive_threshold: int = 3,
        lookback_window: int = 30,
        session_id: str = "",
    ):
        """Initialize the detector.

        Args:
            consecutive_threshold: Number of identical consecutive calls
                before triggering detection.
            lookback_window: Number of recent signatures to analyze.
            session_id: Current session identifier for KG persistence.
        """
        self._consecutive_threshold = consecutive_threshold
        self._lookback = lookback_window
        self._session_id = session_id
        self._signatures: list[ToolCallSignature] = []

    def record_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str | None = None,
        result: str | None = None,
    ) -> None:
        """Record a tool call signature.

        Args:
            tool_name: Name of the tool called.
            arguments: Tool call arguments.
            result: Tool call result (used to distinguish polling from loops).
        """
        args_hash = _hash_string(_normalize_args(arguments))
        result_hash = _hash_string(result) if result else None

        sig = ToolCallSignature(
            name=tool_name,
            args_hash=args_hash,
            result_hash=result_hash,
        )
        self._signatures.append(sig)

        # Trim to lookback window
        if len(self._signatures) > self._lookback * 2:
            self._signatures = self._signatures[-self._lookback :]

    def _detect_identical_consecutive(self) -> str | None:
        """Detect N+ identical consecutive calls.

        Returns:
            Tool name if threshold is exceeded, None otherwise.
        """
        sigs = self._signatures[-self._lookback :]
        if len(sigs) < self._consecutive_threshold:
            return None

        count = 1
        for i in range(1, len(sigs)):
            if sigs[i] == sigs[i - 1]:
                count += 1
                if count >= self._consecutive_threshold:
                    return sigs[i].name
            else:
                count = 1

        return None

    def _detect_repeating_sequence(self) -> list[ToolCallSignature] | None:
        """Detect repeating patterns like [A,B,A,B] for sequences of length 2-5.

        Returns:
            The repeating pattern if found, None otherwise.
        """
        sigs = self._signatures[-self._lookback :]
        n = len(sigs)

        for seq_len in range(2, 6):
            min_required = seq_len * 2
            if n < min_required:
                continue

            tail = sigs[-min_required:]
            pattern = tail[:seq_len]

            # Count repetitions from the end
            reps = 0
            for start in range(n - seq_len, -1, -seq_len):
                chunk = sigs[start : start + seq_len]
                if chunk == pattern:
                    reps += 1
                else:
                    break

            if reps >= 2:
                return pattern

        return None

    def check(self) -> DoomLoopIncidentNode | None:
        """Check for doom loop patterns.

        Returns:
            DoomLoopIncidentNode if a pattern is detected, None otherwise.
        """
        if len(self._signatures) < self._consecutive_threshold:
            return None

        # Check for identical consecutive calls
        tool_name = self._detect_identical_consecutive()
        if tool_name:
            logger.warning(
                "Doom-loop detected: %d+ identical consecutive calls to '%s'",
                self._consecutive_threshold,
                tool_name,
            )
            corrective = (
                f"[SYSTEM: DOOM-LOOP GUARD] You have called '{tool_name}' with the same "
                f"arguments multiple times in a row, getting the same result each time. "
                f"STOP repeating this approach — it is not working. "
                f"Step back and try a fundamentally different strategy. "
                f"Consider: using a different tool, changing your arguments significantly, "
                f"or explaining to the user what you're stuck on and asking for guidance."
            )
            return DoomLoopIncidentNode(
                id=f"doom_{uuid.uuid4().hex[:8]}",
                name=f"Doom loop: {tool_name}",
                description=f"Detected {self._consecutive_threshold}+ identical consecutive calls",
                timestamp=datetime.now(UTC).isoformat(),
                pattern_type="consecutive",
                tool_names=[tool_name],
                signature_hashes=[
                    s.args_hash
                    for s in self._signatures[-self._consecutive_threshold :]
                ],
                repetition_count=self._consecutive_threshold,
                corrective_prompt=corrective,
                session_id=self._session_id,
            )

        # Check for repeating sequences
        pattern = self._detect_repeating_sequence()
        if pattern:
            pattern_desc = " → ".join(s.name for s in pattern)
            logger.warning("Doom-loop detected: repeating sequence [%s]", pattern_desc)
            corrective = (
                f"[SYSTEM: DOOM-LOOP GUARD] You are stuck in a repeating cycle of tool calls: "
                f"[{pattern_desc}]. This pattern has repeated multiple times without progress. "
                f"STOP this cycle and try a fundamentally different approach. "
                f"Consider: breaking down the problem differently, using alternative tools, "
                f"or explaining to the user what you're stuck on and asking for guidance."
            )
            return DoomLoopIncidentNode(
                id=f"doom_{uuid.uuid4().hex[:8]}",
                name=f"Doom loop: {pattern_desc}",
                description="Detected repeating sequence pattern",
                timestamp=datetime.now(UTC).isoformat(),
                pattern_type="sequence",
                tool_names=list({s.name for s in pattern}),
                signature_hashes=[s.args_hash for s in pattern],
                repetition_count=len(pattern),
                corrective_prompt=corrective,
                session_id=self._session_id,
            )

        return None

    def reset(self) -> None:
        """Clear all recorded signatures."""
        self._signatures.clear()

    @property
    def signature_count(self) -> int:
        """Number of recorded tool call signatures."""
        return len(self._signatures)


# --- Merged from execution_stability_engine.py ---

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
            "source": "execution_stability_engine",
            "attempts": outcome.attempts,
            "result": outcome.result.value,
            "session_id": session_id,
        }
