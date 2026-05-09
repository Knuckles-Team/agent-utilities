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

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

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
            "source": "repetition_guard",
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
        from agent_utilities.security.repetition_guard import RepetitionPolicy

        engine = PolicyEngine()
        engine.register(RepetitionPolicy())
        results = engine.evaluate(
            context={"tool_name": "shell", "tool_arguments": {"command": "ls"}}
        )
    """

    name: str = "repetition_guard"
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
