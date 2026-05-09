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

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agent_utilities.models.knowledge_graph import DoomLoopIncidentNode

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
