#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-OS.governance.reactive-multi-axis-budget — Reactive Multi-Axis Budget Guardrails.

Governs cost, token, and wall-clock execution limits dynamically. Integrates
with OS-5.4 token usage analytics to prevent runaway agent execution paths and
maintain systemic guardrails.

Supports:
    - Time-based ceilings (wall-clock elapsed time).
    - Token ceilings (accumulated prompt, response, thought, and tool tokens).
    - Spend ceilings (USD estimated model calling and API pricing).
"""

import logging
import time
from typing import Any

from ...observability.token_tracker import TokenUsageTracker
from .ledger import EventLedger

logger = logging.getLogger(__name__)


class BudgetTrippedException(Exception):
    """Exception raised when an agent execution path breaches budget limits.

    CONCEPT:AU-OS.safety.self-healing-guardrails — Resilient Self-Healing & Guardrails.
    """

    def __init__(
        self, message: str, limit_type: str, limit_value: Any, current_value: Any
    ) -> None:
        super().__init__(message)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value


class BudgetGuard:
    """Three-axis sensory budget governor (Time, Tokens, USD Cost)."""

    def __init__(
        self,
        max_time_seconds: float | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        token_tracker: TokenUsageTracker | None = None,
        prompt_cost_per_token: float = 0.000003,  # Default standard model rates (e.g. GPT-4o / Claude Sonnet)
        response_cost_per_token: float = 0.000015,
    ) -> None:
        """Initialize the Budget Guard.

        Args:
            max_time_seconds: Max elapsed execution wall-clock time.
            max_tokens: Max total tokens (prompt + response + thoughts + tools).
            max_cost_usd: Max estimated total USD spend.
            token_tracker: Instance of TokenUsageTracker for token analytics.
            prompt_cost_per_token: Cost rate per input prompt token.
            response_cost_per_token: Cost rate per output response/thought token.
        """
        self.max_time_seconds = max_time_seconds
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self._token_tracker = token_tracker
        self.prompt_cost_per_token = prompt_cost_per_token
        self.response_cost_per_token = response_cost_per_token

        self._start_time = time.monotonic()

    @property
    def token_tracker(self) -> TokenUsageTracker:
        """Retrieve the active TokenUsageTracker."""
        if self._token_tracker is None:
            self._token_tracker = TokenUsageTracker()
        return self._token_tracker

    def get_elapsed_time(self) -> float:
        """Calculate elapsed seconds since initialization."""
        return time.monotonic() - self._start_time

    def check_limits(self, run_id: str, ledger: EventLedger | None = None) -> None:
        """Assess running constraints and trip exceptions if limits are breached.

        Ontologically records budget alerts as standard 'Event' nodes in the
        ledger to feed downstream self-healing decision heuristics.

        Args:
            run_id: Unique identifier for the execution run.
            ledger: Optional EventLedger to log breach events.

        Raises:
            BudgetTrippedException: If any boundary limit is exceeded.
        """
        # 1. Wall-clock Time check
        elapsed = self.get_elapsed_time()
        if self.max_time_seconds is not None and elapsed > self.max_time_seconds:
            msg = f"Time limit exceeded: {elapsed:.2f}s elapsed (limit: {self.max_time_seconds}s)"
            self._log_and_trip(
                run_id=run_id,
                limit_type="time",
                limit_value=self.max_time_seconds,
                current_value=elapsed,
                message=msg,
                ledger=ledger,
            )

        # 2. Token tracker analytics checks
        # Retrieve session records mapped directly under run_id
        session_records = self.token_tracker._by_session.get(run_id, [])
        total_prompt = sum(r.prompt_tokens for r in session_records)
        total_response = sum(r.response_tokens for r in session_records)
        total_thoughts = sum(r.thoughts_tokens for r in session_records)
        total_tool_use = sum(r.tool_use_tokens for r in session_records)
        total_tokens = sum(r.total_tokens for r in session_records)

        # Token limit check
        if self.max_tokens is not None and total_tokens > self.max_tokens:
            msg = f"Token limit exceeded: {total_tokens} tokens used (limit: {self.max_tokens})"
            self._log_and_trip(
                run_id=run_id,
                limit_type="tokens",
                limit_value=self.max_tokens,
                current_value=total_tokens,
                message=msg,
                ledger=ledger,
            )

        # 3. Spend limit check (USD Cost estimation)
        input_tokens = total_prompt + total_tool_use
        output_tokens = total_response + total_thoughts
        cost = (input_tokens * self.prompt_cost_per_token) + (
            output_tokens * self.response_cost_per_token
        )

        if self.max_cost_usd is not None and cost > self.max_cost_usd:
            msg = f"Spend limit exceeded: ${cost:.6f} spent (limit: ${self.max_cost_usd:.6f})"
            self._log_and_trip(
                run_id=run_id,
                limit_type="cost",
                limit_value=self.max_cost_usd,
                current_value=cost,
                message=msg,
                ledger=ledger,
            )

        logger.debug(
            "[BudgetGuard] Session %s healthy: time=%.2fs/%.2f tokens=%d/%d cost=$%.5f/$%.5f",
            run_id,
            elapsed,
            self.max_time_seconds or float("inf"),
            total_tokens,
            self.max_tokens or 0,
            cost,
            self.max_cost_usd or 0.0,
        )

    def _log_and_trip(
        self,
        run_id: str,
        limit_type: str,
        limit_value: Any,
        current_value: Any,
        message: str,
        ledger: EventLedger | None = None,
    ) -> None:
        """Append error payload to the ledger and raise a BudgetTrippedException."""
        logger.error("[BudgetGuard] TRIP! %s", message)

        if ledger is not None:
            try:
                # Log critical event node aligned under standard 'EventNode' types
                ledger.append_event(
                    run_id=run_id,
                    node_id="budget_guard",
                    event_type="budget.tripped",
                    payload={
                        "limit_type": limit_type,
                        "limit_value": limit_value,
                        "current_value": current_value,
                        "message": message,
                    },
                    severity="critical",
                    source="budget_guard",
                )
            except Exception as e:
                logger.error("Failed to append budget trip event to ledger: %s", e)

        raise BudgetTrippedException(
            message=message,
            limit_type=limit_type,
            limit_value=limit_value,
            current_value=current_value,
        )
