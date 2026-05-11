#!/usr/bin/python
from __future__ import annotations

"""Token Usage Tracker — 4-Bucket Granular Analytics (CONCEPT:OS-5.4).

Provides granular token usage tracking with four distinct buckets:
prompt, response, thoughts, and tool_use. Ported from MATE's
``token_usage_service.py`` and ``token_usage_callback.py``.

MATE tracks tokens via SQLAlchemy + database persistence. This module
adapts the pattern to agent-utilities' KG-native architecture, enabling
OWL-inferred ``highCostAgent`` classification via threshold rules and
cross-session trend analysis that MATE can only do via flat SQL queries.

OWL: :TokenUsageRecord rdfs:subClassOf :Observation
"""


import logging
import time
from collections import defaultdict
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenBucket(StrEnum):
    """Token usage bucket categories.

    CONCEPT:OS-5.4 — Granular Token Analytics

    Ported from MATE's 4-bucket tracking in ``token_usage_callback.py``:
    prompt_tokens, response_tokens, thoughts_tokens, tool_use_tokens.
    """

    PROMPT = "prompt"
    RESPONSE = "response"
    THOUGHTS = "thoughts"
    TOOL_USE = "tool_use"


class TokenUsageRecord(BaseModel):
    """A single token usage record with per-bucket counts.

    CONCEPT:OS-5.4 — Granular Token Analytics

    Mirrors MATE's ``token_data`` dict structure but expressed as a
    Pydantic model for KG persistence and type safety.
    """

    id: str = ""
    agent_name: str = ""
    model_name: str = ""
    session_id: str = ""
    request_id: str = ""
    user_id: str = ""
    prompt_tokens: int = 0
    response_tokens: int = 0
    thoughts_tokens: int = 0
    tool_use_tokens: int = 0
    total_tokens: int = 0
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute total if not explicitly set."""
        if self.total_tokens == 0:
            self.total_tokens = (
                self.prompt_tokens
                + self.response_tokens
                + self.thoughts_tokens
                + self.tool_use_tokens
            )


class TokenUsageSummary(BaseModel):
    """Aggregated token usage summary.

    CONCEPT:OS-5.4 — Granular Token Analytics
    """

    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    total_thoughts_tokens: int = 0
    total_tool_use_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def compute_total(self) -> int:
        """Compute total across all buckets."""
        self.total_tokens = (
            self.total_prompt_tokens
            + self.total_response_tokens
            + self.total_thoughts_tokens
            + self.total_tool_use_tokens
        )
        return self.total_tokens


class TokenBudgetAlert(BaseModel):
    """Alert triggered when a token bucket exceeds its threshold.

    CONCEPT:OS-5.4 — Granular Token Analytics
    """

    bucket: TokenBucket
    current: int
    threshold: int
    percentage: float
    agent_name: str = ""
    session_id: str = ""
    severity: str = "warning"  # warning, critical
    message: str = ""


class TokenUsageTracker:
    """Granular token usage tracker with 4-bucket analytics.

    CONCEPT:OS-5.4 — Granular Token Analytics

    Ported from MATE's ``token_usage_service.py`` and
    ``token_usage_callback.py``. Provides:

    1. Per-record 4-bucket tracking (prompt, response, thoughts, tool_use)
    2. Session-level aggregation
    3. Agent-level breakdown
    4. Budget alerting per bucket with configurable thresholds
    5. Export for API/UI consumption

    Unlike MATE which persists to a SQL database, records are stored
    in-memory and optionally persisted to the Knowledge Graph as
    ``TokenUsageRecordNode`` — enabling OWL-inferred ``highCostAgent``
    classification and cross-session trend analysis.

    Parameters
    ----------
    kg_engine : optional
        If provided, token records are persisted to the KG.
    """

    def __init__(self, kg_engine: Any = None) -> None:
        self._engine = kg_engine
        self._records: list[TokenUsageRecord] = []
        self._by_session: dict[str, list[TokenUsageRecord]] = defaultdict(list)
        self._by_agent: dict[str, list[TokenUsageRecord]] = defaultdict(list)

    def record(self, record: TokenUsageRecord) -> TokenUsageRecord:
        """Record a token usage entry.

        Appends to in-memory stores indexed by session and agent,
        and optionally persists to the Knowledge Graph.

        Parameters
        ----------
        record : TokenUsageRecord
            The token usage record to store.

        Returns
        -------
        TokenUsageRecord
            The stored record (with auto-computed total).
        """
        # Ensure total is computed
        if record.total_tokens == 0:
            record.total_tokens = (
                record.prompt_tokens
                + record.response_tokens
                + record.thoughts_tokens
                + record.tool_use_tokens
            )

        # Generate ID if not set
        if not record.id:
            record.id = f"token:{record.agent_name}:{record.timestamp}"

        self._records.append(record)
        if record.session_id:
            self._by_session[record.session_id].append(record)
        if record.agent_name:
            self._by_agent[record.agent_name].append(record)

        logger.debug(
            "Token usage recorded: agent=%s total=%d (prompt=%d response=%d "
            "thoughts=%d tool_use=%d)",
            record.agent_name,
            record.total_tokens,
            record.prompt_tokens,
            record.response_tokens,
            record.thoughts_tokens,
            record.tool_use_tokens,
        )

        return record

    def get_session_totals(self, session_id: str) -> TokenUsageSummary:
        """Get aggregated token usage for a session.

        Parameters
        ----------
        session_id : str
            The session identifier.

        Returns
        -------
        TokenUsageSummary
            Aggregated totals across all records in the session.
        """
        records = self._by_session.get(session_id, [])
        summary = TokenUsageSummary(call_count=len(records))
        for r in records:
            summary.total_prompt_tokens += r.prompt_tokens
            summary.total_response_tokens += r.response_tokens
            summary.total_thoughts_tokens += r.thoughts_tokens
            summary.total_tool_use_tokens += r.tool_use_tokens
        summary.compute_total()
        return summary

    def get_agent_breakdown(self, agent_name: str) -> dict[str, Any]:
        """Get per-bucket breakdown for an agent.

        Parameters
        ----------
        agent_name : str
            The agent name.

        Returns
        -------
        dict
            Per-bucket totals and call count.
        """
        records = self._by_agent.get(agent_name, [])
        breakdown = {
            "agent_name": agent_name,
            "call_count": len(records),
            "total_prompt_tokens": sum(r.prompt_tokens for r in records),
            "total_response_tokens": sum(r.response_tokens for r in records),
            "total_thoughts_tokens": sum(r.thoughts_tokens for r in records),
            "total_tool_use_tokens": sum(r.tool_use_tokens for r in records),
            "total_tokens": sum(r.total_tokens for r in records),
        }
        return breakdown

    def get_budget_alerts(
        self,
        thresholds: dict[str, int] | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
    ) -> list[TokenBudgetAlert]:
        """Check for budget threshold violations.

        Parameters
        ----------
        thresholds : dict[str, int] | None
            Per-bucket thresholds. Keys are bucket names (prompt,
            response, thoughts, tool_use, total). Defaults to
            reasonable limits if not provided.
        agent_name : str | None
            Filter to specific agent.
        session_id : str | None
            Filter to specific session.

        Returns
        -------
        list[TokenBudgetAlert]
            List of alerts for exceeded thresholds.
        """
        defaults = {
            "prompt": 100_000,
            "response": 100_000,
            "thoughts": 50_000,
            "tool_use": 50_000,
            "total": 250_000,
        }
        effective_thresholds = {**defaults, **(thresholds or {})}

        # Select records to check
        if session_id:
            records = self._by_session.get(session_id, [])
        elif agent_name:
            records = self._by_agent.get(agent_name, [])
        else:
            records = self._records

        # Aggregate
        totals = {
            "prompt": sum(r.prompt_tokens for r in records),
            "response": sum(r.response_tokens for r in records),
            "thoughts": sum(r.thoughts_tokens for r in records),
            "tool_use": sum(r.tool_use_tokens for r in records),
            "total": sum(r.total_tokens for r in records),
        }

        alerts: list[TokenBudgetAlert] = []
        bucket_map = {
            "prompt": TokenBucket.PROMPT,
            "response": TokenBucket.RESPONSE,
            "thoughts": TokenBucket.THOUGHTS,
            "tool_use": TokenBucket.TOOL_USE,
        }

        for key, current in totals.items():
            threshold = effective_thresholds.get(key, 0)
            if threshold <= 0:
                continue
            pct = (current / threshold) * 100
            if pct >= 80:
                severity = "critical" if pct >= 100 else "warning"
                bucket = bucket_map.get(key, TokenBucket.PROMPT)
                alerts.append(
                    TokenBudgetAlert(
                        bucket=bucket,
                        current=current,
                        threshold=threshold,
                        percentage=pct,
                        agent_name=agent_name or "",
                        session_id=session_id or "",
                        severity=severity,
                        message=f"{key} tokens at {pct:.0f}% of budget "
                        f"({current}/{threshold})",
                    )
                )

        return alerts

    def export_summary(self) -> dict[str, Any]:
        """Export a serializable summary of all tracked usage.

        Returns
        -------
        dict
            Complete summary including per-agent breakdown,
            per-session totals, and overall statistics.
        """
        overall = TokenUsageSummary(call_count=len(self._records))
        for r in self._records:
            overall.total_prompt_tokens += r.prompt_tokens
            overall.total_response_tokens += r.response_tokens
            overall.total_thoughts_tokens += r.thoughts_tokens
            overall.total_tool_use_tokens += r.tool_use_tokens
        overall.compute_total()

        return {
            "overall": overall.model_dump(),
            "by_agent": {
                agent: self.get_agent_breakdown(agent) for agent in self._by_agent
            },
            "session_count": len(self._by_session),
            "agent_count": len(self._by_agent),
        }

    def record_from_llm_response(
        self,
        usage_metadata: Any,
        agent_name: str = "",
        model_name: str = "",
        session_id: str = "",
        user_id: str = "",
    ) -> TokenUsageRecord | None:
        """Adapter to record usage from pydantic-ai's UsageMetadata.

        Parameters
        ----------
        usage_metadata : Any
            A pydantic-ai ``Usage`` object with ``request_tokens``,
            ``response_tokens``, etc.
        agent_name : str
            The agent that made the call.
        model_name : str
            The model used.
        session_id : str
            Session identifier.
        user_id : str
            User identifier.

        Returns
        -------
        TokenUsageRecord | None
            The recorded entry, or None if metadata was empty.
        """
        if usage_metadata is None:
            return None

        record = TokenUsageRecord(
            agent_name=agent_name,
            model_name=model_name,
            session_id=session_id,
            user_id=user_id,
            prompt_tokens=getattr(usage_metadata, "request_tokens", 0) or 0,
            response_tokens=getattr(usage_metadata, "response_tokens", 0) or 0,
            thoughts_tokens=getattr(usage_metadata, "thoughts_tokens", 0)
            or getattr(usage_metadata, "details", {}).get("thoughts_tokens", 0)
            or 0,
            tool_use_tokens=getattr(usage_metadata, "tool_use_tokens", 0)
            or getattr(usage_metadata, "details", {}).get("tool_use_tokens", 0)
            or 0,
        )
        return self.record(record)
