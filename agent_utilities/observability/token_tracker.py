#!/usr/bin/python
from __future__ import annotations

"""Token Usage Tracker — 4-Bucket Granular Analytics (CONCEPT:AU-OS.observability.granular-token-analytics).

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


def _bucket_points(
    points: list[tuple[int, list[float]]],
    field_idx: int,
    window_s: float,
    agg: str,
) -> list[tuple[float, float]]:
    """Bucket raw ``(ts_ns, vector)`` points by ``window_s`` over one field.

    Used only for a non-field-0 windowed aggregate (field-0 + the common total case
    are bucketed natively by the engine). Keeps the public ``usage_series`` shape.
    """
    width_ns = int(window_s * 1e9)
    if width_ns <= 0:
        return []
    buckets: dict[int, list[float]] = defaultdict(list)
    for ts, vals in points:
        if field_idx < len(vals):
            buckets[(ts // width_ns) * width_ns].append(vals[field_idx])
    out: list[tuple[float, float]] = []
    for start in sorted(buckets):
        vs = buckets[start]
        if agg == "sum":
            v = sum(vs)
        elif agg == "mean":
            v = sum(vs) / len(vs)
        elif agg == "min":
            v = min(vs)
        elif agg == "max":
            v = max(vs)
        elif agg == "count":
            v = float(len(vs))
        elif agg == "first":
            v = vs[0]
        else:  # last
            v = vs[-1]
        out.append((start / 1e9, v))
    return out


#: The per-agent telemetry series-id prefix in the engine tsdb (CONCEPT:AU-KG.temporal.token-event-tsdb).
_TELEMETRY_SERIES_PREFIX = "telemetry:tokens:"

#: The ordered field vector telemetry points carry (shared by the writer + readers).
TELEMETRY_TS_FIELDS = (
    "prompt_tokens",
    "response_tokens",
    "thoughts_tokens",
    "tool_use_tokens",
    "total_tokens",
)


def query_token_series(
    agent_name: str,
    start_ts: float,
    end_ts: float,
    *,
    field: str = "total_tokens",
    window_s: float | None = None,
    agg: str = "sum",
) -> list[tuple[float, float]]:
    """Per-agent token usage over time from the engine tsdb (CONCEPT:AU-KG.temporal.token-event-tsdb).

    Instance-free reader for the durable telemetry series (any tracker instance's
    writes land in the same engine-keyed series). Returns ``[(epoch_seconds,
    value), ...]`` — native in-engine windowed aggregates when ``window_s`` is set,
    else raw points for ``field``. Empty when no engine / no data.
    """
    try:
        from agent_utilities.knowledge_graph.memory.timeseries import (
            get_timeseries_backend,
        )

        backend = get_timeseries_backend("engine")
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[CONCEPT:AU-KG.temporal.token-event-tsdb] telemetry tsdb unavailable: %s",
            e,
        )
        return []
    client = getattr(backend, "_client", None)
    if client is None:
        return []
    # The engine series id for a tagless symbol is ``ts:<symbol>`` (the backend's
    # own keying, KG-2.246); mirror it here so reads hit the same series the writer
    # appended to.
    sid = f"ts:{_TELEMETRY_SERIES_PREFIX}{agent_name or 'unknown'}"
    try:
        field_idx = TELEMETRY_TS_FIELDS.index(field)
    except ValueError:
        field_idx = TELEMETRY_TS_FIELDS.index("total_tokens")
    from_ns, to_ns = int(start_ts * 1e9), int(end_ts * 1e9)
    try:
        if window_s and field_idx == 0:
            rows = client.timeseries.window(
                sid, from_ns, to_ns + 1, int(window_s * 1e9), agg
            )
            return [(b / 1e9, v) for b, v, _c in rows]
        pts = client.timeseries.range(sid, from_ns, to_ns + 1)
        if window_s:
            return _bucket_points(pts, field_idx, window_s, agg)
        return [
            (ts / 1e9, vals[field_idx]) for ts, vals in pts if field_idx < len(vals)
        ]
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[CONCEPT:AU-KG.temporal.token-event-tsdb] query_token_series failed: %s", e
        )
        return []


class TokenBucket(StrEnum):
    """Token usage bucket categories.

    CONCEPT:AU-OS.observability.granular-token-analytics — Granular Token Analytics

    Ported from MATE's 4-bucket tracking in ``token_usage_callback.py``:
    prompt_tokens, response_tokens, thoughts_tokens, tool_use_tokens.
    """

    PROMPT = "prompt"
    RESPONSE = "response"
    THOUGHTS = "thoughts"
    TOOL_USE = "tool_use"


class TokenUsageRecord(BaseModel):
    """A single token usage record with per-bucket counts.

    CONCEPT:AU-OS.observability.granular-token-analytics — Granular Token Analytics

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
    mementos_generated: int = 0
    kv_cache_saved: int = 0
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

    CONCEPT:AU-OS.observability.granular-token-analytics — Granular Token Analytics
    """

    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    total_thoughts_tokens: int = 0
    total_tool_use_tokens: int = 0
    total_mementos_generated: int = 0
    total_kv_cache_saved: int = 0
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

    CONCEPT:AU-OS.observability.granular-token-analytics — Granular Token Analytics
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

    CONCEPT:AU-OS.observability.granular-token-analytics — Granular Token Analytics

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

    #: The ordered field vector each telemetry point appends to the engine tsdb.
    #: Fixed so a ``range``/``window`` read decodes back to the same bucket names
    #: (CONCEPT:AU-KG.temporal.token-event-tsdb) — shared with the instance-free :func:`query_token_series`.
    _TS_FIELDS = TELEMETRY_TS_FIELDS

    def __init__(self, kg_engine: Any = None) -> None:
        self._engine = kg_engine
        self._records: list[TokenUsageRecord] = []
        self._by_session: dict[str, list[TokenUsageRecord]] = defaultdict(list)
        self._by_agent: dict[str, list[TokenUsageRecord]] = defaultdict(list)
        # Lazily-bound engine time-series backend (CONCEPT:AU-KG.temporal.token-event-tsdb). Telemetry is
        # naturally a time-series — per-agent token counts over time — so it is
        # appended to the engine tsdb and read back via native range/window
        # (in-engine time-bucketing) instead of re-scanning Python lists.
        self._ts_backend: Any = None
        self._ts_disabled = False

    def _series_id(self, agent_name: str) -> str:
        """The per-agent telemetry series id in the engine tsdb."""
        return f"{_TELEMETRY_SERIES_PREFIX}{agent_name or 'unknown'}"

    def _ts(self) -> Any:
        """The engine time-series backend, lazily initialized (``None`` if absent).

        Best-effort: if no engine is reachable, telemetry tsdb is disabled for this
        process (the in-memory aggregation still works) — never raises into the
        record path.
        """
        if self._ts_backend is not None or self._ts_disabled:
            return self._ts_backend
        try:
            from agent_utilities.knowledge_graph.memory.timeseries import (
                get_timeseries_backend,
            )

            self._ts_backend = get_timeseries_backend("engine")
        except Exception as e:  # noqa: BLE001 — engine absent ⇒ disable, don't crash
            logger.debug(
                "[CONCEPT:AU-KG.temporal.token-event-tsdb] telemetry tsdb unavailable: %s",
                e,
            )
            self._ts_disabled = True
            self._ts_backend = None
        return self._ts_backend

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

        # CONCEPT:AU-KG.temporal.token-event-tsdb — ALSO append this event to the engine tsdb as a per-agent
        # telemetry point, so cross-session token trends are a native time-bucketed
        # range/window query in-engine, not a Python re-scan. Best-effort + off the
        # critical path (a missing engine just skips it).
        self._append_ts(record)

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

    def _append_ts(self, record: TokenUsageRecord) -> None:
        """Append one telemetry record to the engine tsdb (best-effort, KG-2.252)."""
        backend = self._ts()
        if backend is None:
            return
        try:
            from datetime import UTC, datetime

            from agent_utilities.knowledge_graph.memory.timeseries.base import (
                TimeSeriesDataPoint,
            )

            metrics = {f: float(getattr(record, f, 0)) for f in self._TS_FIELDS}
            point = TimeSeriesDataPoint(
                symbol=self._series_id(record.agent_name),
                timestamp=datetime.fromtimestamp(record.timestamp, tz=UTC),
                metrics=metrics,
            )
            backend.insert([point])
        except Exception as e:  # noqa: BLE001 — telemetry write must never break a run
            logger.debug(
                "[CONCEPT:AU-KG.temporal.token-event-tsdb] telemetry tsdb append skipped: %s",
                e,
            )

    def usage_series(
        self,
        agent_name: str,
        start_ts: float,
        end_ts: float,
        *,
        field: str = "total_tokens",
        window_s: float | None = None,
        agg: str = "sum",
    ) -> list[tuple[float, float]]:
        """Per-agent token usage over time, computed IN-ENGINE (CONCEPT:AU-KG.temporal.token-event-tsdb).

        Queries the engine tsdb for ``agent_name`` over ``[start_ts, end_ts]``
        (epoch seconds). With ``window_s`` set, returns native time-bucketed
        aggregates (``agg`` ∈ sum/mean/min/max/first/last/count) — the bucketing
        runs in the engine, not by re-scanning Python records. Without a window,
        returns the raw ``(epoch_seconds, value)`` points for ``field``.

        Returns ``[(epoch_seconds, value), ...]`` (empty if no engine / no data).
        Delegates to the instance-free :func:`query_token_series` (the durable series
        is engine-keyed, not tied to this tracker instance).
        """
        return query_token_series(
            agent_name,
            start_ts,
            end_ts,
            field=field,
            window_s=window_s,
            agg=agg,
        )

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
            summary.total_mementos_generated += r.mementos_generated
            summary.total_kv_cache_saved += r.kv_cache_saved
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
            A pydantic-ai ``RunUsage`` object with ``input_tokens``,
            ``output_tokens``, etc. (v1 ``request_tokens``/``response_tokens``
            are still read as a fallback).
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
            # pydantic-ai v2 renamed request_tokens/response_tokens to
            # input_tokens/output_tokens; prefer the new names, fall back to the old.
            prompt_tokens=getattr(usage_metadata, "input_tokens", None)
            or getattr(usage_metadata, "request_tokens", 0)
            or 0,
            response_tokens=getattr(usage_metadata, "output_tokens", None)
            or getattr(usage_metadata, "response_tokens", 0)
            or 0,
            thoughts_tokens=getattr(usage_metadata, "thoughts_tokens", 0)
            or getattr(usage_metadata, "details", {}).get("thoughts_tokens", 0)
            or 0,
            tool_use_tokens=getattr(usage_metadata, "tool_use_tokens", 0)
            or getattr(usage_metadata, "details", {}).get("tool_use_tokens", 0)
            or 0,
        )
        return self.record(record)
