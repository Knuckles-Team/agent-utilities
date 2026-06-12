"""Normalized usage/analytics row + response models.

CONCEPT:ECO-4.39 — Usage analytics store. These pydantic models are the single
shape that both plane A (ingested external agent logs) and plane B (our own
runtime telemetry) write, and that the gateway API returns. Mirrors the
agentsview ``sessions``/``messages``/``tool_calls``/``usage_events`` columns,
plus our additions: ``origin`` (ingested|runtime), ``tenant_id`` and
``correlation_id`` for multi-tenant + cross-agent joins.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Origin discriminates the two data planes that share one store.
ORIGIN_INGESTED = "ingested"
ORIGIN_RUNTIME = "runtime"


class UsageSession(BaseModel):
    id: str
    project: str = ""
    machine: str = "local"
    agent: str = "claude"
    first_message: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    message_count: int = 0
    user_message_count: int = 0
    total_output_tokens: int = 0
    peak_context_tokens: int = 0
    is_automated: bool = False
    outcome: str = "unknown"
    health_grade: str | None = None
    termination_status: str | None = None
    parent_session_id: str | None = None
    relationship_type: str = ""
    origin: str = ORIGIN_INGESTED
    tenant_id: str = ""
    correlation_id: str = ""
    file_path: str | None = None
    file_hash: str | None = None
    file_mtime: int | None = None
    file_inode: int | None = None


class UsageMessage(BaseModel):
    session_id: str
    ordinal: int
    role: str
    content: str = ""
    thinking_text: str = ""
    timestamp: str | None = None
    model: str = ""
    context_tokens: int = 0
    output_tokens: int = 0
    has_tool_use: bool = False
    content_length: int = 0


class UsageToolCall(BaseModel):
    session_id: str
    message_ordinal: int | None = None
    tool_name: str = ""
    # category ∈ read|edit|bash|skill|mcp|tool|db|other
    category: str = "other"
    tool_use_id: str | None = None
    input_json: str | None = None
    skill_name: str | None = None
    result_content_length: int | None = None
    subagent_session_id: str | None = None
    status: str = ""
    occurred_at: str | None = None
    origin: str = ORIGIN_INGESTED
    tenant_id: str = ""
    correlation_id: str = ""


class UsageEvent(BaseModel):
    session_id: str
    message_ordinal: int | None = None
    source: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    cost_status: str = ""
    cost_source: str = ""
    occurred_at: str | None = None
    dedup_key: str = ""
    origin: str = ORIGIN_INGESTED
    tenant_id: str = ""
    correlation_id: str = ""


class ParsedSessionBundle(BaseModel):
    """A session plus all its child rows — the unit ingestion/upload carries."""

    session: UsageSession
    messages: list[UsageMessage] = Field(default_factory=list)
    tool_calls: list[UsageToolCall] = Field(default_factory=list)
    usage_events: list[UsageEvent] = Field(default_factory=list)


# ── API response models ────────────────────────────────────────────────────


class TokenTotals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


class UsageSummary(BaseModel):
    from_date: str | None = None
    to_date: str | None = None
    session_count: int = 0
    totals: TokenTotals = Field(default_factory=TokenTotals)
    cache_hit_rate: float = 0.0


class BreakdownEntry(BaseModel):
    key: str
    session_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ToolStat(BaseModel):
    name: str
    category: str = "other"
    calls: int = 0
    success: int = 0
    success_rate: float = 0.0


class ActivityCell(BaseModel):
    day_of_week: int  # 0=Mon .. 6=Sun
    hour: int  # 0..23
    sessions: int = 0
    cost_usd: float = 0.0


class SessionRow(BaseModel):
    id: str
    project: str = ""
    agent: str = "claude"
    started_at: str | None = None
    ended_at: str | None = None
    message_count: int = 0
    total_output_tokens: int = 0
    cost_usd: float = 0.0
    health_grade: str | None = None
    outcome: str = "unknown"
    origin: str = ORIGIN_INGESTED


class SessionDetail(BaseModel):
    session: SessionRow
    messages: list[UsageMessage] = Field(default_factory=list)
    tool_calls: list[UsageToolCall] = Field(default_factory=list)
    usage_events: list[UsageEvent] = Field(default_factory=list)


class SearchHit(BaseModel):
    session_id: str
    ordinal: int
    role: str
    snippet: str
    project: str = ""
    agent: str = "claude"
