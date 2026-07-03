"""UsageBackend — the storage abstraction for usage/cost analytics.

CONCEPT:ECO-4.39 — Backend-abstracted usage analytics store. Modeled on the
existing KG ``GraphBackend`` interface so the usage subsystem is not pinned to
one engine: SQLite+FTS5 is the zero-dependency native default; Postgres
(``tsvector``) and DuckDB are enterprise-scale options selected by
``USAGE_DB_BACKEND``. All backends preserve query-shape parity (the agentsview
backend-parity discipline) so the gateway API and frontends are identical
regardless of backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from agent_utilities.pricing import ModelPricing

from .models import (
    ActivityCell,
    BreakdownEntry,
    ParsedSessionBundle,
    SearchHit,
    SessionDetail,
    SessionRow,
    ToolStat,
    UsageEvent,
    UsageSummary,
    UsageToolCall,
)


class UsageBackend(ABC):
    """Pluggable durable store for sessions/messages/tool-calls/usage-events."""

    name: str = "abstract"

    # ── schema / lifecycle ──────────────────────────────────────────────
    @abstractmethod
    def ensure_schema(self) -> None:
        """Idempotently create tables/indexes/FTS (safe to call repeatedly)."""

    @abstractmethod
    def close(self) -> None: ...

    # ── writes ──────────────────────────────────────────────────────────
    @abstractmethod
    def write_bundle(self, bundle: ParsedSessionBundle) -> None:
        """Upsert one session and replace its child rows (idempotent by id)."""

    @abstractmethod
    def record_usage_event(self, event: UsageEvent) -> None:
        """Append a single usage event (plane B runtime path; dedup-aware)."""

    @abstractmethod
    def record_tool_call(self, call: UsageToolCall) -> None:
        """Append a single tool/skill/db call (plane B runtime path)."""

    @abstractmethod
    def upsert_pricing(self, entries: Iterable[ModelPricing]) -> None:
        """Persist resolved model pricing rows (refreshed by the daemon)."""

    # ── sync skip cache ─────────────────────────────────────────────────
    @abstractmethod
    def should_sync(self, path: str, mtime: int, size: int) -> bool:
        """True when ``path`` is new/changed since last sync (mtime/size cache)."""

    @abstractmethod
    def mark_synced(self, path: str, mtime: int, size: int) -> None: ...

    # ── queries (parity across backends) ────────────────────────────────
    @abstractmethod
    def summary(self, **filters) -> UsageSummary: ...

    @abstractmethod
    def breakdown(self, dimension: str, **filters) -> list[BreakdownEntry]:
        """``dimension`` ∈ model|project|agent."""

    @abstractmethod
    def tool_stats(self, **filters) -> list[ToolStat]: ...

    @abstractmethod
    def activity(self, **filters) -> list[ActivityCell]: ...

    @abstractmethod
    def top_sessions(self, *, limit: int = 20, **filters) -> list[SessionRow]: ...

    @abstractmethod
    def list_sessions(self, *, limit: int = 100, **filters) -> list[SessionRow]: ...

    @abstractmethod
    def session_detail(self, session_id: str) -> SessionDetail | None: ...

    @abstractmethod
    def search(self, query: str, *, limit: int = 50, **filters) -> list[SearchHit]: ...
