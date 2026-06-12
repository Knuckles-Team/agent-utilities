"""UsageService — read-side aggregation the gateway API delegates to.

CONCEPT:ECO-4.39 / ECO-4.41. A thin layer over the active ``UsageBackend`` so
the router stays declarative and all SQL lives in the backend.
"""

from __future__ import annotations

from .backend import UsageBackend
from .backends import get_usage_backend
from .models import (
    ActivityCell,
    BreakdownEntry,
    SearchHit,
    SessionDetail,
    SessionRow,
    ToolStat,
    UsageSummary,
)


class UsageService:
    def __init__(self, backend: UsageBackend | None = None) -> None:
        self._backend = backend

    @property
    def backend(self) -> UsageBackend:
        if self._backend is None:
            self._backend = get_usage_backend()
        return self._backend

    def summary(self, **filters) -> UsageSummary:
        return self.backend.summary(**filters)

    def by_model(self, **filters) -> list[BreakdownEntry]:
        return self.backend.breakdown("model", **filters)

    def by_project(self, **filters) -> list[BreakdownEntry]:
        return self.backend.breakdown("project", **filters)

    def by_agent(self, **filters) -> list[BreakdownEntry]:
        return self.backend.breakdown("agent", **filters)

    def tools(self, **filters) -> list[ToolStat]:
        return self.backend.tool_stats(**filters)

    def activity(self, **filters) -> list[ActivityCell]:
        return self.backend.activity(**filters)

    def top_sessions(self, *, limit: int = 20, **filters) -> list[SessionRow]:
        return self.backend.top_sessions(limit=limit, **filters)

    def sessions(self, *, limit: int = 100, **filters) -> list[SessionRow]:
        return self.backend.list_sessions(limit=limit, **filters)

    def session_detail(self, session_id: str) -> SessionDetail | None:
        return self.backend.session_detail(session_id)

    def search(self, query: str, *, limit: int = 50, **filters) -> list[SearchHit]:
        return self.backend.search(query, limit=limit, **filters)


_service: UsageService | None = None


def get_usage_service() -> UsageService:
    global _service
    if _service is None:
        _service = UsageService()
    return _service
