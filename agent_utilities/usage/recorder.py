"""UsageRecorder — the single write facade for both data planes.

CONCEPT:ECO-4.39 / OS-5.31. Plane A (ingested external logs) and plane B (our
own runtime telemetry) both go through here so they land in one store with one
shape. Every method is best-effort: a recorder failure must never break a graph
run or an ingest. Runtime rows are stamped with the active correlation id.
"""

from __future__ import annotations

import logging

from .backend import UsageBackend
from .backends import get_usage_backend
from .cost import price_event
from .models import (
    ORIGIN_RUNTIME,
    ParsedSessionBundle,
    UsageEvent,
    UsageSession,
    UsageToolCall,
)

logger = logging.getLogger(__name__)


def _correlation_id() -> str:
    try:
        from agent_utilities.observability.correlation import get_correlation_id

        return get_correlation_id() or ""
    except Exception:  # noqa: BLE001
        return ""


def _enabled() -> bool:
    try:
        from agent_utilities.core.config import config

        return bool(getattr(config, "usage_tracking_enabled", True))
    except Exception:  # noqa: BLE001
        return True


def _bump_call_metric(category: str, tool_name: str, skill_name: str | None) -> None:
    """Mirror a tool/skill/db call to Prometheus (CONCEPT:OS-5.31). No-op when
    the metrics extra is absent."""
    try:
        from agent_utilities.observability import gateway_metrics as gm

        gm.TOOL_CALLS.labels(category=category or "other").inc()
        if category == "skill" and skill_name:
            gm.SKILL_CALLS.labels(skill=skill_name).inc()
        elif category == "db":
            gm.DB_CALLS.labels(store=tool_name or "unknown").inc()
    except Exception:  # noqa: BLE001
        pass


class UsageRecorder:
    """Thin, best-effort writer over a :class:`UsageBackend`."""

    def __init__(self, backend: UsageBackend | None = None) -> None:
        self._backend = backend

    @property
    def backend(self) -> UsageBackend:
        if self._backend is None:
            self._backend = get_usage_backend()
        return self._backend

    # ── plane A: ingested external sessions ─────────────────────────────
    def record_bundle(self, bundle: ParsedSessionBundle) -> bool:
        """Price + persist a fully-parsed session bundle (ingest path)."""
        try:
            bundle.usage_events = [price_event(e) for e in bundle.usage_events]
            self.backend.write_bundle(bundle)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "usage record_bundle failed for %s: %s", bundle.session.id, exc
            )
            return False

    # ── plane B: our own runtime ────────────────────────────────────────
    def record_run(
        self,
        *,
        run_id: str,
        query: str = "",
        status: str = "completed",
        duration_ms: float | None = None,
        token_usage: dict | None = None,
        model: str = "",
        project: str = "",
        tenant_id: str = "",
    ) -> bool:
        """Record one graph run as a runtime session + usage event (plane B)."""
        if not _enabled():
            return False
        try:
            cid = _correlation_id()
            tu = token_usage or {}
            inp = int(tu.get("input_tokens") or tu.get("request_tokens") or 0)
            out = int(tu.get("output_tokens") or tu.get("response_tokens") or 0)
            reasoning = int(
                tu.get("reasoning_tokens") or tu.get("thoughts_tokens") or 0
            )
            cache_read = int(tu.get("cache_read_input_tokens") or 0)
            cache_creation = int(tu.get("cache_creation_input_tokens") or 0)
            session = UsageSession(
                id=run_id,
                project=project or "runtime",
                agent="agent-utilities",
                first_message=query[:200],
                message_count=1,
                total_output_tokens=out,
                outcome=status,
                origin=ORIGIN_RUNTIME,
                tenant_id=tenant_id,
                correlation_id=cid,
            )
            event = price_event(
                UsageEvent(
                    session_id=run_id,
                    source="runtime",
                    model=model,
                    input_tokens=inp,
                    output_tokens=out,
                    cache_creation_input_tokens=cache_creation,
                    cache_read_input_tokens=cache_read,
                    reasoning_tokens=reasoning,
                    origin=ORIGIN_RUNTIME,
                    tenant_id=tenant_id,
                    correlation_id=cid,
                    dedup_key=run_id,
                )
            )
            self.backend.write_bundle(
                ParsedSessionBundle(session=session, usage_events=[event])
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("usage record_run failed for %s: %s", run_id, exc)
            return False

    def record_tool_call(
        self,
        *,
        session_id: str,
        tool_name: str,
        category: str = "tool",
        status: str = "",
        skill_name: str | None = None,
        subagent_session_id: str | None = None,
        tenant_id: str = "",
    ) -> bool:
        """Record one tool/skill/mcp/db call (plane B granular metrics)."""
        if not _enabled():
            return False
        try:
            self.backend.record_tool_call(
                UsageToolCall(
                    session_id=session_id,
                    tool_name=tool_name,
                    category=category,
                    status=status,
                    skill_name=skill_name,
                    subagent_session_id=subagent_session_id,
                    origin=ORIGIN_RUNTIME,
                    tenant_id=tenant_id,
                    correlation_id=_correlation_id(),
                )
            )
            _bump_call_metric(category, tool_name, skill_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("usage record_tool_call failed: %s", exc)
            return False


_recorder: UsageRecorder | None = None


def get_usage_recorder() -> UsageRecorder:
    """Process-wide recorder."""
    global _recorder
    if _recorder is None:
        _recorder = UsageRecorder()
    return _recorder
