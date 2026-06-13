"""Fleet events webhook ingress (CONCEPT:OS-5.15 — Fleet Event Ingress).

Until now nothing could wake the orchestrator except its own timers and
file-mtime watchers — monitoring infrastructure had no way to hand the fleet an
observed incident. This module gives Prometheus Alertmanager, Uptime Kuma,
Portainer (and any generic JSON sender) a single gateway ingress:

    POST /api/fleet/events[?source=<name>]

Every payload is normalized to one internal :class:`FleetEvent` shape, then
each event is

* persisted as a ``FleetEvent`` KG node (the durable observation, following the
  ExecutionSummary/PerformanceAnomaly node-write pattern), and
* enqueued as a durable ``fleet_event_triage`` task on the engine task queue,
  so the host daemon's workers act on it via
  :mod:`agent_utilities.knowledge_graph.adaptation.fleet_event_triage`.

Auth: Alertmanager/Kuma cannot mint JWTs, so on top of the OS-5.14 identity
middleware (open unless ``KG_AUTH_REQUIRED``) the endpoint supports an optional
shared-secret header ``X-Fleet-Events-Token`` checked against
``AgentConfig.fleet_events_token`` (``FLEET_EVENTS_TOKEN``; default ``None`` =
not required). A naive in-memory per-source per-minute counter caps event
storms (429 when exceeded).
"""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Canonical severity vocabulary (normalized from each sender's own wording).
_SEVERITY_MAP = {
    "critical": "critical",
    "fatal": "critical",
    "emergency": "critical",
    "page": "critical",
    "error": "error",
    "major": "error",
    "warning": "warning",
    "warn": "warning",
    "minor": "warning",
    "info": "info",
    "information": "info",
    "ok": "info",
    "none": "info",
}

# Naive storm cap: at most this many accepted events per source per minute.
# In-memory and deliberately simple — its job is to keep a misconfigured
# Alertmanager/Kuma from flooding the KG + task queue, not to be a real WAF.
RATE_CAP_PER_MINUTE = 120
_rate_counters: dict[str, list[int]] = {}  # source -> [minute_epoch, count]


def _normalize_severity(raw: Any, default: str = "info") -> str:
    return _SEVERITY_MAP.get(str(raw or "").strip().lower(), default)


@dataclass
class FleetEvent:
    """One normalized fleet event, whatever monitoring system sent it."""

    source: str  # alertmanager | uptime-kuma | portainer | <generic>
    severity: str  # critical | error | warning | info
    subject: str  # the affected service/monitor/instance
    status: str  # firing | resolved | up | down | unknown
    summary: str
    raw: dict[str, Any] = field(default_factory=dict)
    received_at: str = ""

    def __post_init__(self) -> None:
        if not self.received_at:
            self.received_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())
            )


# ── payload normalization ────────────────────────────────────────────


def _normalize_alertmanager(payload: dict[str, Any]) -> list[FleetEvent]:
    """Prometheus Alertmanager v4 webhook JSON → one event per ``alerts[]``."""
    events: list[FleetEvent] = []
    for alert in payload.get("alerts") or []:
        labels = alert.get("labels") or {}
        annotations = alert.get("annotations") or {}
        subject = (
            labels.get("service")
            or labels.get("instance")
            or labels.get("job")
            or labels.get("alertname")
            or "unknown"
        )
        status = str(alert.get("status") or payload.get("status") or "firing")
        severity = _normalize_severity(labels.get("severity"), default="warning")
        if status == "resolved":
            severity = "info"
        summary = (
            annotations.get("summary")
            or annotations.get("description")
            or labels.get("alertname")
            or "alert"
        )
        events.append(
            FleetEvent(
                source="alertmanager",
                severity=severity,
                subject=str(subject),
                status=status,
                summary=str(summary),
                raw=alert,
            )
        )
    return events


# Uptime Kuma heartbeat status codes (0=down, 1=up, 2=pending, 3=maintenance).
_KUMA_STATUS = {0: "down", 1: "up", 2: "pending", 3: "maintenance"}


def _normalize_uptime_kuma(payload: dict[str, Any]) -> list[FleetEvent]:
    """Uptime Kuma webhook JSON (``heartbeat``/``monitor``) → one event."""
    heartbeat = payload.get("heartbeat") or {}
    monitor = payload.get("monitor") or {}
    raw_status = heartbeat.get("status")
    status = (
        _KUMA_STATUS.get(raw_status, "unknown")
        if isinstance(raw_status, int)
        else "unknown"
    )
    severity = "critical" if status == "down" else "info"
    subject = monitor.get("name") or monitor.get("url") or "unknown"
    summary = payload.get("msg") or heartbeat.get("msg") or f"monitor {status}"
    return [
        FleetEvent(
            source="uptime-kuma",
            severity=severity,
            subject=str(subject),
            status=status,
            summary=str(summary),
            raw=payload,
        )
    ]


def _normalize_generic(
    payload: dict[str, Any], source_hint: str | None
) -> list[FleetEvent]:
    """Fallback: accept any JSON object (Portainer & friends)."""
    source = str(source_hint or payload.get("source") or "generic")
    subject = (
        payload.get("service")
        or payload.get("subject")
        or payload.get("name")
        or payload.get("host")
        or "unknown"
    )
    summary = (
        payload.get("summary")
        or payload.get("message")
        or payload.get("msg")
        or json.dumps(payload, default=str)[:200]
    )
    return [
        FleetEvent(
            source=source,
            severity=_normalize_severity(payload.get("severity")),
            subject=str(subject),
            status=str(payload.get("status") or "unknown"),
            summary=str(summary),
            raw=payload,
        )
    ]


def normalize_payload(payload: Any, source_hint: str | None = None) -> list[FleetEvent]:
    """Detect the sender format and normalize to :class:`FleetEvent` records.

    Detection is structural: an ``alerts`` list with Alertmanager envelope keys
    means Alertmanager v4; a ``heartbeat``/``monitor`` object means Uptime Kuma;
    anything else falls back to the generic normalizer (``source`` taken from
    the ``?source=`` query param / ``X-Event-Source`` header hint).
    """
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("alerts"), list) and (
        "receiver" in payload or "version" in payload or "groupKey" in payload
    ):
        return _normalize_alertmanager(payload)
    if isinstance(payload.get("heartbeat"), dict) or isinstance(
        payload.get("monitor"), dict
    ):
        return _normalize_uptime_kuma(payload)
    return _normalize_generic(payload, source_hint)


# ── persistence + queue ──────────────────────────────────────────────


def _get_engine() -> Any:
    """Resolve the gateway's KG engine (kept as a seam for tests)."""
    from agent_utilities.mcp.kg_server import _get_engine as _kg_engine

    return _kg_engine()


def _correlation_stamp() -> dict[str, str]:
    """Correlation + identity to stamp on persisted effect nodes (CONCEPT:OS-5.11).

    Stamping the originating ``correlation_id`` (+ actor/tenant) onto the durable
    node is what makes the swarm-wide ``/api/fleet/trace`` and ``/api/fleet/touched``
    queries answerable from the graph rather than only from external traces.
    """
    stamp: dict[str, str] = {}
    try:
        from agent_utilities.observability import correlation

        stamp["correlation_id"] = correlation.ensure_correlation_id()
        try:
            from agent_utilities.security.brain_context import current_actor

            actor = current_actor()
            if actor.actor_id and actor.actor_id != "system":
                stamp["actor_id"] = actor.actor_id
            if actor.tenant_id:
                stamp["tenant_id"] = actor.tenant_id
        except Exception:  # noqa: BLE001 — identity is best-effort context
            pass
    except Exception:  # noqa: BLE001 — correlation is best-effort context
        pass
    return stamp


def persist_event(engine: Any, event: FleetEvent) -> str:
    """Write the event as a ``FleetEvent`` KG node; returns the node id."""
    event_id = f"fleet_event:{uuid.uuid4().hex[:12]}"
    properties = {
        "source": event.source,
        "severity": event.severity,
        "subject": event.subject,
        "status": event.status,
        "summary": event.summary[:500],
        "raw": json.dumps(event.raw, default=str)[:4000],
        "received_at": event.received_at,
        "triage_status": "pending",
    }
    properties.update(_correlation_stamp())
    engine.add_node(event_id, "FleetEvent", properties=properties)
    return event_id


def enqueue_triage(engine: Any, event_id: str, event: FleetEvent) -> str | None:
    """Enqueue a durable ``fleet_event_triage`` task for the daemon workers."""
    submit = getattr(engine, "submit_task", None)
    if not callable(submit):
        return None
    return submit(
        event_id,
        is_codebase=False,
        provenance={"source": "fleet_events", "event_source": event.source},
        task_type="fleet_event_triage",
        skip_dedupe=True,
    )


def _storm_capped(source: str, n: int = 1) -> bool:
    """True when accepting ``n`` more events from ``source`` exceeds the cap."""
    minute = int(time.time() // 60)
    window = _rate_counters.get(source)
    if window is None or window[0] != minute:
        window = [minute, 0]
        _rate_counters[source] = window
    if window[1] + n > RATE_CAP_PER_MINUTE:
        return True
    window[1] += n
    return False


# ── HTTP handler ─────────────────────────────────────────────────────


async def fleet_events_receive(request: Request) -> JSONResponse:
    """``POST /api/fleet/events`` — normalize, persist, and enqueue triage."""
    from agent_utilities.core.config import AgentConfig

    # Optional shared-secret check (fresh config so a rotated token applies
    # without a gateway restart; constant-time compare).
    required = AgentConfig().fleet_events_token
    if required:
        offered = request.headers.get("x-fleet-events-token") or ""
        if not hmac.compare_digest(offered, required):
            return JSONResponse(
                {
                    "status": "error",
                    "message": "invalid or missing X-Fleet-Events-Token",
                },
                status_code=401,
            )

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "body must be JSON"}, status_code=400
        )

    source_hint = request.query_params.get("source") or request.headers.get(
        "x-event-source"
    )
    events = normalize_payload(payload, source_hint=source_hint)
    if not events:
        return JSONResponse(
            {"status": "error", "message": "no events recognized in payload"},
            status_code=400,
        )

    if _storm_capped(events[0].source, len(events)):
        return JSONResponse(
            {
                "status": "error",
                "message": f"event storm: per-source cap "
                f"({RATE_CAP_PER_MINUTE}/min) exceeded",
            },
            status_code=429,
        )

    try:
        engine = _get_engine()
    except Exception as e:  # noqa: BLE001 — engine genuinely unavailable
        logger.warning("fleet_events: engine unavailable: %s", e)
        engine = None
    if engine is None:
        # 503 so well-behaved senders (Alertmanager) retry instead of dropping.
        return JSONResponse(
            {"status": "error", "message": "knowledge-graph engine unavailable"},
            status_code=503,
        )

    accepted: list[dict[str, Any]] = []
    for ev in events:
        try:
            event_id = persist_event(engine, ev)
        except Exception as e:  # noqa: BLE001 — one bad event never drops the batch
            logger.warning("fleet_events: persist failed: %s", e)
            continue
        job_id = None
        try:
            job_id = enqueue_triage(engine, event_id, ev)
        except Exception as e:  # noqa: BLE001
            logger.warning("fleet_events: triage enqueue failed: %s", e)
        accepted.append(
            {
                "event_id": event_id,
                "job_id": job_id,
                "source": ev.source,
                "severity": ev.severity,
                "subject": ev.subject,
            }
        )

    status_code = 200 if accepted else 500
    return JSONResponse(
        {
            "status": "success" if accepted else "error",
            "accepted": len(accepted),
            "events": accepted,
        },
        status_code=status_code,
    )
