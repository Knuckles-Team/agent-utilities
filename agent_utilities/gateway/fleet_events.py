"""Fleet events webhook ingress (CONCEPT:AU-OS.config.fleet-event-ingress — Fleet Event Ingress).

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

Auth: Alertmanager/Kuma cannot usually mint JWTs, so the endpoint accepts a
shared-secret header ``X-Fleet-Events-Token`` resolved through
``FLEET_EVENTS_TOKEN_REF``. Authenticated identity middleware callers are also
accepted. With neither boundary, ingress is denied. A bounded,
concurrency-safe, privacy-keyed per-source counter caps event
storms (429 when exceeded). Persisted nodes contain only pseudonymous references
and normalized classifications; raw webhook content is never retained.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import threading
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
MAX_EVENTS_PER_REQUEST = 256
MAX_EVENT_BODY_BYTES = 16 * 1024 * 1024
_rate_counters: dict[str, list[int]] = {}  # opaque source ref -> [minute, count]
_rate_lock = threading.Lock()
_rate_privacy_key = secrets.token_bytes(32)


def _source_type(source: str) -> str:
    normalized = source.strip().lower()
    return (
        normalized
        if normalized in {"alertmanager", "uptime-kuma", "portainer"}
        else "generic"
    )


def _ephemeral_source_ref(source: str) -> str:
    digest = hashlib.blake2s(
        source.encode("utf-8"), key=_rate_privacy_key, digest_size=12
    ).hexdigest()
    return f"source_{digest}"


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
        self.source = str(self.source)[:256]
        self.severity = _normalize_severity(self.severity)
        self.subject = str(self.subject)[:1024]
        self.status = str(self.status)[:64]
        self.summary = str(self.summary)[:4096]
        if not self.received_at:
            self.received_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())
            )


# ── payload normalization ────────────────────────────────────────────


def _normalize_alertmanager(payload: dict[str, Any]) -> list[FleetEvent]:
    """Prometheus Alertmanager v4 webhook JSON → one event per ``alerts[]``."""
    events: list[FleetEvent] = []
    for alert in (payload.get("alerts") or [])[:MAX_EVENTS_PER_REQUEST]:
        if not isinstance(alert, dict):
            continue
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
        return _normalize_alertmanager(payload)[:MAX_EVENTS_PER_REQUEST]
    if isinstance(payload.get("heartbeat"), dict) or isinstance(
        payload.get("monitor"), dict
    ):
        return _normalize_uptime_kuma(payload)[:MAX_EVENTS_PER_REQUEST]
    return _normalize_generic(payload, source_hint)[:MAX_EVENTS_PER_REQUEST]


# ── persistence + queue ──────────────────────────────────────────────


def _get_engine() -> Any:
    """Resolve the gateway's KG engine (kept as a seam for tests)."""
    from agent_utilities.mcp.kg_server import _get_engine as _kg_engine

    return _kg_engine()


def _correlation_stamp() -> dict[str, str]:
    """Correlation + identity to stamp on persisted effect nodes (CONCEPT:AU-OS.observability.run-wide-correlation-id).

    Stamping the originating ``correlation_id`` (+ actor/tenant) onto the durable
    node is what makes the swarm-wide ``/api/fleet/trace`` and ``/api/fleet/touched``
    queries answerable from the graph rather than only from external traces.
    """
    stamp: dict[str, str] = {}
    try:
        from agent_utilities.observability import correlation
        from agent_utilities.security.persistence_privacy import (
            persistence_reference,
        )

        stamp["correlation_ref"] = persistence_reference(
            "correlation", correlation.ensure_correlation_id(), namespace="fleet-event"
        )
        try:
            from agent_utilities.security.brain_context import current_actor

            actor = current_actor()
            if actor.actor_id and actor.actor_id != "system":
                stamp["actor_ref"] = persistence_reference(
                    "actor", actor.actor_id, namespace="fleet-event"
                )
            if actor.tenant_id:
                stamp["tenant_ref"] = persistence_reference(
                    "tenant", actor.tenant_id, namespace="fleet-event"
                )
        except Exception:  # noqa: BLE001 — identity is best-effort context
            pass
    except Exception:  # noqa: BLE001 — correlation is best-effort context
        pass
    return stamp


def persist_event(engine: Any, event: FleetEvent) -> str:
    """Write the event as a ``FleetEvent`` KG node; returns the node id."""
    from agent_utilities.security.persistence_privacy import persistence_reference

    event_id = f"fleet_event:{uuid.uuid4().hex}"
    source_type = _source_type(event.source)
    properties = {
        "source_type": source_type,
        "source_ref": persistence_reference(
            "fleet_source", event.source, namespace="fleet-event"
        ),
        "severity": event.severity,
        "subject_ref": persistence_reference(
            "fleet_subject", event.subject, namespace=source_type
        ),
        "status": event.status,
        "summary_ref": persistence_reference(
            "fleet_summary", event.summary, namespace=source_type
        ),
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
        provenance={
            "source": "fleet_events",
            "event_source": _source_type(event.source),
        },
        task_type="fleet_event_triage",
        skip_dedupe=True,
    )


def _storm_capped(source: str, n: int = 1) -> bool:
    """True when accepting ``n`` more events from ``source`` exceeds the cap."""
    minute = int(time.time() // 60)
    source_ref = _ephemeral_source_ref(source)
    with _rate_lock:
        if len(_rate_counters) >= 4096:
            stale = [key for key, value in _rate_counters.items() if value[0] != minute]
            for key in stale:
                _rate_counters.pop(key, None)
            if len(_rate_counters) >= 4096 and source_ref not in _rate_counters:
                return True
        window = _rate_counters.get(source_ref)
        if window is None or window[0] != minute:
            window = [minute, 0]
            _rate_counters[source_ref] = window
        if window[1] + n > RATE_CAP_PER_MINUTE:
            return True
        window[1] += n
        return False


# ── HTTP handler ─────────────────────────────────────────────────────


def _resolve_webhook_secret(cfg: Any) -> str | None:
    reference = str(cfg.fleet_events_token_ref or "").strip()
    if reference:
        from agent_utilities.security.secrets_client import create_secrets_client

        secret = create_secrets_client().resolve_ref(reference)
        if not secret:
            raise RuntimeError("fleet-events secret reference did not resolve")
        return str(secret)
    return None


async def fleet_events_receive(request: Request) -> JSONResponse:
    """``POST /api/fleet/events`` — normalize, persist, and enqueue triage."""
    from agent_utilities.core.config import AgentConfig

    # Fresh config lets secret rotation take effect without a gateway restart.
    cfg = AgentConfig()
    try:
        required = _resolve_webhook_secret(cfg)
    except Exception:
        logger.error("fleet-events webhook secret is unavailable")
        return JSONResponse(
            {"status": "error", "message": "webhook authentication unavailable"},
            status_code=503,
        )
    if required:
        offered = request.headers.get("x-fleet-events-token") or ""
        if not hmac.compare_digest(offered, required):
            return JSONResponse(
                {"status": "error", "message": "authentication required"},
                status_code=401,
            )
    else:
        try:
            from agent_utilities.security.brain_context import current_actor

            authenticated = current_actor().authenticated
        except Exception:
            authenticated = False
        if not authenticated:
            return JSONResponse(
                {"status": "error", "message": "authentication required"},
                status_code=401,
            )

    try:
        limit = max(1, min(int(cfg.max_upload_size), MAX_EVENT_BODY_BYTES))
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                return JSONResponse(
                    {"status": "error", "message": "request body is too large"},
                    status_code=413,
                )
        payload = json.loads(body)
    except Exception:
        return JSONResponse(
            {"status": "error", "message": "body must be JSON"}, status_code=400
        )

    source_hint = (
        request.query_params.get("source")
        or request.headers.get("x-event-source")
        or ""
    )[:256]
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
        logger.warning("fleet-events engine unavailable (%s)", type(e).__name__)
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
            logger.warning("fleet-events persist failed (%s)", type(e).__name__)
            continue
        job_id = None
        try:
            job_id = enqueue_triage(engine, event_id, ev)
        except Exception as e:  # noqa: BLE001
            logger.warning("fleet-events enqueue failed (%s)", type(e).__name__)
        accepted.append(
            {
                "event_id": event_id,
                "job_id": job_id,
                "source": _source_type(ev.source),
                "severity": ev.severity,
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
