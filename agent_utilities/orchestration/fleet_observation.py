#!/usr/bin/python
from __future__ import annotations

"""Fleet observers — the injectable eyes of the autonomy control plane.

CONCEPT:OS-5.25 — Desired-state fleet reconciler (observation seam).

The reconciler (and the OS-5.26 playbooks / OS-5.27 deploy watches) never
shell out to monitoring systems directly; they read a :class:`FleetObserver`:

* :class:`KGFleetObserver` — the DEFAULT, zero-infra observer: folds the
  recent ``FleetEvent`` ingress stream (OS-5.15 — Alertmanager / Uptime Kuma /
  Portainer webhooks already normalized into the KG) into a per-service
  status snapshot, including a flap count for the playbooks' back-off logic.
* :class:`DockerFleetObserver` — optional local docker CLI/socket snapshot
  (running containers + swarm service replica counts); inert without docker.
* :class:`CompositeFleetObserver` — merges several observers; positive
  evidence of *unhealth* wins over silence.

A deployment with a richer source of truth (Portainer API, Prometheus, …)
injects its own implementation via :func:`set_fleet_observer`.

The crucial conservative property: a service NOBODY observed reports
``status='unknown'`` — the reconciler only acts on positive evidence.
"""

import json
import logging
import shutil
import subprocess  # nosec B404 — argv-only docker CLI calls, no shell
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

STATUS_UP = "up"
STATUS_DOWN = "down"
STATUS_UNKNOWN = "unknown"

# Statuses on FleetEvent nodes that mean the subject is unhealthy / healthy.
_DOWN_STATUSES = {"down", "firing"}
_UP_STATUSES = {"up", "resolved", "ok"}

# How far back KG fleet events count as "current" observation, and how many
# recent events one snapshot folds.
DEFAULT_EVENT_WINDOW_S = 1800.0
_EVENT_SCAN_LIMIT = 500


@dataclass
class ServiceObservation:
    """What one observer (or a merge of them) currently believes about a service."""

    name: str
    status: str = STATUS_UNKNOWN  # up | down | unknown
    replicas: int | None = None  # observed replicas, when knowable
    flap_count: int = 0  # down-transitions inside the window
    last_seen_unix: float = 0.0
    sources: list[str] = field(default_factory=list)
    detail: str = ""


@runtime_checkable
class FleetObserver(Protocol):
    """Anything that can snapshot the observed fleet state."""

    name: str

    def observe(self) -> dict[str, ServiceObservation]:
        """Return ``{service_name: ServiceObservation}``. Never raises."""
        ...  # ABSTRACT-OK

    def service_status(self, name: str) -> ServiceObservation:
        """Current observation for one service (``unknown`` when unobserved)."""
        ...  # ABSTRACT-OK


class KGFleetObserver:
    """Default observer: folds recent ``FleetEvent`` nodes into service status.

    The latest event per subject wins (down/firing ⇒ down, up/resolved ⇒ up);
    the count of down-events inside the window is exposed as ``flap_count``.
    """

    name = "kg"

    def __init__(self, engine: Any, window_s: float = DEFAULT_EVENT_WINDOW_S):
        self.engine = engine
        self.window_s = window_s

    def _recent_events(self) -> list[dict[str, Any]]:
        if self.engine is None:
            return []
        try:
            rows = self.engine.query_cypher(
                "MATCH (e:FleetEvent) RETURN e.subject AS subject, "
                "e.status AS status, e.severity AS severity, "
                f"e.received_at AS received_at LIMIT {_EVENT_SCAN_LIMIT}"
            )
        except Exception as e:  # noqa: BLE001 — observation is best-effort
            logger.debug("KGFleetObserver: event scan failed: %s", e)
            return []
        return [r for r in rows or [] if isinstance(r, dict) and r.get("subject")]

    @staticmethod
    def _to_unix(received_at: Any) -> float:
        """Parse the FleetEvent's UTC ``...Z`` timestamp to epoch seconds."""
        try:
            import calendar

            return float(
                calendar.timegm(time.strptime(str(received_at), "%Y-%m-%dT%H:%M:%SZ"))
            )
        except (ValueError, TypeError):
            return 0.0

    def observe(self) -> dict[str, ServiceObservation]:
        since = time.time() - self.window_s
        out: dict[str, ServiceObservation] = {}
        for ev in self._recent_events():
            subject = str(ev["subject"])
            ts = self._to_unix(ev.get("received_at"))
            if ts and ts < since:
                continue
            obs = out.setdefault(
                subject, ServiceObservation(name=subject, sources=[self.name])
            )
            status = str(ev.get("status") or "").lower()
            if status in _DOWN_STATUSES:
                obs.flap_count += 1
            if ts >= obs.last_seen_unix:
                obs.last_seen_unix = ts
                if status in _DOWN_STATUSES:
                    obs.status = STATUS_DOWN
                elif status in _UP_STATUSES:
                    obs.status = STATUS_UP
                obs.detail = f"latest fleet event status={status or 'unknown'}"
        return out

    def service_status(self, name: str) -> ServiceObservation:
        return self.observe().get(name) or ServiceObservation(name=name)


class DockerFleetObserver:
    """Optional observer over the local docker CLI (containers + swarm services)."""

    name = "docker"

    def __init__(self, docker_bin: str | None = None, timeout: float = 20.0):
        self.docker_bin = docker_bin or shutil.which("docker")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.docker_bin)

    def _lines(self, *args: str) -> list[str]:
        try:
            proc = subprocess.run(  # nosec B603 — fixed binary, fixed argv
                [str(self.docker_bin), *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                return []
            return [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        except Exception as e:  # noqa: BLE001
            logger.debug("DockerFleetObserver: %s failed: %s", args[0], e)
            return []

    def observe(self) -> dict[str, ServiceObservation]:
        if not self.available:
            return {}
        now = time.time()
        out: dict[str, ServiceObservation] = {}
        for ln in self._lines("ps", "--format", "{{json .}}"):
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            name = str(row.get("Names") or "").split(",")[0]
            if not name:
                continue
            state = str(row.get("State") or "").lower()
            out[name] = ServiceObservation(
                name=name,
                status=STATUS_UP if state == "running" else STATUS_DOWN,
                replicas=1 if state == "running" else 0,
                last_seen_unix=now,
                sources=[self.name],
                detail=f"docker ps state={state}",
            )
        for ln in self._lines("service", "ls", "--format", "{{json .}}"):
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            name = str(row.get("Name") or "")
            if not name:
                continue
            running, _, wanted = str(row.get("Replicas") or "0/0").partition("/")
            try:
                running_n = int(running)
                wanted_n = int(wanted or 0)
            except ValueError:
                running_n, wanted_n = 0, 0
            out[name] = ServiceObservation(
                name=name,
                status=STATUS_UP if running_n >= max(wanted_n, 1) else STATUS_DOWN,
                replicas=running_n,
                last_seen_unix=now,
                sources=[self.name],
                detail=f"swarm replicas {running_n}/{wanted_n}",
            )
        return out

    def service_status(self, name: str) -> ServiceObservation:
        return self.observe().get(name) or ServiceObservation(name=name)


class CompositeFleetObserver:
    """Merge several observers; positive evidence beats silence, down beats up."""

    name = "composite"

    def __init__(self, observers: list[FleetObserver]):
        self.observers = observers

    def observe(self) -> dict[str, ServiceObservation]:
        merged: dict[str, ServiceObservation] = {}
        for observer in self.observers:
            try:
                snapshot = observer.observe()
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "observer %s failed: %s", getattr(observer, "name", "?"), e
                )
                continue
            for name, obs in snapshot.items():
                prev = merged.get(name)
                if prev is None:
                    merged[name] = obs
                    continue
                prev.sources = sorted(set(prev.sources) | set(obs.sources))
                prev.flap_count = max(prev.flap_count, obs.flap_count)
                prev.last_seen_unix = max(prev.last_seen_unix, obs.last_seen_unix)
                if obs.replicas is not None:
                    prev.replicas = obs.replicas
                # Down anywhere = down; up only upgrades unknown.
                if obs.status == STATUS_DOWN:
                    prev.status, prev.detail = STATUS_DOWN, obs.detail
                elif obs.status == STATUS_UP and prev.status == STATUS_UNKNOWN:
                    prev.status, prev.detail = STATUS_UP, obs.detail
        return merged

    def service_status(self, name: str) -> ServiceObservation:
        return self.observe().get(name) or ServiceObservation(name=name)


# ── registry (deployment injection point) ───────────────────────────

_OBSERVER: FleetObserver | None = None


def set_fleet_observer(observer: FleetObserver | None) -> None:
    """Register the process-wide observer (``None`` resets to the default)."""
    global _OBSERVER
    _OBSERVER = observer


def get_fleet_observer(engine: Any = None) -> FleetObserver:
    """Resolve the active observer: injected > KG events (+ docker if present)."""
    if _OBSERVER is not None:
        return _OBSERVER
    observers: list[FleetObserver] = [KGFleetObserver(engine)]
    docker = DockerFleetObserver()
    if docker.available:
        observers.append(docker)
    return observers[0] if len(observers) == 1 else CompositeFleetObserver(observers)
