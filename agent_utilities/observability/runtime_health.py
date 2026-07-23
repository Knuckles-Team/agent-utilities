#!/usr/bin/python
from __future__ import annotations

"""The ONE shared runtime health-check core for graph-os (CONCEPT:AU-OS.observability.no-op-without-metrics).

Both the REST gateway (``agent_utilities/server/routers/core.py``'s ``/health``
and ``/health/ready``, plus ``graph_configure(action="health")``'s REST twin
``POST /api/graph/configure``) and the MCP server
(``agent_utilities/mcp/kg_server.py``'s ``/health``/``/health/ready`` routes and
its ``graph_configure`` tool) dispatch into :func:`collect_health` — there is
exactly one implementation of "is this deployment actually healthy", never two
that can drift (CONCEPT:AU-OS.config.two-surfaces-by-default).

Replaces the historical stub that returned ``{"status": "ok"}`` unconditionally
and never checked anything — which reported healthy through a ~2.5-minute real
engine outage. The hard rule this module exists to enforce:

    A probe that errors is UNHEALTHY with the reason. NEVER a probe whose
    failure is swallowed into "ok".

Every individual check function may itself misbehave (import error, a driver
bug, a genuinely hung socket) — :func:`collect_health` treats an exception OR a
timeout from a check exactly the same as an explicit ``unhealthy`` result: it
never lets a broken probe read as healthy, and it never lets a probe hang the
endpoint past its bound (each check runs in its own thread with a hard wall-
clock ceiling; a stuck check is abandoned, not awaited).

Four-state semantics per check (CONCEPT:AU-OS.observability.co-service-visibility):

* ``ok``             — configured (or mandatory) and verified reachable/healthy.
* ``unhealthy``       — configured (or mandatory) and NOT healthy. Always carries
  a ``reason``.
* ``not_configured``  — optional co-service/dependency that is simply not turned
  on in this deployment. Informational only; never fails the overall rollup.
* ``degraded``        — configured and PARTIALLY impaired in a way that is, by
  design, never part of the mandatory contract — e.g. an optional read-only
  kg_connections mirror (neo4j/falkordb/ladybug/...) that failed to construct.
  The epistemic-graph engine authority is independent of these by
  construction, so readiness must not pull graph-os out of Service routing
  over one; ``degraded`` carries a ``reason``/``detail`` like ``unhealthy``
  does, but — same as ``not_configured`` — never fails the overall rollup.

Liveness vs readiness (CONCEPT:AU-OS.deployment.liveness-vs-readiness-split — see the callers):
this module computes ONE truthful report; it is the callers (the HTTP routes)
that decide what to do with it. A liveness probe must keep returning 200 (the
process itself is alive and answering) even when the body says "unhealthy" —
killing/restarting a process because a *downstream dependency* is unreachable
just crash-loops a perfectly fine pod. A readiness probe maps the SAME report's
overall status onto the HTTP status code (200/503) so kubelet stops routing
traffic without touching the process.

No secret values ever appear in the payload: connection strings/DSNs/tokens are
never included; only counts, booleans, resolved modes, and platform/check
names. Raw engine endpoint strings are deliberately redacted to counts, matching
the same convention already used by ``deployment/doctor.py``'s ``_check_engine``
(an endpoint string can embed a local path or user-specific topology detail).
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

__all__ = ["collect_health", "is_overall_healthy"]

# One small shared worker pool for every probe — created once, reused across
# every /health hit. Sized generously above the current check count so every
# check always gets its own thread and none queue behind another.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="au-health-probe")

# Transport-level connect timeout per probe attempt (engine/kafka/stardog).
_PROBE_TIMEOUT_S = 0.75
# Hard wall-clock ceiling per check, enforced from OUTSIDE the check function
# itself — this is what makes "/health must never hang" true even if a check's
# own internal bound is buggy or missing.
_CHECK_WALL_TIMEOUT_S = 2.0


# --------------------------------------------------------------------------- #
# result builders
# --------------------------------------------------------------------------- #
def _ok(name: str, *, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "status": "ok"}
    if detail:
        out["detail"] = detail
    return out


def _unhealthy(
    name: str, reason: str, *, detail: dict[str, Any] | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "status": "unhealthy", "reason": reason}
    if detail:
        out["detail"] = detail
    return out


def _not_configured(name: str, reason: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "status": "not_configured"}
    if reason:
        out["reason"] = reason
    return out


def _degraded(
    name: str, reason: str, *, detail: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Configured-but-partially-impaired, where the impairment is, by design,
    outside the mandatory contract (see the module docstring's four-state
    semantics). Never counted as ``unhealthy`` by :func:`collect_health`'s
    rollup — informational only, same as ``not_configured``."""
    out: dict[str, Any] = {"name": name, "status": "degraded", "reason": reason}
    if detail:
        out["detail"] = detail
    return out


def _tcp_reachable(host: str, port: int, timeout: float) -> bool:
    """Bounded raw TCP connect — proves a listener exists, nothing more."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# individual checks — each is synchronous and returns a result dict; each is
# run inside the bounded wrapper below, so an uncaught exception or a hang is
# converted to "unhealthy", never swallowed into "ok".
# --------------------------------------------------------------------------- #
def _check_engine(cfg: Any) -> dict[str, Any]:
    """Real reachability of the epistemic-graph engine — a live socket connect
    per configured endpoint, plus the shared per-endpoint circuit-breaker state
    (CONCEPT:AU-OS.observability.no-op-without-metrics's ``engine_breaker.py``).
    This is the mandatory check: the engine is never "not configured".
    """
    from agent_utilities.knowledge_graph.core.engine_resolver import resolve_engine
    from agent_utilities.knowledge_graph.core.shard_topology import (
        default_graph_name,
        shard_topology_status,
    )

    graph_name = default_graph_name(cfg)
    topology = shard_topology_status(cfg, probe=True, timeout=_PROBE_TIMEOUT_S)
    resolved = resolve_engine(cfg, graph_name)
    endpoints = topology.get("endpoints") or []
    reachable = [e for e in endpoints if e.get("reachable")]
    open_breakers = [e for e in endpoints if e.get("breaker") == "open"]

    # Endpoint strings can embed local socket paths or topology-identifying
    # detail — report counts only (matches deployment/doctor.py's redaction
    # convention for this exact same data).
    detail = {
        "resolved_mode": resolved.mode,
        "endpoint_count": len(endpoints),
        "reachable_count": len(reachable),
        "open_breaker_count": len(open_breakers),
    }

    if not reachable:
        reason = (
            "configured remote engine is unreachable (remote mode never "
            "autostarts a local stand-in — fail-loud)"
            if resolved.mode == "remote"
            else f"no engine endpoint reachable (resolved_mode={resolved.mode})"
        )
        return _unhealthy("engine", reason, detail=detail)

    if open_breakers:
        return _unhealthy(
            "engine",
            f"circuit breaker OPEN on {len(open_breakers)}/{len(endpoints)} "
            "endpoint(s) — recent real engine calls are failing fast",
            detail=detail,
        )

    return _ok("engine", detail=detail)


def _check_kg_host_daemon(cfg: Any) -> dict[str, Any]:
    """The consolidated KG background daemon (queue drain / checkpoint /
    maintenance — ``gateway/daemon.py`` + ``knowledge_graph/core/host_lock.py``).

    Only "configured" (i.e. expected to exist SOMEWHERE in this deployment)
    when the engine resolves to something other than a remote, externally
    managed engine — a remote engine's own deployment owns its own daemon.
    Liveness is read via the shared advisory flock, which is real and
    cross-process (it detects ANY live holder, not just one started by this
    process), never by trying to acquire the lock ourselves.
    """
    from agent_utilities.knowledge_graph.core.engine_resolver import resolve_engine
    from agent_utilities.knowledge_graph.core.host_lock import host_daemon_running
    from agent_utilities.knowledge_graph.core.shard_topology import default_graph_name

    resolved = resolve_engine(cfg, default_graph_name(cfg))
    if resolved.mode == "remote":
        return _not_configured(
            "kg_host_daemon",
            "engine is remote-managed; no local host daemon is expected here",
        )
    if host_daemon_running():
        return _ok("kg_host_daemon")
    return _unhealthy(
        "kg_host_daemon", "no live process currently holds the KG host-daemon lock"
    )


def _check_messaging(cfg: Any) -> dict[str, Any]:
    """Messaging co-service (Telegram/Slack/Mattermost/...) — configured iff any
    platform has credentials in the environment; healthy iff every configured
    platform has a connected backend instance IN THIS PROCESS.

    This intentionally reflects only this process's view, matching the
    self-composing entrypoint (graph-os starting messaging in-process when
    configured): each process's own health is authoritative for itself. Never
    instantiates a backend or performs network I/O — read-only registry scan.
    """
    from agent_utilities.messaging.registry import MessagingRegistry

    registry = MessagingRegistry.instance()
    configured = registry.configured_backend_ids()
    if not configured:
        return _not_configured(
            "messaging", "no messaging platform credentials configured"
        )
    down = [
        backend_id
        for backend_id in configured
        if not bool(getattr(registry.get_backend(backend_id), "is_connected", False))
    ]
    detail = {"configured": configured, "down": down}
    if down:
        return _unhealthy(
            "messaging",
            f"{len(down)}/{len(configured)} configured platform(s) not connected "
            f"in this process: {', '.join(down)}",
            detail=detail,
        )
    return _ok("messaging", detail=detail)


def _check_state_store(cfg: Any) -> dict[str, Any]:
    """Externalized Postgres state store (``core/state_store.py``) — only
    checked when ``STATE_DB_URI`` selects it; a real bounded connection-pool
    checkout + ``SELECT 1`` against the SAME pool the app already serves from.

    "Configured" is read off the SAME ``cfg`` snapshot every other check in
    this module uses (not the process-wide reloadable config singleton
    ``state_store.py`` itself reads) so one health report is never internally
    inconsistent about which configuration it is describing.
    """
    from agent_utilities.core import state_store

    uri = str(getattr(cfg, "state_db_uri", "") or "")
    if not uri.startswith(("postgresql://", "postgres://")):
        return _not_configured(
            "state_store", "STATE_DB_URI not set (per-host SQLite default in effect)"
        )
    try:
        pool = state_store.state_pool()
        with pool.connection(timeout=_PROBE_TIMEOUT_S) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        return _unhealthy(
            "state_store",
            f"could not reach externalized Postgres state store ({type(exc).__name__})",
            detail={"backend": "postgres"},
        )
    return _ok("state_store", detail={"backend": "postgres"})


def _check_kafka_bus(cfg: Any) -> dict[str, Any]:
    """Kafka bus/queue backend — only checked when either
    ``AGENT_BUS_LOG_BACKEND`` or ``TASK_QUEUE_BACKEND`` selects kafka. A bounded
    raw TCP connect to the first configured bootstrap broker.
    """
    bus_backend = str(getattr(cfg, "agent_bus_log_backend", "") or "").strip().lower()
    queue_backend = str(getattr(cfg, "task_queue_backend", "") or "").strip().lower()
    if bus_backend != "kafka" and queue_backend != "kafka":
        return _not_configured(
            "kafka_bus",
            "AGENT_BUS_LOG_BACKEND/TASK_QUEUE_BACKEND is not kafka",
        )
    servers = str(getattr(cfg, "kafka_bootstrap_servers", "") or "").strip()
    if not servers:
        return _unhealthy(
            "kafka_bus",
            "kafka backend selected but KAFKA_BOOTSTRAP_SERVERS is empty",
        )
    server_list = [s.strip() for s in servers.split(",") if s.strip()]
    first = server_list[0] if server_list else ""
    host, _sep, port_s = first.rpartition(":")
    if not host or not port_s.isdigit():
        return _unhealthy(
            "kafka_bus", "KAFKA_BOOTSTRAP_SERVERS is not a valid host:port list"
        )
    detail = {"configured_broker_count": len(server_list)}
    if _tcp_reachable(host, int(port_s), _PROBE_TIMEOUT_S):
        return _ok("kafka_bus", detail=detail)
    return _unhealthy("kafka_bus", "kafka bootstrap broker unreachable", detail=detail)


def _check_stardog_mirror(cfg: Any) -> dict[str, Any]:
    """Continuous Stardog mirror — only checked when ``CONTINUOUS_STARDOG_MIRROR``
    is on. A bounded raw TCP connect to the configured Stardog endpoint (no
    authenticated call — credentials are never exercised by a health probe).
    """
    if not bool(getattr(cfg, "continuous_stardog_mirror", False)):
        return _not_configured("stardog_mirror", "CONTINUOUS_STARDOG_MIRROR is off")

    from agent_utilities.core.config import setting

    endpoint = str(setting("STARDOG_ENDPOINT", "http://localhost:5820") or "")
    parts = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    host = parts.hostname
    port = parts.port or 5820
    if not host:
        return _unhealthy("stardog_mirror", "STARDOG_ENDPOINT is not a valid URL")
    if _tcp_reachable(host, port, _PROBE_TIMEOUT_S):
        return _ok("stardog_mirror")
    return _unhealthy("stardog_mirror", "stardog mirror endpoint unreachable")


def _check_kg_mirrors(cfg: Any) -> dict[str, Any]:
    """Optional read-only ``kg_connections`` mirrors (neo4j/falkordb/ladybug/
    pg-age/stardog, CONCEPT:AU-KG.backend.mirror-health-repair) — interop/BI/DR
    projections the epistemic-graph engine authority is completely independent
    of by construction. Read-only snapshot of the last real fan-out build
    attempt (``knowledge_graph/backends/__init__.py``'s ``_build_mirror_set``,
    run once when the operational backend is constructed); never builds,
    reconnects, or does any network I/O of its own — same read-only-registry-
    scan convention as the ``messaging`` check above.

    A mirror that failed to construct (missing driver / unreachable host / bad
    credentials / invalid ``connection_profile_ref``) is reported here as
    ``degraded`` detail, NEVER as ``unhealthy``: an optional mirror going down
    must not pull graph-os out of Service routing (see the module docstring's
    four-state semantics — the same "optional co-service never fails
    readiness" rationale ``not_configured`` already encodes elsewhere in this
    file, applied to a partially-impaired optional dependency instead of an
    absent one).
    """
    from agent_utilities.knowledge_graph.backends import (
        _resolve_mirror_target_names,
        get_mirror_build_status,
    )

    configured = _resolve_mirror_target_names()
    if not configured:
        return _not_configured(
            "kg_mirrors",
            "no kg_connections mirror role / GRAPH_MIRROR_TARGETS configured",
        )

    status = get_mirror_build_status()
    healthy = [name for name in configured if status.get(name, {}).get("ok")]
    failed = {
        name: status[name].get("reason", "unknown cause")
        for name in configured
        if name in status and not status[name].get("ok")
    }
    unobserved = [name for name in configured if name not in status]
    detail = {
        "configured": configured,
        "healthy": healthy,
        "failed": failed,
        "unobserved": unobserved,
    }
    if failed:
        return _degraded(
            "kg_mirrors",
            f"{len(failed)}/{len(configured)} optional mirror(s) unavailable "
            f"(engine authority unaffected): {', '.join(sorted(failed))}",
            detail=detail,
        )
    return _ok("kg_mirrors", detail=detail)


# Ordered so the mandatory check reports first; order is otherwise cosmetic.
def _check_bundled_skills(cfg: Any) -> dict[str, Any]:
    """Report packaged-skill readiness.

    This deliberately does NOT gate the process: a bundled skill failing to
    ingest is a capability gap, not a reason to refuse every unrelated tool. But
    it must be visible, so an incomplete bundle reports ``unhealthy`` here and
    names the skills — "serving degraded" has to be distinguishable from "fully
    ready" from outside the process.
    """
    try:
        from agent_utilities.mcp.kg_server import bundled_skill_readiness
    except Exception:  # noqa: BLE001 - the MCP surface may not be importable
        return {"status": "not_configured", "reason": "graph-os MCP surface not loaded"}
    report = bundled_skill_readiness()
    if not report:
        return {"status": "not_configured", "reason": "skill bootstrap has not run"}
    not_ready = list(report.get("not_ready") or [])
    detail = {
        "ready": report.get("ready"),
        "required": report.get("required"),
        "not_ready": not_ready,
    }
    if report.get("error"):
        detail["error"] = report["error"]
    if not_ready:
        # Reported as ok-with-degraded-detail ON PURPOSE. /health/ready drives
        # kubelet's readiness probe, so returning unhealthy here would pull the
        # pod out of Service routing over a capability gap — re-creating, at the
        # routing layer, exactly the boot gate we just removed. The gap stays
        # fully visible in `detail` (and is logged loudly at bootstrap); it just
        # does not take the server out of rotation.
        detail["degraded"] = True
        detail["reason"] = f"{len(not_ready)} packaged skill(s) not ready"
        return {"status": "ok", "detail": detail}
    return {"status": "ok", "detail": detail}


_CHECKS: tuple[tuple[str, Callable[[Any], dict[str, Any]]], ...] = (
    ("engine", _check_engine),
    ("kg_host_daemon", _check_kg_host_daemon),
    ("messaging", _check_messaging),
    ("state_store", _check_state_store),
    ("kafka_bus", _check_kafka_bus),
    ("stardog_mirror", _check_stardog_mirror),
    ("bundled_skills", _check_bundled_skills),
    ("kg_mirrors", _check_kg_mirrors),
)


def _run_bounded(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one check on the shared pool; NEVER let it read as healthy on
    failure or hang the caller past ``_CHECK_WALL_TIMEOUT_S``.
    """
    started = time.monotonic()
    future: Future = _EXECUTOR.submit(fn)
    try:
        result = future.result(timeout=_CHECK_WALL_TIMEOUT_S)
    except _FutureTimeoutError:
        result = _unhealthy(
            name, f"probe exceeded its {_CHECK_WALL_TIMEOUT_S:.1f}s bound"
        )
    except Exception as exc:
        # The one place a bug INSIDE a check function still cannot report
        # healthy — any exception here becomes an explicit "unhealthy", never
        # a swallowed pass. (CONCEPT:AU-OS.observability.no-op-without-metrics)
        result = _unhealthy(name, f"probe raised {type(exc).__name__}: {exc}")
    result.setdefault("name", name)
    result["latency_ms"] = round((time.monotonic() - started) * 1000.0, 1)
    return result


def collect_health() -> dict[str, Any]:
    """Run every runtime-health check concurrently and return one truthful report.

    Returns::

        {
          "status": "healthy" | "unhealthy",
          "checks": [
            {"name": str, "status": "ok"|"unhealthy"|"not_configured"|"degraded",
             "latency_ms": float,
             "reason": str (on unhealthy/not_configured/degraded),
             "detail": {...} (optional, non-secret)},
            ...
          ],
          "generated_at": "<iso8601 UTC>",
        }

    Only ``unhealthy`` checks flip the overall rollup — ``degraded`` and
    ``not_configured`` are both informational-only (four-state semantics, see
    the module docstring).

    Never raises. Never blocks past ``_CHECK_WALL_TIMEOUT_S`` per check (all
    checks run in parallel, so the wall-clock cost of this whole function is
    bounded by the SLOWEST single check, not their sum).
    """
    from agent_utilities.core.config import AgentConfig

    try:
        cfg = AgentConfig()
    except Exception as exc:
        # Config itself is unreadable — that is unambiguously unhealthy, not a
        # reason to fall back to "ok".
        return {
            "status": "unhealthy",
            "checks": [
                _unhealthy(
                    "config", f"AgentConfig() raised {type(exc).__name__}: {exc}"
                )
            ],
            "generated_at": _now_iso(),
        }

    checks = [
        _run_bounded(name, lambda fn=fn, cfg=cfg: fn(cfg)) for name, fn in _CHECKS
    ]
    overall = (
        "unhealthy" if any(c["status"] == "unhealthy" for c in checks) else "healthy"
    )
    return {"status": overall, "checks": checks, "generated_at": _now_iso()}


def is_overall_healthy(report: dict[str, Any]) -> bool:
    """True iff a :func:`collect_health` report's rollup is healthy."""
    # Readiness answers ONE question: can this process serve requests right now.
    # That depends on the engine it reads and writes through — not on optional
    # co-services which run in their own deployments. Gating routing on those
    # means an unrelated outage (a down messaging daemon, a stopped mirror) pulls
    # a perfectly serving graph-os out of the Service, turning one component's
    # failure into a total one. They stay fully reported in /health; they just do
    # not decide rotation.
    essential = {"engine"}
    return not any(
        check["status"] == "unhealthy" and check["name"] in essential
        for check in report.get("checks", [])
    )


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
