#!/usr/bin/python
from __future__ import annotations

"""Health-gated deploy/restart watch + policy-gated rollback.

CONCEPT:OS-5.27 — Health-gated deploy and rollback watch where every deploy or restart the autonomy plane triggers is followed by a durable health watch whose failure invokes a policy-gated rollback and escalation

This is the safety net under the reconciler (OS-5.25) and the remediation
playbooks (OS-5.26): they never fire-and-forget a mutating action. The flow:

1. an actuator deploy/restart succeeds;
2. :func:`watch_deploy` enqueues a durable ``deploy_watch`` task on the
   engine task queue (the queue gives durability: if the host dies mid-watch
   the zombie-task reaper requeues it and the new worker resumes against the
   same recorded deadline);
3. the worker-side :func:`run_deploy_watch` probes the
   :class:`~agent_utilities.orchestration.fleet_observation.FleetObserver`
   every ``DEPLOY_WATCH_POLL`` seconds until the window closes:

   * any ``down`` observation ⇒ **failed** → ``on_fail`` (default: a
     ``rollback_service`` action through the ActionPolicy gate + actuator,
     plus an operator notification);
   * the window closes with ≥1 healthy and 0 down probes ⇒ **success**;
   * no observation at all ⇒ **unobserved** — escalates a notification but
     does NOT roll back (never mutate on zero evidence).

Every outcome is a ``DeployWatch`` KG node linked to the watched service's
action trail.
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

OUTCOME_SUCCESS = "success"
OUTCOME_FAILED = "failed"
OUTCOME_UNOBSERVED = "unobserved"

# Tasks carry their watch spec in this Task-node property.
WATCH_PROP = "deploy_watch_json"

OnFail = Callable[[Any, dict[str, Any]], dict[str, Any]]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _config_float(name: str, fallback: float) -> float:
    try:
        from agent_utilities.core.config import config as _cfg

        return float(getattr(_cfg, name, fallback) or fallback)
    except Exception:  # noqa: BLE001
        return fallback


def watch_deploy(
    engine: Any,
    service: str,
    version: str = "",
    window_s: float | None = None,
    source: str = "manual",
    rollback_params: dict[str, Any] | None = None,
) -> str | None:
    """Schedule a durable health watch for ``service`` after a deploy/restart.

    Returns the queued task's job id (``None`` when the engine cannot queue).
    The watch spec rides on the Task node so a requeued/resumed watch keeps
    its original deadline.
    """
    submit = getattr(engine, "submit_task", None)
    if not callable(submit):
        logger.warning("watch_deploy: engine cannot queue durable tasks")
        return None
    window = float(window_s or _config_float("deploy_watch_window", 300.0))
    spec = {
        "service": service,
        "version": version,
        "window_s": window,
        "deadline_unix": time.time() + window,
        "source": source,
        "rollback_params": rollback_params or {},
    }
    return submit(
        service,
        is_codebase=False,
        provenance={WATCH_PROP: json.dumps(spec, default=str), "source": source},
        task_type="deploy_watch",
        skip_dedupe=True,
    )


def _load_spec(engine: Any, job_id: str, service: str) -> dict[str, Any]:
    """Read the watch spec back off the Task node (durable across requeues)."""
    try:
        rows = engine.query_cypher("MATCH (t:Task {id: $id}) RETURN t", {"id": job_id})
        props = rows[0].get("t") if rows else None
        if isinstance(props, dict) and props.get(WATCH_PROP):
            spec = json.loads(props[WATCH_PROP])
            if isinstance(spec, dict):
                return spec
    except Exception as e:  # noqa: BLE001
        logger.debug("deploy_watch: spec load failed for %s: %s", job_id, e)
    window = _config_float("deploy_watch_window", 300.0)
    return {
        "service": service,
        "version": "",
        "window_s": window,
        "deadline_unix": time.time() + window,
        "source": "unknown",
        "rollback_params": {},
    }


def default_on_fail(engine: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Default failure handler: policy-gated rollback + operator escalation."""
    from agent_utilities.orchestration.action_policy import (
        ActionRequest,
        get_action_policy,
    )
    from agent_utilities.orchestration.fleet_actuation import execute_action

    service = str(spec.get("service") or "")
    request = ActionRequest(
        kind="rollback_service",
        target=service,
        params={
            "version": spec.get("version") or "",
            **(spec.get("rollback_params") or {}),
        },
        source="deploy_watch",
        reason=f"health watch failed within {spec.get('window_s')}s",
    )
    decision = get_action_policy(engine).decide(request)
    out: dict[str, Any] = {
        "rollback_decision": decision.decision,
        "approval_id": decision.approval_id,
    }
    if decision.allowed:
        out["rollback_execution"] = execute_action(engine, request)
    _notify(
        f"[fleet-autonomy] deploy watch FAILED for {service} "
        f"(version={spec.get('version') or '?'}) — rollback {decision.decision}"
    )
    return out


def _notify(message: str) -> None:
    try:
        from agent_utilities.knowledge_graph.actions.dispatch import send_notification
        from agent_utilities.knowledge_graph.actions.models import NotificationSpec

        send_notification(
            NotificationSpec(channel="fleet", recipient="operators"), message
        )
    except Exception as e:  # noqa: BLE001
        logger.info("deploy_watch notification (fallback log): %s (%s)", message, e)


def _record(engine: Any, spec: dict[str, Any], outcome: str, detail: str) -> str | None:
    watch_id = f"deploy_watch:{uuid.uuid4().hex[:12]}"
    try:
        engine.add_node(
            watch_id,
            "DeployWatch",
            properties={
                "service": str(spec.get("service") or ""),
                "version": str(spec.get("version") or ""),
                "window_s": float(spec.get("window_s") or 0),
                "outcome": outcome,
                "detail": detail[:500],
                "source": str(spec.get("source") or ""),
                "completed_at": _now_iso(),
                "completed_unix": time.time(),
            },
        )
        return watch_id
    except Exception as e:  # noqa: BLE001
        logger.debug("deploy_watch: record write failed: %s", e)
        return None


def run_deploy_watch(
    engine: Any,
    service: str,
    job_id: str,
    on_fail: OnFail | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Worker-side watch loop: probe until the recorded deadline, then verdict.

    Bounded by the spec's ``deadline_unix`` (not by when this worker picked the
    task up), so a watch resumed after a host crash doesn't restart its window.
    """
    from agent_utilities.orchestration.fleet_observation import get_fleet_observer

    spec = _load_spec(engine, job_id, service)
    deadline = float(spec.get("deadline_unix") or 0) or (
        time.time() + float(spec.get("window_s") or 300.0)
    )
    poll = max(1.0, _config_float("deploy_watch_poll", 15.0))
    observer = get_fleet_observer(engine)

    healthy_probes = 0
    probes = 0
    failure_detail = ""
    while True:
        obs = observer.service_status(service)
        probes += 1
        if obs.status == "down":
            failure_detail = obs.detail or "observed down"
            break
        if obs.status == "up":
            healthy_probes += 1
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        sleep(min(poll, remaining))

    if failure_detail:
        outcome, detail = OUTCOME_FAILED, failure_detail
    elif healthy_probes > 0:
        outcome = OUTCOME_SUCCESS
        detail = f"sustained green ({healthy_probes}/{probes} healthy probes)"
    else:
        outcome = OUTCOME_UNOBSERVED
        detail = f"no observation for {service} during the window"

    watch_id = _record(engine, spec, outcome, detail)
    result: dict[str, Any] = {
        "service": service,
        "outcome": outcome,
        "detail": detail,
        "watch_id": watch_id,
        "probes": probes,
    }
    if outcome == OUTCOME_FAILED:
        handler = on_fail or default_on_fail
        try:
            result["on_fail"] = handler(engine, spec)
        except Exception as e:  # noqa: BLE001 — a failing handler never kills the worker
            logger.warning("deploy_watch: on_fail handler error: %s", e)
            result["on_fail"] = {"error": str(e)}
    elif outcome == OUTCOME_UNOBSERVED:
        _notify(
            f"[fleet-autonomy] deploy watch for {service} saw NO observations — "
            "verify monitoring coverage (no rollback on zero evidence)"
        )
    return result
