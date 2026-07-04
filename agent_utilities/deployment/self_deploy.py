#!/usr/bin/python
from __future__ import annotations

"""Governed self-deploy — close the build loop's last mile (CONCEPT:AU-OS.deployment.os-2).

The missing step after build→test→merge: make the merged change *live*. The served
daemon source-mounts the code but only picks it up on restart, and that restart is
the opaque, manual gap that breaks the loop every time. This provides:

* :func:`plan_redeploy` — the exact, safe answer to "how do I make my change live"
  (the restart command, the health endpoint, the rollback note) — used by the
  ``deploy`` context provider so the answer is one query.
* :func:`execute_redeploy` — the *governed* mutation: it passes through the
  fail-closed :class:`ActionPolicy` gate (``kind="restart_service"``), is **dry-run
  by default** (``confirm=False`` plans only), and on a real run health-checks after
  the restart and reports a rollback path. Credential/host access stays human-gated
  by design — this never force-restarts a shared daemon without explicit confirm AND
  a policy allow.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SERVICE = "graph-os"
# The restart mechanic is deployment-specific; this is the documented default for
# the fleet (source-mounted compose → restart picks up code). Operators override.
_RESTART_HINT = "docker service update --force <stack>_{service} (on the manager)"
_HEALTH_HINT = "graph_configure action=system_doctor (or GET /graph/configure/doctor)"


def plan_redeploy(service: str = DEFAULT_SERVICE) -> dict[str, Any]:
    """The safe, read-only plan to make a merged change live (never mutates)."""
    return {
        "service": service,
        "why": (
            "source mounts update files, not the running process — the served "
            f"{service} must restart to load merged code"
        ),
        "restart": _RESTART_HINT.format(service=service),
        "verify": _HEALTH_HINT,
        "rollback": (
            "if the health check fails after restart, redeploy the previous image/"
            "revision (the prior commit) and restart again"
        ),
        "governed": "execute_redeploy() runs this through the ActionPolicy gate",
    }


def execute_redeploy(
    service: str = DEFAULT_SERVICE,
    *,
    confirm: bool = False,
    engine: Any = None,
    restart_fn: Any = None,
    health_fn: Any = None,
) -> dict[str, Any]:
    """Governed redeploy: policy-gated, dry-run by default, health-gated + rollback.

    Returns a structured result. With ``confirm=False`` (default) it only plans.
    With ``confirm=True`` it (1) asks :class:`ActionPolicy` to ``decide`` a
    ``restart_service`` action — proceeding ONLY on an allowing verdict — then (2)
    runs ``restart_fn`` (injected; no built-in shell-out to a shared host), (3)
    health-checks via ``health_fn``, and (4) reports a rollback path on failure.
    """
    plan = plan_redeploy(service)
    if not confirm:
        return {"status": "planned", "executed": False, "plan": plan}

    # 1. Governance gate (fail-closed).
    try:
        from agent_utilities.orchestration.action_policy import (
            ActionRequest,
            get_action_policy,
        )

        decision = get_action_policy(engine).decide(
            ActionRequest(
                kind="restart_service",
                target=service,
                source="self_deploy",
                reason="make merged change live",
                actor_id="self_deploy",
            )
        )
        verdict = getattr(decision, "decision", decision)
        if str(verdict) not in ("allow", "allow_notify"):
            return {
                "status": "blocked",
                "executed": False,
                "verdict": str(verdict),
                "plan": plan,
                "note": "ActionPolicy did not allow restart_service — human approval required",
            }
    except Exception as exc:  # noqa: BLE001 — no gate available → fail closed
        return {"status": "blocked", "executed": False, "error": str(exc), "plan": plan}

    # 2. Restart (injected mechanic only — never a built-in shared-host shell-out).
    if not callable(restart_fn):
        return {
            "status": "blocked",
            "executed": False,
            "plan": plan,
            "note": "no restart_fn provided — the host restart mechanic is human-gated",
        }
    try:
        restart_fn(service)
    except Exception as exc:  # pragma: no cover - depends on injected fn
        return {
            "status": "failed",
            "executed": True,
            "stage": "restart",
            "error": str(exc),
        }

    # 3. Health gate + rollback advice.
    healthy = True
    if callable(health_fn):
        try:
            healthy = bool(health_fn(service))
        except Exception as exc:  # pragma: no cover
            healthy = False
            logger.warning("self_deploy health check raised: %s", exc)
    return {
        "status": "deployed" if healthy else "unhealthy",
        "executed": True,
        "healthy": healthy,
        "rollback": None if healthy else plan["rollback"],
    }
