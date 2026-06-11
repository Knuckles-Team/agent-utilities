#!/usr/bin/python
from __future__ import annotations

"""Desired-state fleet reconciler.

CONCEPT:OS-5.25 — Desired-state fleet reconciler: a leader-only daemon tick
diffs the declared fleet (registry + optional override) against the observed
fleet and converges through the ActionPolicy gate and the actuator seam.

Until now ``deploy/mcp-fleet.registry.yml`` was a deploy-time input only —
nothing at runtime ever compared "what should be running" against "what is".
This module is that runtime contract:

* **desired state** — the registry's ``services:`` list (every entry is
  expected ``running`` with 1 replica unless said otherwise), layered with an
  optional override file (``FLEET_DESIRED_STATE_PATH``) carrying per-service
  ``replicas`` / ``desired: running|stopped`` / ``version``.
* **observed state** — a pluggable
  :class:`~agent_utilities.orchestration.fleet_observation.FleetObserver`
  (default: KG fleet events + local docker when present; Portainer observers
  are deployment-wired via ``set_fleet_observer``).
* **divergence → action** — service down ⇒ ``restart_service``; replica
  mismatch ⇒ ``scale_service``; running-but-undesired ⇒ ``stop_service``.
  Services with NO observation are skipped (never act on zero evidence).
* **gate → actuate** — every proposal passes the ActionPolicy decision point
  (CONCEPT:OS-5.24); allowed actions run through the
  :class:`~agent_utilities.orchestration.fleet_actuation.FleetActuator` (the
  default dry-run actuator records intent without mutating), and restarts
  schedule an OS-5.27 health watch. Queue-approval decisions land in the
  fleet approvals flow; this tick also DRAINS granted approvals, closing the
  human-in-the-loop circle.
* **storm guard** — at most ``FLEET_RECONCILER_MAX_ACTIONS`` proposals are
  processed per tick; the rest defer to the next tick.

Wiring: registered as the leader-only ``fleet_reconciler`` maintenance job in
``knowledge_graph/core/engine_tasks.py``, opt-in via ``FLEET_RECONCILER``
(default off until a deployment wires real actuators).
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utilities.orchestration.action_policy import (
    ActionRequest,
    get_action_policy,
)
from agent_utilities.orchestration.fleet_actuation import (
    execute_action,
    get_fleet_actuator,
)
from agent_utilities.orchestration.fleet_observation import (
    STATUS_DOWN,
    STATUS_UP,
    get_fleet_observer,
)

logger = logging.getLogger(__name__)

_APPROVAL_DRAIN_LIMIT = 20

# Action kinds whose execution warrants a follow-up health watch (OS-5.27).
_WATCHED_KINDS = {
    "restart_service",
    "deploy_service",
    "redeploy_stack",
    "rollback_service",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class DesiredService:
    """One service's desired state after registry + override layering."""

    name: str
    desired: str = "running"  # running | stopped
    replicas: int = 1
    version: str = ""
    profiles: list[str] = field(default_factory=list)


def resolve_registry_path(explicit: str | None = None) -> Path | None:
    """Resolve the fleet registry YAML: explicit flag → repo shipped file."""
    if explicit:
        return Path(explicit)
    shipped = Path(__file__).resolve().parents[2] / "deploy" / "mcp-fleet.registry.yml"
    return shipped if shipped.is_file() else None


def load_desired_state(
    registry_path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> dict[str, DesiredService]:
    """Parse registry + optional override into ``{name: DesiredService}``."""
    import yaml

    if registry_path is None or override_path is None:
        try:
            from agent_utilities.core.config import config as _cfg

            registry_path = registry_path or (
                getattr(_cfg, "fleet_registry_path", "") or None
            )
            override_path = override_path or (
                getattr(_cfg, "fleet_desired_state_path", "") or None
            )
        except Exception:  # noqa: BLE001
            pass

    desired: dict[str, DesiredService] = {}
    path = resolve_registry_path(str(registry_path) if registry_path else None)
    if path is not None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for raw in data.get("services") or []:
                if not isinstance(raw, dict) or not raw.get("name"):
                    continue
                name = str(raw["name"])
                desired[name] = DesiredService(
                    name=name,
                    desired=str(raw.get("desired") or "running"),
                    replicas=int(raw.get("replicas") or 1),
                    version=str(raw.get("version") or ""),
                    profiles=[str(p) for p in raw.get("profiles") or []],
                )
        except Exception as e:  # noqa: BLE001 — a broken registry reconciles nothing
            logger.warning("fleet_reconciler: registry parse failed (%s): %s", path, e)

    if override_path:
        try:
            data = yaml.safe_load(Path(override_path).read_text(encoding="utf-8")) or {}
            for raw in data.get("services") or []:
                if not isinstance(raw, dict) or not raw.get("name"):
                    continue
                name = str(raw["name"])
                entry = desired.setdefault(name, DesiredService(name=name))
                if raw.get("desired"):
                    entry.desired = str(raw["desired"])
                if raw.get("replicas") is not None:
                    entry.replicas = int(raw["replicas"])
                if raw.get("version"):
                    entry.version = str(raw["version"])
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "fleet_reconciler: override parse failed (%s): %s", override_path, e
            )
    return desired


class FleetReconciler:
    """One reconcile pass: diff → policy gate → actuate/queue → record."""

    def __init__(
        self,
        engine: Any,
        observer: Any = None,
        actuator: Any = None,
        policy: Any = None,
        max_actions: int | None = None,
    ):
        self.engine = engine
        self.observer = observer or get_fleet_observer(engine)
        self.actuator = actuator or get_fleet_actuator()
        self.policy = policy or get_action_policy(engine)
        if max_actions is None:
            try:
                from agent_utilities.core.config import config as _cfg

                max_actions = int(getattr(_cfg, "fleet_reconciler_max_actions", 5))
            except Exception:  # noqa: BLE001
                max_actions = 5
        self.max_actions = max(1, int(max_actions))

    # ── divergence detection ────────────────────────────────────────

    def diff(self) -> list[ActionRequest]:
        """Desired vs observed → ordered convergence proposals (conservative).

        Only positive evidence diverges: a service the observer never saw is
        skipped, not restarted.
        """
        desired = load_desired_state()
        observed: dict[str, Any] = {}
        try:
            observed = self.observer.observe() or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("fleet_reconciler: observer failed: %s", e)

        proposals: list[ActionRequest] = []
        for name, want in sorted(desired.items()):
            obs = observed.get(name)
            if obs is None:
                continue  # no evidence — never act blind
            if want.desired == "stopped":
                if obs.status == STATUS_UP:
                    proposals.append(
                        ActionRequest(
                            kind="stop_service",
                            target=name,
                            source="reconciler",
                            reason="desired stopped but observed up",
                        )
                    )
                continue
            if obs.status == STATUS_DOWN:
                proposals.append(
                    ActionRequest(
                        kind="restart_service",
                        target=name,
                        params={"version": want.version} if want.version else {},
                        source="reconciler",
                        reason=f"observed down ({obs.detail})",
                    )
                )
            elif (
                obs.status == STATUS_UP
                and obs.replicas is not None
                and obs.replicas != want.replicas
            ):
                proposals.append(
                    ActionRequest(
                        kind="scale_service",
                        target=name,
                        params={"replicas": want.replicas},
                        source="reconciler",
                        reason=f"replicas {obs.replicas} != desired {want.replicas}",
                    )
                )
        return proposals

    # ── convergence ─────────────────────────────────────────────────

    def _converge_one(self, request: ActionRequest) -> dict[str, Any]:
        decision = self.policy.decide(request)
        entry: dict[str, Any] = {
            "kind": request.kind,
            "target": request.target,
            "reason": request.reason,
            "decision": decision.decision,
            "approval_id": decision.approval_id,
        }
        if decision.allowed:
            entry["execution"] = execute_action(self.engine, request, self.actuator)
            if request.kind in _WATCHED_KINDS and entry["execution"].get("ok"):
                from agent_utilities.orchestration.deploy_watch import watch_deploy

                entry["watch_job"] = watch_deploy(
                    self.engine,
                    request.target,
                    version=str(request.params.get("version") or ""),
                    source="reconciler",
                )
        return entry

    def _drain_approved(self, budget: int) -> list[dict[str, Any]]:
        """Execute fleet actions a human approved via /api/fleet/approvals/grant."""
        if budget <= 0 or self.engine is None:
            return []
        try:
            rows = self.engine.query_cypher(
                "MATCH (a:ActionApproval {status: 'approved'}) "
                f"RETURN a LIMIT {_APPROVAL_DRAIN_LIMIT}"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("fleet_reconciler: approval drain scan failed: %s", e)
            return []
        drained: list[dict[str, Any]] = []
        for row in rows or []:
            if budget <= 0:
                break
            props = row.get("a") if isinstance(row, dict) else None
            if not isinstance(props, dict) or not props.get("id"):
                continue
            if str(props.get("kind") or "") == "merge_promotion":
                # Code-evolution publications are NOT fleet actuations: a
                # granted merge_promotion approval is consumed by the
                # evolution→branch bridge's ``publish_proposal`` action
                # (CONCEPT:AHE-3.21), never by the fleet actuator — which
                # would dry-run/fail it and silently eat the grant.
                continue
            try:
                params = json.loads(props.get("params_json") or "{}")
            except (TypeError, ValueError):
                params = {}
            request = ActionRequest(
                kind=str(props.get("kind") or ""),
                target=str(props.get("target") or ""),
                params=params if isinstance(params, dict) else {},
                source=f"approved:{props.get('source') or 'unknown'}",
                reason=str(props.get("reason") or ""),
            )
            execution = execute_action(self.engine, request, self.actuator)
            budget -= 1
            new_status = "executed" if execution.get("ok") else "failed"
            try:
                self.engine.backend.execute(
                    "MATCH (a:ActionApproval {id: $id}) "
                    "SET a.status = $status, a.executed_at = $ts",
                    {"id": props["id"], "status": new_status, "ts": _now_iso()},
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("fleet_reconciler: approval stamp failed: %s", e)
            if request.kind in _WATCHED_KINDS and execution.get("ok"):
                from agent_utilities.orchestration.deploy_watch import watch_deploy

                watch_deploy(
                    self.engine,
                    request.target,
                    version=str(request.params.get("version") or ""),
                    source="approval",
                )
            drained.append(
                {
                    "approval_id": props["id"],
                    "kind": request.kind,
                    "target": request.target,
                    "status": new_status,
                    "execution": execution,
                }
            )
        return drained

    def reconcile(self) -> dict[str, Any]:
        """One full pass; returns (and durably records) the convergence report."""
        proposals = self.diff()
        processed = proposals[: self.max_actions]
        deferred = proposals[self.max_actions :]

        actions = [self._converge_one(p) for p in processed]
        # Human-granted approvals get their own budget: a backlog of new
        # divergences must not starve actions an operator already sanctioned.
        approved = self._drain_approved(self.max_actions)

        report = {
            "divergences": len(proposals),
            "processed": len(actions),
            "deferred": [p.summary() for p in deferred],
            "actions": actions,
            "approved_drained": approved,
            "actuator": getattr(self.actuator, "name", "?"),
        }
        self._record(report)
        return report

    def _record(self, report: dict[str, Any]) -> None:
        if self.engine is None:
            return
        try:
            self.engine.add_node(
                f"reconcile_report:{uuid.uuid4().hex[:12]}",
                "ReconcileReport",
                properties={
                    "divergences": report["divergences"],
                    "processed": report["processed"],
                    "deferred": len(report["deferred"]),
                    "approved_drained": len(report["approved_drained"]),
                    "details_json": json.dumps(report, default=str)[:4000],
                    "created_at": _now_iso(),
                    "created_unix": time.time(),
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("fleet_reconciler: report write failed: %s", e)


def reconcile_fleet(engine: Any) -> dict[str, Any]:
    """The leader-only maintenance-tick entry point (see ``engine_tasks``)."""
    return FleetReconciler(engine).reconcile()
