#!/usr/bin/python
from __future__ import annotations

"""Desired-state fleet reconciler.

CONCEPT:AU-OS.config.desired-state-fleet-reconciler — Desired-state fleet reconciler: a leader-only daemon tick
diffs the declared fleet (registry + optional override) against the observed
fleet and converges through the ActionPolicy gate and the actuator seam.

Until now ``deploy/mcp-fleet.registry.yml`` was a deploy-time input only —
nothing at runtime ever compared "what should be running" against "what is".
This module is that runtime contract:

* **desired state** — the registry's ``services:`` list (every entry is
  expected ``running`` with 1 replica unless said otherwise), layered with an
  optional override file (``FLEET_DESIRED_STATE_PATH``) carrying per-service
  ``replicas`` / ``desired: running|stopped`` / ``version`` / ``scaling``
  (reactive-autoscaling bounds, CONCEPT:AU-OS.scaling.fleet-reconciler — consumed by the
  ``fleet_autoscaler`` tick, not by this reconciler).
* **observed state** — a pluggable
  :class:`~agent_utilities.orchestration.fleet_observation.FleetObserver`
  (default: KG fleet events + local docker when present; Portainer observers
  are deployment-wired via ``set_fleet_observer``).
* **divergence → action** — service down ⇒ ``restart_service``; replica
  mismatch ⇒ ``scale_service``; running-but-undesired ⇒ ``stop_service``.
  Services with NO observation are skipped (never act on zero evidence).
* **gate → actuate** — every proposal passes the ActionPolicy decision point
  (CONCEPT:AU-OS.deployment.fleet-lifecycle-control); allowed actions run through the
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
class ScalingSpec:
    """Registry-declared reactive-autoscaling bounds for one service.

    CONCEPT:AU-OS.scaling.fleet-reconciler — consumed by the leader-only ``fleet_autoscaler`` tick
    (``orchestration/fleet_autoscaler.py``). ``max``, ``signal`` and ``target``
    are deliberately explicit (no implicit ceiling, no implicit metric): a
    service only autoscales when its owner declared how far and on what.
    """

    min_replicas: int = 1
    max_replicas: int = 1
    signal: str = ""  # queue_depth | consumer_lag | cpu | custom metric name
    target: float = 0.0  # per-replica target value for the signal
    scale_up_step: int = 1  # max replicas added per evaluation
    scale_down_step: int = 1  # max replicas removed per evaluation
    cooldown_s: float = 300.0  # min seconds between scale actions


def parse_scaling_spec(raw: Any, service: str) -> ScalingSpec | None:
    """Validate one registry ``scaling:`` block into a :class:`ScalingSpec`.

    Required: ``max`` (ceiling), ``signal`` and ``target`` (>0). Defaults:
    ``min=1``, steps ``1``, ``cooldown_s=300``. Invariant ``max >= min >= 0``.
    Any invalid block is dropped with a warning — the service then keeps the
    static replica reconcile (OS-5.25) and is simply never autoscaled; a typo
    must never produce surprise scaling.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning("scaling spec for %s is not a mapping — ignored", service)
        return None
    try:
        spec = ScalingSpec(
            min_replicas=int(raw.get("min", 1)),
            max_replicas=int(raw["max"]),  # required: no implicit ceiling
            signal=str(raw.get("signal") or ""),
            target=float(raw.get("target") or 0.0),
            scale_up_step=int(raw.get("scale_up_step", 1)),
            scale_down_step=int(raw.get("scale_down_step", 1)),
            cooldown_s=float(raw.get("cooldown_s", 300.0)),
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("scaling spec for %s is invalid (%s) — ignored", service, e)
        return None
    problems: list[str] = []
    if spec.min_replicas < 0:
        problems.append(f"min={spec.min_replicas} < 0")
    if spec.max_replicas < spec.min_replicas:
        problems.append(f"max={spec.max_replicas} < min={spec.min_replicas}")
    if not spec.signal:
        problems.append("signal missing")
    if spec.target <= 0:
        problems.append(f"target={spec.target} must be > 0")
    if spec.scale_up_step < 1 or spec.scale_down_step < 1:
        problems.append("steps must be >= 1")
    if spec.cooldown_s < 0:
        problems.append(f"cooldown_s={spec.cooldown_s} < 0")
    if problems:
        logger.warning(
            "scaling spec for %s rejected: %s — ignored", service, "; ".join(problems)
        )
        return None
    return spec


@dataclass
class DesiredService:
    """One service's desired state after registry + override layering."""

    name: str
    desired: str = "running"  # running | stopped
    replicas: int = 1
    version: str = ""
    profiles: list[str] = field(default_factory=list)
    scaling: ScalingSpec | None = (
        None  # CONCEPT:AU-OS.scaling.fleet-reconciler (None = never autoscale)
    )


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
                    scaling=parse_scaling_spec(raw.get("scaling"), name),
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
                if "scaling" in raw:
                    # The registry file is machine-generated, so the override
                    # file is where a deployment normally declares scaling
                    # bounds. ``scaling: null`` explicitly disables.
                    entry.scaling = parse_scaling_spec(raw.get("scaling"), name)
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
        # C3/Phase 3b (D13): CDC-fired :AgentTask dependency firing when the
        # engine's change-feed is reachable, falling back to the poll sweep
        # otherwise — one watcher per reconciler instance so its CDC cursor
        # persists across ticks (see ``AgentTaskDepWatcher``).
        self._agent_task_watcher = AgentTaskDepWatcher(engine)

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
                # (CONCEPT:AU-AHE.harness.evolution-branch-bridge), never by the fleet actuator — which
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
        # C3/Phase 3b (D13): CDC-fired :AgentTask dependency firing when the
        # engine's change-feed is reachable; poll sweep as the fallback (see
        # ``AgentTaskDepWatcher``/``fire_ready_agent_tasks`` docstrings).
        fired_agent_tasks = self._agent_task_watcher.fire()

        report = {
            "divergences": len(proposals),
            "processed": len(actions),
            "deferred": [p.summary() for p in deferred],
            "actions": actions,
            "approved_drained": approved,
            "fired_agent_tasks": fired_agent_tasks,
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


# ── C3/Phase 3a→3b: :AgentTask dependency firing — CDC-first, poll fallback ──
#
# CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects
#
# Phase 3a shipped a POLLING sweep only: every tick blindly re-scanned every
# 'blocked' ``:AgentTask`` node, whether or not anything had actually completed
# since the last tick. Phase 3b (D13) closes the gap with
# :class:`AgentTaskDepWatcher`: it rides the SAME engine change-feed primitive
# every other reactive consumer in this codebase uses
# (:class:`agent_utilities.graph.reactive.engine_subscription.EngineSubscription`,
# label="AgentTask") so a tick with NO completed dependency since the last one
# does ZERO Cypher work instead of a full sweep. ``fire_ready_agent_tasks``
# itself (the sweep body) is UNCHANGED and kept as the fallback — a non-engine
# backend, or an engine build without the streaming feature, degrades the
# watcher straight back to Phase 3a's always-sweep behavior. Wired into both
# the leader-only ``FleetReconciler.reconcile()`` tick (fleet-wide) and the
# per-scheduler ``RecoveryDaemon.stabilize()`` tick (local) — same watcher
# class, two callers, no duplicated dependency logic.

_AGENT_TASK_DEP_SWEEP_LIMIT = 200


def _agent_task_dependencies_satisfied(
    engine: Any, depends_on_task_ids: list[str]
) -> bool:
    """True iff every dependency id resolves to an ``:AgentTask`` with status 'completed'.

    Conservative like the reconciler's ``diff()`` above: a missing/unknown
    dependency counts as NOT satisfied (never fire on absent evidence).
    """
    if not depends_on_task_ids:
        return True
    rows = engine.query_cypher(
        "MATCH (t:AgentTask) WHERE t.id IN $ids RETURN t.id AS id, t.status AS status",
        {"ids": list(depends_on_task_ids)},
    )
    statuses = {r.get("id"): r.get("status") for r in (rows or [])}
    return all(statuses.get(tid) == "completed" for tid in depends_on_task_ids)


def fire_ready_agent_tasks(
    engine: Any, limit: int = _AGENT_TASK_DEP_SWEEP_LIMIT
) -> list[str]:
    """Sweep 'blocked' ``:AgentTask`` nodes and fire the ones whose deps completed.

    Returns the ids flipped to 'ready' this sweep (empty if the engine is
    unavailable or the query fails — never load-bearing for the caller's
    tick). See the module-level note above for the poll-vs-CDC rationale.
    """
    if engine is None:
        return []
    try:
        rows = (
            engine.query_cypher(
                "MATCH (t:AgentTask {status: 'blocked'}) RETURN t.id AS id, "
                "t.depends_on_task_ids AS depends_on_task_ids "
                f"LIMIT {int(limit)}"
            )
            or []
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("fleet_reconciler: agent-task dependency sweep failed: %s", e)
        return []

    fired: list[str] = []
    for row in rows:
        task_id = row.get("id")
        if not task_id:
            continue
        deps = list(row.get("depends_on_task_ids") or [])
        if not _agent_task_dependencies_satisfied(engine, deps):
            continue
        try:
            engine.add_node(task_id, "AgentTask", properties={"status": "ready"})
            fired.append(task_id)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "fleet_reconciler: failed to fire agent task %s: %s", task_id, e
            )
    return fired


class AgentTaskDepWatcher:
    """CDC-first ``:AgentTask`` dependency firing, poll sweep as the fallback (D13).

    CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects (C3/Phase 3b)

    Wraps one :class:`~agent_utilities.graph.reactive.engine_subscription.
    EngineSubscription` (``label="AgentTask"``) per instance so its CDC cursor
    persists across ticks — construct ONCE per reconciler/daemon (not per
    tick) and call :meth:`fire` on each tick.

    * **engine change-feed reachable** (``subscription.available``) — a tick
      polls the subscription (``block_ms=0``, non-blocking); when NO
      ``:AgentTask`` changed since the last tick this is a single cheap
      long-poll round-trip and :func:`fire_ready_agent_tasks` (the Cypher
      sweep) is skipped entirely. When at least one ``:AgentTask`` changed,
      the sweep runs once (still the same conservative
      ``_agent_task_dependencies_satisfied`` check — the CDC signal only
      gates WHETHER to look, never what "satisfied" means) to fire every task
      now eligible, since one completion can unblock several depends_on
      chains at once.
    * **engine change-feed unavailable** (non-engine backend / an engine build
      without ``streaming``) — ``subscription.available`` is ``False`` and
      this degrades straight back to Phase 3a: an unconditional sweep every
      tick, byte-identical to calling :func:`fire_ready_agent_tasks` directly.

    Never raises: subscription construction/polling failures degrade to the
    poll fallback, mirroring every other engine-surface consumer here.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._subscription = self._build_subscription(engine)
        self._dirty = False

    @staticmethod
    def _build_subscription(engine: Any) -> Any:
        try:
            from agent_utilities.graph.reactive.engine_subscription import subscribe
        except Exception as e:  # noqa: BLE001 — subsystem unimportable ⇒ poll fallback
            logger.debug("fleet_reconciler: engine_subscription unavailable: %s", e)
            return None
        try:
            return subscribe(engine, "AgentTask", handler=None)
        except Exception as e:  # noqa: BLE001
            logger.debug("fleet_reconciler: AgentTask subscription failed: %s", e)
            return None

    def _on_change(self, event: dict[str, Any]) -> None:
        self._dirty = True

    def fire(self, limit: int = _AGENT_TASK_DEP_SWEEP_LIMIT) -> list[str]:
        """One tick: CDC-gated sweep when the engine change-feed is reachable, else always-sweep."""
        sub = self._subscription
        if sub is None or not getattr(sub, "available", False):
            return fire_ready_agent_tasks(self.engine, limit=limit)

        # Route delivered events through this instance's dirty flag (bound
        # here rather than at construction so a monkeypatched/rebuilt
        # subscription in tests still wires correctly).
        sub.handler = self._on_change
        try:
            sub.poll(block_ms=0)
        except Exception as e:  # noqa: BLE001 — a feed hiccup ⇒ fall back to the sweep
            logger.debug("fleet_reconciler: AgentTask CDC poll failed: %s", e)
            return fire_ready_agent_tasks(self.engine, limit=limit)

        if not self._dirty:
            return []  # nothing changed since the last tick — zero Cypher work
        self._dirty = False
        return fire_ready_agent_tasks(self.engine, limit=limit)
