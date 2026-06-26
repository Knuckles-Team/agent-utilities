#!/usr/bin/python
from __future__ import annotations

"""Reactive replica autoscaler — load signals → bounds → gated scale actions.

CONCEPT:OS-5.29 — Reactive replica autoscaling: a leader-only tick reads
pluggable load signals, applies registry-declared min/max replica bounds via
target tracking, and converges through the ActionPolicy gate, the actuator
seam and the deploy-watch safety net.

The last autonomy-gap item: the registry's replica counts were static — the
platform could *converge* on a declared number (OS-5.25) but never *choose*
one from load. This module is the smallest viable autoscaler, composed
entirely from the existing autonomy primitives:

* **bounds** — each service's optional registry/override ``scaling:`` block
  (:class:`~agent_utilities.orchestration.fleet_reconciler.ScalingSpec`):
  {min, max, signal, target, scale_up_step, scale_down_step, cooldown_s}.
  No block ⇒ never autoscaled.
* **signal** — a pluggable
  :class:`~agent_utilities.orchestration.scaling_signals.ScalingSignalProvider`
  (zero-infra local gauges by default, Prometheus via
  ``SCALING_PROMETHEUS_URL``, deployment-injected otherwise). ``None`` ⇒ NO
  action — never scale on missing data, mirroring the reconciler's
  unobserved⇒skip rule.
* **target tracking** — classic per-replica formula::

      desired = ceil(current * value_per_replica / target)

  where fleet-total signals (``queue_depth``, ``consumer_lag``) are first
  normalized to per-replica (``value / max(current, 1)``); the result is
  clamped to [min, max] and step-capped (at most ``scale_up_step`` added /
  ``scale_down_step`` removed per evaluation).
* **cooldown + flap guard** — no scale action (either direction) within
  ``cooldown_s`` of the service's last allowed/executed ``scale_service``
  entry in the durable ActionDecision/ActionExecution ledger — which also
  guarantees no opposite-direction flapping inside the window.
* **gate → actuate → watch** — proposals go through ActionPolicy
  (CONCEPT:OS-5.24; ``scale_service`` is approval_required under the shipped
  default policy) and the FleetActuator seam; successful scale-UPs schedule
  an OS-5.27 deploy watch (scale-downs too when the policy file sets
  ``options: {watch_scale_down: true}``).
* **audit** — at most one compact ``AutoscaleEvaluation`` node per tick (the
  per-action audit already lives in the ActionDecision/ActionExecution
  ledger), keeping KG noise low.

Wiring: leader-only ``fleet_autoscaler`` maintenance job in
``knowledge_graph/core/engine_tasks.py``, opt-in via ``FLEET_AUTOSCALER``
(default off; with the default dry-run actuator it records intent only).
"""

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
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
    get_fleet_observer,
)
from agent_utilities.orchestration.fleet_reconciler import (
    ScalingSpec,
    load_desired_state,
)
from agent_utilities.orchestration.scaling_signals import (
    SIGNAL_CONSUMER_LAG,
    SIGNAL_QUEUE_DEPTH,
    get_scaling_signal_provider,
)

logger = logging.getLogger(__name__)

# Signals whose provider value is a FLEET-TOTAL (normalized to per-replica by
# dividing by current replicas before target tracking); everything else (cpu,
# custom metrics) is already a per-replica average by convention.
AGGREGATE_SIGNALS = {SIGNAL_QUEUE_DEPTH, SIGNAL_CONSUMER_LAG}

# How many ledger rows the cooldown probe scans per service.
_LEDGER_SCAN_LIMIT = 200

_ALLOWING = {"allow", "allow_notify"}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compute_desired_replicas(current: int, value: float, spec: ScalingSpec) -> int:
    """Target-tracking replica count for one service, clamped and step-capped.

    ``desired = ceil(effective_current * per_replica_value / target)`` with
    ``effective_current = max(current, 1)`` (so a service scaled to zero can
    scale back up from an aggregate backlog), then clamped to
    [min_replicas, max_replicas] and capped to one step in either direction.
    """
    eff = max(int(current), 1)
    per_replica = value / eff if spec.signal in AGGREGATE_SIGNALS else float(value)
    desired = math.ceil(eff * per_replica / spec.target)
    desired = max(spec.min_replicas, min(spec.max_replicas, desired))
    if desired > current:
        desired = min(desired, current + spec.scale_up_step)
    elif desired < current:
        desired = max(desired, current - spec.scale_down_step)
    return desired


@dataclass
class ServiceEvaluation:
    """One service's autoscale verdict inside a tick (compact audit row)."""

    service: str
    outcome: str  # scaled | proposed | skipped
    reason: str = ""
    current: int | None = None
    desired: int | None = None
    value: float | None = None

    def compact(self) -> dict[str, Any]:
        row: dict[str, Any] = {"service": self.service, "outcome": self.outcome}
        if self.reason:
            row["reason"] = self.reason
        if self.current is not None:
            row["current"] = self.current
        if self.desired is not None:
            row["desired"] = self.desired
        if self.value is not None:
            row["value"] = round(self.value, 3)
        return row


class FleetAutoscaler:
    """One autoscale pass: signal → target tracking → policy gate → actuate."""

    def __init__(
        self,
        engine: Any,
        observer: Any = None,
        actuator: Any = None,
        policy: Any = None,
        signal_provider: Any = None,
        max_actions: int | None = None,
    ):
        self.engine = engine
        self.observer = observer or get_fleet_observer(engine)
        self.actuator = actuator or get_fleet_actuator()
        self.policy = policy or get_action_policy(engine)
        self.signals = signal_provider or get_scaling_signal_provider()
        if max_actions is None:
            try:
                from agent_utilities.core.config import config as _cfg

                max_actions = int(getattr(_cfg, "fleet_reconciler_max_actions", 5))
            except Exception:  # noqa: BLE001
                max_actions = 5
        self.max_actions = max(1, int(max_actions))
        # CONCEPT:OS-5.35 — cost-aware scale-up budget (opt-in; unset ⇒ no cap, so
        # the autoscaler behaves exactly as before). ``setting()`` keeps these
        # config.json-driven without a new typed field.
        from agent_utilities.core.config import setting

        self._scale_budget_usd_per_hour = setting(
            "FLEET_SCALE_BUDGET_USD_PER_HOUR", None, cast=float
        )
        self._replica_cost_usd_per_hour = setting(
            "FLEET_REPLICA_COST_USD_PER_HOUR", 0.05, cast=float
        )

    # ── cooldown (durable, shared across processes) ─────────────────

    def _last_scale_unix(self, service: str) -> float:
        """Latest allowed/executed ``scale_service`` timestamp for ``service``.

        Reads BOTH ledgers: ActionDecision (covers allow/allow_notify gates,
        including dry-run actuation) and ActionExecution (covers
        approval-granted actions drained later by the reconciler, whose
        decision row predates the actual scale). 0.0 = never scaled.
        """
        if self.engine is None:
            return 0.0
        latest = 0.0
        try:
            rows = self.engine.query_cypher(
                "MATCH (d:ActionDecision {kind: $kind, target: $target}) "
                "RETURN d.decision AS decision, d.params_json AS params_json, "
                f"d.decided_unix AS ts LIMIT {_LEDGER_SCAN_LIMIT}",
                {"kind": "scale_service", "target": service},
            )
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                if row.get("decision") in _ALLOWING:
                    latest = max(latest, float(row.get("ts") or 0))
        except Exception as e:  # noqa: BLE001 — cooldown probe is best-effort
            logger.debug("fleet_autoscaler: decision ledger scan failed: %s", e)
        try:
            rows = self.engine.query_cypher(
                "MATCH (x:ActionExecution {kind: $kind, target: $target}) "
                "RETURN x.ok AS ok, x.executed_unix AS ts "
                f"LIMIT {_LEDGER_SCAN_LIMIT}",
                {"kind": "scale_service", "target": service},
            )
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                if row.get("ok"):
                    latest = max(latest, float(row.get("ts") or 0))
        except Exception as e:  # noqa: BLE001
            logger.debug("fleet_autoscaler: execution ledger scan failed: %s", e)
        return latest

    # ── one service ─────────────────────────────────────────────────

    def _evaluate_service(
        self, name: str, spec: ScalingSpec, observation: Any
    ) -> ServiceEvaluation:
        if observation is None or observation.replicas is None:
            return ServiceEvaluation(
                name, "skipped", "unobserved (no replica evidence)"
            )
        if observation.status == STATUS_DOWN:
            # A down service is the reconciler's (restart) problem, not a
            # scaling problem — scaling a dead service masks the failure.
            return ServiceEvaluation(name, "skipped", "observed down — not scaling")
        current = int(observation.replicas)

        try:
            value = self.signals.signal_value(name, spec.signal)
        except Exception as e:  # noqa: BLE001 — protocol says never raise; belt+braces
            logger.debug("fleet_autoscaler: signal provider error for %s: %s", name, e)
            value = None
        if value is None:
            return ServiceEvaluation(
                name, "skipped", f"no data for signal {spec.signal!r}", current=current
            )

        desired = compute_desired_replicas(current, value, spec)
        # CONCEPT:OS-5.35 — cost-aware scale-up cap. Keep the target-tracking math
        # unchanged; only trim a scale-up that would breach the hourly budget, and
        # carry the cost estimate forward for the audit row + ActionRequest.
        cost_reason = ""
        cost_per_hour = desired * self._replica_cost_usd_per_hour
        if self._scale_budget_usd_per_hour is not None:
            from agent_utilities.orchestration.cost_governor import cost_aware_cap

            verdict = cost_aware_cap(
                desired,
                current,
                cost_per_replica_hour=self._replica_cost_usd_per_hour,
                budget_per_hour=self._scale_budget_usd_per_hour,
                load_value=float(value),
            )
            desired, cost_per_hour, cost_reason = (
                verdict.replicas,
                verdict.cost_per_hour,
                verdict.reason,
            )
        if desired == current:
            return ServiceEvaluation(
                name,
                "skipped",
                cost_reason or "at target",
                current=current,
                desired=desired,
                value=value,
            )

        last_scale = self._last_scale_unix(name)
        if last_scale and (time.time() - last_scale) < spec.cooldown_s:
            return ServiceEvaluation(
                name,
                "skipped",
                f"cooldown ({spec.cooldown_s}s since last scale; flap guard)",
                current=current,
                desired=desired,
                value=value,
            )

        direction = "up" if desired > current else "down"
        request = ActionRequest(
            kind="scale_service",
            target=name,
            params={
                "replicas": desired,
                "from_replicas": current,
                "direction": direction,
                "signal": spec.signal,
                "value": round(float(value), 3),
                "target": spec.target,
                # CONCEPT:OS-5.35 — cost lens on every scaling action.
                "est_cost_usd_per_hour": round(cost_per_hour, 4),
            },
            source="autoscaler",
            reason=(
                f"target tracking: {spec.signal}={value:.3g} vs target "
                f"{spec.target:g}/replica → {current}→{desired} "
                f"(bounds {spec.min_replicas}-{spec.max_replicas})"
            ),
        )
        decision = self.policy.decide(request)
        evaluation = ServiceEvaluation(
            name,
            "proposed",
            f"decision={decision.decision}",
            current=current,
            desired=desired,
            value=value,
        )
        if not decision.allowed:
            return evaluation
        execution = execute_action(self.engine, request, self.actuator)
        evaluation.outcome = "scaled" if execution.get("ok") else "proposed"
        evaluation.reason = (
            f"decision={decision.decision} ok={bool(execution.get('ok'))}"
            f"{' dry_run' if execution.get('dry_run') else ''}"
        )
        if execution.get("ok") and (
            direction == "up" or bool(self.policy.option("watch_scale_down", False))
        ):
            from agent_utilities.orchestration.deploy_watch import watch_deploy

            watch_deploy(self.engine, name, source="autoscaler")
        return evaluation

    # ── one tick ────────────────────────────────────────────────────

    def evaluate(self) -> dict[str, Any]:
        """One autoscale pass over every service with a scaling block."""
        desired_state = load_desired_state()
        observed: dict[str, Any] = {}
        try:
            observed = self.observer.observe() or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("fleet_autoscaler: observer failed: %s", e)

        evaluations: list[ServiceEvaluation] = []
        actions = 0
        for name, want in sorted(desired_state.items()):
            spec = want.scaling
            if spec is None or want.desired != "running":
                continue
            if actions >= self.max_actions:
                evaluations.append(
                    ServiceEvaluation(
                        name, "skipped", "per-tick action budget exhausted"
                    )
                )
                continue
            evaluation = self._evaluate_service(name, spec, observed.get(name))
            evaluations.append(evaluation)
            if evaluation.outcome in ("scaled", "proposed"):
                actions += 1

        report = {
            "evaluated": len(evaluations),
            "actions": actions,
            "scaled": sum(1 for e in evaluations if e.outcome == "scaled"),
            "evaluations": [e.compact() for e in evaluations],
            "actuator": getattr(self.actuator, "name", "?"),
            "signal_provider": getattr(self.signals, "name", "?"),
        }
        self._record(report)
        return report

    def _record(self, report: dict[str, Any]) -> None:
        """At most ONE compact AutoscaleEvaluation node per tick (no per-service
        nodes — the action-level audit already lives in the policy ledger);
        ticks that evaluated nothing write nothing."""
        if self.engine is None or not report["evaluations"]:
            return
        try:
            self.engine.add_node(
                f"autoscale_evaluation:{uuid.uuid4().hex[:12]}",
                "AutoscaleEvaluation",
                properties={
                    "evaluated": report["evaluated"],
                    "actions": report["actions"],
                    "scaled": report["scaled"],
                    "details_json": json.dumps(report["evaluations"], default=str)[
                        :4000
                    ],
                    "actuator": report["actuator"],
                    "signal_provider": report["signal_provider"],
                    "created_at": _now_iso(),
                    "created_unix": time.time(),
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("fleet_autoscaler: evaluation write failed: %s", e)


def autoscale_fleet(engine: Any) -> dict[str, Any]:
    """The leader-only maintenance-tick entry point (see ``engine_tasks``)."""
    return FleetAutoscaler(engine).evaluate()


#: The control-plane node label whose committed changes move queue depth — a new
#: :Task enqueued / claimed / completed changes the backlog the autoscaler tracks.
TASK_LABEL = "Task"


def fleet_autoscale_subscription(engine: Any) -> Any:
    """Reactive change-feed subscription over control-plane ``:Task`` mutations.

    CONCEPT:KG-2.253 — the poll→push seam for autoscaling: instead of waiting for
    the next leader poll interval, the daemon polls this subscription and, when the
    engine pushes a ``:Task`` change (the queue-depth signal moved), fires an
    autoscale evaluation immediately — so scaling reacts to the change-EVENT, not
    a fixed interval. The slow periodic ``_tick_fleet_autoscaler`` stays as the
    safety-net reconcile.

    Subscribes on the engine's **control graph** (``__control__`` — where ``:Task``
    lives, CONCEPT:KG-2.148), resolved via the engine's control backend. The
    handler bumps ``sub.pending_state["pending"]``; the caller reads it to decide
    whether to evaluate now. Returns a
    :class:`~agent_utilities.graph.reactive.EngineSubscription` whose ``.available``
    is ``False`` (a permanent no-op) when no engine streaming surface exists — so
    the periodic tick remains the correctness guarantee.
    """
    from agent_utilities.graph.reactive import subscribe

    # The control plane (``__control__``) is where :Task is written; fall back to
    # the engine's content compute, then to the passed object itself (e.g. a bare
    # GraphComputeEngine), when no isolated control backend exists.
    source = (
        getattr(engine, "_control", None)
        or getattr(engine, "graph_compute", None)
        or engine
    )

    state = {"pending": 0}

    def _on_task_change(_event: dict[str, Any]) -> None:
        state["pending"] += 1

    sub = subscribe(source, TASK_LABEL, _on_task_change)
    sub.pending_state = state  # type: ignore[attr-defined]
    return sub
