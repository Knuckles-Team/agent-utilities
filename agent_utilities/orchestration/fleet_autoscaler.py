#!/usr/bin/python
from __future__ import annotations

"""Reactive replica autoscaler — load signals → bounds → gated scale actions.

CONCEPT:AU-OS.scaling.reactive-replica-autoscaling — Reactive replica autoscaling: a leader-only tick reads
pluggable load signals, applies registry-declared min/max replica bounds via
target tracking, and converges through the ActionPolicy gate, the actuator
seam and the deploy-watch safety net.

The last autonomy-gap item: the registry's replica counts were static — the
platform could *converge* on a declared number (OS-5.25) but never *choose*
one from load. This module is the smallest viable autoscaler, composed
entirely from the existing autonomy primitives:

* **bounds** — each service's optional registry/override ``scaling:`` block
  (:class:`~agent_utilities.orchestration.fleet_reconciler.ScalingSpec`):
  {min, max, signal, target, scale_up_step, scale_down_step, cooldown_s,
  deadband, scale_up_stabilization_samples, scale_down_stabilization_samples}.
  No block ⇒ never autoscaled.
* **signal** — a pluggable
  :class:`~agent_utilities.orchestration.scaling_signals.ScalingSignalProvider`
  (zero-infra local gauges by default, Prometheus via
  ``SCALING_PROMETHEUS_URL``, deployment-injected otherwise). Each tick reads
  all requested samples through one bounded bulk call. ``None`` ⇒ NO action —
  never scale on missing data, mirroring the reconciler's unobserved⇒skip rule.
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
* **deadband + stabilization** — values inside the declared relative deadband
  hold steady; fresh consecutive samples stabilize direction, with one sample
  for scale-up by default and three for scale-down. No-data and direction
  changes reset the streak.
* **gate → actuate → watch** — proposals go through ActionPolicy
  (CONCEPT:AU-OS.deployment.fleet-lifecycle-control; ``scale_service`` is approval_required under the shipped
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
from collections.abc import Callable
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
from agent_utilities.orchestration.fleet_health import (
    FleetHealthEvidence,
    FleetHealthSnapshot,
    collect_fleet_health,
    unavailable_fleet_health,
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
    ScalingSignalSample,
    SignalAggregation,
    SignalDefinition,
    get_scaling_signal_provider,
    read_scaling_signal_samples,
    validate_scaling_signal_sample,
)

logger = logging.getLogger(__name__)

# How many ledger rows the cooldown probe scans per service.
_LEDGER_SCAN_LIMIT = 200

_ALLOWING = {"allow", "allow_notify"}
_SAMPLE_UNSET = object()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compute_desired_replicas(
    current: int,
    value: float,
    spec: ScalingSpec,
    *,
    aggregation: SignalAggregation,
) -> int:
    """Target-tracking replica count for one service, clamped and step-capped.

    ``desired = ceil(effective_current * per_replica_value / target)`` with
    ``effective_current = max(current, 1)`` (so a service scaled to zero can
    scale back up from an aggregate backlog), then clamped to
    [min_replicas, max_replicas] and capped to one step in either direction.
    ``aggregation`` is required metadata from the validated signal definition;
    target tracking never infers it from the signal name.
    """
    if aggregation not in ("fleet_total", "per_replica"):
        raise ValueError("signal aggregation must be fleet_total or per_replica")
    eff = max(int(current), 1)
    per_replica = value / eff if aggregation == "fleet_total" else float(value)
    desired = math.ceil(eff * per_replica / spec.target)
    desired = max(spec.min_replicas, min(spec.max_replicas, desired))
    if desired > current:
        desired = min(desired, current + spec.scale_up_step)
    elif desired < current:
        desired = max(desired, current - spec.scale_down_step)
    return desired


def _per_replica_value(
    current: int, value: float, aggregation: SignalAggregation
) -> float:
    effective_current = max(int(current), 1)
    return value / effective_current if aggregation == "fleet_total" else float(value)


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
        health_provider: Callable[[], FleetHealthEvidence | FleetHealthSnapshot]
        | None = None,
    ):
        self.engine = engine
        self.observer = observer or get_fleet_observer(engine)
        self.actuator = actuator or get_fleet_actuator()
        self.policy = policy or get_action_policy(engine)
        self.signals = signal_provider or get_scaling_signal_provider()
        self.health_provider = health_provider or (
            lambda: collect_fleet_health().evidence
        )
        self._last_signal_observed: dict[tuple[str, str], float] = {}
        self._stabilization: dict[tuple[str, str], tuple[str, int]] = {}
        if max_actions is None:
            try:
                from agent_utilities.core.config import config as _cfg

                max_actions = int(getattr(_cfg, "fleet_reconciler_max_actions", 5))
            except Exception:  # noqa: BLE001
                max_actions = 5
        self.max_actions = max(1, int(max_actions))
        # CONCEPT:AU-OS.scaling.cost-aware-autoscaling — cost-aware scale-up budget (opt-in; unset ⇒ no cap, so
        # the autoscaler behaves exactly as before). ``setting()`` keeps these
        # config.json-driven without a new typed field.
        from agent_utilities.core.config import setting

        self._scale_budget_usd_per_hour = setting(
            "FLEET_SCALE_BUDGET_USD_PER_HOUR", None, cast=float
        )
        self._replica_cost_usd_per_hour = setting(
            "FLEET_REPLICA_COST_USD_PER_HOUR", 0.05, cast=float
        )

    def _fleet_health(self) -> FleetHealthEvidence:
        """Read the shared supervisory contract; provider failure is unavailable."""

        try:
            result = self.health_provider()
            if isinstance(result, FleetHealthSnapshot):
                return result.evidence
            if isinstance(result, FleetHealthEvidence):
                return result
            raise TypeError("health provider returned an untyped result")
        except Exception as exc:  # noqa: BLE001 - autonomy must fail closed
            logger.warning(
                "fleet_autoscaler: supervisory evidence unavailable (%s)",
                type(exc).__name__,
            )
            return unavailable_fleet_health("autoscaler.health")

    def _reset_stabilization(self, key: tuple[str, str]) -> None:
        self._stabilization.pop(key, None)

    def _stabilization_ready(
        self,
        key: tuple[str, str],
        direction: str,
        required: int,
    ) -> tuple[bool, int]:
        previous = self._stabilization.get(key)
        count = (
            previous[1] + 1 if previous is not None and previous[0] == direction else 1
        )
        self._stabilization[key] = (direction, count)
        return count >= required, count

    def _signal_definition(
        self, name: str, spec: ScalingSpec
    ) -> SignalDefinition | None:
        definition_fn = getattr(self.signals, "signal_definition", None)
        if not callable(definition_fn):
            return None
        try:
            definition = definition_fn(spec.signal, name)
        except Exception as exc:  # noqa: BLE001 — malformed provider is no data
            logger.debug(
                "fleet_autoscaler: signal definition failed for %s: %s",
                name,
                type(exc).__name__,
            )
            return None
        return definition if isinstance(definition, SignalDefinition) else None

    # ── cooldown (durable, shared across processes) ─────────────────

    def _last_scale_unix(self, service: str) -> float | None:
        """Latest allowed/executed ``scale_service`` timestamp for ``service``.

        Reads BOTH ledgers: ActionDecision (covers allow/allow_notify gates,
        including dry-run actuation) and ActionExecution (covers
        approval-granted actions drained later by the reconciler, whose
        decision row predates the actual scale). 0.0 = never scaled;
        ``None`` = cooldown state UNKNOWN (a ledger scan failed — D-DST-6:
        the caller must fail closed on ``None``, never treat it as "clear").
        """
        if self.engine is None:
            return 0.0
        # D-DST-6: track whether EACH ledger scan actually completed, not just
        # accumulate `latest`. This cooldown/flap-guard is a safety check — the
        # same "guardrail crash reads as clean pass" shape as ActionPolicy's
        # rate/blast-radius reads. Silently treating a failed scan as "no prior
        # scale found" (the old behavior) reports "never scaled" during exactly
        # the KG-outage window when the underlying actuator (docker/k8s) is
        # still fully capable of firing repeated, unthrottled scale actions.
        # `_evaluate_service` below now fails CLOSED (skips) when either scan
        # didn't complete, instead of assuming the cooldown is clear.
        latest = 0.0
        decision_ok = execution_ok = False
        try:
            rows = self.engine.query_cypher(
                "MATCH (d:ActionDecision {kind: $kind, target: $target}) "
                "RETURN d.id AS id, d.decision AS decision, d.params_json AS params_json, "
                f"d.decided_unix AS ts LIMIT {_LEDGER_SCAN_LIMIT}",
                {"kind": "scale_service", "target": service},
            )
            decision_ok = True
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                if row.get("decision") in _ALLOWING:
                    latest = max(latest, float(row.get("ts") or 0))
        except Exception as e:  # noqa: BLE001 — decision-ledger read failed; caller fails closed below if the execution ledger doesn't cover it either
            logger.warning("fleet_autoscaler: decision ledger scan failed: %s", e)
        try:
            rows = self.engine.query_cypher(
                "MATCH (x:ActionExecution {kind: $kind, target: $target}) "
                "RETURN x.id AS id, x.ok AS ok, x.executed_unix AS ts "
                f"LIMIT {_LEDGER_SCAN_LIMIT}",
                {"kind": "scale_service", "target": service},
            )
            execution_ok = True
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                if row.get("ok"):
                    latest = max(latest, float(row.get("ts") or 0))
        except Exception as e:  # noqa: BLE001 — cooldown state is UNKNOWN when this read fails; treating it as "never scaled" would defeat the flap guard exactly when the KG backend is degraded
            logger.warning("fleet_autoscaler: execution ledger scan failed: %s", e)
        if not (decision_ok and execution_ok):
            return None
        return latest

    # ── one service ─────────────────────────────────────────────────

    def _evaluate_service(
        self,
        name: str,
        spec: ScalingSpec,
        observation: Any,
        *,
        sample: ScalingSignalSample | None | object = _SAMPLE_UNSET,
        definition: SignalDefinition | None | object = _SAMPLE_UNSET,
    ) -> ServiceEvaluation:
        key = (name, spec.signal)
        if observation is None or observation.replicas is None:
            self._reset_stabilization(key)
            return ServiceEvaluation(
                name, "skipped", "unobserved (no replica evidence)"
            )
        if observation.status == STATUS_DOWN:
            # A down service is the reconciler's (restart) problem, not a
            # scaling problem — scaling a dead service masks the failure.
            self._reset_stabilization(key)
            return ServiceEvaluation(name, "skipped", "observed down — not scaling")
        current = int(observation.replicas)

        if definition is _SAMPLE_UNSET:
            definition = self._signal_definition(name, spec)
        if sample is _SAMPLE_UNSET:
            values = read_scaling_signal_samples(self.signals, [key])
            sample = values.get(key)
        if not isinstance(definition, SignalDefinition) or not definition.binds_service(
            name
        ):
            sample = None
        if isinstance(sample, ScalingSignalSample) and isinstance(
            definition, SignalDefinition
        ):
            sample = validate_scaling_signal_sample(
                sample,
                service=name,
                signal=spec.signal,
                aggregation=definition.aggregation,
                unit=definition.unit,
                scope=definition.scope,
                previous_observed_at=self._last_signal_observed.get(key),
            )
        if not isinstance(sample, ScalingSignalSample):
            sample = None
        if sample is None:
            self._reset_stabilization(key)
            return ServiceEvaluation(
                name, "skipped", f"no data for signal {spec.signal!r}", current=current
            )
        self._last_signal_observed[(name, spec.signal)] = sample.observed_at
        value = sample.value

        desired = compute_desired_replicas(
            current, value, spec, aggregation=sample.aggregation
        )
        per_replica = _per_replica_value(current, value, sample.aggregation)
        if current > 0 and (
            abs(per_replica - spec.target) <= spec.target * spec.deadband
        ):
            self._reset_stabilization(key)
            return ServiceEvaluation(
                name,
                "skipped",
                f"deadband ({spec.deadband:.3g} of target)",
                current=current,
                desired=current,
                value=value,
            )
        # CONCEPT:AU-OS.scaling.cost-aware-autoscaling — cost-aware scale-up cap. Keep the target-tracking math
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
            self._reset_stabilization(key)
            return ServiceEvaluation(
                name,
                "skipped",
                cost_reason or "at target",
                current=current,
                desired=desired,
                value=value,
            )

        direction = "up" if desired > current else "down"
        required_samples = (
            spec.scale_up_stabilization_samples
            if direction == "up"
            else spec.scale_down_stabilization_samples
        )
        ready, consecutive = self._stabilization_ready(key, direction, required_samples)
        if not ready:
            return ServiceEvaluation(
                name,
                "skipped",
                f"stabilizing {direction} ({consecutive}/{required_samples} consecutive samples)",
                current=current,
                desired=desired,
                value=value,
            )

        last_scale = self._last_scale_unix(name)
        if last_scale is None:
            return ServiceEvaluation(
                name,
                "skipped",
                "cooldown state unknown (ledger unavailable) — failing closed",
                current=current,
                desired=desired,
                value=value,
            )
        if last_scale and (time.time() - last_scale) < spec.cooldown_s:
            return ServiceEvaluation(
                name,
                "skipped",
                f"cooldown ({spec.cooldown_s}s since last scale; flap guard)",
                current=current,
                desired=desired,
                value=value,
            )

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
                # CONCEPT:AU-OS.scaling.cost-aware-autoscaling — cost lens on every scaling action.
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
        if execution.get("ok"):
            # A successful action consumed the streak. The next action in the
            # same direction must observe its full declared sample count again;
            # approval-pending/proposed actions intentionally retain state.
            self._reset_stabilization(key)
        if execution.get("ok") and (
            direction == "up" or bool(self.policy.option("watch_scale_down", False))
        ):
            from agent_utilities.orchestration.deploy_watch import watch_deploy

            watch_deploy(self.engine, name, source="autoscaler")
        return evaluation

    # ── one tick ────────────────────────────────────────────────────

    def evaluate(self) -> dict[str, Any]:
        """One autoscale pass over every service with a scaling block."""
        health = self._fleet_health()
        health_payload = health.model_dump(mode="json")
        if not health.autoscaling_ready:
            report: dict[str, Any] = {
                "evaluated": 0,
                "actions": 0,
                "scaled": 0,
                "evaluations": [],
                "actuator": getattr(self.actuator, "name", "?"),
                "signal_provider": getattr(self.signals, "name", "?"),
                "health": health_payload,
                "reason": "fleet supervisory evidence is not ready; autoscaling skipped",
            }
            self._record(report)
            return report
        desired_state = load_desired_state()
        observed: dict[str, Any] = {}
        try:
            observed = self.observer.observe() or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("fleet_autoscaler: observer failed: %s", e)

        candidates = [
            (name, want.scaling)
            for name, want in sorted(desired_state.items())
            if want.scaling is not None and want.desired == "running"
        ]
        requests = [(name, spec.signal) for name, spec in candidates]
        definitions = {
            (name, spec.signal): self._signal_definition(name, spec)
            for name, spec in candidates
        }
        samples = (
            read_scaling_signal_samples(self.signals, requests) if requests else {}
        )

        evaluations: list[ServiceEvaluation] = []
        actions = 0
        for name, spec in candidates:
            if actions >= self.max_actions:
                evaluations.append(
                    ServiceEvaluation(
                        name, "skipped", "per-tick action budget exhausted"
                    )
                )
                continue
            key = (name, spec.signal)
            evaluation = self._evaluate_service(
                name,
                spec,
                observed.get(name),
                sample=samples.get(key),
                definition=definitions.get(key),
            )
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
            "health": health_payload,
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
                f"autoscale_evaluation:{uuid.uuid4().hex}",
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
        except Exception as e:  # noqa: BLE001 — AutoscaleEvaluation is a per-tick summary node; confirmed via repo-wide grep that nothing ever reads it back, so no decision depends on this write succeeding
            logger.debug("fleet_autoscaler: evaluation write failed: %s", e)


def autoscale_fleet(engine: Any) -> dict[str, Any]:
    """The leader-only maintenance-tick entry point (see ``engine_tasks``)."""
    return FleetAutoscaler(engine).evaluate()


#: The sole control-plane work label whose committed changes move queue depth.
WORK_ITEM_LABEL = "WorkItem"


def fleet_autoscale_subscription(engine: Any) -> Any:
    """Reactive change-feed subscription over control-plane WorkItem mutations.

    CONCEPT:AU-KG.compute.change-feed-subscription — the poll→push seam for autoscaling: instead of waiting for
    the next leader poll interval, the daemon polls this subscription and, when the
    engine pushes a WorkItem change (the queue-depth signal moved), fires an
    autoscale evaluation immediately — so scaling reacts to the change-EVENT, not
    a fixed interval. The slow periodic ``_tick_fleet_autoscaler`` stays as the
    safety-net reconcile.

    Subscribes on the engine's **control graph** (``__control__`` — where WorkItem
    lives, CONCEPT:AU-KG.backend.schedule-on-control-graph), resolved via the engine's control backend. The
    handler bumps ``sub.pending_state["pending"]``; the caller reads it to decide
    whether to evaluate now. Returns a
    :class:`~agent_utilities.graph.reactive.EngineSubscription` whose ``.available``
    is ``False`` (a permanent no-op) when no engine streaming surface exists — so
    the periodic tick remains the correctness guarantee.
    """
    from agent_utilities.graph.reactive import subscribe

    # The control plane (``__control__``) is where WorkItem is written; fall back to
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

    sub = subscribe(source, WORK_ITEM_LABEL, _on_task_change)
    sub.pending_state = state  # type: ignore[attr-defined]
    return sub
