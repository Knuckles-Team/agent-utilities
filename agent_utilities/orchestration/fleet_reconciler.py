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
import math
import time
import uuid
from collections.abc import Callable
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
from agent_utilities.orchestration.fleet_health import (
    FleetHealthEvidence,
    FleetHealthSnapshot,
    collect_fleet_health,
    unavailable_fleet_health,
)
from agent_utilities.orchestration.fleet_observation import (
    STATUS_DOWN,
    STATUS_UP,
    get_fleet_observer,
)

logger = logging.getLogger(__name__)

_APPROVAL_DRAIN_LIMIT = 20

_MAX_SCALING_REPLICAS = 100_000
_MAX_SCALING_STEP = 100_000
_MAX_SCALING_TARGET = 1_000_000_000_000.0
_MAX_SCALING_COOLDOWN_S = 86_400.0
_MAX_SCALING_DEADBAND = 1.0
_MAX_STABILIZATION_SAMPLES = 60

# Action kinds whose execution warrants a follow-up health watch (OS-5.27).
_WATCHED_KINDS = {
    "restart_service",
    "deploy_service",
    "redeploy_stack",
    "rollback_service",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _strict_scaling_int(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < lower or value > upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _strict_scaling_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


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
    deadband: float = 0.05  # relative target band (5% by default)
    scale_up_stabilization_samples: int = 1  # fast scale-up by default
    scale_down_stabilization_samples: int = 3  # conservative scale-down

    def __post_init__(self) -> None:
        integer_fields = (
            ("min_replicas", self.min_replicas, 0, _MAX_SCALING_REPLICAS),
            ("max_replicas", self.max_replicas, 0, _MAX_SCALING_REPLICAS),
            ("scale_up_step", self.scale_up_step, 1, _MAX_SCALING_STEP),
            ("scale_down_step", self.scale_down_step, 1, _MAX_SCALING_STEP),
            (
                "scale_up_stabilization_samples",
                self.scale_up_stabilization_samples,
                1,
                _MAX_STABILIZATION_SAMPLES,
            ),
            (
                "scale_down_stabilization_samples",
                self.scale_down_stabilization_samples,
                1,
                _MAX_STABILIZATION_SAMPLES,
            ),
        )
        for field_name, value, lower, upper in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
            if value < lower or value > upper:
                raise ValueError(f"{field_name} must be between {lower} and {upper}")
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        if not isinstance(self.signal, str) or not self.signal.strip():
            raise ValueError("signal must be a non-empty string")
        if len(self.signal) > 128:
            raise ValueError("signal is too long")
        for field_name, numeric_value in (
            ("target", self.target),
            ("cooldown_s", self.cooldown_s),
            ("deadband", self.deadband),
        ):
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise ValueError(f"{field_name} must be numeric")
            try:
                parsed = float(numeric_value)
            except OverflowError as exc:
                raise ValueError(f"{field_name} must be finite") from exc
            if not math.isfinite(parsed):
                raise ValueError(f"{field_name} must be finite")
        if self.target <= 0 or self.target > _MAX_SCALING_TARGET:
            raise ValueError("target is outside its bounded range")
        if self.cooldown_s < 0 or self.cooldown_s > _MAX_SCALING_COOLDOWN_S:
            raise ValueError("cooldown_s is outside its bounded range")
        if self.deadband < 0 or self.deadband > _MAX_SCALING_DEADBAND:
            raise ValueError("deadband must be between 0 and 1")


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
            min_replicas=_strict_scaling_int(
                raw.get("min", 1), "min", 0, _MAX_SCALING_REPLICAS
            ),
            max_replicas=_strict_scaling_int(
                raw["max"], "max", 0, _MAX_SCALING_REPLICAS
            ),  # required: no implicit ceiling
            signal=raw.get("signal") or "",
            target=_strict_scaling_float(raw.get("target"), "target"),
            scale_up_step=_strict_scaling_int(
                raw.get("scale_up_step", 1),
                "scale_up_step",
                1,
                _MAX_SCALING_STEP,
            ),
            scale_down_step=_strict_scaling_int(
                raw.get("scale_down_step", 1),
                "scale_down_step",
                1,
                _MAX_SCALING_STEP,
            ),
            cooldown_s=_strict_scaling_float(
                raw.get("cooldown_s", 300.0), "cooldown_s"
            ),
            deadband=_strict_scaling_float(raw.get("deadband", 0.05), "deadband"),
            scale_up_stabilization_samples=_strict_scaling_int(
                raw.get("scale_up_stabilization_samples", 1),
                "scale_up_stabilization_samples",
                1,
                _MAX_STABILIZATION_SAMPLES,
            ),
            scale_down_stabilization_samples=_strict_scaling_int(
                raw.get("scale_down_stabilization_samples", 3),
                "scale_down_stabilization_samples",
                1,
                _MAX_STABILIZATION_SAMPLES,
            ),
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


class FleetRegistryError(RuntimeError):
    """The fleet registry cannot answer a question it is the authority for."""


def registry_server_aliases(registry_path: str | Path | None = None) -> dict[str, str]:
    """Map every provider distribution name to its ONE registered server alias.

    CONCEPT:AU-KG.ontology.registry-derived-server-alias — ``mcp-fleet.registry.yml``
    is the single authority for what a fleet server is *called*. Anything that needs
    that alias derives it here instead of restating it: a restated alias is exactly
    the D-OB-7 drift, where 27 providers' signed manifests named a server the
    registry did not have (``github-agent`` where the fleet runs ``github-mcp``) and
    9 more named servers the registry had never heard of at all.
    """
    import yaml

    path = resolve_registry_path(str(registry_path) if registry_path else None)
    if path is None:
        raise FleetRegistryError("the MCP fleet registry is not available")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        services = data["services"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as exc:
        raise FleetRegistryError("the MCP fleet registry is unreadable") from exc
    if not isinstance(services, list) or not services:
        raise FleetRegistryError("the MCP fleet registry declares no services")
    aliases: dict[str, str] = {}
    for entry in services:
        if not isinstance(entry, dict):
            raise FleetRegistryError("the MCP fleet registry has an invalid entry")
        package = str(entry.get("package") or "")
        name = str(entry.get("name") or "")
        if not package or not name:
            raise FleetRegistryError("a registry service is missing name or package")
        if package in aliases and aliases[package] != name:
            raise FleetRegistryError("a provider maps to two registry server aliases")
        aliases[package] = name
    return aliases


def registry_server_alias(package: str, registry_path: str | Path | None = None) -> str:
    """The registered server alias for one provider distribution — or fail closed."""

    aliases = registry_server_aliases(registry_path)
    alias = aliases.get(str(package))
    if not alias:
        raise FleetRegistryError("provider is not registered in the MCP fleet registry")
    return alias


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
            logger.warning(
                "fleet_reconciler: registry parse failed (%s)", type(e).__name__
            )

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
        health_provider: Callable[[], FleetHealthEvidence | FleetHealthSnapshot]
        | None = None,
    ):
        self.engine = engine
        self.observer = observer or get_fleet_observer(engine)
        self.actuator = actuator or get_fleet_actuator()
        self.policy = policy or get_action_policy(engine)
        self.health_provider = health_provider or (
            lambda: collect_fleet_health().evidence
        )
        self._last_health: FleetHealthEvidence | None = None
        if max_actions is None:
            try:
                from agent_utilities.core.config import config as _cfg

                max_actions = int(getattr(_cfg, "fleet_reconciler_max_actions", 5))
            except Exception:  # noqa: BLE001
                max_actions = 5
        self.max_actions = max(1, int(max_actions))

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
                "fleet_reconciler: supervisory evidence unavailable (%s)",
                type(exc).__name__,
            )
            return unavailable_fleet_health("reconciler.health")

    # ── divergence detection ────────────────────────────────────────

    def diff(self) -> list[ActionRequest]:
        """Desired vs observed → ordered convergence proposals (conservative).

        Only positive evidence diverges: a service the observer never saw is
        skipped, not restarted.
        """
        self._last_health = self._fleet_health()
        if not self._last_health.convergence_ready:
            return []
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
        except Exception as e:  # noqa: BLE001 — read-only candidate scan; on failure nothing is mutated (no approvals drained, budget untouched) and every approved row is re-selected on the next tick
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
            # D-DST-6: execute_action has ALREADY run by this point (and this
            # actuation may not be idempotent). If the status stamp below
            # fails, the ActionApproval row stays 'approved' and gets
            # re-selected by the next tick's scan, causing execute_action to
            # fire a SECOND time for the same grant. Retry the stamp once
            # before giving up, and log loud (not DEBUG) when it's still
            # unstamped so a stuck 'approved' row is discoverable before the
            # next tick re-fires it.
            for _attempt in (1, 2):
                try:
                    self.engine.backend.execute(
                        "MATCH (a:ActionApproval {id: $id}) "
                        "SET a.status = $status, a.executed_at = $ts",
                        {"id": props["id"], "status": new_status, "ts": _now_iso()},
                    )
                    break
                except Exception as e:  # noqa: BLE001 — retried once above; still logged loud below since a failed stamp risks a duplicate actuation on the next tick
                    if _attempt == 2:
                        logger.warning(
                            "fleet_reconciler: approval %s stamp failed twice, "
                            "status may still read 'approved' (risk: duplicate "
                            "actuation next tick): %s",
                            props.get("id"),
                            e,
                        )
                    else:
                        logger.debug(
                            "fleet_reconciler: approval stamp failed, retrying once: %s",
                            e,
                        )
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
        health = self._last_health or self._fleet_health()
        if not health.convergence_ready:
            report: dict[str, Any] = {
                "divergences": 0,
                "processed": 0,
                "deferred": [],
                "actions": [],
                "approved_drained": [],
                "actuator": getattr(self.actuator, "name", "?"),
                "fired_agent_tasks": [],
                "health": health.model_dump(mode="json"),
                "reason": "fleet supervisory evidence is not ready; convergence skipped",
            }
            self._record(report)
            return report
        processed = proposals[: self.max_actions]
        deferred = proposals[self.max_actions :]

        actions = [self._converge_one(p) for p in processed]
        # Human-granted approvals get their own budget: a backlog of new
        # divergences must not starve actions an operator already sanctioned.
        approved = self._drain_approved(self.max_actions)
        # C3/Phase 3a: the leader-only tick this reconcile() pass IS also
        # sweeps 'blocked' :AgentTask nodes whose dependencies just
        # completed, firing them to 'ready'. fire_ready_agent_tasks() never
        # raises (degrades to [] on an unreachable engine/failed query), so
        # this never destabilizes the rest of the report.
        fired_agent_tasks = fire_ready_agent_tasks(self.engine)
        report = {
            "divergences": len(proposals),
            "processed": len(actions),
            "deferred": [p.summary() for p in deferred],
            "actions": actions,
            "approved_drained": approved,
            "actuator": getattr(self.actuator, "name", "?"),
            "fired_agent_tasks": fired_agent_tasks,
            "health": health.model_dump(mode="json"),
        }
        self._record(report)
        return report

    def _record(self, report: dict[str, Any]) -> None:
        if self.engine is None:
            return
        try:
            self.engine.add_node(
                f"reconcile_report:{uuid.uuid4().hex}",
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
        except Exception as e:  # noqa: BLE001 — reconcile() already built and returns the full report dict independent of this write; this is only the durable KG audit copy
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
# watcher straight back to Phase 3a's always-sweep behavior. Wired into the
# leader-only ``FleetReconciler.reconcile()`` tick (fleet-wide). (The standalone
# ``RecoveryDaemon.stabilize()`` local-tick caller this watcher class also once
# supported was deleted as orphaned/never-instantiated dead code — this
# reconciler tick was always the live path.)

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

    Routed through WorkItem (AU-P1-1, report §9 #4): the readiness event first
    shadow-creates/advances this task's ``WorkItem`` via
    :func:`~agent_utilities.orchestration.work_item.ensure_agent_task_work_item`
    (so the engine-native dependency graph reflects readiness immediately,
    rather than lazily at claim time) — WorkItem is the write authority. The
    legacy ``:AgentTask.status`` flip stays as a best-effort MIRROR (same
    pattern as ``work_item.claim_agent_task_via_work_item``'s own "running"
    mirror) so unmigrated readers (dashboards) keep seeing 'ready' unchanged;
    this sweep itself never reads that mirror back.

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
    except Exception as e:  # noqa: BLE001 — read-only scan for 'blocked' AgentTask nodes; on failure no task is mutated and the same nodes are re-selected on the next tick
        logger.debug("fleet_reconciler: agent-task dependency sweep failed: %s", e)
        return []

    from agent_utilities.orchestration.work_item import ensure_agent_task_work_item

    fired: list[str] = []
    for row in rows:
        task_id = row.get("id")
        if not task_id:
            continue
        deps = list(row.get("depends_on_task_ids") or [])
        if not _agent_task_dependencies_satisfied(engine, deps):
            continue
        try:
            ensure_agent_task_work_item(engine, task_id)
        except Exception as e:
            # D-DST-6: this docstring calls WorkItem "the write authority" and the
            # legacy status flip a "best-effort MIRROR" -- but falling through here
            # (the prior behavior) let the mirror flip to 'ready' even when the
            # authority write failed. Because this sweep's OWN selection query is
            # `WHERE status = 'blocked'`, once the legacy status flips to 'ready'
            # the task permanently drops out of the retry pool even though its
            # WorkItem was never created -- a transient KG hiccup here could
            # orphan a task forever (looks 'ready' to legacy readers, invisible to
            # the WorkItem-based claim path). `continue` so the task stays
            # 'blocked' (and thus retried next tick) whenever the authority write
            # fails, instead of letting the mirror advance anyway.
            logger.warning(
                "fleet_reconciler: work_item shadow-create failed for %s, "
                "leaving task 'blocked' for retry: %s",
                task_id,
                e,
            )
            continue
        try:
            engine.add_node(task_id, "AgentTask", properties={"status": "ready"})
            fired.append(task_id)
        except Exception as e:  # noqa: BLE001 — on failure the task is not appended to 'fired' and its legacy status stays 'blocked' (the WorkItem authority write above already succeeded), so it is naturally re-swept on the next tick
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
        self._dirty = False
        self._subscription = self._build_subscription(engine)

    def _build_subscription(self, engine: Any) -> Any:
        try:
            from agent_utilities.graph.reactive.engine_subscription import subscribe
        except Exception as e:  # noqa: BLE001 — subsystem unimportable ⇒ poll fallback
            logger.debug("fleet_reconciler: engine_subscription unavailable: %s", e)
            return None
        try:
            return subscribe(engine, "AgentTask", self._on_change)
        except Exception as e:  # noqa: BLE001 — documented fallback: a construction failure returns None, and fire() degrades to the always-sweep Phase-3a-equivalent behavior below
            logger.debug("fleet_reconciler: AgentTask subscription failed: %s", e)
            return None

    def _on_change(self, event: dict[str, Any]) -> None:
        self._dirty = True

    def fire(self, limit: int = _AGENT_TASK_DEP_SWEEP_LIMIT) -> list[str]:
        """One tick: CDC-gated sweep when the engine change-feed is reachable, else always-sweep."""
        sub = self._subscription
        if sub is None or not getattr(sub, "available", False):
            return fire_ready_agent_tasks(self.engine, limit=limit)

        try:
            sub.poll(block_ms=0)
        except Exception as e:  # noqa: BLE001 — a feed hiccup ⇒ fall back to the sweep
            logger.debug("fleet_reconciler: AgentTask CDC poll failed: %s", e)
            return fire_ready_agent_tasks(self.engine, limit=limit)

        if not self._dirty:
            return []  # nothing changed since the last tick — zero Cypher work
        self._dirty = False
        return fire_ready_agent_tasks(self.engine, limit=limit)
