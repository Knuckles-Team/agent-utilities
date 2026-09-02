#!/usr/bin/python
from __future__ import annotations

"""Desired-state fleet reconciler.

CONCEPT:AU-OS.config.desired-state-fleet-reconciler — Desired-state fleet reconciler: a leader-only daemon tick
diffs the declared fleet (registry + optional override) against the observed
fleet and converges through the native intent authority or the ActionPolicy
and actuator seams.

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
* **divergence → action** — service down ⇒ ``restart_service``;
  running-but-undesired ⇒ ``stop_service``; a native ``intent_accepted`` ``ScaleIntent``
  may reconcile a replica mismatch. Services with NO observation are skipped
  (never act on zero evidence). A service delegated to external HPA/KEDA is
  observation-only and never receives a replica action from AU.
* **gate → intent/reconcile → actuate** — native autoscaler proposals are
  durable, revisioned intents; this reconciler accepts queued intents and is
  the sole native replica actuator, recording simulated/executed/observed/
  verified outcomes. Other proposals pass the ActionPolicy
  decision point (CONCEPT:AU-OS.deployment.fleet-lifecycle-control); allowed
  actions run through the
  :class:`~agent_utilities.orchestration.fleet_actuation.FleetActuator` (the
  default dry-run actuator records a simulation without mutating), and restarts
  schedule an OS-5.27 health watch. Queue-approval decisions land in the
  fleet approvals flow; this tick also DRAINS granted approvals, closing the
  human-in-the-loop circle.
* **storm guard** — at most ``FLEET_RECONCILER_MAX_ACTIONS`` proposals are
  processed per tick; the rest defer to the next tick.

Wiring: registered as the leader-only ``fleet_reconciler`` maintenance job in
``knowledge_graph/core/engine_tasks.py``, opt-in via ``FLEET_RECONCILER``
(default off until a deployment wires real actuators).
"""

import hashlib
import json
import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agent_utilities.orchestration.action_policy import (
    ActionRequest,
    get_action_policy,
)
from agent_utilities.orchestration.fleet_actuation import (
    ActionOutboxStore,
    _action_request_digest,
    _outbox_completion_matches,
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

# Replica ownership is explicit.  ``native`` means AU's desired-state intent
# store is authoritative and this reconciler is the only actuator.  The two
# external modes are observation/reporting integrations: HPA/KEDA owns the
# replica field and AU must never write it.
SCALE_CONTROLLER_NATIVE = "native"
SCALE_CONTROLLER_EXTERNAL_HPA = "external_hpa"
SCALE_CONTROLLER_EXTERNAL_KEDA = "external_keda"
SCALE_CONTROLLER_MODES = frozenset(
    {
        SCALE_CONTROLLER_NATIVE,
        SCALE_CONTROLLER_EXTERNAL_HPA,
        SCALE_CONTROLLER_EXTERNAL_KEDA,
    }
)
_EXTERNAL_CONTROLLER_ALIASES = {
    "hpa": SCALE_CONTROLLER_EXTERNAL_HPA,
    "keda": SCALE_CONTROLLER_EXTERNAL_KEDA,
    SCALE_CONTROLLER_EXTERNAL_HPA: SCALE_CONTROLLER_EXTERNAL_HPA,
    SCALE_CONTROLLER_EXTERNAL_KEDA: SCALE_CONTROLLER_EXTERNAL_KEDA,
    SCALE_CONTROLLER_NATIVE: SCALE_CONTROLLER_NATIVE,
}
_SCALE_INTENT_PROPOSED = "proposed"
_SCALE_INTENT_SIMULATED = "simulated"
_SCALE_INTENT_ACCEPTED = "intent_accepted"
_SCALE_INTENT_EXECUTED = "executed"
_SCALE_INTENT_OBSERVED = "observed"
_SCALE_INTENT_VERIFIED = "verified"
_SCALE_INTENT_FAILED = "failed"
_SCALE_INTENT_REJECTED = "rejected"
_SCALE_INTENT_SUPERSEDED = "superseded"
_SCALE_INTENT_RECOVERY_PENDING = "recovery_pending"
_SCALE_INTENT_STATES = frozenset(
    {
        _SCALE_INTENT_PROPOSED,
        _SCALE_INTENT_SIMULATED,
        _SCALE_INTENT_ACCEPTED,
        _SCALE_INTENT_EXECUTED,
        _SCALE_INTENT_OBSERVED,
        _SCALE_INTENT_VERIFIED,
        _SCALE_INTENT_FAILED,
        _SCALE_INTENT_REJECTED,
        _SCALE_INTENT_SUPERSEDED,
        _SCALE_INTENT_RECOVERY_PENDING,
    }
)
_SCALE_INTENT_ACTIVE = frozenset(
    {
        _SCALE_INTENT_PROPOSED,
        _SCALE_INTENT_ACCEPTED,
        _SCALE_INTENT_EXECUTED,
        _SCALE_INTENT_OBSERVED,
    }
)
_SCALE_INTENT_VISIBLE = frozenset({_SCALE_INTENT_ACCEPTED})


def normalize_scale_controller_mode(raw: Any) -> str | None:
    """Canonicalize the one allowed replica-controller declaration."""

    value = SCALE_CONTROLLER_NATIVE if raw is None else str(raw).strip().lower()
    return _EXTERNAL_CONTROLLER_ALIASES.get(value)


def scale_intent_key(
    service: str,
    controller_mode: str,
    desired_replicas: int,
    expected_revision: int,
) -> str:
    """Stable idempotency key for one CAS proposal, without runtime secrets."""

    body = json.dumps(
        {
            "service": service,
            "controller_mode": controller_mode,
            "desired_replicas": desired_replicas,
            "expected_revision": expected_revision,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "scale-intent:" + hashlib.sha256(body).hexdigest()[:32]


class ScaleIntentStore(Protocol):
    """Native durable read/CAS seam for autoscaler desired replica intents."""

    def latest(self, service: str) -> tuple[bool, dict[str, Any] | None]:
        """Return ``(read_complete, latest_intent)``; incomplete reads fail closed."""

    def cas(self, request: dict[str, Any]) -> dict[str, Any]:
        """Atomically put/transition an intent at ``expected_revision``.

        The engine implementation must return the existing result for an
        exact replay of the same intent identity and reject a different
        payload that reuses that identity.
        """


class EngineScaleIntentStore:
    """Adapter requiring an engine-native ScaleIntent CAS implementation.

    A read-only Cypher query is acceptable for discovery, but a read-then-add
    fallback is deliberately not: two autoscaler leaders could both win and
    the reconciler would have no authoritative revision to act on.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def _latest_via_reader(
        self, reader: Callable[[str], Any], service: str
    ) -> tuple[bool, dict[str, Any] | None]:
        try:
            value = reader(service)
            if isinstance(value, dict) and "intent" in value:
                value = value.get("intent")
            return True, value if isinstance(value, dict) else None
        except Exception as exc:  # noqa: BLE001 — incomplete read is fail-closed
            logger.warning("scale intent read failed for %s: %s", service, exc)
            return False, None

    def _latest_via_query(
        self, query: Callable[..., Any], service: str
    ) -> tuple[bool, dict[str, Any] | None]:
        try:
            rows = (
                query(
                    "MATCH (i:ScaleIntent {service: $service}) "
                    "RETURN i ORDER BY i.revision DESC LIMIT 1",
                    {"service": service},
                )
                or []
            )
            row = rows[0] if rows else None
            value = row.get("i") if isinstance(row, dict) else row
            return True, value if isinstance(value, dict) else None
        except Exception as exc:  # noqa: BLE001 — incomplete read is fail-closed
            logger.warning("scale intent query failed for %s: %s", service, exc)
            return False, None

    def latest(self, service: str) -> tuple[bool, dict[str, Any] | None]:
        reader = getattr(self.engine, "read_scale_intent", None)
        if callable(reader):
            return self._latest_via_reader(reader, service)
        query = getattr(self.engine, "query_cypher", None)
        if not callable(query):
            return False, None
        return self._latest_via_query(query, service)

    def cas(self, request: dict[str, Any]) -> dict[str, Any]:
        writer = getattr(self.engine, "cas_scale_intent", None)
        if not callable(writer):
            backend = getattr(self.engine, "backend", None)
            writer = getattr(backend, "cas_scale_intent", None)
        if not callable(writer):
            return {
                "accepted": False,
                "reason": "native ScaleIntent CAS is unavailable",
            }
        try:
            result = writer(dict(request))
        except Exception as exc:  # noqa: BLE001 — CAS failures never actuate
            logger.warning("scale intent CAS failed: %s", exc)
            return {"accepted": False, "reason": "native ScaleIntent CAS failed"}
        return result if isinstance(result, dict) else {"accepted": bool(result)}


def _intent_metadata_valid(intent: dict[str, Any] | None) -> bool:
    if not isinstance(intent, dict):
        return False
    try:
        return (
            bool(str(intent.get("service") or ""))
            and bool(str(intent.get("intent_id") or ""))
            and intent.get("controller_mode") is not None
            and normalize_scale_controller_mode(intent.get("controller_mode"))
            == SCALE_CONTROLLER_NATIVE
            and int(intent["revision"]) >= 1
            and int(intent["desired_replicas"]) >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _intent_is_valid(intent: dict[str, Any] | None) -> bool:
    if intent is None or not _intent_metadata_valid(intent):
        return False
    return str(intent.get("status") or "") in _SCALE_INTENT_VISIBLE


def _cas_succeeded(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("accepted") is False or result.get("ok") is False:
        return False
    return bool(
        result.get("accepted") is True
        or result.get("ok") is True
        or result.get("status") in _SCALE_INTENT_STATES
    )


# Action kinds whose execution warrants a follow-up health watch (OS-5.27).
_WATCHED_KINDS = {
    "restart_service",
    "deploy_service",
    "redeploy_stack",
    "rollback_service",
}

# `scale_service` is watched CONDITIONALLY, so it is deliberately not a member
# of `_WATCHED_KINDS`. Before actuation moved here from the autoscaler, a
# successful scale scheduled a deploy watch when the direction was up, or when
# the policy file set `options: {watch_scale_down: true}`. Moving actuation to
# the reconciler dropped that watch entirely and left `watch_scale_down` a dead
# option read by nothing -- a silent capability loss rather than a design
# decision, since the fleet-scale-authority document records every other
# behaviour change and says nothing about removing it.
_SCALE_KIND = "scale_service"


def _should_watch(request: Any, policy: Any) -> bool:
    """Whether a successful actuation of ``request`` schedules a deploy watch."""
    if request.kind in _WATCHED_KINDS:
        return True
    if request.kind != _SCALE_KIND:
        return False
    if str(request.params.get("direction") or "").lower() == "up":
        return True
    return bool(policy.option("watch_scale_down", False))


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
    ``controller_mode`` is the replica authority: ``native`` delegates the
    write to this reconciler through a durable intent; ``external_hpa`` and
    ``external_keda`` are observation-only.
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
    controller_mode: str = SCALE_CONTROLLER_NATIVE

    def __post_init__(self) -> None:
        self._validate_replica_bounds()
        self._validate_signal()
        self._validate_numeric_fields()

    def _validate_replica_bounds(self) -> None:
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

    def _validate_signal(self) -> None:
        if not isinstance(self.signal, str) or not self.signal.strip():
            raise ValueError("signal must be a non-empty string")
        if len(self.signal) > 128:
            raise ValueError("signal is too long")

    def _validate_numeric_fields(self) -> None:
        self._validate_numeric_types()
        self._validate_numeric_bounds()

    def _validate_numeric_types(self) -> None:
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

    def _validate_numeric_bounds(self) -> None:
        if self.target <= 0 or self.target > _MAX_SCALING_TARGET:
            raise ValueError("target is outside its bounded range")
        if self.cooldown_s < 0 or self.cooldown_s > _MAX_SCALING_COOLDOWN_S:
            raise ValueError("cooldown_s is outside its bounded range")
        if self.deadband < 0 or self.deadband > _MAX_SCALING_DEADBAND:
            raise ValueError("deadband must be between 0 and 1")


@dataclass(frozen=True)
class KubernetesResourceRef:
    """Resource-registry binding carried into a Kubernetes ActionRequest.

    This is deliberately an identity reference, not a mutable desired-state
    cache.  The object UID and resourceVersion are refreshed by the deployment
    or resource-registry integration; if they are absent or stale, the k8s
    actuator refuses the action.  Manifests are not modified here because
    their registry publication is an environment-owned prerequisite.
    """

    cluster: str
    context: str
    namespace: str
    workload_kind: str
    name: str
    uid: str
    resource_version: str
    controller_mode: str
    quorum_required: bool = False

    def to_params(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "context": self.context,
            "namespace": self.namespace,
            "workload_kind": self.workload_kind,
            "resource_name": self.name,
            "uid": self.uid,
            "resource_version": self.resource_version,
            "controller_mode": self.controller_mode,
            "quorum_required": self.quorum_required,
        }


def _kubernetes_resource_fields(
    raw: dict[str, Any], service: str
) -> dict[str, str] | None:
    """Resolve the aliased field set for one k8s resource block, or None if incomplete."""

    aliases = {
        "cluster": ("cluster", "kube_cluster"),
        "context": ("context", "kube_context"),
        "namespace": ("namespace", "kube_namespace"),
        "workload_kind": ("workload_kind", "kind", "resource_kind"),
        "name": ("name", "resource_name", "workload_name"),
        "uid": ("uid", "resource_uid"),
        "resource_version": ("resource_version", "resourceVersion"),
        "controller_mode": ("controller_mode", "controller", "mode"),
    }

    def pick(*names: str) -> str:
        for name in names:
            value = raw.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    values = {field: pick(*names) for field, names in aliases.items()}
    missing = [field for field, value in values.items() if not value]
    if missing:
        logger.warning(
            "kubernetes resource for %s is incomplete: %s",
            service,
            ", ".join(missing),
        )
        return None
    return values


def _kubernetes_kind_and_mode(
    values: dict[str, str], service: str
) -> tuple[str, str] | None:
    """Normalize the workload kind + controller mode, or None if unsupported."""

    kind_aliases = {
        "deployment": "Deployment",
        "deployments": "Deployment",
        "statefulset": "StatefulSet",
        "statefulsets": "StatefulSet",
    }
    kind = kind_aliases.get(values["workload_kind"].lower())
    mode = normalize_scale_controller_mode(values["controller_mode"])
    if kind is None or mode is None:
        logger.warning(
            "kubernetes resource for %s has unsupported kind/controller", service
        )
        return None
    return kind, mode


def _kubernetes_quorum_required(raw: dict[str, Any], service: str) -> tuple[bool, bool]:
    """Return (valid, value); value is only meaningful when valid is True."""

    quorum_raw = raw.get("quorum_required", raw.get("quorum", False))
    if isinstance(quorum_raw, bool):
        return True, quorum_raw
    if str(quorum_raw).strip().lower() in {"1", "true", "yes"}:
        return True, True
    if str(quorum_raw).strip().lower() in {"0", "false", "no", ""}:
        return True, False
    logger.warning("kubernetes resource for %s has invalid quorum_required", service)
    return False, False


def parse_kubernetes_resource(raw: Any, service: str) -> KubernetesResourceRef | None:
    """Parse a registry-owned Kubernetes identity without guessing fields.

    A malformed block is omitted, which intentionally makes any later k8s
    action fail closed in ``KubernetesActuator`` rather than silently deriving
    identity from a service name or configured namespace.
    """

    if raw is None:
        return None
    if not isinstance(raw, dict):
        logger.warning("kubernetes resource for %s is not a mapping", service)
        return None
    values = _kubernetes_resource_fields(raw, service)
    if values is None:
        return None
    kind_mode = _kubernetes_kind_and_mode(values, service)
    if kind_mode is None:
        return None
    kind, mode = kind_mode
    if values["name"] != service:
        logger.warning("kubernetes resource for %s names a different workload", service)
        return None
    quorum_valid, quorum_required = _kubernetes_quorum_required(raw, service)
    if not quorum_valid:
        return None
    return KubernetesResourceRef(
        cluster=values["cluster"],
        context=values["context"],
        namespace=values["namespace"],
        workload_kind=kind,
        name=values["name"],
        uid=values["uid"],
        resource_version=values["resource_version"],
        controller_mode=mode,
        quorum_required=quorum_required,
    )


def _resolve_declared_controller_mode(raw: dict[str, Any]) -> str:
    """Resolve the ``controller_mode``/``controller``/``mode`` alias set.

    Multiple aliases must agree; an invalid or conflicting declaration cannot
    silently choose a replica authority.
    """

    declared_modes = [
        raw[key]
        for key in ("controller_mode", "controller", "mode")
        if key in raw and raw[key] is not None
    ]
    normalized_modes = [
        normalize_scale_controller_mode(value) for value in declared_modes
    ]
    if not declared_modes:
        return SCALE_CONTROLLER_NATIVE
    if (
        normalized_modes
        and normalized_modes[0] in SCALE_CONTROLLER_MODES
        and all(value == normalized_modes[0] for value in normalized_modes)
    ):
        return normalized_modes[0]
    return "__invalid__"


def _build_scaling_spec(raw: dict[str, Any], controller_mode: str) -> ScalingSpec:
    return ScalingSpec(
        min_replicas=_strict_scaling_int(
            raw.get("min", 1), "min", 0, _MAX_SCALING_REPLICAS
        ),
        max_replicas=_strict_scaling_int(
            raw["max"], "max", 0, _MAX_SCALING_REPLICAS
        ),  # required: no implicit ceiling
        signal=str(raw.get("signal") or ""),
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
        cooldown_s=_strict_scaling_float(raw.get("cooldown_s", 300.0), "cooldown_s"),
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
        controller_mode=controller_mode,
    )


def _scaling_spec_problems(spec: ScalingSpec) -> list[str]:
    problems: list[str] = []
    if spec.min_replicas < 0:
        problems.append(f"min={spec.min_replicas} < 0")
    if spec.max_replicas < spec.min_replicas:
        problems.append(f"max={spec.max_replicas} < min={spec.min_replicas}")
    if not spec.signal:
        problems.append("signal missing")
    if spec.target <= 0:
        problems.append(f"target={spec.target} must be > 0")
    if spec.controller_mode not in SCALE_CONTROLLER_MODES:
        problems.append(
            "controller_mode must be native, hpa, keda, external_hpa, or external_keda"
        )
    return problems


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
        controller_mode = _resolve_declared_controller_mode(raw)
        spec = _build_scaling_spec(raw, controller_mode)
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("scaling spec for %s is invalid (%s) — ignored", service, e)
        return None
    problems = _scaling_spec_problems(spec)
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
    kubernetes_resource: KubernetesResourceRef | None = None
    operator_override: bool = False


def resolve_registry_path(explicit: str | None = None) -> Path | None:
    """Resolve the fleet registry YAML: explicit flag → repo shipped file."""
    if explicit:
        return Path(explicit)
    shipped = Path(__file__).resolve().parents[2] / "deploy" / "mcp-fleet.registry.yml"
    return shipped if shipped.is_file() else None


class FleetRegistryError(RuntimeError):
    """The fleet registry cannot answer a question it is the authority for."""


def _collect_registry_aliases(services: list[Any]) -> dict[str, str]:
    """Build the package -> server-alias map from a registry's ``services:`` list."""

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
    return _collect_registry_aliases(services)


def registry_server_alias(package: str, registry_path: str | Path | None = None) -> str:
    """The registered server alias for one provider distribution — or fail closed."""

    aliases = registry_server_aliases(registry_path)
    alias = aliases.get(str(package))
    if not alias:
        raise FleetRegistryError("provider is not registered in the MCP fleet registry")
    return alias


def _resolve_desired_state_paths(
    registry_path: str | Path | None, override_path: str | Path | None
) -> tuple[str | Path | None, str | Path | None]:
    """Fall back to AgentConfig for whichever path the caller left unset."""

    if registry_path is not None and override_path is not None:
        return registry_path, override_path
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
    return registry_path, override_path


def _registry_entry_to_desired_service(
    raw: dict[str, Any], name: str
) -> DesiredService:
    return DesiredService(
        name=name,
        desired=str(raw.get("desired") or "running"),
        replicas=int(raw.get("replicas") or 1),
        version=str(raw.get("version") or ""),
        profiles=[str(p) for p in raw.get("profiles") or []],
        scaling=parse_scaling_spec(raw.get("scaling"), name),
        kubernetes_resource=parse_kubernetes_resource(
            raw.get("kubernetes", raw.get("k8s")), name
        ),
    )


def _load_registry_desired_services(
    registry_path: str | Path | None, yaml: Any
) -> dict[str, DesiredService]:
    desired: dict[str, DesiredService] = {}
    path = resolve_registry_path(str(registry_path) if registry_path else None)
    if path is None:
        return desired
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for raw in data.get("services") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            name = str(raw["name"])
            desired[name] = _registry_entry_to_desired_service(raw, name)
    except Exception as e:  # noqa: BLE001 — a broken registry reconciles nothing
        logger.warning("fleet_reconciler: registry parse failed (%s)", type(e).__name__)
    return desired


def _apply_override_entry(
    entry: DesiredService, raw: dict[str, Any], name: str
) -> None:
    if raw.get("desired"):
        entry.desired = str(raw["desired"])
        entry.operator_override = True
    if raw.get("replicas") is not None:
        entry.replicas = int(raw["replicas"])
        entry.operator_override = True
    if raw.get("version"):
        entry.version = str(raw["version"])
    if "scaling" in raw:
        # The registry file is machine-generated, so the override
        # file is where a deployment normally declares scaling
        # bounds. ``scaling: null`` explicitly disables.
        entry.scaling = parse_scaling_spec(raw.get("scaling"), name)
    if "kubernetes" in raw or "k8s" in raw:
        entry.kubernetes_resource = parse_kubernetes_resource(
            raw.get("kubernetes", raw.get("k8s")), name
        )


def _apply_override_desired_state(
    desired: dict[str, DesiredService], override_path: str | Path, yaml: Any
) -> None:
    try:
        data = yaml.safe_load(Path(override_path).read_text(encoding="utf-8")) or {}
        for raw in data.get("services") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            name = str(raw["name"])
            entry = desired.setdefault(name, DesiredService(name=name))
            _apply_override_entry(entry, raw, name)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "fleet_reconciler: override parse failed (%s): %s", override_path, e
        )


def load_desired_state(
    registry_path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> dict[str, DesiredService]:
    """Parse registry + optional override into ``{name: DesiredService}``."""
    import yaml

    registry_path, override_path = _resolve_desired_state_paths(
        registry_path, override_path
    )
    desired = _load_registry_desired_services(registry_path, yaml)
    if override_path:
        _apply_override_desired_state(desired, override_path, yaml)
    return desired


class FleetReconciler:
    """One reconcile pass with one replica authority per service.

    Native scaling consumes an ``intent_accepted`` durable ``ScaleIntent`` and
    this reconciler is the only replica actuator. It records explicit
    simulated/executed/observed/verified outcomes. External HPA/KEDA modes are
    observation-only; their replica mismatches never become AU actions. Real
    actions also require the engine-native idempotent action outbox before the
    actuator is called.
    """

    def __init__(
        self,
        engine: Any,
        observer: Any = None,
        actuator: Any = None,
        policy: Any = None,
        max_actions: int | None = None,
        health_provider: Callable[[], FleetHealthEvidence | FleetHealthSnapshot]
        | None = None,
        intent_store: ScaleIntentStore | None = None,
        action_outbox_store: ActionOutboxStore | None = None,
    ):
        self.engine = engine
        self.observer = observer or get_fleet_observer(engine)
        self.actuator = actuator or get_fleet_actuator()
        self.policy = policy or get_action_policy(engine)
        self.health_provider = health_provider or (
            lambda: collect_fleet_health().evidence
        )
        self._last_health: FleetHealthEvidence | None = None
        self.intent_store = intent_store or EngineScaleIntentStore(engine)
        self.action_outbox_store = action_outbox_store
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

    @staticmethod
    def _bind_kubernetes_resource(
        request: ActionRequest, want: DesiredService | None
    ) -> ActionRequest:
        """Bind registry identity into the immutable action request.

        Registry identity wins over caller-supplied duplicate fields.  The
        resulting request digest therefore fences the UID/resourceVersion and
        controller mode in the durable ActionOutbox record, while preserving
        action-specific parameters such as ``replicas`` and ``image``.
        """

        if want is None or want.kubernetes_resource is None:
            return request
        params = dict(request.params or {})
        params.update(want.kubernetes_resource.to_params())
        return ActionRequest(
            kind=request.kind,
            target=request.target,
            params=params,
            source=request.source,
            reason=request.reason,
            actor_id=request.actor_id,
        )

    def _resolve_valid_intent(
        self, want: DesiredService
    ) -> tuple[dict[str, Any] | None, bool]:
        """Read the latest intent and validate it belongs to `want`.

        Returns ``(intent, read_complete)``. ``intent`` is non-None if and
        only if the read completed and the intent is usable (present and
        valid for `want`); every unusable case returns ``intent=None``. When
        ``intent`` is None the caller must return ``(None, None,
        read_complete)`` unchanged.
        """
        complete, intent = self.intent_store.latest(want.name)
        if not complete:
            return None, False
        if intent is None:
            return None, True
        if (
            not _intent_metadata_valid(intent)
            or str(intent.get("service")) != want.name
        ):
            return None, True
        return intent, True

    def _intent_target(
        self, want: DesiredService, observed_replicas: int | None = None
    ) -> tuple[int | None, dict[str, Any] | None, bool]:
        """Resolve a native target and advance observation states safely.

        The third value says whether the intent read completed. A native
        service with no active intent deliberately suppresses static replica
        convergence: the autoscaler must establish the desired revision before
        the reconciler can write replicas. ``executed`` → ``observed`` →
        ``verified`` is driven only by positive observer evidence. Simulation,
        failure, and terminal states never become convergence targets.
        """

        spec = want.scaling
        if spec is None or want.operator_override:
            return want.replicas, None, True
        if spec.controller_mode != SCALE_CONTROLLER_NATIVE:
            return None, None, True
        intent, complete = self._resolve_valid_intent(want)
        if intent is None:
            return None, None, complete
        status = str(intent.get("status") or "")
        if status in {
            _SCALE_INTENT_PROPOSED,
            _SCALE_INTENT_SIMULATED,
            _SCALE_INTENT_FAILED,
            _SCALE_INTENT_REJECTED,
            _SCALE_INTENT_SUPERSEDED,
            _SCALE_INTENT_VERIFIED,
        }:
            return None, intent, True
        if status == _SCALE_INTENT_ACCEPTED:
            return int(intent["desired_replicas"]), intent, True
        if status in {_SCALE_INTENT_EXECUTED, _SCALE_INTENT_OBSERVED}:
            return self._advance_executed_observed(
                want, intent, status, observed_replicas
            )
        if status == _SCALE_INTENT_RECOVERY_PENDING:
            return self._advance_recovery_pending(want, intent, observed_replicas)
        # Unknown state is never an authorization to write replicas.
        return None, intent, True

    def _advance_executed_observed(
        self,
        want: DesiredService,
        intent: dict[str, Any],
        status: str,
        observed_replicas: int | None,
    ) -> tuple[int | None, dict[str, Any] | None, bool]:
        desired = int(intent["desired_replicas"])
        if observed_replicas != desired:
            # A real execution is in flight from the control-plane point
            # of view. Never issue a competing convergence action while
            # the observer is stale or reports a failed mutation.
            return None, intent, True
        next_status = (
            _SCALE_INTENT_OBSERVED
            if status == _SCALE_INTENT_EXECUTED
            else _SCALE_INTENT_VERIFIED
        )
        transition = self.intent_store.cas(
            {
                "operation": "transition",
                "service": want.name,
                "intent_id": intent["intent_id"],
                "expected_revision": int(intent["revision"]),
                "status": next_status,
                "observed_replicas": observed_replicas,
                "updated_unix": time.time(),
            }
        )
        if not _cas_succeeded(transition):
            return None, intent, False
        advanced = dict(intent)
        advanced["status"] = next_status
        advanced["observed_replicas"] = observed_replicas
        return None, advanced, True

    def _complete_recovery_outbox_from_intent(
        self, intent: dict[str, Any], observed_replicas: int | None
    ) -> dict[str, Any] | None:
        outbox = self.action_outbox_store
        if outbox is None:
            from agent_utilities.orchestration.fleet_actuation import (
                EngineActionOutboxStore,
            )

            outbox = EngineActionOutboxStore(self.engine)
        try:
            return outbox.complete(
                {
                    "operation": "reconcile",
                    "idempotency_key": str(
                        intent.get("idempotency_key") or intent["intent_id"]
                    ),
                    "execution_id": str(intent.get("execution_id") or ""),
                    "request_digest": str(intent.get("request_digest") or ""),
                    "state": _SCALE_INTENT_OBSERVED,
                    "ok": True,
                    "dry_run": False,
                    "observed_replicas": observed_replicas,
                    "recovery": True,
                }
            )
        except Exception as exc:  # noqa: BLE001 — recovery evidence is authoritative
            logger.warning("fleet action outbox recovery failed: %s", exc)
            return None

    def _advance_recovery_pending(
        self,
        want: DesiredService,
        intent: dict[str, Any],
        observed_replicas: int | None,
    ) -> tuple[int | None, dict[str, Any] | None, bool]:
        desired = int(intent["desired_replicas"])
        if observed_replicas != desired:
            return None, intent, True
        completion = self._complete_recovery_outbox_from_intent(
            intent, observed_replicas
        )
        if completion is None:
            return None, intent, False
        if not _outbox_completion_matches(completion, _SCALE_INTENT_OBSERVED):
            return None, intent, False
        transition = self.intent_store.cas(
            {
                "operation": "transition",
                "service": want.name,
                "intent_id": intent["intent_id"],
                "expected_revision": int(intent["revision"]),
                "status": _SCALE_INTENT_OBSERVED,
                "observed_replicas": observed_replicas,
                "updated_unix": time.time(),
            }
        )
        if not _cas_succeeded(transition):
            return None, intent, False
        advanced = dict(intent)
        advanced["status"] = _SCALE_INTENT_OBSERVED
        advanced["observed_replicas"] = observed_replicas
        return None, advanced, True

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
        observed = self._observe_fleet()

        proposals: list[ActionRequest] = []
        for name, want in sorted(desired.items()):
            obs = observed.get(name)
            if obs is None:
                continue  # no evidence — never act blind
            proposal = self._diff_one_service(name, want, obs)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def _observe_fleet(self) -> dict[str, Any]:
        try:
            return self.observer.observe() or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("fleet_reconciler: observer failed: %s", e)
            return {}

    def _diff_one_service(
        self, name: str, want: DesiredService, obs: Any
    ) -> ActionRequest | None:
        if want.desired == "stopped":
            return self._diff_stopped_service(name, want, obs)
        if obs.status == STATUS_DOWN:
            return self._bind_kubernetes_resource(
                ActionRequest(
                    kind="restart_service",
                    target=name,
                    params={"version": want.version} if want.version else {},
                    source="reconciler",
                    reason=f"observed down ({obs.detail})",
                ),
                want,
            )
        if want.scaling is not None and want.scaling.controller_mode in {
            SCALE_CONTROLLER_EXTERNAL_HPA,
            SCALE_CONTROLLER_EXTERNAL_KEDA,
        }:
            # HPA/KEDA owns replicas. AU may still restart a down service,
            # but it must never write a competing replica value.
            return None
        return self._diff_scale_service(name, want, obs)

    def _diff_stopped_service(
        self, name: str, want: DesiredService, obs: Any
    ) -> ActionRequest | None:
        if obs.status != STATUS_UP:
            return None
        return self._bind_kubernetes_resource(
            ActionRequest(
                kind="stop_service",
                target=name,
                source="reconciler",
                reason="desired stopped but observed up",
            ),
            want,
        )

    @staticmethod
    def _scale_action_params(
        intent: dict[str, Any] | None, obs: Any, target: int
    ) -> dict[str, Any]:
        # Record the direction the reconciler is actuating. The
        # autoscaler's own request carried it, but the reconciler
        # rebuilds this request from the intent and previously dropped
        # it -- which left `_should_watch` unable to tell a scale-up
        # from a scale-down, and made the audit row poorer than the
        # proposal it came from.
        params = {
            "replicas": target,
            "from_replicas": obs.replicas,
            "direction": "up" if target > obs.replicas else "down",
        }
        if intent is not None and str(intent.get("status")) == _SCALE_INTENT_ACCEPTED:
            params.update(
                {
                    "scale_intent_id": intent.get("intent_id"),
                    "scale_intent_revision": int(intent["revision"]),
                }
            )
        return params

    def _diff_scale_service(
        self, name: str, want: DesiredService, obs: Any
    ) -> ActionRequest | None:
        target, intent, complete = self._intent_target(want, obs.replicas)
        if not complete or target is None:
            return None
        if not (
            obs.status == STATUS_UP
            and obs.replicas is not None
            and obs.replicas != target
        ):
            return None
        params = self._scale_action_params(intent, obs, target)
        return self._bind_kubernetes_resource(
            ActionRequest(
                kind="scale_service",
                target=name,
                params=params,
                source="intent-reconciler" if intent is not None else "reconciler",
                reason=(
                    f"intent_accepted scale intent revision {intent['revision']} "
                    f"requires replicas {target}"
                    if intent is not None
                    else f"replicas {obs.replicas} != desired {target}"
                ),
            ),
            want,
        )

    # ── convergence ─────────────────────────────────────────────────

    @staticmethod
    def _lacks_native_scale_authority(desired: DesiredService | None) -> bool:
        return (
            desired is None
            or desired.operator_override
            or desired.scaling is None
            or desired.scaling.controller_mode != SCALE_CONTROLLER_NATIVE
        )

    @classmethod
    def _native_scale_authority_reason(cls, target: str) -> str:
        """Empty when `target` declares native scale-intent authority, else why not."""
        desired = load_desired_state().get(target)
        if cls._lacks_native_scale_authority(desired):
            return "scale intent is not the declared native replica authority"
        return ""

    @staticmethod
    def _intent_fields_match(
        intent: dict[str, Any],
        target: str,
        intent_id: str,
        expected_revision: Any,
        replicas_param: Any,
    ) -> bool:
        return bool(
            str(intent.get("service")) == target
            and str(intent["status"]) == _SCALE_INTENT_ACCEPTED
            and str(intent["intent_id"]) == intent_id
            and int(intent["revision"]) == int(expected_revision)
            and int(intent["desired_replicas"]) == int(replicas_param)
        )

    @staticmethod
    def _scale_intent_matches(
        complete: bool,
        intent: dict[str, Any] | None,
        target: str,
        intent_id: str,
        expected_revision: Any,
        replicas_param: Any,
    ) -> bool:
        try:
            if expected_revision is None:
                raise ValueError("scale_intent_revision is required")
            if replicas_param is None:
                raise ValueError("replicas is required")
            if not complete or intent is None or not _intent_is_valid(intent):
                return False
            return FleetReconciler._intent_fields_match(
                intent, target, intent_id, expected_revision, replicas_param
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _authorized_intent(self, request: ActionRequest) -> tuple[bool, str]:
        intent_id = str(request.params.get("scale_intent_id") or "")
        if not intent_id:
            return False, ""
        authority_reason = self._native_scale_authority_reason(request.target)
        if authority_reason:
            return False, authority_reason
        complete, intent = self.intent_store.latest(request.target)
        expected_revision = request.params.get("scale_intent_revision")
        replicas_param = request.params.get("replicas")
        matches = self._scale_intent_matches(
            complete,
            intent,
            request.target,
            intent_id,
            expected_revision,
            replicas_param,
        )
        if not matches:
            return False, "accepted scale intent is stale, concurrent, or unavailable"
        return True, ""

    @staticmethod
    def _direct_scale_allowed(target: str) -> tuple[bool, str]:
        """Return whether a replica write may bypass a native intent.

        Static services and explicit operator replica overrides retain the
        ordinary reconciler path. A scaling block delegates replica ownership
        either to the native intent flow or to HPA/KEDA; neither may be
        bypassed by an old/manual approval or a caller-supplied action.
        Missing desired-state authority fails closed for a scale write.
        """

        desired = load_desired_state().get(target)
        if desired is None:
            return False, "scale target is not present in desired-state authority"
        spec = desired.scaling
        if spec is None:
            return True, ""
        if spec.controller_mode != SCALE_CONTROLLER_NATIVE:
            return False, f"replica authority delegated to {spec.controller_mode}"
        if desired.operator_override:
            return True, ""
        return False, "native scale requires an accepted ScaleIntent"

    def _converge_one(self, request: ActionRequest) -> dict[str, Any]:
        # The resource registry is the only source allowed to bind a
        # Kubernetes identity into an action.  Do this before policy/outbox
        # digesting so UID/resourceVersion/controller ownership are fenced by
        # the same durable request identity that will be actuated.
        request = self._bind_kubernetes_resource(
            request, load_desired_state().get(request.target)
        )
        rejection = self._direct_scale_rejection(request)
        if rejection is not None:
            return rejection
        is_intent, intent_error = self._authorized_intent(request)
        if request.params.get("scale_intent_id"):
            return self._converge_via_intent(request, is_intent, intent_error)
        return self._converge_via_policy(request)

    def _direct_scale_rejection(self, request: ActionRequest) -> dict[str, Any] | None:
        if request.kind != "scale_service" or request.params.get("scale_intent_id"):
            return None
        direct_allowed, direct_reason = self._direct_scale_allowed(request.target)
        if direct_allowed:
            return None
        return {
            "kind": request.kind,
            "target": request.target,
            "reason": direct_reason,
            "decision": "rejected",
            "state": _SCALE_INTENT_REJECTED,
            "approval_id": None,
        }

    @staticmethod
    def _next_intent_status(execution: dict[str, Any]) -> str:
        if (
            execution.get("dry_run")
            or execution.get("state") == _SCALE_INTENT_SIMULATED
        ):
            return _SCALE_INTENT_SIMULATED
        if execution.get("state") == _SCALE_INTENT_RECOVERY_PENDING:
            return _SCALE_INTENT_RECOVERY_PENDING
        if execution.get("ok"):
            return _SCALE_INTENT_EXECUTED
        return _SCALE_INTENT_FAILED

    def _maybe_schedule_watch(
        self, request: ActionRequest, execution: dict[str, Any], state: str
    ) -> tuple[bool, Any]:
        if not (
            _should_watch(request, self.policy)
            and execution.get("ok")
            and state != _SCALE_INTENT_SIMULATED
        ):
            return False, None
        from agent_utilities.orchestration.deploy_watch import watch_deploy

        return True, watch_deploy(
            self.engine,
            request.target,
            version=str(request.params.get("version") or ""),
            source="reconciler",
        )

    def _converge_via_intent(
        self, request: ActionRequest, is_intent: bool, intent_error: str
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "kind": request.kind,
            "target": request.target,
            "reason": request.reason,
            "decision": "accepted_intent" if is_intent else "stale_intent",
            "state": _SCALE_INTENT_ACCEPTED if is_intent else "stale",
            "approval_id": None,
        }
        if not is_intent:
            entry["reason"] = intent_error
            return entry
        execution = execute_action(
            self.engine,
            request,
            self.actuator,
            outbox_store=self.action_outbox_store,
        )
        entry["execution"] = execution
        next_status = self._next_intent_status(execution)
        transition = self.intent_store.cas(
            {
                "operation": "transition",
                "service": request.target,
                "intent_id": request.params["scale_intent_id"],
                "expected_revision": int(request.params["scale_intent_revision"]),
                "status": next_status,
                "execution_id": execution.get("execution_id", ""),
                "idempotency_key": execution.get(
                    "idempotency_key", request.params["scale_intent_id"]
                ),
                "request_digest": execution.get("request_digest", ""),
                "executed_unix": execution.get("executed_unix", time.time()),
                "updated_unix": time.time(),
            }
        )
        entry["state"] = (
            next_status if _cas_succeeded(transition) else "transition_conflict"
        )
        entry["intent_transition"] = (
            next_status if _cas_succeeded(transition) else "conflict"
        )
        # The accepted-intent branch is the PRIMARY actuation path for
        # native autoscaling, and it carried no health watch at all -- the
        # scale-up watch the autoscaler used to schedule was lost when
        # actuation moved here. Same predicate and same simulated/ok guards
        # as the policy-decision path below, so a dry run still never
        # schedules one.
        scheduled, watch_job = self._maybe_schedule_watch(
            request, execution, next_status
        )
        if scheduled:
            entry["watch_job"] = watch_job
        return entry

    def _converge_via_policy(self, request: ActionRequest) -> dict[str, Any]:
        decision = self.policy.decide(request)
        entry: dict[str, Any] = {
            "kind": request.kind,
            "target": request.target,
            "reason": request.reason,
            "decision": decision.decision,
            "approval_id": decision.approval_id,
        }
        if not decision.allowed:
            return entry
        entry["execution"] = execute_action(
            self.engine,
            request,
            self.actuator,
            outbox_store=self.action_outbox_store,
        )
        entry["state"] = entry["execution"].get(
            "state",
            _SCALE_INTENT_SIMULATED
            if entry["execution"].get("dry_run")
            else _SCALE_INTENT_EXECUTED
            if entry["execution"].get("ok")
            else _SCALE_INTENT_FAILED,
        )
        scheduled, watch_job = self._maybe_schedule_watch(
            request, entry["execution"], entry["state"]
        )
        if scheduled:
            entry["watch_job"] = watch_job
        return entry

    def _accept_approved_scale_intent(self, request: ActionRequest) -> dict[str, Any]:
        """Convert a granted autoscaler approval into an accepted intent.

        Approval draining is deliberately not an actuator path for native
        autoscaling. The reconciler's normal intent path performs the only
        replica mutation after this CAS succeeds.
        """

        desired_state = load_desired_state()
        want = desired_state.get(request.target)
        if self._lacks_native_scale_authority(want):
            return {"ok": False, "detail": "scale intent is not native-authorized"}
        complete, intent = self.intent_store.latest(request.target)
        intent_id = str(request.params.get("scale_intent_id") or "")
        revision, identity_matches = self._scale_intent_identity_matches(
            complete, intent, request.target, intent_id, request.params
        )
        if not identity_matches or not isinstance(intent, dict):
            return {"ok": False, "detail": "scale intent is stale or concurrent"}
        if str(intent.get("status")) == _SCALE_INTENT_ACCEPTED:
            # The intent CAS may have committed before the approval status
            # stamp was lost. Replaying the approval is an acceptance no-op;
            # the drain can safely repair only the approval row.
            return {
                "ok": True,
                "dry_run": False,
                "intent_accepted": True,
                "replayed": True,
                "state": _SCALE_INTENT_ACCEPTED,
                "detail": "scale intent was already accepted",
            }
        if str(intent.get("status")) != _SCALE_INTENT_PROPOSED:
            return {"ok": False, "detail": "scale intent is no longer approvable"}
        result = self._transition_scale_intent_to_accepted(
            request.target, intent_id, revision
        )
        if not _cas_succeeded(result):
            return {"ok": False, "detail": "scale intent acceptance CAS conflicted"}
        return {
            "ok": True,
            "dry_run": False,
            "intent_accepted": True,
            "state": _SCALE_INTENT_ACCEPTED,
            "detail": "approved scale intent accepted for reconciler actuation",
        }

    @staticmethod
    def _scale_intent_identity_matches(
        complete: bool,
        intent: Any,
        target: str,
        intent_id: str,
        params: dict[str, Any],
    ) -> tuple[int, bool]:
        try:
            revision = int(params["scale_intent_revision"])
            replicas = int(params["replicas"])
            identity_matches = (
                complete
                and isinstance(intent, dict)
                and str(intent.get("service")) == target
                and str(intent.get("intent_id")) == intent_id
                and int(intent["revision"]) == revision
                and int(intent["desired_replicas"]) == replicas
            )
        except (KeyError, TypeError, ValueError):
            identity_matches = False
            revision = -1
        return revision, identity_matches

    def _transition_scale_intent_to_accepted(
        self, target: str, intent_id: str, revision: int
    ) -> dict[str, Any]:
        return self.intent_store.cas(
            {
                "operation": "transition",
                "service": target,
                "intent_id": intent_id,
                "expected_revision": revision,
                "status": _SCALE_INTENT_ACCEPTED,
                "updated_unix": time.time(),
            }
        )

    def _reconcile_recovery_approval(
        self, request: ActionRequest, execution: dict[str, Any], approval_id: str
    ) -> dict[str, Any]:
        """Close an ambiguous approval only after positive observer evidence."""

        if execution.get("state") != _SCALE_INTENT_RECOVERY_PENDING:
            return execution
        try:
            observed = (self.observer.observe() or {}).get(request.target)
        except Exception as exc:  # noqa: BLE001 — recovery reads fail closed
            logger.warning("fleet recovery observation failed: %s", exc)
            return execution
        if observed is None:
            return execution
        if not self._recovery_confirmed(request, observed):
            return execution
        completion = self._complete_recovery_outbox(request, execution, approval_id)
        if completion is None:
            return execution
        if not _outbox_completion_matches(completion, _SCALE_INTENT_OBSERVED):
            return execution
        if not bool(completion.get("approval_committed")):
            return execution
        return {
            **execution,
            "ok": True,
            "state": _SCALE_INTENT_OBSERVED,
            "real_execution": True,
            "approval_committed": True,
            "outbox_status": _SCALE_INTENT_OBSERVED,
            "observed": True,
        }

    @staticmethod
    def _recovery_confirmed(request: ActionRequest, observed: Any) -> bool:
        if request.kind == "scale_service":
            try:
                replicas_param = request.params.get("replicas")
                if replicas_param is None:
                    raise ValueError("replicas is required")
                return bool(
                    observed.status == STATUS_UP
                    and observed.replicas is not None
                    and int(observed.replicas) == int(replicas_param)
                )
            except (TypeError, ValueError):
                return False
        if request.kind == "stop_service":
            return observed.status == STATUS_DOWN
        return observed.status == STATUS_UP

    def _complete_recovery_outbox(
        self, request: ActionRequest, execution: dict[str, Any], approval_id: str
    ) -> dict[str, Any] | None:
        outbox = self.action_outbox_store
        if outbox is None:
            from agent_utilities.orchestration.fleet_actuation import (
                EngineActionOutboxStore,
            )

            outbox = EngineActionOutboxStore(self.engine)
        try:
            return outbox.complete(
                {
                    "operation": "reconcile",
                    "idempotency_key": execution.get("idempotency_key", ""),
                    "execution_id": execution.get("execution_id", ""),
                    "request_digest": _action_request_digest(request),
                    "state": _SCALE_INTENT_OBSERVED,
                    "ok": True,
                    "dry_run": False,
                    "approval_id": approval_id,
                    "approval_status": _SCALE_INTENT_OBSERVED,
                    "observed": True,
                }
            )
        except Exception as exc:  # noqa: BLE001 — completion remains pending
            logger.warning("fleet recovery completion failed: %s", exc)
            return None

    def _scan_approved_candidates(self) -> list[Any]:
        try:
            return self.engine.query_cypher(
                "MATCH (a:ActionApproval {status: 'approved'}) "
                f"RETURN a LIMIT {_APPROVAL_DRAIN_LIMIT}"
            )
        except Exception as e:  # noqa: BLE001 — read-only candidate scan; on failure nothing is mutated (no approvals drained, budget untouched) and every approved row is re-selected on the next tick
            logger.debug("fleet_reconciler: approval drain scan failed: %s", e)
            return []

    def _build_approval_request(self, props: dict[str, Any]) -> ActionRequest:
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
        return self._bind_kubernetes_resource(
            request, load_desired_state().get(request.target)
        )

    def _execute_direct_scale_approval(
        self, request: ActionRequest, props: dict[str, Any]
    ) -> dict[str, Any]:
        direct_allowed, direct_reason = self._direct_scale_allowed(request.target)
        if not direct_allowed:
            return {
                "ok": False,
                "dry_run": False,
                "state": _SCALE_INTENT_REJECTED,
                "detail": direct_reason,
            }
        return execute_action(
            self.engine,
            request,
            self.actuator,
            outbox_store=self.action_outbox_store,
            idempotency_key=f"approval:{props['id']}",
            approval_id=str(props["id"]),
        )

    def _execute_approved_action(
        self, request: ActionRequest, props: dict[str, Any]
    ) -> dict[str, Any]:
        if request.kind == "scale_service" and request.params.get("scale_intent_id"):
            return self._accept_approved_scale_intent(request)
        if request.kind == "scale_service":
            return self._execute_direct_scale_approval(request, props)
        return execute_action(
            self.engine,
            request,
            self.actuator,
            outbox_store=self.action_outbox_store,
            idempotency_key=f"approval:{props['id']}",
            approval_id=str(props["id"]),
        )

    @staticmethod
    def _new_approval_status(execution: dict[str, Any]) -> str:
        if execution.get("intent_accepted"):
            return _SCALE_INTENT_ACCEPTED
        state = execution.get("state")
        if state == _SCALE_INTENT_SIMULATED:
            return _SCALE_INTENT_SIMULATED
        if state == _SCALE_INTENT_REJECTED:
            return _SCALE_INTENT_REJECTED
        if state == _SCALE_INTENT_RECOVERY_PENDING:
            return _SCALE_INTENT_RECOVERY_PENDING
        if state == _SCALE_INTENT_OBSERVED:
            return _SCALE_INTENT_OBSERVED
        if execution.get("ok"):
            return "executed"
        return "failed"

    def _stamp_approval_if_needed(
        self, props: dict[str, Any], new_status: str, execution: dict[str, Any]
    ) -> None:
        # A real action's outbox completion is the atomic approval-close
        # boundary. Never perform a second best-effort approval write after
        # the side effect: if completion was lost, leave the row approved
        # so the next delivery retries completion without re-running the
        # actuator. Dry-runs, intent acceptance, rejections, and legacy
        # non-actuating paths still use the existing status stamp.
        stamp_approval = not execution.get("outbox_prepared") and not execution.get(
            "durability_unavailable"
        )
        if not stamp_approval:
            return
        for _attempt in (1, 2):
            try:
                self.engine.backend.execute(
                    "MATCH (a:ActionApproval {id: $id}) "
                    "SET a.status = $status, a.executed_at = $ts",
                    {
                        "id": props["id"],
                        "status": new_status,
                        "ts": _now_iso(),
                    },
                )
                break
            except Exception as e:  # noqa: BLE001 — retry once for non-outbox status-only paths
                if _attempt == 2:
                    logger.warning(
                        "fleet_reconciler: approval %s stamp failed twice: %s",
                        props.get("id"),
                        e,
                    )
                else:
                    logger.debug(
                        "fleet_reconciler: approval stamp failed, retrying once: %s",
                        e,
                    )

    def _watch_approved_action(
        self, request: ActionRequest, execution: dict[str, Any]
    ) -> None:
        if not (
            _should_watch(request, self.policy)
            and execution.get("ok")
            and execution.get("state") != _SCALE_INTENT_SIMULATED
        ):
            return
        from agent_utilities.orchestration.deploy_watch import watch_deploy

        watch_deploy(
            self.engine,
            request.target,
            version=str(request.params.get("version") or ""),
            source="approval",
        )

    def _process_one_approval(self, props: dict[str, Any]) -> dict[str, Any]:
        request = self._build_approval_request(props)
        execution = self._execute_approved_action(request, props)
        execution = self._reconcile_recovery_approval(
            request, execution, str(props["id"])
        )
        new_status = self._new_approval_status(execution)
        self._stamp_approval_if_needed(props, new_status, execution)
        self._watch_approved_action(request, execution)
        return {
            "approval_id": props["id"],
            "kind": request.kind,
            "target": request.target,
            "status": new_status,
            "execution": execution,
        }

    @staticmethod
    def _approval_candidate_props(row: Any) -> dict[str, Any] | None:
        """None means: skip this row — invalid, or not actuatable via this drain path."""
        props = row.get("a") if isinstance(row, dict) else None
        if not isinstance(props, dict) or not props.get("id"):
            return None
        if str(props.get("kind") or "") == "merge_promotion":
            # Code-evolution publications are NOT fleet actuations: a
            # granted merge_promotion approval is consumed by the
            # evolution→branch bridge's ``publish_proposal`` action
            # (CONCEPT:AU-AHE.harness.evolution-branch-bridge), never by the fleet actuator — which
            # would dry-run/fail it and silently eat the grant.
            return None
        return props

    def _drain_approved(self, budget: int) -> list[dict[str, Any]]:
        """Execute fleet actions a human approved via /api/fleet/approvals/grant."""
        if budget <= 0 or self.engine is None:
            return []
        rows = self._scan_approved_candidates()
        drained: list[dict[str, Any]] = []
        for row in rows or []:
            if budget <= 0:
                break
            props = self._approval_candidate_props(row)
            if props is None:
                continue
            drained.append(self._process_one_approval(props))
            budget -= 1
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
    :func:`~agent_utilities.knowledge_graph.core.work_durability.ensure_agent_task_work_item`
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

    from agent_utilities.knowledge_graph.core.work_durability import (
        ensure_agent_task_work_item,
    )

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
