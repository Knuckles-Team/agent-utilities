#!/usr/bin/python
from __future__ import annotations

"""Dynamic Topology Engine (CONCEPT:AU-ORCH.execution.dynamic-topology-materialization).

Replaces static ``create_graph_agent()`` topology with KG-driven dynamic
graph materialization.  Instead of all execution paths existing simultaneously,
the engine selects and materializes only the relevant subgraph based on:

    - Task domain (general, finance, medical, legal, government)
    - Task complexity (1-5 scale)
    - KG-stored ``TopologyTemplateNode`` success rates
    - Available adaptive_agent_router and tools

The engine supports all pydantic-graph execution patterns:
    - **Sequential**: A → B → C (simple pipeline)
    - **Parallel**: [A, B] → C (fan-out/fan-in)
    - **Mixed**: A → [B, C] → D → [E, F] → G (arbitrary DAG)
    - **Fan-out**: A → [B₁, B₂, ..., Bₙ] (scatter)
    - **Fan-in**: [B₁, B₂, ..., Bₙ] → C (gather)

Each materialized topology creates isolated subgraph instances with:
    - Per-specialist system prompts
    - Per-specialist MCP tool assignments
    - Per-specialist model selection
    - Shared KG memory channels for P2P communication
"""


import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..models.knowledge_graph import (
    TeamComposition,
)

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class TopologyAdmissionError(ValueError):
    """A topology or sandbox request exceeded its immutable admission contract."""


def _check_sandbox_int_types(
    memory_bytes: object, max_pids: object, max_wasm_pages: object
) -> None:
    """Type-check the strictly-integer fields of
    ``SandboxResourceLimits.__post_init__``."""
    if (
        isinstance(memory_bytes, bool)
        or not isinstance(memory_bytes, int)
        or isinstance(max_pids, bool)
        or not isinstance(max_pids, int)
        or isinstance(max_wasm_pages, bool)
        or not isinstance(max_wasm_pages, int)
    ):
        raise TopologyAdmissionError("sandbox integer limits must be integers")


def _check_sandbox_float_types(cpu_cores: object, deadline_s: object) -> None:
    """Type-check the int-or-float fields of
    ``SandboxResourceLimits.__post_init__``."""
    if (
        isinstance(cpu_cores, bool)
        or not isinstance(cpu_cores, int | float)
        or isinstance(deadline_s, bool)
        or not isinstance(deadline_s, int | float)
    ):
        raise TopologyAdmissionError("sandbox integer limits must be integers")


def _check_sandbox_limit_types(
    memory_bytes: object,
    max_pids: object,
    max_wasm_pages: object,
    cpu_cores: object,
    deadline_s: object,
) -> None:
    """Type-check ``SandboxResourceLimits.__post_init__``'s raw fields before
    any numeric coercion or range check runs."""
    _check_sandbox_int_types(memory_bytes, max_pids, max_wasm_pages)
    _check_sandbox_float_types(cpu_cores, deadline_s)


def _check_sandbox_limit_ranges(
    cpu: float, memory: int, pids: int, pages: int, deadline: float
) -> None:
    """Range-check ``SandboxResourceLimits.__post_init__``'s coerced fields."""
    if not math.isfinite(cpu) or not 0.1 <= cpu <= 16.0:
        raise TopologyAdmissionError("sandbox CPU limit is out of range")
    if not 64 * 1024 * 1024 <= memory <= 64 * 1024 * 1024 * 1024:
        raise TopologyAdmissionError("sandbox memory limit is out of range")
    if not 16 <= pids <= 1_024:
        raise TopologyAdmissionError("sandbox PID limit is out of range")
    if not 1 <= pages <= 1_048_576:
        raise TopologyAdmissionError("sandbox WASM page limit is out of range")
    if pages * 65_536 > memory:
        raise TopologyAdmissionError("sandbox WASM page limit exceeds the memory limit")
    if not math.isfinite(deadline) or not 1.0 <= deadline <= 600.0:
        raise TopologyAdmissionError("sandbox deadline is out of range")


@dataclass(frozen=True, slots=True)
class SandboxResourceLimits:
    """Actual per-sandbox resource ceilings carried by one admission.

    The limits are deliberately a value object rather than a second mutable
    policy.  Container backends translate them to kernel/runtime limits,
    forkserver applies the process limits before user code starts, and
    Wasmtime applies the memory/page and epoch/fuel limits to the real store.
    A backend that cannot apply a requested limit must fail closed; recording a
    number in metadata is not enforcement.
    """

    cpu_cores: float = 1.0
    memory_bytes: int = 512 * 1024 * 1024
    max_pids: int = 256
    max_wasm_pages: int = 8_192
    deadline_s: float = 120.0

    def __post_init__(self) -> None:
        _check_sandbox_limit_types(
            self.memory_bytes,
            self.max_pids,
            self.max_wasm_pages,
            self.cpu_cores,
            self.deadline_s,
        )
        try:
            cpu = float(self.cpu_cores)
            memory = self.memory_bytes
            pids = self.max_pids
            pages = self.max_wasm_pages
            deadline = float(self.deadline_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TopologyAdmissionError("sandbox resource limits are invalid") from exc
        _check_sandbox_limit_ranges(cpu, memory, pids, pages, deadline)
        object.__setattr__(self, "cpu_cores", cpu)
        object.__setattr__(self, "memory_bytes", memory)
        object.__setattr__(self, "max_pids", pids)
        object.__setattr__(self, "max_wasm_pages", pages)
        object.__setattr__(self, "deadline_s", deadline)

    def as_dict(self) -> dict[str, int | float]:
        """Return only the runtime limits needed by a sandbox adapter."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_bytes": self.memory_bytes,
            "max_pids": self.max_pids,
            "max_wasm_pages": self.max_wasm_pages,
            "deadline_s": self.deadline_s,
        }


def _check_tenant_and_delegation(
    tenant_raw: object, delegation_id_raw: object
) -> tuple[str, str]:
    """Normalize and validate ``ElasticTopologyAdmission.__post_init__``'s
    ``tenant``/``delegation_id`` fields."""
    if not isinstance(tenant_raw, str) or not isinstance(delegation_id_raw, str):
        raise TopologyAdmissionError("tenant and delegation_id must be strings")
    tenant = tenant_raw.strip()
    delegation_id = delegation_id_raw.strip()
    if not tenant or not delegation_id:
        raise TopologyAdmissionError(
            "tenant and delegation_id are required for topology admission"
        )
    if any(ord(c) < 32 or ord(c) == 127 for c in tenant + delegation_id):
        raise TopologyAdmissionError(
            "tenant and delegation_id contain control characters"
        )
    return tenant, delegation_id


def _check_capabilities(capabilities_raw: Sequence[str]) -> tuple[str, ...]:
    """Normalize and validate ``ElasticTopologyAdmission.__post_init__``'s
    ``capabilities`` field."""
    if any(not isinstance(value, str) for value in capabilities_raw):
        raise TopologyAdmissionError("capability identifiers must be strings")
    capabilities = tuple(sorted({value.strip() for value in capabilities_raw}))
    if any(not value or any(ord(c) < 32 for c in value) for value in capabilities):
        raise TopologyAdmissionError(
            "capability identifiers must be non-empty and printable"
        )
    return capabilities


def _check_integer_limits(
    max_nodes: int,
    max_depth: int,
    max_fan_out: int,
    max_parallelism: int,
    max_tokens: int,
    max_payload_bytes: int,
) -> None:
    """Range-check ``ElasticTopologyAdmission.__post_init__``'s six bounded
    integer limits."""
    integer_limits = {
        "max_nodes": (max_nodes, 1, 1_024),
        "max_depth": (max_depth, 1, 64),
        "max_fan_out": (max_fan_out, 1, 256),
        "max_parallelism": (max_parallelism, 1, 256),
        "max_tokens": (max_tokens, 1, 10_000_000),
        "max_payload_bytes": (max_payload_bytes, 1, 64 * 1024 * 1024),
    }
    for name, (raw, lower, upper) in integer_limits.items():
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int)
            or not lower <= raw <= upper
        ):
            raise TopologyAdmissionError(f"{name} is out of range")


def _compute_issued_and_deadline(
    issued_at_raw: object,
    deadline_unix_raw: float | None,
    resource_limits: SandboxResourceLimits,
) -> tuple[float, float]:
    """Coerce and validate ``ElasticTopologyAdmission.__post_init__``'s
    ``issued_at``/``deadline_unix`` pair."""
    try:
        issued_at = float(issued_at_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TopologyAdmissionError("issued_at must be finite") from exc
    if not math.isfinite(issued_at):
        raise TopologyAdmissionError("issued_at must be finite")
    try:
        deadline = (
            issued_at + resource_limits.deadline_s
            if deadline_unix_raw is None
            else float(deadline_unix_raw)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise TopologyAdmissionError("deadline_unix must be finite") from exc
    if (
        not math.isfinite(deadline)
        or deadline <= issued_at
        or deadline > issued_at + resource_limits.deadline_s
    ):
        raise TopologyAdmissionError("deadline_unix must be after issued_at")
    return issued_at, deadline


def _check_schema_version(schema_version: object) -> None:
    """Validate ``ElasticTopologyAdmission.__post_init__``'s ``schema_version``
    field."""
    if (
        not isinstance(schema_version, str)
        or not schema_version.strip()
        or any(ord(c) < 32 or ord(c) == 127 for c in schema_version)
    ):
        raise TopologyAdmissionError("schema_version must be printable and non-empty")


def _collect_topology_roles(specialists: Sequence[Mapping[str, Any]]) -> set[str]:
    """Validate and collect the unique role set for
    ``ElasticTopologyAdmission.require_topology``."""
    roles: list[str] = []
    for specialist in specialists:
        if not isinstance(specialist, Mapping):
            raise TopologyAdmissionError("topology nodes must be mappings")
        raw_role = specialist.get("role", "")
        if not isinstance(raw_role, str):
            raise TopologyAdmissionError("topology node roles must be strings")
        role = raw_role.strip()
        if not role:
            raise TopologyAdmissionError("every topology node requires a role")
        roles.append(role)
    if len(set(roles)) != len(roles):
        raise TopologyAdmissionError("topology roles must be unique")
    return set(roles)


def _check_step_fanout(
    step: Mapping[str, Any],
    step_roles: Sequence[str],
    max_fan_out: int,
    max_parallelism: int,
) -> None:
    """Fan-out/parallelism half of ``_check_topology_step``."""
    fan_out = len(step_roles)
    if fan_out > max_fan_out:
        raise TopologyAdmissionError(
            f"topology fan-out exceeds admission ({fan_out} > {max_fan_out})"
        )
    if str(step.get("mode", "")) == "parallel" and fan_out > max_parallelism:
        raise TopologyAdmissionError(
            f"topology parallelism exceeds admission ({fan_out} > {max_parallelism})"
        )


def _check_step_roles(step_roles: Sequence[str], role_set: set[str]) -> None:
    """Role type/uniqueness/membership half of ``_check_topology_step``."""
    if any(not isinstance(role, str) for role in step_roles):
        raise TopologyAdmissionError("topology step roles must be strings")
    if len(set(step_roles)) != len(step_roles):
        raise TopologyAdmissionError("topology step roles must be unique")
    if any(str(role) not in role_set for role in step_roles):
        raise TopologyAdmissionError("topology plan references an unknown role")


def _check_topology_step(
    step: Mapping[str, Any],
    role_set: set[str],
    max_fan_out: int,
    max_parallelism: int,
) -> None:
    """Validate one plan step for
    ``ElasticTopologyAdmission.require_topology``."""
    if not isinstance(step, Mapping):
        raise TopologyAdmissionError("topology steps must be mappings")
    step_roles = step.get("roles", ())
    if not isinstance(step_roles, Sequence) or isinstance(step_roles, (str, bytes)):
        raise TopologyAdmissionError("topology step roles must be a sequence")
    _check_step_fanout(step, step_roles, max_fan_out, max_parallelism)
    _check_step_roles(step_roles, role_set)


def _check_parallel_group(
    group: Sequence[str],
    role_set: set[str],
    max_parallelism: int,
    grouped_roles: set[str],
) -> None:
    """Validate one parallel group for
    ``ElasticTopologyAdmission.require_topology``."""
    if len(group) > max_parallelism:
        raise TopologyAdmissionError(
            f"parallel group exceeds admission ({len(group)} > {max_parallelism})"
        )
    if any(str(role) not in role_set for role in group):
        raise TopologyAdmissionError("parallel group references an unknown role")
    if len(set(group)) != len(group) or grouped_roles.intersection(group):
        raise TopologyAdmissionError("parallel groups must be disjoint and unique")


@dataclass(frozen=True, slots=True)
class ElasticTopologyAdmission:
    """Immutable tenant/delegation/capability/budget admission for elastic work.

    This is the one contract shared by dynamic topology materialization, RLM
    recursion/fan-out, and sandbox resource adapters.  It is content-addressed
    so a resumed or retried run cannot silently adopt a newer budget.  Live
    callers should construct it from their verified session/delegation and
    pass the same object through the complete run; the ``local`` constructor is
    only for isolated, engine-less unit use.
    """

    tenant: str = ""
    delegation_id: str = ""
    capabilities: tuple[str, ...] = ()
    max_nodes: int = 64
    max_depth: int = 8
    max_fan_out: int = 16
    max_parallelism: int = 16
    max_tokens: int = 500_000
    max_payload_bytes: int = 4 * 1024 * 1024
    issued_at: float = field(default_factory=time.time)
    deadline_unix: float | None = None
    resource_limits: SandboxResourceLimits = field(
        default_factory=SandboxResourceLimits
    )
    schema_version: str = "elastic-topology-admission.v1"

    def __post_init__(self) -> None:
        tenant, delegation_id = _check_tenant_and_delegation(
            self.tenant, self.delegation_id
        )
        capabilities = _check_capabilities(self.capabilities)
        _check_integer_limits(
            self.max_nodes,
            self.max_depth,
            self.max_fan_out,
            self.max_parallelism,
            self.max_tokens,
            self.max_payload_bytes,
        )
        if not isinstance(self.resource_limits, SandboxResourceLimits):
            raise TopologyAdmissionError(
                "resource_limits must be SandboxResourceLimits"
            )
        issued_at, deadline = _compute_issued_and_deadline(
            self.issued_at, self.deadline_unix, self.resource_limits
        )
        _check_schema_version(self.schema_version)

        object.__setattr__(self, "tenant", tenant)
        object.__setattr__(self, "delegation_id", delegation_id)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "deadline_unix", deadline)

    @classmethod
    def local(cls, *, issued_at: float | None = None) -> ElasticTopologyAdmission:
        """Create the bounded, non-engine admission used by isolated tests."""
        kwargs: dict[str, Any] = {}
        if issued_at is not None:
            kwargs["issued_at"] = issued_at
        return cls(
            tenant="local",
            delegation_id="local:rlm",
            capabilities=("rlm.execute", "topology.materialize"),
            **kwargs,
        )

    @property
    def digest(self) -> str:
        """Stable identity of the exact authority and every enforced bound."""
        payload = {
            "schema_version": self.schema_version,
            "tenant": self.tenant,
            "delegation_id": self.delegation_id,
            "capabilities": list(self.capabilities),
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "max_fan_out": self.max_fan_out,
            "max_parallelism": self.max_parallelism,
            "max_tokens": self.max_tokens,
            "max_payload_bytes": self.max_payload_bytes,
            "issued_at": self.issued_at,
            "deadline_unix": self.deadline_unix,
            "resource_limits": self.resource_limits.as_dict(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def require_capabilities(self, required: Sequence[str]) -> None:
        missing = sorted(set(required) - set(self.capabilities))
        if missing:
            raise TopologyAdmissionError(
                f"delegation {self.delegation_id!r} lacks capabilities: {', '.join(missing)}"
            )

    def remaining_seconds(self, *, now: float | None = None) -> float:
        try:
            current = time.time() if now is None else float(now)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TopologyAdmissionError("admission clock value is invalid") from exc
        if not math.isfinite(current):
            raise TopologyAdmissionError("admission clock value is not finite")
        if self.deadline_unix is None:
            raise TopologyAdmissionError("admission has no resolved deadline")
        remaining = self.deadline_unix - current
        if remaining <= 0:
            raise TopologyAdmissionError("topology admission deadline expired")
        return remaining

    @staticmethod
    def payload_size(payload: Any) -> int:
        if isinstance(payload, bytes):
            return len(payload)
        if isinstance(payload, str):
            return len(payload.encode("utf-8"))
        try:
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=None,
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise TopologyAdmissionError(
                "payload is not deterministically serializable"
            ) from exc
        return len(encoded)

    def require_payload(self, payload: Any, *, label: str = "payload") -> int:
        size = self.payload_size(payload)
        if size > self.max_payload_bytes:
            raise TopologyAdmissionError(
                f"{label} exceeds admission payload limit ({size} > {self.max_payload_bytes})"
            )
        return size

    def work_item_metadata(self) -> dict[str, Any]:
        """Render the queue-visible budget without creating a second authority."""
        return {
            "tenant": self.tenant,
            "deadline_unix": self.deadline_unix,
            "budget": {
                "max_nodes": self.max_nodes,
                "max_depth": self.max_depth,
                "max_fan_out": self.max_fan_out,
                "max_parallelism": self.max_parallelism,
                "max_tokens": self.max_tokens,
                "max_payload_bytes": self.max_payload_bytes,
            },
            "resource_limits": self.resource_limits.as_dict(),
            "metadata": {
                "admission_digest": self.digest,
                "delegation_id": self.delegation_id,
                "schema_version": self.schema_version,
            },
        }

    def _check_work_item_identity(self, item: Mapping[str, Any]) -> None:
        """Tenant/deadline half of ``require_work_item``."""
        if str(item.get("tenant") or "") != self.tenant:
            raise TopologyAdmissionError(
                "WorkItem tenant does not match topology admission"
            )
        if item.get("deadline_unix") != self.deadline_unix:
            raise TopologyAdmissionError(
                "WorkItem deadline does not match topology admission"
            )

    def _check_work_item_metadata(self, item: Mapping[str, Any]) -> Mapping[str, Any]:
        """Metadata-presence-and-match half of ``require_work_item``."""
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TopologyAdmissionError(
                "native WorkItem admission metadata is missing"
            )
        expected = self.work_item_metadata()
        for key in ("tenant", "deadline_unix", "budget", "resource_limits"):
            if metadata.get(key) != expected[key]:
                raise TopologyAdmissionError(
                    f"WorkItem {key} does not match topology admission"
                )
        return metadata

    def _check_work_item_digest(self, metadata: Mapping[str, Any]) -> None:
        """Digest-match half of ``require_work_item``."""
        nested = metadata.get("metadata")
        observed_digest = metadata.get("admission_digest")
        if isinstance(nested, Mapping):
            observed_digest = observed_digest or nested.get("admission_digest")
        if observed_digest != self.digest:
            raise TopologyAdmissionError("WorkItem admission digest does not match")

    def require_work_item(self, item: Mapping[str, Any] | None) -> None:
        """Require a native WorkItem to carry this exact admission identity."""
        if not isinstance(item, Mapping):
            raise TopologyAdmissionError("native WorkItem is missing")
        self._check_work_item_identity(item)
        metadata = self._check_work_item_metadata(item)
        self._check_work_item_digest(metadata)

    def require_topology(
        self,
        specialists: Sequence[Mapping[str, Any]],
        plan: Sequence[Mapping[str, Any]],
        *,
        parallel_groups: Sequence[Sequence[str]] = (),
    ) -> None:
        node_count = len(specialists)
        if node_count > self.max_nodes:
            raise TopologyAdmissionError(
                f"topology node count exceeds admission ({node_count} > {self.max_nodes})"
            )
        if len(plan) > self.max_depth:
            raise TopologyAdmissionError(
                f"topology depth exceeds admission ({len(plan)} > {self.max_depth})"
            )
        role_set = _collect_topology_roles(specialists)
        for step in plan:
            _check_topology_step(step, role_set, self.max_fan_out, self.max_parallelism)
        grouped_roles: set[str] = set()
        for group in parallel_groups:
            _check_parallel_group(group, role_set, self.max_parallelism, grouped_roles)
            grouped_roles.update(group)
        self.require_payload(specialists, label="topology specialists")


def _find_role_group(
    role: str, parallel_groups: list[list[str]], scheduled: set[str]
) -> list[str] | None:
    """First not-fully-scheduled parallel group containing ``role``, for
    ``_schedule_parallel_role``. ``None`` if no such group exists."""
    for group in parallel_groups:
        if role in group and not all(r in scheduled for r in group):
            return group
    return None


def _schedule_parallel_role(
    role: str,
    parallel_groups: list[list[str]],
    role_to_spec: dict[str, dict[str, Any]],
    scheduled: set[str],
    steps: list[dict[str, Any]],
    step_idx: int,
) -> int:
    """Schedule one parallel-group role for
    ``TopologyEngine._build_mixed_plan``. Appends to ``steps`` and
    ``scheduled`` in place; returns the (possibly advanced) ``step_idx``."""
    group = _find_role_group(role, parallel_groups, scheduled)
    if group is None:
        return step_idx

    group_specs = [
        role_to_spec[r] for r in group if r in role_to_spec and r not in scheduled
    ]
    if group_specs:
        steps.append(
            {
                "step": step_idx,
                "roles": [s["role"] for s in group_specs],
                "mode": "parallel",
                "agent_ids": [s.get("agent_id", s["role"]) for s in group_specs],
            }
        )
        step_idx += 1
        scheduled.update(s["role"] for s in group_specs)
    return step_idx


def _schedule_sequential_role(
    specialist: dict[str, Any],
    role: str,
    scheduled: set[str],
    steps: list[dict[str, Any]],
    step_idx: int,
) -> int:
    """Schedule one sequential role for
    ``TopologyEngine._build_mixed_plan``. Appends to ``steps`` and
    ``scheduled`` in place; returns the advanced ``step_idx``."""
    steps.append(
        {
            "step": step_idx,
            "roles": [role],
            "mode": "sequential",
            "agent_ids": [specialist.get("agent_id", role)],
        }
    )
    scheduled.add(role)
    return step_idx + 1


class TopologyEngine:
    """Materializes KG-stored topology templates into executable graphs.

    CONCEPT:AU-ORCH.execution.dynamic-topology-materialization — Dynamic Topology Materialization

    The engine is the bridge between the KG's declarative topology
    descriptions and the pydantic-graph's runtime execution model.

    Usage::

        engine = TopologyEngine(knowledge_engine)

        # Materialize a topology for a task
        materialized = engine.materialize(
            team_composition=composer.compose_team(query),
            session_id="sess:abc123",
        )

        # Get execution plan
        plan = materialized["execution_plan"]
        # => [{"step": 0, "role": "router", "mode": "sequential"}, ...]

    Args:
        engine: The IntelligenceGraphEngine for KG queries.
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        admission: ElasticTopologyAdmission | None = None,
    ):
        self.engine = engine
        self.admission = admission

    def materialize(
        self,
        team_composition: TeamComposition,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Materialize a team composition into an executable topology.

        Converts the declarative ``TeamComposition`` into an ordered
        execution plan that the pydantic-graph runner can consume.

        Args:
            team_composition: The composed team to materialize.
            session_id: Session ID for provenance tracking.

        Returns:
            A dict with:
                - ``execution_plan``: Ordered list of execution steps
                - ``specialist_configs``: Per-specialist configuration
                - ``memory_channels``: Shared KG channels
                - ``topology_id``: Template ID for tracking
        """
        adaptive_agent_router = team_composition.adaptive_agent_router
        mode = team_composition.execution_mode
        parallel_groups = team_composition.parallel_groups
        if self.admission is not None:
            self.admission.require_capabilities(("topology.materialize",))
            self.admission.remaining_seconds()
            if len(adaptive_agent_router) > self.admission.max_nodes:
                raise TopologyAdmissionError(
                    "topology node count exceeds admission before planning"
                )
        execution_plan = self._build_execution_plan(
            adaptive_agent_router, mode, parallel_groups
        )
        specialist_configs = self._build_specialist_configs(adaptive_agent_router)

        if self.admission is not None:
            self.admission.require_topology(
                adaptive_agent_router,
                execution_plan,
                parallel_groups=parallel_groups,
            )
            self.admission.require_payload(
                {
                    "execution_plan": execution_plan,
                    "specialist_configs": specialist_configs,
                    "memory_channels": team_composition.memory_channels,
                    "session_id": session_id,
                    "team_id": team_composition.team_id,
                },
                label="materialized topology payload",
            )
        elif self.engine is not None:
            # A live engine path cannot materialize an unscoped topology.  The
            # engine-less path remains usable for pure planning/unit callers.
            raise TopologyAdmissionError(
                "live topology materialization requires an immutable admission"
            )

        result = {
            "execution_plan": execution_plan,
            "specialist_configs": specialist_configs,
            "memory_channels": team_composition.memory_channels,
            "topology_id": team_composition.topology_template_id,
            "team_id": team_composition.team_id,
            "session_id": session_id,
            "execution_mode": mode,
            "materialized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if self.admission is not None:
            result["admission_digest"] = self.admission.digest
            result["tenant"] = self.admission.tenant
            result["delegation_id"] = self.admission.delegation_id
            result["resource_limits"] = self.admission.resource_limits.as_dict()
            result["work_item_metadata"] = self.admission.work_item_metadata()

        # Track in KG
        if self.engine:
            self._record_materialization(result)

        logger.info(
            "[CONCEPT:AU-ORCH.execution.dynamic-topology-materialization] Materialized topology: %d steps, "
            "%d adaptive_agent_router, mode=%s",
            len(execution_plan),
            len(specialist_configs),
            mode,
        )

        return result

    def retire(self, work_item_id: str, *, reason: str = "topology-retired") -> bool:
        """Retire a materialized run through the native WorkItem authority.

        Topology state is observational; it never grows a parallel lifecycle.
        A missing engine/item is a failed retirement, while the native cancel
        verb owns idempotency, fencing, and terminal-state semantics.
        """
        if self.engine is None or not str(work_item_id).strip():
            return False
        if self.admission is None:
            raise TopologyAdmissionError(
                "live topology retirement requires an immutable admission"
            )
        from ..knowledge_graph.core.work_durability import (
            cancel_work_item,
            get_work_item,
        )

        item = get_work_item(self.engine, str(work_item_id))
        self.admission.require_work_item(item)

        return bool(
            cancel_work_item(
                self.engine,
                str(work_item_id),
                reason=reason,
            )
        )

    def _build_sequential_plan(
        self, adaptive_agent_router: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """``"sequential"`` shape of ``_build_execution_plan``."""
        return [
            {
                "step": i,
                "roles": [s["role"]],
                "mode": "sequential",
                "agent_ids": [s.get("agent_id", s["role"])],
            }
            for i, s in enumerate(adaptive_agent_router)
        ]

    def _build_parallel_plan(
        self, adaptive_agent_router: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """``"parallel"`` shape of ``_build_execution_plan``: all
        adaptive_agent_router execute in parallel, then join."""
        return [
            {
                "step": 0,
                "roles": [s["role"] for s in adaptive_agent_router],
                "mode": "parallel",
                "agent_ids": [
                    s.get("agent_id", s["role"]) for s in adaptive_agent_router
                ],
            }
        ]

    def _build_fan_out_plan(
        self, adaptive_agent_router: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """``"fan_out"`` shape of ``_build_execution_plan``: the first
        specialist fans out to the rest."""
        if len(adaptive_agent_router) < 2:
            return self._build_sequential_plan(adaptive_agent_router)

        return [
            {
                "step": 0,
                "roles": [adaptive_agent_router[0]["role"]],
                "mode": "sequential",
                "agent_ids": [
                    adaptive_agent_router[0].get(
                        "agent_id", adaptive_agent_router[0]["role"]
                    )
                ],
            },
            {
                "step": 1,
                "roles": [s["role"] for s in adaptive_agent_router[1:]],
                "mode": "parallel",
                "agent_ids": [
                    s.get("agent_id", s["role"]) for s in adaptive_agent_router[1:]
                ],
            },
        ]

    def _build_fan_in_plan(
        self, adaptive_agent_router: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """``"fan_in"`` shape of ``_build_execution_plan``: all but the last
        execute in parallel, then the last gathers."""
        if len(adaptive_agent_router) < 2:
            return self._build_sequential_plan(adaptive_agent_router)

        return [
            {
                "step": 0,
                "roles": [s["role"] for s in adaptive_agent_router[:-1]],
                "mode": "parallel",
                "agent_ids": [
                    s.get("agent_id", s["role"]) for s in adaptive_agent_router[:-1]
                ],
            },
            {
                "step": 1,
                "roles": [adaptive_agent_router[-1]["role"]],
                "mode": "sequential",
                "agent_ids": [
                    adaptive_agent_router[-1].get(
                        "agent_id", adaptive_agent_router[-1]["role"]
                    )
                ],
            },
        ]

    def _build_execution_plan(
        self,
        adaptive_agent_router: list[dict[str, Any]],
        mode: str,
        parallel_groups: list[list[str]],
    ) -> list[dict[str, Any]]:
        """Build an ordered execution plan from adaptive_agent_router and mode.

        Returns a list of execution steps, where each step may contain:
        - A single specialist (sequential)
        - Multiple adaptive_agent_router (parallel group)
        """
        if mode == "sequential":
            return self._build_sequential_plan(adaptive_agent_router)
        if mode == "parallel":
            return self._build_parallel_plan(adaptive_agent_router)
        if mode == "fan_out":
            return self._build_fan_out_plan(adaptive_agent_router)
        if mode == "fan_in":
            return self._build_fan_in_plan(adaptive_agent_router)
        if mode == "mixed":
            return self._build_mixed_plan(adaptive_agent_router, parallel_groups)

        # Fallback: sequential
        return self._build_sequential_plan(adaptive_agent_router)

    def _build_mixed_plan(
        self,
        adaptive_agent_router: list[dict[str, Any]],
        parallel_groups: list[list[str]],
    ) -> list[dict[str, Any]]:
        """Build a mixed sequential/parallel execution plan.

        Specialists in parallel_groups execute concurrently.
        Others execute sequentially in order.
        """
        # Create a set of roles that are in parallel groups
        parallel_roles: set[str] = set()
        for group in parallel_groups:
            parallel_roles.update(group)

        steps: list[dict[str, Any]] = []
        step_idx = 0
        role_to_spec = {s["role"]: s for s in adaptive_agent_router}

        # Track which roles have been scheduled
        scheduled: set[str] = set()

        for specialist in adaptive_agent_router:
            role = specialist["role"]
            if role in scheduled:
                continue

            if role in parallel_roles:
                step_idx = _schedule_parallel_role(
                    role, parallel_groups, role_to_spec, scheduled, steps, step_idx
                )
            else:
                step_idx = _schedule_sequential_role(
                    specialist, role, scheduled, steps, step_idx
                )

        return steps

    def _build_specialist_configs(
        self, adaptive_agent_router: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Build per-specialist configuration from the team composition."""
        configs: dict[str, dict[str, Any]] = {}

        for spec in adaptive_agent_router:
            role = spec["role"]
            configs[role] = {
                "agent_id": spec.get("agent_id", role),
                "model_id": spec.get("model_id", ""),
                "tools": spec.get("tools", []),
                "system_prompt": spec.get("system_prompt", ""),
                "memory_channels": spec.get("memory_channels", ["episodic"]),
                "role": role,
            }

        return configs

    def _record_materialization(self, result: dict[str, Any]) -> None:
        """Record the materialization event in the KG for provenance."""
        if not self.engine:
            return

        try:
            node_id = f"mat:{result.get('team_id', '')}"
            self.engine.add_node(
                node_id,
                "topology_materialization",
                {
                    "name": f"Materialization: {result.get('execution_mode', '')}",
                    "topology_id": result.get("topology_id", ""),
                    "session_id": result.get("session_id", ""),
                    "specialist_count": len(result.get("specialist_configs", {})),
                    "execution_mode": result.get("execution_mode", ""),
                    "materialized_at": result.get("materialized_at", ""),
                    "admission_digest": result.get("admission_digest", ""),
                    "tenant": result.get("tenant", ""),
                    "delegation_id": result.get("delegation_id", ""),
                },
            )
        except Exception as e:  # noqa: BLE001 — docstring: "Record the materialization event in the KG for provenance"; no caller reads the return value, and a write failure doesn't affect the materialization it's merely describing
            logger.debug("Failed to record materialization: %s", e)

    def record_outcome(
        self,
        topology_id: str,
        success: bool,
        quality_score: float = 0.5,
    ) -> None:
        """Record execution outcome to update topology template success rates.

        This is the evolutionary feedback loop — successful topologies get
        higher success_rate and are preferred in future selections.

        Args:
            topology_id: The TopologyTemplate ID.
            success: Whether execution succeeded.
            quality_score: Quality of the result (0-1).
        """
        if not self.engine or not topology_id:
            return

        if self.engine.backend:
            try:
                # Update rolling success rate with exponential moving average
                alpha = 0.15
                score = quality_score if success else 0.0

                self.engine.backend.execute(
                    "MATCH (t:TopologyTemplate) WHERE t.id = $tid "
                    "SET t.success_rate = (1 - $alpha) * t.success_rate + $alpha * $score, "
                    "t.usage_count = t.usage_count + 1",
                    {"tid": topology_id, "alpha": alpha, "score": score},
                )
                logger.info(
                    "[CONCEPT:AU-ORCH.execution.dynamic-topology-materialization] Updated topology '%s': success=%s, quality=%.2f",
                    topology_id,
                    success,
                    quality_score,
                )
            except Exception as e:  # noqa: BLE001 — the EMA success_rate/usage_count update is skipped on failure with no other consumer reading a return value (method returns None unconditionally); this stalls the evolutionary feedback loop this method drives, but nothing downstream is falsely marked successful
                logger.debug("Failed to record topology outcome: %s", e)

    def get_topology_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all topology templates.

        Returns:
            List of topology stats (id, name, success_rate, usage_count).
        """
        stats: list[dict[str, Any]] = []

        if self.engine and self.engine.backend:
            try:
                results = self.engine.backend.execute(
                    "MATCH (t:TopologyTemplate) "
                    "RETURN t.id AS id, t.name AS name, "
                    "t.success_rate AS success_rate, "
                    "t.usage_count AS usage_count, "
                    "t.execution_mode AS mode "
                    "ORDER BY t.success_rate DESC",
                    {},
                )
                for r in results:
                    stats.append(
                        {
                            "id": r.get("id", ""),
                            "name": r.get("name", ""),
                            "success_rate": r.get("success_rate", 0),
                            "usage_count": r.get("usage_count", 0),
                            "mode": r.get("mode", ""),
                        }
                    )
            except Exception:
                pass  # nosec

        return stats
