#!/usr/bin/python
"""SCALE-P2-1 workload contract: the DEFINED 1M-resident workload (not linear arithmetic).

``capacity_model.py`` sizes infrastructure (shards/workers/nodes/partitions) from a
population + active fraction. It never claimed 1M was *run* — it is a first-order
linear model with explicitly documented caveats. This module is the companion piece
Codex's SCALE-P2-1 asked for: a machine-readable WORKLOAD (turns/s, tool-calls/s,
graph-mutations/s, messages/s, tokens/s, tenant skew, per-agent footprint,
interactive/background mix, availability/RPO/RTO, and SLO percentile targets) that
can actually be GENERATED against a running fleet (:mod:`scripts.scale.loadgen`) and
whose SLOs are ASSERTED by a soak/chaos harness (``tests/scale/soak/``).

The source of truth for the numbers is ``workload_contract.yml`` (same directory);
this module only loads, validates, and scales it. See that file's header comment for
which figures are anchored to :mod:`capacity_model`'s measured/modeled constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONTRACT_PATH = Path(__file__).with_name("workload_contract.yml")


class WorkloadContractError(ValueError):
    """Raised when the workload contract YAML is missing a required field or is inconsistent."""


@dataclass(frozen=True)
class ElephantTenant:
    residents_fraction: float
    active_fraction: float
    messages_fraction: float


@dataclass(frozen=True)
class TenantSpec:
    count: int
    skew_model: str
    skew_exponent: float
    elephant: ElephantTenant


@dataclass(frozen=True)
class SloTarget:
    """One SLO axis's percentile targets, in milliseconds."""

    p50: float
    p95: float
    p99: float
    p99_9: float

    def as_dict(self) -> dict[str, float]:
        return {"p50": self.p50, "p95": self.p95, "p99": self.p99, "p99_9": self.p99_9}


@dataclass(frozen=True)
class WorkloadContract:
    """The full, unscaled 1M-resident workload contract."""

    name: str
    version: int
    reference_active_fraction: float

    registered_agents: int
    resident_metadata_bytes_avg: int

    concurrent_active_sessions: int
    concurrent_turns_in_flight: int
    avg_turn_duration_s: float

    turns_per_sec: float
    tool_calls_per_sec: float
    graph_mutations_per_sec: float
    messages_per_sec: float
    tokens_per_sec: float

    tenants: TenantSpec

    working_set_bytes_avg: int
    history_bytes_avg: int
    history_bytes_p99: int
    media_bytes_avg: int
    media_bytes_p99: int

    interactive_fraction: float
    background_fraction: float

    availability_target_percent: float
    rpo_seconds: float
    rto_seconds: float

    slo: dict[str, SloTarget] = field(default_factory=dict)

    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


_REQUIRED_TOP_LEVEL = (
    "population",
    "concurrency",
    "rates",
    "tenants",
    "per_agent",
    "mix",
    "availability",
    "slo",
)
_REQUIRED_SLO_AXES = (
    "queue_latency_ms",
    "query_latency_ms",
    "write_latency_ms",
    "end_to_end_latency_ms",
)


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise WorkloadContractError(f"workload contract missing {ctx}.{key!r}")
    return d[key]


def load_workload_contract(path: str | Path | None = None) -> WorkloadContract:
    """Load + validate ``workload_contract.yml`` into a typed :class:`WorkloadContract`.

    Raises :class:`WorkloadContractError` (never a bare ``KeyError``/``TypeError``) on
    any missing/malformed field so a broken contract fails loudly at load time, not
    deep inside the load generator.
    """
    contract_path = Path(path) if path is not None else _DEFAULT_CONTRACT_PATH
    if not contract_path.is_file():
        raise WorkloadContractError(f"workload contract not found: {contract_path}")
    raw = yaml.safe_load(contract_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise WorkloadContractError(
            f"workload contract must be a YAML mapping, got {type(raw).__name__}"
        )
    for key in _REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise WorkloadContractError(f"workload contract missing top-level {key!r}")

    pop = raw["population"]
    conc = raw["concurrency"]
    rates = raw["rates"]
    ten = raw["tenants"]
    per_agent = raw["per_agent"]
    mix = raw["mix"]
    avail = raw["availability"]
    slo_raw = raw["slo"]

    for axis in _REQUIRED_SLO_AXES:
        if axis not in slo_raw:
            raise WorkloadContractError(f"workload contract missing slo.{axis!r}")

    elephant_raw = _require(ten, "elephant_tenant", "tenants")
    elephant = ElephantTenant(
        residents_fraction=float(
            _require(elephant_raw, "residents_fraction", "tenants.elephant_tenant")
        ),
        active_fraction=float(
            _require(elephant_raw, "active_fraction", "tenants.elephant_tenant")
        ),
        messages_fraction=float(
            _require(elephant_raw, "messages_fraction", "tenants.elephant_tenant")
        ),
    )
    tenants = TenantSpec(
        count=int(_require(ten, "count", "tenants")),
        skew_model=str(_require(ten, "skew_model", "tenants")),
        skew_exponent=float(_require(ten, "skew_exponent", "tenants")),
        elephant=elephant,
    )

    slo: dict[str, SloTarget] = {}
    for axis in _REQUIRED_SLO_AXES:
        axis_raw = slo_raw[axis]
        slo[axis] = SloTarget(
            p50=float(_require(axis_raw, "p50", f"slo.{axis}")),
            p95=float(_require(axis_raw, "p95", f"slo.{axis}")),
            p99=float(_require(axis_raw, "p99", f"slo.{axis}")),
            p99_9=float(_require(axis_raw, "p99_9", f"slo.{axis}")),
        )

    contract = WorkloadContract(
        name=str(raw.get("name", "")),
        version=int(raw.get("version", 1)),
        reference_active_fraction=float(raw.get("reference_active_fraction", 0.02)),
        registered_agents=int(_require(pop, "registered_agents", "population")),
        resident_metadata_bytes_avg=int(
            _require(pop, "resident_metadata_bytes_avg", "population")
        ),
        concurrent_active_sessions=int(
            _require(conc, "concurrent_active_sessions", "concurrency")
        ),
        concurrent_turns_in_flight=int(
            _require(conc, "concurrent_turns_in_flight", "concurrency")
        ),
        avg_turn_duration_s=float(_require(conc, "avg_turn_duration_s", "concurrency")),
        turns_per_sec=float(_require(rates, "turns_per_sec", "rates")),
        tool_calls_per_sec=float(_require(rates, "tool_calls_per_sec", "rates")),
        graph_mutations_per_sec=float(
            _require(rates, "graph_mutations_per_sec", "rates")
        ),
        messages_per_sec=float(_require(rates, "messages_per_sec", "rates")),
        tokens_per_sec=float(_require(rates, "tokens_per_sec", "rates")),
        tenants=tenants,
        working_set_bytes_avg=int(
            _require(per_agent, "working_set_bytes_avg", "per_agent")
        ),
        history_bytes_avg=int(_require(per_agent, "history_bytes_avg", "per_agent")),
        history_bytes_p99=int(_require(per_agent, "history_bytes_p99", "per_agent")),
        media_bytes_avg=int(_require(per_agent, "media_bytes_avg", "per_agent")),
        media_bytes_p99=int(_require(per_agent, "media_bytes_p99", "per_agent")),
        interactive_fraction=float(_require(mix, "interactive_fraction", "mix")),
        background_fraction=float(_require(mix, "background_fraction", "mix")),
        availability_target_percent=float(
            _require(avail, "target_percent", "availability")
        ),
        rpo_seconds=float(_require(avail, "rpo_seconds", "availability")),
        rto_seconds=float(_require(avail, "rto_seconds", "availability")),
        slo=slo,
        raw=raw,
    )
    _validate(contract)
    return contract


def _validate(c: WorkloadContract) -> None:
    if abs((c.interactive_fraction + c.background_fraction) - 1.0) > 1e-6:
        raise WorkloadContractError(
            "mix.interactive_fraction + mix.background_fraction must sum to 1.0, "
            f"got {c.interactive_fraction + c.background_fraction}"
        )
    if c.tenants.count <= 0:
        raise WorkloadContractError("tenants.count must be > 0")
    if not (0.0 < c.tenants.elephant.residents_fraction < 1.0):
        raise WorkloadContractError(
            "tenants.elephant_tenant.residents_fraction must be in (0, 1)"
        )
    for axis, target in c.slo.items():
        if not (target.p50 <= target.p95 <= target.p99 <= target.p99_9):
            raise WorkloadContractError(
                f"slo.{axis} percentiles must be non-decreasing p50<=p95<=p99<=p99_9, "
                f"got {target.as_dict()}"
            )


@dataclass(frozen=True)
class ScaledWorkload:
    """A :class:`WorkloadContract` with the population/rate axes scaled by ``scale``.

    SLO percentile targets, per-agent byte sizes, and the tenant skew SHAPE never
    scale — those are per-operation/per-unit contracts, not totals. ``scale`` lets
    the same contract drive a CI-sized run (e.g. ``scale=0.0005`` -> 500 residents)
    or a real hardware soak at ``scale=1.0`` (the full 1,000,000).
    """

    contract: WorkloadContract
    scale: float

    registered_agents: int
    concurrent_active_sessions: int
    concurrent_turns_in_flight: int
    turns_per_sec: float
    tool_calls_per_sec: float
    graph_mutations_per_sec: float
    messages_per_sec: float
    tokens_per_sec: float
    tenant_count: int

    @classmethod
    def for_scale(
        cls, contract: WorkloadContract, scale: float, *, min_tenants: int = 2
    ) -> ScaledWorkload:
        if not (0.0 < scale <= 1.0):
            raise WorkloadContractError(f"scale must be in (0, 1], got {scale}")
        return cls(
            contract=contract,
            scale=scale,
            registered_agents=max(1, round(contract.registered_agents * scale)),
            concurrent_active_sessions=max(
                1, round(contract.concurrent_active_sessions * scale)
            ),
            concurrent_turns_in_flight=max(
                1, round(contract.concurrent_turns_in_flight * scale)
            ),
            turns_per_sec=max(contract.turns_per_sec * scale, 0.001),
            tool_calls_per_sec=contract.tool_calls_per_sec * scale,
            graph_mutations_per_sec=contract.graph_mutations_per_sec * scale,
            messages_per_sec=contract.messages_per_sec * scale,
            tokens_per_sec=contract.tokens_per_sec * scale,
            tenant_count=max(min_tenants, round(contract.tenants.count * scale)),
        )

    def elephant_tenant_index(self) -> int:
        """Tenant index (0-based) designated the elephant tenant at this scale."""
        return 0

    def tenant_weight(self, index: int) -> float:
        """Relative share of *ordinary* load for tenant ``index`` (Zipf skew).

        The elephant tenant (index 0) is handled separately by the caller via
        ``contract.tenants.elephant`` fractions — this only shapes the long tail.
        """
        if index == self.elephant_tenant_index():
            return 0.0
        rank = index  # 0-based rank among the non-elephant tenants
        exponent = self.contract.tenants.skew_exponent
        return 1.0 / ((rank + 1) ** exponent)


def summarize(contract: WorkloadContract) -> str:
    """Human-readable one-shot summary (mirrors ``capacity_model.py``'s ``__main__``)."""
    lines = [
        f"{contract.name} (v{contract.version})",
        f"  registered_agents        = {contract.registered_agents:,}",
        f"  concurrent_active_sessions= {contract.concurrent_active_sessions:,}",
        f"  concurrent_turns_in_flight= {contract.concurrent_turns_in_flight:,}",
        f"  turns/s                  = {contract.turns_per_sec:,.1f}",
        f"  tool_calls/s              = {contract.tool_calls_per_sec:,.1f}",
        f"  graph_mutations/s         = {contract.graph_mutations_per_sec:,.1f}",
        f"  messages/s                = {contract.messages_per_sec:,.1f}",
        f"  tokens/s                  = {contract.tokens_per_sec:,.1f}",
        f"  tenants                   = {contract.tenants.count:,} "
        f"(elephant={contract.tenants.elephant.residents_fraction:.0%} residents)",
        f"  availability              = {contract.availability_target_percent}% "
        f"RPO={contract.rpo_seconds}s RTO={contract.rto_seconds}s",
    ]
    for axis, target in contract.slo.items():
        lines.append(f"  slo.{axis:<22}= {target.as_dict()}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(summarize(load_workload_contract()))
