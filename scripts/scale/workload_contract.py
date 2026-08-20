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

The source of truth for the numbers is the packaged ``workload_contract.yml``;
this module only loads, validates, and scales it. See that file's header comment for
which figures are anchored to :mod:`capacity_model`'s measured/modeled constants.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


class WorkloadContractError(ValueError):
    """Raised when the workload contract YAML is missing a required field or is inconsistent."""


EXPECTED_RESIDENT_POPULATION = 1_000_000
_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_EVIDENCE_FIELDS = frozenset(
    {
        "contract_digest",
        "release_digest",
        "topology_digest",
        "image_digest",
        "execution_mode",
        "live_authority",
        "mock_authority",
    }
)


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
class WorkloadEvidence:
    """Runtime proof binding one workload result to immutable authorities.

    ``execution_mode`` is explicit because a mock run is useful for CI but
    cannot certify a live deployment.  Exactly one of ``live_authority`` and
    ``mock_authority`` must be true and it must agree with the mode.  Digests
    are opaque content identities; locations, endpoints, and credentials are
    intentionally not part of this evidence object.
    """

    execution_mode: str
    live_authority: bool
    mock_authority: bool
    contract_digest: str
    release_digest: str
    topology_digest: str
    image_digest: str

    @classmethod
    def from_mapping(
        cls, contract: WorkloadContract, raw: Mapping[str, Any]
    ) -> WorkloadEvidence:
        if not hasattr(contract, "contract_digest"):
            raise WorkloadContractError("workload evidence requires a loaded contract")
        if not isinstance(raw, Mapping):
            raise WorkloadContractError("workload evidence must be a mapping")
        unknown = set(raw) - _EVIDENCE_FIELDS
        if unknown:
            raise WorkloadContractError(
                "workload evidence contains unknown fields: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        missing = sorted(field for field in _EVIDENCE_FIELDS if field not in raw)
        if missing:
            raise WorkloadContractError(
                "workload evidence missing required fields: " + ", ".join(missing)
            )
        mode = raw.get("execution_mode")
        if not isinstance(mode, str) or mode not in {"live", "mock"}:
            raise WorkloadContractError(
                "workload evidence execution_mode must be exactly 'live' or 'mock'"
            )
        live = raw.get("live_authority")
        mock = raw.get("mock_authority")
        if not isinstance(live, bool) or not isinstance(mock, bool):
            raise WorkloadContractError(
                "workload evidence live_authority/mock_authority must be booleans"
            )
        if live == mock or live != (mode == "live") or mock != (mode == "mock"):
            raise WorkloadContractError(
                "workload evidence has ambiguous mock/live authority"
            )
        contract_digest = _require_digest(raw.get("contract_digest"), "contract_digest")
        if contract_digest != contract.contract_digest:
            raise WorkloadContractError(
                "workload evidence contract_digest does not match the loaded contract"
            )
        return cls(
            execution_mode=mode,
            live_authority=live,
            mock_authority=mock,
            contract_digest=contract_digest,
            release_digest=_require_digest(raw.get("release_digest"), "release_digest"),
            topology_digest=_require_digest(
                raw.get("topology_digest"), "topology_digest"
            ),
            image_digest=_require_digest(raw.get("image_digest"), "image_digest"),
        )


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
    contract_digest: str = ""


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


def _mapping(value: Any, ctx: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkloadContractError(
            f"workload contract {ctx} must be a mapping, got {type(value).__name__}"
        )
    return dict(value)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise WorkloadContractError(f"workload contract {field} must be an integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise WorkloadContractError(f"workload contract {field} must be an integer")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
        raise WorkloadContractError(f"workload contract {field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkloadContractError(
            f"workload contract {field} must be an integer"
        ) from exc


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise WorkloadContractError(f"workload contract {field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkloadContractError(
            f"workload contract {field} must be numeric"
        ) from exc
    if not math.isfinite(number):
        raise WorkloadContractError(f"workload contract {field} must be finite")
    return number


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkloadContractError(f"workload contract {field} must be non-empty text")
    return value.strip()


def _require_digest(value: Any, field: str) -> str:
    digest = str(value or "")
    if not _DIGEST.fullmatch(digest) or digest.endswith("0" * 64):
        raise WorkloadContractError(f"workload evidence {field} is not a valid digest")
    return digest


def _contract_digest(raw: Mapping[str, Any]) -> str:
    """Hash contract material while excluding runtime evidence (no recursion)."""

    material = dict(raw)
    material.pop("evidence", None)
    try:
        encoded = json.dumps(
            material, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkloadContractError(
            "workload contract cannot be canonically hashed"
        ) from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def bind_workload_evidence(
    contract: WorkloadContract, evidence: Mapping[str, Any] | None = None, **fields: Any
) -> WorkloadEvidence:
    """Bind a result to one explicit mock/live authority and four digests.

    No default mode or default authority is provided.  Callers must identify
    whether the run is a live deployment or an isolated mock and provide the
    exact release, topology, and image digests that were exercised.
    """

    if evidence is None:
        payload: dict[str, Any] = {}
    elif isinstance(evidence, Mapping):
        payload = dict(evidence)
    else:
        raise WorkloadContractError("workload evidence must be a mapping")
    payload.update(fields)
    return WorkloadEvidence.from_mapping(contract, payload)


def bind_evidence(
    contract: WorkloadContract, evidence: Mapping[str, Any] | None = None, **fields: Any
) -> WorkloadEvidence:
    """Compatibility spelling for callers that use the shorter evidence name."""

    return bind_workload_evidence(contract, evidence, **fields)


def load_workload_contract(path: str | Path | None = None) -> WorkloadContract:
    """Load + validate ``workload_contract.yml`` into a typed :class:`WorkloadContract`.

    Raises :class:`WorkloadContractError` (never a bare ``KeyError``/``TypeError``) on
    any missing/malformed field so a broken contract fails loudly at load time, not
    deep inside the load generator.
    """
    if path is None:
        try:
            payload = (
                files("scripts.scale")
                .joinpath("workload_contract.yml")
                .read_text(encoding="utf-8")
            )
        except (ModuleNotFoundError, OSError) as exc:
            raise WorkloadContractError(
                "packaged workload contract is unavailable"
            ) from exc
    else:
        contract_path = Path(path)
        if not contract_path.is_file():
            raise WorkloadContractError("workload contract not found")
        payload = contract_path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(payload) or {}
    except yaml.YAMLError as exc:
        raise WorkloadContractError("workload contract YAML is malformed") from exc
    raw = _mapping(raw, "root")
    for key in _REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise WorkloadContractError(f"workload contract missing top-level {key!r}")

    pop = _mapping(raw["population"], "population")
    conc = _mapping(raw["concurrency"], "concurrency")
    rates = _mapping(raw["rates"], "rates")
    ten = _mapping(raw["tenants"], "tenants")
    per_agent = _mapping(raw["per_agent"], "per_agent")
    mix = _mapping(raw["mix"], "mix")
    avail = _mapping(raw["availability"], "availability")
    slo_raw = _mapping(raw["slo"], "slo")

    for axis in _REQUIRED_SLO_AXES:
        if axis not in slo_raw:
            raise WorkloadContractError(f"workload contract missing slo.{axis!r}")

    elephant_raw = _mapping(
        _require(ten, "elephant_tenant", "tenants"), "tenants.elephant_tenant"
    )
    elephant = ElephantTenant(
        residents_fraction=_number(
            _require(elephant_raw, "residents_fraction", "tenants.elephant_tenant"),
            "tenants.elephant_tenant.residents_fraction",
        ),
        active_fraction=_number(
            _require(elephant_raw, "active_fraction", "tenants.elephant_tenant"),
            "tenants.elephant_tenant.active_fraction",
        ),
        messages_fraction=_number(
            _require(elephant_raw, "messages_fraction", "tenants.elephant_tenant"),
            "tenants.elephant_tenant.messages_fraction",
        ),
    )
    tenants = TenantSpec(
        count=_integer(_require(ten, "count", "tenants"), "tenants.count"),
        skew_model=_text(_require(ten, "skew_model", "tenants"), "tenants.skew_model"),
        skew_exponent=_number(
            _require(ten, "skew_exponent", "tenants"), "tenants.skew_exponent"
        ),
        elephant=elephant,
    )

    slo: dict[str, SloTarget] = {}
    for axis in _REQUIRED_SLO_AXES:
        axis_raw = _mapping(slo_raw[axis], f"slo.{axis}")
        slo[axis] = SloTarget(
            p50=_number(_require(axis_raw, "p50", f"slo.{axis}"), f"slo.{axis}.p50"),
            p95=_number(_require(axis_raw, "p95", f"slo.{axis}"), f"slo.{axis}.p95"),
            p99=_number(_require(axis_raw, "p99", f"slo.{axis}"), f"slo.{axis}.p99"),
            p99_9=_number(
                _require(axis_raw, "p99_9", f"slo.{axis}"), f"slo.{axis}.p99_9"
            ),
        )

    contract = WorkloadContract(
        name=_text(raw.get("name", ""), "name"),
        version=_integer(raw.get("version", 1), "version"),
        contract_digest=_contract_digest(raw),
        reference_active_fraction=_number(
            raw.get("reference_active_fraction", 0.02), "reference_active_fraction"
        ),
        registered_agents=_integer(
            _require(pop, "registered_agents", "population"),
            "population.registered_agents",
        ),
        resident_metadata_bytes_avg=_integer(
            _require(pop, "resident_metadata_bytes_avg", "population"),
            "population.resident_metadata_bytes_avg",
        ),
        concurrent_active_sessions=_integer(
            _require(conc, "concurrent_active_sessions", "concurrency"),
            "concurrency.concurrent_active_sessions",
        ),
        concurrent_turns_in_flight=_integer(
            _require(conc, "concurrent_turns_in_flight", "concurrency"),
            "concurrency.concurrent_turns_in_flight",
        ),
        avg_turn_duration_s=_number(
            _require(conc, "avg_turn_duration_s", "concurrency"),
            "concurrency.avg_turn_duration_s",
        ),
        turns_per_sec=_number(
            _require(rates, "turns_per_sec", "rates"), "rates.turns_per_sec"
        ),
        tool_calls_per_sec=_number(
            _require(rates, "tool_calls_per_sec", "rates"), "rates.tool_calls_per_sec"
        ),
        graph_mutations_per_sec=_number(
            _require(rates, "graph_mutations_per_sec", "rates"),
            "rates.graph_mutations_per_sec",
        ),
        messages_per_sec=_number(
            _require(rates, "messages_per_sec", "rates"), "rates.messages_per_sec"
        ),
        tokens_per_sec=_number(
            _require(rates, "tokens_per_sec", "rates"), "rates.tokens_per_sec"
        ),
        tenants=tenants,
        working_set_bytes_avg=_integer(
            _require(per_agent, "working_set_bytes_avg", "per_agent"),
            "per_agent.working_set_bytes_avg",
        ),
        history_bytes_avg=_integer(
            _require(per_agent, "history_bytes_avg", "per_agent"),
            "per_agent.history_bytes_avg",
        ),
        history_bytes_p99=_integer(
            _require(per_agent, "history_bytes_p99", "per_agent"),
            "per_agent.history_bytes_p99",
        ),
        media_bytes_avg=_integer(
            _require(per_agent, "media_bytes_avg", "per_agent"),
            "per_agent.media_bytes_avg",
        ),
        media_bytes_p99=_integer(
            _require(per_agent, "media_bytes_p99", "per_agent"),
            "per_agent.media_bytes_p99",
        ),
        interactive_fraction=_number(
            _require(mix, "interactive_fraction", "mix"),
            "mix.interactive_fraction",
        ),
        background_fraction=_number(
            _require(mix, "background_fraction", "mix"),
            "mix.background_fraction",
        ),
        availability_target_percent=_number(
            _require(avail, "target_percent", "availability"),
            "availability.target_percent",
        ),
        rpo_seconds=_number(
            _require(avail, "rpo_seconds", "availability"),
            "availability.rpo_seconds",
        ),
        rto_seconds=_number(
            _require(avail, "rto_seconds", "availability"),
            "availability.rto_seconds",
        ),
        slo=slo,
        raw=raw,
    )
    _validate(contract)
    return contract


def _validate(c: WorkloadContract) -> None:
    if not isinstance(c.name, str) or c.name.strip() == "":
        raise WorkloadContractError("workload contract name must not be empty")
    if c.version < 1:
        raise WorkloadContractError("workload contract version must be >= 1")
    if c.registered_agents != EXPECTED_RESIDENT_POPULATION:
        raise WorkloadContractError(
            "population.registered_agents must be exactly "
            f"{EXPECTED_RESIDENT_POPULATION:,}, got {c.registered_agents:,}"
        )
    if not (0.0 < c.reference_active_fraction <= 1.0):
        raise WorkloadContractError(
            "reference_active_fraction must be finite and in (0, 1]"
        )

    for field_name, value in (
        ("concurrency.concurrent_active_sessions", c.concurrent_active_sessions),
        ("concurrency.concurrent_turns_in_flight", c.concurrent_turns_in_flight),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkloadContractError(f"{field_name} must be a non-negative integer")
    if c.concurrent_turns_in_flight > c.concurrent_active_sessions:
        raise WorkloadContractError(
            "concurrency.concurrent_turns_in_flight cannot exceed active sessions"
        )
    expected_active = c.registered_agents * c.reference_active_fraction
    if not math.isclose(
        c.concurrent_active_sessions, expected_active, rel_tol=0.0, abs_tol=0.5
    ):
        raise WorkloadContractError(
            "concurrency.concurrent_active_sessions is incoherent with population "
            "and reference_active_fraction"
        )
    if c.avg_turn_duration_s < 0:
        raise WorkloadContractError(
            "concurrency.avg_turn_duration_s must be non-negative"
        )
    if c.concurrent_turns_in_flight > 0 and c.avg_turn_duration_s <= 0:
        raise WorkloadContractError(
            "concurrency.avg_turn_duration_s must be > 0 when turns are in flight"
        )

    rate_fields = (
        "turns_per_sec",
        "tool_calls_per_sec",
        "graph_mutations_per_sec",
        "messages_per_sec",
        "tokens_per_sec",
    )
    for field_name in rate_fields:
        value = getattr(c, field_name)
        if not math.isfinite(value) or value < 0:
            raise WorkloadContractError(
                f"rates.{field_name} must be finite and non-negative"
            )
    if c.concurrent_turns_in_flight == 0 and c.turns_per_sec != 0:
        raise WorkloadContractError(
            "rates.turns_per_sec must be zero when no turns are in flight"
        )
    if c.concurrent_turns_in_flight > 0:
        expected_turns = c.concurrent_turns_in_flight / c.avg_turn_duration_s
        if not math.isclose(c.turns_per_sec, expected_turns, rel_tol=0.05, abs_tol=1.0):
            raise WorkloadContractError(
                "rates.turns_per_sec is incoherent with in-flight turns and duration"
            )

    byte_fields = (
        "resident_metadata_bytes_avg",
        "working_set_bytes_avg",
        "history_bytes_avg",
        "history_bytes_p99",
        "media_bytes_avg",
        "media_bytes_p99",
    )
    for field_name in byte_fields:
        value = getattr(c, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkloadContractError(
                f"{field_name} must be a finite non-negative byte count"
            )
    if c.working_set_bytes_avg != c.resident_metadata_bytes_avg:
        raise WorkloadContractError(
            "per_agent.working_set_bytes_avg must match population."
            "resident_metadata_bytes_avg"
        )
    if c.history_bytes_p99 < c.history_bytes_avg:
        raise WorkloadContractError("per_agent.history_bytes_p99 must be >= average")
    if c.media_bytes_p99 < c.media_bytes_avg:
        raise WorkloadContractError("per_agent.media_bytes_p99 must be >= average")

    for field_name, value in (
        ("mix.interactive_fraction", c.interactive_fraction),
        ("mix.background_fraction", c.background_fraction),
        (
            "tenants.elephant_tenant.residents_fraction",
            c.tenants.elephant.residents_fraction,
        ),
        ("tenants.elephant_tenant.active_fraction", c.tenants.elephant.active_fraction),
        (
            "tenants.elephant_tenant.messages_fraction",
            c.tenants.elephant.messages_fraction,
        ),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise WorkloadContractError(
                f"{field_name} must be finite and in the range [0, 1]"
            )
    if abs((c.interactive_fraction + c.background_fraction) - 1.0) > 1e-6:
        raise WorkloadContractError(
            "mix.interactive_fraction + mix.background_fraction must sum to 1.0, "
            f"got {c.interactive_fraction + c.background_fraction}"
        )
    if c.tenants.count <= 0:
        raise WorkloadContractError("tenants.count must be > 0")
    if not isinstance(c.tenants.skew_model, str) or not c.tenants.skew_model.strip():
        raise WorkloadContractError("tenants.skew_model must not be empty")
    if not math.isfinite(c.tenants.skew_exponent) or c.tenants.skew_exponent < 0:
        raise WorkloadContractError(
            "tenants.skew_exponent must be finite and non-negative"
        )
    if not (0.0 < c.tenants.elephant.residents_fraction <= 1.0):
        raise WorkloadContractError(
            "tenants.elephant_tenant.residents_fraction must be in (0, 1]"
        )
    if not (
        math.isfinite(c.availability_target_percent)
        and 0.0 < c.availability_target_percent <= 100.0
    ):
        raise WorkloadContractError(
            "availability.target_percent must be finite and in (0, 100]"
        )
    if not math.isfinite(c.rpo_seconds) or c.rpo_seconds < 0:
        raise WorkloadContractError(
            "availability.rpo_seconds must be finite and non-negative"
        )
    if not math.isfinite(c.rto_seconds) or c.rto_seconds < 0:
        raise WorkloadContractError(
            "availability.rto_seconds must be finite and non-negative"
        )
    if c.rto_seconds < c.rpo_seconds:
        raise WorkloadContractError("availability.rto_seconds must be >= rpo_seconds")
    for axis, target in c.slo.items():
        if any(
            not math.isfinite(value) or value < 0
            for value in (target.p50, target.p95, target.p99, target.p99_9)
        ):
            raise WorkloadContractError(
                f"slo.{axis} percentile targets must be finite and non-negative"
            )
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
