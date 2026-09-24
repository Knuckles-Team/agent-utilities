"""Typed L4/L5 contracts shared by every harness adapter (RF-ADR-010 §6-§8).

One vocabulary for the harness port: what a harness claims it can do
(:class:`HarnessDescriptor`), what a run requires (:class:`RunSpec`), what was
agreed before launch (:class:`NegotiatedRunSpec`), what a run emitted
(:class:`RunEvent`, :class:`UsageRecord`) and how it ended (:class:`RunResult`).

Nothing here executes anything. The models are frozen and strict so a spec
cannot be widened after its digest is taken, and every enum-like field is a
closed ``Literal`` so a new value is a visible contract change.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

#: Adapter trace fidelity, strongest first (RF-ADR-010 §3 "make it visible").
TraceFidelity = Literal["full-step", "tool-calls", "final-output"]
FIDELITY_RANK: dict[TraceFidelity, int] = {
    "full-step": 3,
    "tool-calls": 2,
    "final-output": 1,
}

#: Provenance class of a record (§3): reproducible proof, captured
#: observation, or unverified claim (provider report, model output).
EvidenceClass = Literal["proof", "observation", "claim"]

AccountMode = Literal["api_key", "subscription"]

#: Exactly one execution-environment mode per negotiated run (§7).
EnvironmentMode = Literal[
    "local-sandbox", "caller-managed-host", "provider-managed-remote"
]

#: Usage measurement quality (§8); ``unavailable`` is never coerced to zero.
UsageQuality = Literal["measured", "estimated", "unavailable"]

HarnessCapability = Literal[
    "code_edit",
    "shell",
    "browse",
    "sub_agents",
    "mcp_client",
    "skills",
    "tool_allowlist",
    "cancellation",
    "resume",
    "structured_output",
]

#: How an adapter proves tool/skill availability before work starts (§6.5).
InventoryProof = Literal["startup_inventory", "runtime_contract", "none"]

BudgetUnit = Literal["tokens", "cost_usd", "wall_time"]
BudgetMode = Literal["strict", "advisory"]
SideEffectPolicy = Literal["none", "idempotent", "irreversible"]
ReconciliationSupport = Literal["provider_session", "none"]

RunEventKind = Literal[
    "started",
    "step",
    "message",
    "tool_call",
    "tool_result",
    "usage",
    "artifact",
    "error",
    "completed",
]
RunStatus = Literal["succeeded", "failed", "cancelled", "outcome_uncertain"]
TraceCompleteness = Literal["complete", "incomplete"]

#: Where the evidence behind a conformance or run record came from.
EvidenceSource = Literal[
    "real-harness", "recorded-live-transcript", "synthetic-transcript"
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class McpEndpoint(_Frozen):
    """One MCP server a run may reach; credentials only by reference."""

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    url: str = Field(min_length=1, max_length=2048)
    transport: Literal["http", "sse"] = "http"
    bearer_ref: str | None = Field(default=None, max_length=1024)


class SkillRef(_Frozen):
    """A digest-pinned ``SKILL.md`` body resolved from a ``skill://`` resource."""

    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9-]*$")
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    body: str = Field(min_length=1, max_length=256_000)


class RunToolset(_Frozen):
    """The run toolset manifest EG resolved and authorized (§6.5)."""

    context_endpoint: McpEndpoint | None = None
    mcp_servers: tuple[McpEndpoint, ...] = Field(default=(), max_length=64)
    allowed_tools: tuple[str, ...] | None = Field(default=None, max_length=256)
    required_tools: tuple[str, ...] = Field(default=(), max_length=256)
    skills: tuple[SkillRef, ...] = Field(default=(), max_length=64)

    def endpoints(self) -> tuple[McpEndpoint, ...]:
        """Every endpoint the run may reach, the EG context endpoint first."""
        head = (self.context_endpoint,) if self.context_endpoint else ()
        return head + self.mcp_servers


class RunBudget(_Frozen):
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_wall_s: float = Field(default=1800.0, gt=0, le=86_400, allow_inf_nan=False)
    mode: BudgetMode = "advisory"

    def units(self) -> frozenset[BudgetUnit]:
        declared: list[BudgetUnit] = ["wall_time"]
        if self.max_tokens is not None:
            declared.append("tokens")
        if self.max_cost_usd is not None:
            declared.append("cost_usd")
        return frozenset(declared)


class RunSpec(_Frozen):
    """Transport-neutral, digest-bound run requirements (§6.3)."""

    run_id: str = Field(min_length=1, max_length=512)
    task: str = Field(min_length=1, max_length=200_000)
    agent_ref: str = Field(min_length=1, max_length=512)
    component_digests: tuple[str, ...] = Field(default=(), max_length=256)
    required_capabilities: frozenset[HarnessCapability] = frozenset()
    optional_capabilities: frozenset[HarnessCapability] = frozenset()
    toolset: RunToolset = RunToolset()
    min_fidelity: TraceFidelity = "final-output"
    allowed_environments: frozenset[EnvironmentMode] = frozenset(
        {"caller-managed-host"}
    )
    budget: RunBudget = RunBudget()
    model: str | None = Field(default=None, max_length=256)
    account_mode: AccountMode = "subscription"
    account_ref: str | None = Field(default=None, max_length=1024)
    policy_ref: str = Field(default="policy:default", min_length=1, max_length=512)
    authorization_ref: str = Field(
        default="authz:unverified", min_length=1, max_length=512
    )
    side_effects: SideEffectPolicy = "none"
    #: Adapter-specific execution options; negotiation refuses any key the
    #: selected harness does not declare in ``HarnessDescriptor.runtime_options``.
    runtime_options: dict[str, JsonValue] = Field(default_factory=dict)

    def digest(self) -> str:
        """SHA-256 over the canonical JSON of the whole spec.

        Sets are serialized sorted so the digest never depends on hash order.
        """
        encoded = json.dumps(
            self.model_dump(mode="python"),
            sort_keys=True,
            separators=(",", ":"),
            default=_canonical_set,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _canonical_set(value: object) -> list[str]:
    if isinstance(value, frozenset | set):
        return sorted(str(item) for item in value)
    raise TypeError(f"not canonically encodable: {type(value).__name__}")


class VendorTerms(_Frozen):
    """Recorded vendor terms per account mode, checked before launch (§6.4)."""

    subscription_automation_allowed: bool
    note: str = Field(default="", max_length=1024)


class HarnessDescriptor(_Frozen):
    """A harness's provider claim about itself; negotiation checks it (§6.2)."""

    name: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=128)
    fidelity: TraceFidelity
    capabilities: frozenset[HarnessCapability]
    usage_quality: UsageQuality
    enforceable_budgets: frozenset[BudgetUnit]
    account_modes: frozenset[AccountMode]
    environment_modes: frozenset[EnvironmentMode]
    tool_proof: InventoryProof
    skill_proof: InventoryProof
    max_skills: int = Field(ge=0, le=64)
    reconciliation: ReconciliationSupport
    vendor_terms: VendorTerms
    runtime_options: frozenset[str] = frozenset()


class NegotiatedRunSpec(_Frozen):
    """The immutable intersection of a spec and a descriptor that may execute."""

    spec: RunSpec
    spec_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    harness: str
    harness_version: str
    environment: EnvironmentMode
    fidelity: TraceFidelity
    granted_capabilities: frozenset[HarnessCapability]
    absent_optional: frozenset[HarnessCapability]
    usage_quality: UsageQuality


class UsageRecord(_Frozen):
    """Usage with its measurement quality and source (§8)."""

    quality: UsageQuality
    source: str = Field(min_length=1, max_length=128)
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    wall_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)


UNAVAILABLE_USAGE = UsageRecord(quality="unavailable", source="not-reported")


class RunEvent(_Frozen):
    """One normalized adapter event, stamped with the negotiated spec digest."""

    run_id: str
    seq: int = Field(ge=0)
    kind: RunEventKind
    evidence: EvidenceClass
    fidelity: TraceFidelity
    spec_digest: str
    name: str = ""
    detail: str = Field(default="", max_length=20_000)
    data: dict[str, JsonValue] = Field(default_factory=dict)


class RunResult(_Frozen):
    """Terminal outcome of one harness run (§7-§8)."""

    run_id: str
    spec_digest: str
    harness: str
    status: RunStatus
    output: str = ""
    usage: UsageRecord = UNAVAILABLE_USAGE
    error_kind: str | None = None
    error: str = ""
    provider_session: str | None = None
    environment: EnvironmentMode
    fidelity: TraceFidelity
    trace: TraceCompleteness
    high_watermark: int = Field(ge=-1)
    gap_reason: str = ""


# ---------------------------------------------------------------------------
# Typed errors -- every adapter fails closed with one of these.
# ---------------------------------------------------------------------------


class HarnessError(RuntimeError):
    """Base class for every harness-port failure."""


class HarnessNotInstalled(HarnessError):
    """The harness binary/SDK is absent on this host."""


class HarnessNotConfigured(HarnessError):
    """The harness is present but its account/credentials are not configured."""


class HarnessRefused(HarnessError):
    """Negotiation refused the run; ``reasons`` names every unmet requirement."""

    def __init__(self, harness: str, reasons: tuple[str, ...]) -> None:
        self.harness = harness
        self.reasons = reasons
        super().__init__(f"{harness} refused the run: " + "; ".join(reasons))


class HarnessRunFailed(HarnessError):
    """The run started and failed with no possible unreconciled effect."""


class HarnessToolInventoryMismatch(HarnessRunFailed):
    """Startup inventory did not prove every required tool or skill (§6.5)."""


class HarnessOutcomeUncertain(HarnessError):
    """A dispatched run may have had effects; failover is forbidden (§7)."""


class SandboxBoundaryError(HarnessError):
    """A run tried to reach outside its leased workspace or environment."""


__all__ = [
    "FIDELITY_RANK",
    "UNAVAILABLE_USAGE",
    "AccountMode",
    "BudgetMode",
    "BudgetUnit",
    "EnvironmentMode",
    "EvidenceClass",
    "EvidenceSource",
    "HarnessCapability",
    "HarnessDescriptor",
    "HarnessError",
    "HarnessNotConfigured",
    "HarnessNotInstalled",
    "HarnessOutcomeUncertain",
    "HarnessRefused",
    "HarnessRunFailed",
    "HarnessToolInventoryMismatch",
    "InventoryProof",
    "McpEndpoint",
    "NegotiatedRunSpec",
    "ReconciliationSupport",
    "RunBudget",
    "RunEvent",
    "RunEventKind",
    "RunResult",
    "RunSpec",
    "RunStatus",
    "RunToolset",
    "SandboxBoundaryError",
    "SideEffectPolicy",
    "SkillRef",
    "TraceCompleteness",
    "TraceFidelity",
    "UsageQuality",
    "UsageRecord",
    "VendorTerms",
]
