"""Fail-closed capability negotiation of a :class:`RunSpec` (RF-ADR-010 §6.3).

A descriptor is a provider claim. Negotiation intersects it with the spec and
the policy and either returns the one immutable :class:`NegotiatedRunSpec` that
may execute, or raises :class:`HarnessRefused` naming *every* unmet
requirement. Each rule is one row of :data:`_RULES`; adding a rule never
touches the others.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from agent_utilities.layers.contracts import (
    FIDELITY_RANK,
    EnvironmentMode,
    HarnessDescriptor,
    HarnessRefused,
    NegotiatedRunSpec,
    RunSpec,
    SubagentGrant,
)

#: Strongest containment first; the first mode both sides accept is chosen.
_ENVIRONMENT_PREFERENCE: tuple[EnvironmentMode, ...] = (
    "local-sandbox",
    "caller-managed-host",
    "provider-managed-remote",
)


class HarnessPolicy(BaseModel):
    """The policy side of negotiation, resolved by the caller's authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    allowed_harnesses: frozenset[str] | None = None
    #: Every harness run reads/writes context through EG's MCP (§6.5).
    require_context_endpoint: bool = True


Rule = Callable[[RunSpec, HarnessDescriptor, HarnessPolicy], str | None]


def _capabilities(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> str | None:
    missing = sorted(spec.required_capabilities - desc.capabilities)
    return f"missing required capabilities {missing}" if missing else None


def _fidelity(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> str | None:
    if FIDELITY_RANK[desc.fidelity] >= FIDELITY_RANK[spec.min_fidelity]:
        return None
    return f"fidelity {desc.fidelity} is below the required {spec.min_fidelity}"


def _environment(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> str | None:
    if spec.allowed_environments & desc.environment_modes:
        return None
    return (
        f"no shared execution environment (spec {sorted(spec.allowed_environments)},"
        f" harness {sorted(desc.environment_modes)})"
    )


def _account(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> str | None:
    if spec.account_mode not in desc.account_modes:
        return f"account mode {spec.account_mode} is not supported"
    terms = desc.vendor_terms
    if (
        spec.account_mode == "subscription"
        and not terms.subscription_automation_allowed
    ):
        return f"vendor terms forbid automated subscription use: {terms.note}"
    return None


def _budget(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> str | None:
    if spec.budget.mode != "strict":
        return None
    unenforced = sorted(spec.budget.units() - desc.enforceable_budgets)
    if unenforced:
        return f"strict budget units {unenforced} cannot be enforced"
    metered = spec.budget.units() - {"wall_time"}
    if metered and desc.usage_quality != "measured":
        return (
            f"strict budget needs measured usage, harness reports {desc.usage_quality}"
        )
    return None


def _runtime_options(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> str | None:
    unknown = sorted(set(spec.runtime_options) - desc.runtime_options)
    return f"unsupported runtime options {unknown}" if unknown else None


def _skills(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> str | None:
    count = len(spec.toolset.skills)
    if count > desc.max_skills:
        return f"{count} skills requested, harness delivers at most {desc.max_skills}"
    if count and desc.skill_proof == "none":
        return "the harness cannot prove skill delivery before launch"
    return None


#: Single-predicate rules: (refuse when true, reason).
_PREDICATES: tuple[
    tuple[Callable[[RunSpec, HarnessDescriptor, HarnessPolicy], bool], str], ...
] = (
    (
        lambda s, d, p: (
            p.allowed_harnesses is not None and d.name not in p.allowed_harnesses
        ),
        "policy does not allow this harness",
    ),
    (
        lambda s, d, p: (
            bool(s.toolset.endpoints()) and "mcp_client" not in d.capabilities
        ),
        "the run needs MCP endpoints but the harness has no MCP client",
    ),
    (
        lambda s, d, p: (
            p.require_context_endpoint and s.toolset.context_endpoint is None
        ),
        "policy requires the EG context MCP endpoint in every run",
    ),
    (
        lambda s, d, p: bool(s.toolset.required_tools) and d.tool_proof == "none",
        "the harness cannot prove required tools before launch",
    ),
    (
        lambda s, d, p: (
            s.toolset.allowed_tools is not None
            and "tool_allowlist" not in d.capabilities
        ),
        "the harness cannot enforce a tool allowlist",
    ),
)

_RULES: tuple[Rule, ...] = (
    _capabilities,
    _fidelity,
    _environment,
    _account,
    _budget,
    _skills,
    _runtime_options,
)


def _budget_caps_subagents(spec: RunSpec, desc: HarnessDescriptor) -> bool:
    """A strict token or cost budget the harness enforces caps every native
    sub-agent together (their count is then only observed, in L5)."""
    metered = spec.budget.units() - {"wall_time"}
    return spec.budget.mode == "strict" and bool(metered & desc.enforceable_budgets)


def grant_subagents(spec: RunSpec, desc: HarnessDescriptor) -> SubagentGrant:
    """Native sub-agents under the plan's allowance (ruling 2026-09-24, Q1).

    Enforced only where the harness bounds count and depth from outside;
    otherwise DISABLED, unless the plan recorded the harness's policy opt-in
    (``token_budget``) and the run carries a strict budget the harness
    enforces. No plan, a zero allowance or no native sub-agents: disabled.
    """
    allowance = spec.subagents
    if (
        allowance is None
        or allowance.max_children == 0
        or "sub_agents" not in desc.capabilities
    ):
        return SubagentGrant()
    if {"count", "depth"} <= desc.subagent_limits:
        return SubagentGrant(
            mode="enforced",
            max_children=allowance.max_children,
            max_depth=allowance.max_depth,
            max_tokens=allowance.max_tokens,
        )
    if allowance.fallback == "token_budget" and _budget_caps_subagents(spec, desc):
        return SubagentGrant(mode="budget_capped", max_tokens=allowance.max_tokens)
    return SubagentGrant()


def refusal_reasons(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> tuple[str, ...]:
    """Every unmet requirement, in rule order (empty when negotiable)."""
    flagged = tuple(reason for test, reason in _PREDICATES if test(spec, desc, policy))
    reasons = (rule(spec, desc, policy) for rule in _RULES)
    return flagged + tuple(reason for reason in reasons if reason)


def negotiate(
    spec: RunSpec, desc: HarnessDescriptor, policy: HarnessPolicy
) -> NegotiatedRunSpec:
    """The executable intersection, or :class:`HarnessRefused`."""
    reasons = refusal_reasons(spec, desc, policy)
    if reasons:
        raise HarnessRefused(desc.name, reasons)
    shared = spec.allowed_environments & desc.environment_modes
    environment = next(mode for mode in _ENVIRONMENT_PREFERENCE if mode in shared)
    granted_optional = spec.optional_capabilities & desc.capabilities
    return NegotiatedRunSpec(
        spec=spec,
        spec_digest=spec.digest(),
        harness=desc.name,
        harness_version=desc.version,
        environment=environment,
        fidelity=desc.fidelity,
        granted_capabilities=spec.required_capabilities | granted_optional,
        absent_optional=spec.optional_capabilities - desc.capabilities,
        usage_quality=desc.usage_quality,
        subagents=grant_subagents(spec, desc),
    )


__all__ = ["HarnessPolicy", "grant_subagents", "negotiate", "refusal_reasons"]
