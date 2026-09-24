"""Fail-closed RunSpec negotiation (RF-ADR-010 §6.3) and the spec digest."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.layers.adapters.claude_code import DESCRIPTOR as CLAUDE
from agent_utilities.layers.adapters.codex import DESCRIPTOR as CODEX
from agent_utilities.layers.adapters.devin import DESCRIPTOR as DEVIN
from agent_utilities.layers.adapters.grok import DESCRIPTOR as GROK
from agent_utilities.layers.adapters.pydantic_ai import DESCRIPTOR as IN_PROCESS
from agent_utilities.layers.contracts import (
    HarnessDescriptor,
    HarnessRefused,
    McpEndpoint,
    RunBudget,
    RunSpec,
    RunToolset,
    SkillRef,
    SubagentAllowance,
    VendorTerms,
)
from agent_utilities.layers.negotiation import (
    HarnessPolicy,
    negotiate,
    refusal_reasons,
)

OPEN = HarnessPolicy(require_context_endpoint=False)
EG = McpEndpoint(name="eg", url="https://eg.example/mcp", bearer_ref="env://EG_TOKEN")
SKILL = SkillRef(name="planted", digest="a" * 64, body="---\nname: planted\n---\n")


def _spec(**overrides: Any) -> RunSpec:
    fields: dict[str, Any] = {"run_id": "r1", "task": "t", "agent_ref": "a"}
    return RunSpec(**{**fields, **overrides})


def test_spec_digest_is_stable_and_order_independent() -> None:
    one = _spec(required_capabilities=frozenset({"shell", "code_edit", "skills"}))
    two = _spec(required_capabilities=frozenset({"skills", "shell", "code_edit"}))
    assert one.digest() == two.digest()
    assert one.digest() != _spec(task="other").digest()


def test_negotiation_grants_the_intersection() -> None:
    spec = _spec(
        required_capabilities=frozenset({"shell"}),
        optional_capabilities=frozenset({"resume", "mcp_client"}),
    )
    negotiated = negotiate(spec, CODEX, OPEN)
    assert negotiated.spec_digest == spec.digest()
    assert negotiated.granted_capabilities == {"shell", "mcp_client"}
    assert negotiated.absent_optional == {"resume"}
    assert negotiated.environment == "caller-managed-host"
    assert negotiated.usage_quality == "measured"


@pytest.mark.parametrize(
    ("descriptor", "spec", "reason"),
    [
        (CODEX, _spec(required_capabilities=frozenset({"resume"})), "missing required"),
        (DEVIN, _spec(min_fidelity="tool-calls"), "fidelity final-output"),
        (DEVIN, _spec(), "no shared execution environment"),
        (CODEX, _spec(toolset=RunToolset(required_tools=("x",))), "cannot prove"),
        (CODEX, _spec(toolset=RunToolset(allowed_tools=("x",))), "allowlist"),
        (CODEX, _spec(toolset=RunToolset(skills=(SKILL,))), "skills requested"),
        (DEVIN, _spec(toolset=RunToolset(mcp_servers=(EG,))), "no MCP client"),
        (GROK, _spec(budget=RunBudget(max_tokens=10, mode="strict")), "strict"),
        (CODEX, _spec(runtime_options={"max_steps": 3}), "runtime options"),
    ],
)
def test_negotiation_refuses_unmet_requirements(
    descriptor: HarnessDescriptor, spec: RunSpec, reason: str
) -> None:
    with pytest.raises(HarnessRefused) as caught:
        negotiate(spec, descriptor, OPEN)
    assert any(reason in item for item in caught.value.reasons), caught.value.reasons


def test_policy_requires_the_eg_context_endpoint_by_default() -> None:
    reasons = refusal_reasons(_spec(), CLAUDE, HarnessPolicy())
    assert reasons == ("policy requires the EG context MCP endpoint in every run",)
    with_eg = _spec(toolset=RunToolset(context_endpoint=EG))
    assert refusal_reasons(with_eg, CLAUDE, HarnessPolicy()) == ()


def test_policy_harness_allowlist_and_vendor_terms() -> None:
    policy = HarnessPolicy(
        allowed_harnesses=frozenset({"codex"}), require_context_endpoint=False
    )
    assert "policy does not allow this harness" in refusal_reasons(
        _spec(), CLAUDE, policy
    )
    forbidden = CLAUDE.model_copy(
        update={
            "vendor_terms": VendorTerms(
                subscription_automation_allowed=False, note="plan forbids automation"
            )
        }
    )
    reasons = refusal_reasons(_spec(account_mode="subscription"), forbidden, OPEN)
    assert any("plan forbids automation" in reason for reason in reasons)


def test_strict_cost_budget_is_accepted_only_where_enforced() -> None:
    spec = _spec(budget=RunBudget(max_cost_usd=1.0, mode="strict"))
    assert refusal_reasons(spec, CLAUDE, OPEN) == ()
    assert any("cannot be enforced" in r for r in refusal_reasons(spec, CODEX, OPEN))


def test_in_process_runtime_accepts_its_declared_options_only() -> None:
    spec = _spec(runtime_options={"max_steps": 5, "grounding": "required"})
    assert refusal_reasons(spec, IN_PROCESS, OPEN) == ()
    bad = _spec(runtime_options={"shell_escape": True})
    assert refusal_reasons(bad, IN_PROCESS, OPEN) == (
        "unsupported runtime options ['shell_escape']",
    )


def test_devin_is_provider_managed_remote_only() -> None:
    spec = _spec(allowed_environments=frozenset({"provider-managed-remote"}))
    assert negotiate(spec, DEVIN, OPEN).environment == "provider-managed-remote"


# --- ST-10: native sub-agents under the plan's allowance (ruling 2026-09-24 Q1)

PLAN_ALLOWANCE = SubagentAllowance(
    max_children=3, max_depth=1, max_tokens=8000, record_ref="decision:ab"
)
COUNTING = HarnessDescriptor(
    **{
        **CLAUDE.model_dump(),
        "name": "counting",
        "subagent_limits": frozenset({"count", "depth", "tokens"}),
    }
)


def test_no_plan_grants_no_native_sub_agents() -> None:
    assert negotiate(_spec(), CLAUDE, OPEN).subagents.mode == "disabled"
    assert negotiate(_spec(), COUNTING, OPEN).subagents.mode == "disabled"


def test_a_harness_that_bounds_count_and_depth_is_granted_the_allowance() -> None:
    grant = negotiate(_spec(subagents=PLAN_ALLOWANCE), COUNTING, OPEN).subagents
    assert (grant.mode, grant.max_children, grant.max_depth, grant.max_tokens) == (
        "enforced",
        3,
        1,
        8000,
    )


def test_a_harness_that_cannot_bound_a_child_count_is_disabled_by_default() -> None:
    grant = negotiate(_spec(subagents=PLAN_ALLOWANCE), CLAUDE, OPEN).subagents
    assert grant.mode == "disabled"


def test_the_budget_opt_in_needs_a_strict_budget_the_harness_enforces() -> None:
    opted = PLAN_ALLOWANCE.model_copy(update={"fallback": "token_budget"})
    advisory = _spec(subagents=opted)
    assert negotiate(advisory, CLAUDE, OPEN).subagents.mode == "disabled"
    strict = _spec(subagents=opted, budget=RunBudget(max_cost_usd=2.0, mode="strict"))
    grant = negotiate(strict, CLAUDE, OPEN).subagents
    assert (grant.mode, grant.max_children, grant.max_tokens) == (
        "budget_capped",
        0,
        8000,
    )


def test_a_harness_without_native_sub_agents_is_never_granted_them() -> None:
    assert negotiate(_spec(subagents=PLAN_ALLOWANCE), CODEX, OPEN).subagents.mode == (
        "disabled"
    )


def test_an_execution_request_carries_its_plan_node_s_allowance() -> None:
    from agent_utilities.api.agent_control_contracts import AgentExecutionRequest
    from agent_utilities.api.harness_executor import run_spec_for

    request = AgentExecutionRequest(
        agent_name="agent-lead",
        task="survey",
        subagents={"max_children": 2, "max_depth": 1, "record_ref": "decision:ab"},
    )
    spec = run_spec_for(request, "run-1")
    assert spec.subagents == SubagentAllowance(
        max_children=2, max_depth=1, record_ref="decision:ab"
    )
    assert (
        run_spec_for(request.model_copy(update={"subagents": None}), "r").subagents
        is None
    )
