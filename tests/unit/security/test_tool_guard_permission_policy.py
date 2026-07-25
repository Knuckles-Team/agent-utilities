"""PermissionPolicy adoption (PA-R1): retires tool_guard's raise-inside-bool hack.

CONCEPT:AU-OS.identity.identity-policy-check — :class:`PermissionPolicy` is au's
adapter of the pydantic-ai-harness ``permission_policy.PermissionPolicy`` capability's
decision shape (``rules`` + ``default_verdict`` + ``context_policy``, merged
most-restrictive-wins). Covers the engine directly (denies-by-default, tighten-only
merge) and ``flag_mcp_tool_definitions``'s preserved external behavior (DENY raises,
REQUIRE_APPROVAL/ontological-hit ask, ALLOW executes) via both a scriptable fake
``PermissionsKernel``-like object and the real ``PermissionsKernel`` class.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.security.tool_guard import (
    ContextPolicy,
    Decision,
    PermissionPolicy,
    Rule,
    Verdict,
    flag_mcp_tool_definitions,
    more_restrictive,
)

# ---------------------------------------------------------------------------
# more_restrictive / Decision merge law
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "other", "expected"),
    [
        ("allow", "ask", "ask"),
        ("allow", "deny", "deny"),
        ("ask", "deny", "deny"),
        ("deny", "allow", "deny"),  # never loosens
        ("deny", "ask", "deny"),  # never loosens
        ("ask", "allow", "ask"),  # never loosens
        ("allow", "allow", "allow"),  # tie keeps current
        ("deny", "deny", "deny"),
    ],
)
def test_more_restrictive_merge_law(
    current: Verdict, other: Verdict, expected: Verdict
) -> None:
    result = more_restrictive(
        Decision(current, reason="c", source="test"),
        Decision(other, reason="o", source="test"),
    )
    assert result.verdict == expected


# ---------------------------------------------------------------------------
# PermissionPolicy.decide -- the three VERIFY scenarios
# ---------------------------------------------------------------------------


def test_denies_by_default_with_no_rules_and_no_context_policy() -> None:
    policy = PermissionPolicy(rules=[], default_verdict="deny")
    decision = policy.decide(None, "any_tool", {})
    assert decision.verdict == "deny"
    assert decision.source == "default"


def test_context_policy_deny_tightens_a_rule_allow() -> None:
    def deny_everything(_ctx: Any, _tool: str, _args: dict[str, Any]) -> Verdict:
        return "deny"

    policy = PermissionPolicy(
        rules=[Rule("allow", tool="foo")],
        default_verdict="deny",
        context_policy=deny_everything,
    )
    decision = policy.decide(None, "foo", {})
    assert decision.verdict == "deny"
    assert decision.source == "context-policy"


def test_context_policy_cannot_loosen_a_rule_deny() -> None:
    def allow_everything(_ctx: Any, _tool: str, _args: dict[str, Any]) -> Verdict:
        return "allow"

    policy = PermissionPolicy(
        rules=[Rule("deny", tool="foo")],
        default_verdict="deny",
        context_policy=allow_everything,
    )
    decision = policy.decide(None, "foo", {})
    assert decision.verdict == "deny"
    assert decision.source == "rule"  # the rule channel's decision stands, unloosened


# ---------------------------------------------------------------------------
# PermissionPolicy.decide -- rule matching, abstain, and the ontological channel
# ---------------------------------------------------------------------------


def test_last_matching_rule_wins() -> None:
    policy = PermissionPolicy(
        rules=[Rule("deny", tool="git *"), Rule("allow", tool="git status")]
    )
    assert policy.decide(None, "git status", {}).verdict == "allow"
    assert policy.decide(None, "git push", {}).verdict == "deny"


def test_context_policy_none_abstains_and_rule_decision_stands() -> None:
    def abstain(_ctx: Any, _tool: str, _args: dict[str, Any]) -> Verdict | None:
        return None

    policy = PermissionPolicy(rules=[Rule("allow", tool="*")], context_policy=abstain)
    decision = policy.decide(None, "anything", {})
    assert decision.verdict == "allow"
    assert decision.source == "rule"


def test_ontological_guardrail_channel_tightens_allow_to_ask() -> None:
    policy = PermissionPolicy(rules=[Rule("allow", tool="*")])
    decision = policy.decide(
        None,
        "read_file",
        {},
        ontological_guardrail=Decision(
            "ask", reason="hit", source="ontological-guardrail"
        ),
    )
    assert decision.verdict == "ask"


def test_ontological_guardrail_channel_cannot_loosen_a_deny() -> None:
    policy = PermissionPolicy(rules=[Rule("deny", tool="*")])
    decision = policy.decide(
        None,
        "read_file",
        {},
        ontological_guardrail=Decision(
            "ask", reason="hit", source="ontological-guardrail"
        ),
    )
    assert decision.verdict == "deny"


def test_context_policy_type_is_the_adopted_three_arg_shape() -> None:
    """``(ctx, tool_name, args) -> Verdict | None`` -- PR #460's contributed shape."""

    def policy_fn(ctx: Any, tool_name: str, args: dict[str, Any]) -> Verdict | None:
        return "ask" if tool_name == "risky" else None

    policy: ContextPolicy = policy_fn
    assert policy(None, "risky", {}) == "ask"
    assert policy(None, "safe", {}) is None


# ---------------------------------------------------------------------------
# flag_mcp_tool_definitions -- preserved external behavior via a fake toolset
# ---------------------------------------------------------------------------


class FakeGovernedToolset:
    """Recognized as identity-governed by ``is_identity_governed_toolset``."""

    def list_tools(self) -> list[Any]:  # pragma: no cover - not exercised
        return []


class FakeToolDef:
    def __init__(self, name: str, required_capability: str | None = None) -> None:
        self.name = name
        self.required_capability = required_capability
        self.metadata: dict[str, Any] = {}


class FakeCtx:
    tool_call_approved = False


class ScriptedKernel:
    """A ``PermissionsKernel``-shaped double: returns a scripted decision, or raises."""

    def __init__(self, decision: Any = None, raises: Exception | None = None) -> None:
        self.decision = decision
        self.raises = raises
        self.calls: list[tuple[str, str | None]] = []

    def authorize_tool(
        self, identity: Any, tool_name: str, *, required_capability: str | None = None
    ) -> Any:
        self.calls.append((tool_name, required_capability))
        if self.raises is not None:
            raise self.raises
        return self.decision


def _approval_func(wrapped_toolsets: list[Any]) -> Any:
    """The single ``ApprovalRequiredToolset.approval_required_func`` from a
    ``flag_mcp_tool_definitions`` call over one governed toolset."""
    from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

    assert len(wrapped_toolsets) == 1
    wrapped = wrapped_toolsets[0]
    assert isinstance(wrapped, ApprovalRequiredToolset)
    return wrapped.approval_required_func


def test_allow_executes_without_approval() -> None:
    kernel = ScriptedKernel(decision="allow")
    wrapped = flag_mcp_tool_definitions(
        [FakeGovernedToolset()], permissions_kernel=kernel, agent_identity=object()
    )
    func = _approval_func(wrapped)
    assert func(FakeCtx(), FakeToolDef("search_docs"), {}) is False


def test_require_approval_returns_true() -> None:
    kernel = ScriptedKernel(decision="require_approval")
    wrapped = flag_mcp_tool_definitions(
        [FakeGovernedToolset()], permissions_kernel=kernel, agent_identity=object()
    )
    func = _approval_func(wrapped)
    assert func(FakeCtx(), FakeToolDef("delete_file"), {}) is True


def test_deny_raises_permission_error_with_original_message() -> None:
    kernel = ScriptedKernel(decision="deny")
    wrapped = flag_mcp_tool_definitions(
        [FakeGovernedToolset()], permissions_kernel=kernel, agent_identity=object()
    )
    func = _approval_func(wrapped)
    with pytest.raises(
        PermissionError, match="Tool execution denied by identity policy"
    ):
        func(FakeCtx(), FakeToolDef("reboot_host"), {})


def test_unrecognized_decision_denies() -> None:
    kernel = ScriptedKernel(decision="not_a_real_decision")
    wrapped = flag_mcp_tool_definitions(
        [FakeGovernedToolset()], permissions_kernel=kernel, agent_identity=object()
    )
    func = _approval_func(wrapped)
    with pytest.raises(PermissionError):
        func(FakeCtx(), FakeToolDef("whatever"), {})


def test_kernel_error_is_normalized_to_permission_error() -> None:
    kernel = ScriptedKernel(raises=RuntimeError("kernel exploded"))
    wrapped = flag_mcp_tool_definitions(
        [FakeGovernedToolset()], permissions_kernel=kernel, agent_identity=object()
    )
    func = _approval_func(wrapped)
    with pytest.raises(
        PermissionError, match="Tool authorization unavailable; execution denied"
    ):
        func(FakeCtx(), FakeToolDef("whatever"), {})


def test_required_capability_is_forwarded_from_tool_def() -> None:
    kernel = ScriptedKernel(decision="allow")
    wrapped = flag_mcp_tool_definitions(
        [FakeGovernedToolset()], permissions_kernel=kernel, agent_identity=object()
    )
    func = _approval_func(wrapped)
    func(FakeCtx(), FakeToolDef("search_docs", required_capability="kg.read"), {})
    assert kernel.calls == [("search_docs", "kg.read")]


def test_missing_identity_policy_raises_before_any_call() -> None:
    with pytest.raises(PermissionError, match="Tool identity policy is required"):
        flag_mcp_tool_definitions(
            [FakeGovernedToolset()], permissions_kernel=None, agent_identity=object()
        )


def test_ontological_guardrail_still_asks_even_when_identity_would_allow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ontological hit -> ask, merged with an identity ALLOW via tighten-only
    semantics -- matches the pre-adoption short-circuit's observable result."""
    from agent_utilities.security import tool_guard as tg

    monkeypatch.setattr(tg, "check_ontological_guardrails", lambda *a, **k: True)
    kernel = ScriptedKernel(decision="allow")
    wrapped = flag_mcp_tool_definitions(
        [FakeGovernedToolset()], permissions_kernel=kernel, agent_identity=object()
    )
    func = _approval_func(wrapped)
    assert (
        func(FakeCtx(), FakeToolDef("read_etc_passwd"), {"path": "/etc/passwd"}) is True
    )


# ---------------------------------------------------------------------------
# flag_mcp_tool_definitions -- real PermissionsKernel + SPECIALIST role policy
# ---------------------------------------------------------------------------


def test_real_permissions_kernel_specialist_role_policy() -> None:
    """End-to-end with the unmodified ``PermissionsKernel`` class and its shipped
    SPECIALIST policy (allow *, deny *reboot*/*shutdown*, ask *delete*/*remove*)."""
    from agent_utilities.security.permissions_kernel import AgentRole, PermissionsKernel

    kernel = PermissionsKernel(signing_key="x" * 32)
    identity = kernel.issue_identity("agent-1", role=AgentRole.SPECIALIST)

    wrapped = flag_mcp_tool_definitions(
        [FakeGovernedToolset()], permissions_kernel=kernel, agent_identity=identity
    )
    func = _approval_func(wrapped)

    assert func(FakeCtx(), FakeToolDef("search_docs"), {}) is False  # allow
    assert func(FakeCtx(), FakeToolDef("delete_record"), {}) is True  # ask
    with pytest.raises(
        PermissionError, match="Tool execution denied by identity policy"
    ):
        func(FakeCtx(), FakeToolDef("reboot_host"), {})  # deny
