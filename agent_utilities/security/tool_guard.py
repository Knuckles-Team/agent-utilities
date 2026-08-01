#!/usr/bin/python
from __future__ import annotations

"""Tool Guard Module.

This module implements a security middleware layer for agent tools. It combines
identity policy, ontological guardrails, and pydantic-ai's Human-in-the-Loop
mechanism. Pattern sensitivity is limited to native function-tool annotation;
MCP authorization is identity-policy based.

Two mechanisms are provided:

1. :func:`apply_tool_guard_approvals` — flags *function* tools (``@agent.tool``)
   with ``requires_approval=True``.  Used for the top-level agent.
2. :func:`flag_mcp_tool_definitions` — requires a signed agent identity and
   ``PermissionsKernel``, then wraps MCP and explicitly marked in-process
   GraphOS toolsets with pydantic-ai's
   ``ApprovalRequiredToolset``. DENY fails closed, REQUIRE_APPROVAL yields a
   deferred request, and ALLOW executes. Missing identity policy is rejected.
   :func:`build_sensitive_tool_names` remains the registry projection helper
   for native tool metadata; it is not an MCP authorization fallback.

CONCEPT:AU-OS.identity.identity-policy-check — PA-R1 adoption note: the per-call verdict
``flag_mcp_tool_definitions`` feeds into ``ApprovalRequiredToolset`` is now computed by
:class:`PermissionPolicy`, an adapter shaped like the pydantic-ai-harness
``permission_policy.PermissionPolicy`` capability (``rules`` + ``default_verdict`` +
``context_policy``, merged most-restrictive-wins — see ``open-source-libraries/
pydantic-ai-harness-ppctx`` branch ``contrib/permission-policy-context``, PR #460's
``context_policy`` contribution). This retires the prior raise-inside-bool hack (a
``PermissionError`` raised from inside what is otherwise a bool-returning predicate,
with the ontological-guardrail check, the identity decision, and error handling
tangled into one branch) in favor of a clean three-way :class:`Decision`
(``allow``/``ask``/``deny``) computed from two channels — the built-in
ontological-guardrail check and a ``context_policy`` wrapping ``PermissionsKernel.
authorize_tool`` — and merged via :func:`more_restrictive`. This is an ADAPTER, not a
governance rewrite: ``PermissionsKernel.authorize_tool`` (identity/RBAC) and
``orchestration/action_policy.py`` (the operational ActionPolicy decision point) are
unchanged, deny-by-default is preserved, and the ``PermissionError``-on-deny /
``ApprovalRequired``-on-ask / execute-on-allow enforcement contract at the
``ApprovalRequiredToolset`` boundary — including "authority denial is a terminal
contract, never retried" (``graph/_router_impl.py``) — is unchanged.
"""


import contextlib
import fnmatch
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

Verdict = Literal["allow", "ask", "deny"]
"""Three-way tool-call decision. Mirrors the adopted pydantic-ai-harness
``permission_policy.Verdict`` shape (``'allow' | 'ask' | 'deny'``) so au's gating
composes with the same merge semantics the harness capability uses."""

_RANK: dict[Verdict, int] = {"allow": 0, "ask": 1, "deny": 2}


@dataclass(frozen=True)
class Decision:
    """A resolved :data:`Verdict` with its reason and the channel that produced it."""

    verdict: Verdict
    reason: str
    source: str  # "rule" | "ontological-guardrail" | "context-policy" | "default"


def more_restrictive(current: Decision, other: Decision) -> Decision:
    """Merge two decisions most-restrictive-wins (``deny > ask > allow``); a tie keeps ``current``.

    Mirrors ``pydantic_ai_harness.permission_policy.more_restrictive`` — the merge law
    that lets a ``context_policy`` TIGHTEN a decision (deny a call a rule would allow)
    but never loosen one (an ``allow`` from ``context_policy`` cannot override a rule
    or default that already says ``deny``/``ask``).
    """
    return other if _RANK[other.verdict] > _RANK[current.verdict] else current


ContextPolicy = Callable[[Any, str, dict[str, Any]], Verdict | None]
"""A decision channel keyed on the run context -- caller identity/tenant/role -- for
access decisions the tool name and args alone can't express. Returns a verdict to
contribute, or ``None`` to abstain. Merged most-restrictive-wins (:func:`more_restrictive`),
so it can only TIGHTEN a decision, never loosen one -- PR #460's contribution to
pydantic-ai-harness's ``PermissionPolicy.context_policy``. Sync-only here: unlike upstream's
``ContextPolicy`` (which also accepts an ``Awaitable[Verdict | None]``), au's integration
point (``ApprovalRequiredToolset.approval_required_func``) is itself a synchronous
predicate, so only the synchronous half of the adopted shape applies."""


@dataclass(frozen=True)
class Rule:
    """One static allow/ask/deny rule, matched by an ``fnmatch`` glob over the tool name."""

    verdict: Verdict
    tool: str = "*"


@dataclass
class PermissionPolicy:
    """Allow/ask/deny decision engine over tool calls (adapted from the pydantic-ai-harness
    ``permission_policy.PermissionPolicy`` capability's decision shape).

    Unlike the harness capability, this is a plain decision engine, not an
    ``AbstractCapability``: au already has a native HITL wrap (``ApprovalRequiredToolset``
    / ``ApprovalRequired``, CONCEPT:AU-OS.state.cognitive-scheduler-preemption) and a
    KG-durable operational decision point (``orchestration/action_policy.py``) — this class
    only cleans up HOW a per-call verdict is computed, not how it is enforced or audited.

    Resolution: with no matching ``rules`` entry, the decision falls to
    ``default_verdict`` (**deny-by-default**). ``context_policy`` is then merged in via
    :func:`more_restrictive`, so it can only tighten what the rules produced, never
    loosen it -- an identity/RBAC channel is a restriction layer, not a grant layer.
    """

    rules: list[Rule] = field(default_factory=list)
    default_verdict: Verdict = "deny"
    context_policy: ContextPolicy | None = None

    def _rule_decision(self, tool_name: str) -> Decision | None:
        """Last-match-wins verdict over ``rules`` (opencode semantics), or ``None`` when none match."""
        verdict: Verdict | None = None
        for rule in self.rules:
            if fnmatch.fnmatchcase(tool_name, rule.tool):
                verdict = rule.verdict
        if verdict is None:
            return None
        return Decision(
            verdict, reason=f"matched policy rule (tool={tool_name!r})", source="rule"
        )

    def decide(
        self,
        ctx: Any,
        tool_name: str,
        args: dict[str, Any],
        *,
        ontological_guardrail: Decision | None = None,
    ) -> Decision:
        """Resolve a verdict: ``rules``, the ontological-guardrail channel, and
        ``context_policy`` -- most-restrictive-wins."""
        decision = self._rule_decision(tool_name) or Decision(
            self.default_verdict,
            reason="no rule matched; using the default verdict",
            source="default",
        )
        if ontological_guardrail is not None:
            decision = more_restrictive(decision, ontological_guardrail)
        if self.context_policy is not None:
            verdict = self.context_policy(ctx, tool_name, args)
            if verdict is not None:
                decision = more_restrictive(
                    decision,
                    Decision(
                        verdict,
                        reason="matched context policy",
                        source="context-policy",
                    ),
                )
        return decision


def is_identity_governed_toolset(toolset: Any) -> bool:
    """Return whether a toolset requires the signed identity-policy wrapper."""

    if hasattr(toolset, "list_tools") or hasattr(toolset, "direct_call_tool"):
        return True
    metadata = getattr(toolset, "metadata", None)
    return isinstance(metadata, dict) and metadata.get("graphos_native") is True


def is_sensitive_tool(name: str) -> bool:
    """Check if a tool name matches any sensitive pattern."""
    from agent_utilities.core.config import SENSITIVE_TOOL_PATTERNS, TOOL_GUARD_MODE

    if TOOL_GUARD_MODE == "strict":
        # In strict mode, everything is sensitive unless it's explicitly safe
        return not is_safe_tool(name)

    for pattern in SENSITIVE_TOOL_PATTERNS:
        if re.match(pattern, name.lower()):
            return True
    return False


def is_safe_tool(name: str) -> bool:
    """Check if a tool name is explicitly safe (read-only)."""
    safe_patterns = [
        r"^read_.*",
        r"^list_.*",
        r"^get_.*",
        r"^describe_.*",
        r"^search_.*",
        r"^inspect_.*",
        r"^view_.*",
        r"^show_.*",
    ]
    for pattern in safe_patterns:
        if re.match(pattern, name.lower()):
            return True
    return False


def build_sensitive_tool_names() -> set[str]:
    """Build the authoritative set of tool names that require approval.

    Projects tools whose discovery-registry metadata sets
    ``requires_approval=True``. MCP authorization does not consume this set;
    it is decided by ``PermissionsKernel`` and the signed caller identity.

    Returns:
        A set of lowercase tool names that should be flagged for approval.

    """
    sensitive: set[str] = set()

    # Source 1: Knowledge Graph
    with contextlib.suppress(Exception):
        from agent_utilities.core.config import get_discovery_registry

        registry = get_discovery_registry()
        for tool in registry.tools:
            if tool.requires_approval:
                sensitive.add(tool.name.lower())

    return sensitive


def check_ontological_guardrails(
    tool_name: str,
    tool_args: dict[str, Any],
    engine: Any | None = None,
    ctx: Any | None = None,
) -> bool:
    """Performs real-time OWL reasoning / classification on tool arguments.

    CONCEPT:AU-OS.safety.ontological-guardrail — Ontological Guardrail Engine.
    Checks target files, directories, network hosts, or database tables against
    active SecurityPolicyNode classifications in the Knowledge Graph.
    """
    try:
        # Extract potential targets from arguments
        targets = []
        for key, val in tool_args.items():
            if isinstance(val, str):
                if key in (
                    "path",
                    "filepath",
                    "dir",
                    "directory",
                    "host",
                    "hostname",
                    "url",
                    "db",
                    "database",
                    "table",
                ):
                    targets.append(val.lower())

        if not targets:
            return False

        # 1. Query Knowledge Graph for active SecurityPolicyNode restrictions
        if engine and hasattr(engine, "graph") and engine.graph is not None:
            for nid, ndata in engine.graph.nodes(data=True):
                if ndata.get("node_type") == "SecurityPolicyNode":
                    restricted_target = ndata.get("target", "").lower()
                    if restricted_target:
                        for target in targets:
                            if restricted_target in target:
                                logger.warning(
                                    "Ontological Guardrail (KG): blocked a policy-matched target in tool '%s'",
                                    tool_name,
                                )
                                return True

        # 2. Check active fallback security policy constraints (restricted system paths)
        restricted_keywords = ("/etc", "/var/run", "admin", "db_root", "production_db")
        for target in targets:
            for kw in restricted_keywords:
                if kw in target:
                    logger.warning(
                        "Ontological Guardrail (Fallback): flagged a restricted target in tool '%s'",
                        tool_name,
                    )
                    return True
    except Exception as e:
        logger.warning(
            "Ontological guardrail evaluation failed closed (%s)",
            type(e).__name__,
        )
        # If a caller supplied a policy engine, an unavailable/malformed policy
        # graph is an enforcement outage, not permission to execute.
        if engine is not None:
            raise PermissionError(
                "Ontological guardrail unavailable; execution denied"
            ) from None

    return False


def flag_mcp_tool_definitions(
    toolsets: list[Any],
    *,
    permissions_kernel: Any,
    agent_identity: Any,
    engine: Any | None = None,
) -> list[Any]:
    """Wrap MCP/native-GraphOS toolsets with mandatory identity authorization.

    Uses pydantic-ai's native :class:`ApprovalRequiredToolset` wrapper.
    When policy requires approval, the wrapper raises ``ApprovalRequired``
    (unless ``ctx.tool_call_approved`` is already ``True`` from a prior
    approval round). This causes pydantic-ai to return
    ``DeferredToolRequests`` instead of executing the tool.

    The ``approval_required_func`` applies the mandatory identity policy
    (CONCEPT:AU-OS.state.cognitive-scheduler-preemption): DENY raises,
    REQUIRE_APPROVAL returns true, and ALLOW returns false.

    Args:
        toolsets: The original toolsets. MCP transports and toolsets carrying
            the internal ``graphos_native`` marker are wrapped; unrelated
            function toolsets remain unchanged.
        permissions_kernel: Required ``PermissionsKernel`` for identity-based
            authorization (CONCEPT:AU-OS.identity.identity-policy-check).
        agent_identity: Required signed ``AgentIdentity`` for the calling agent.
        engine: Optional KG engine for ontological guardrails.

    Returns:
        A new list where governed MCP/native GraphOS toolsets are wrapped with
        ``ApprovalRequiredToolset``.

    """
    governed_toolsets = [ts for ts in toolsets if is_identity_governed_toolset(ts)]
    if not governed_toolsets:
        return toolsets
    if permissions_kernel is None or agent_identity is None:
        raise PermissionError("Tool identity policy is required")

    # Renewal seam (CONCEPT:AU-OS.identity.permissions-kernel): the signing KEY is
    # stable/long-lived but an ISSUED identity carries a bounded TTL. A long-running
    # governed task (multi-hour agent run, KG_LOOP) must not die when that TTL
    # lapses, so refresh-on-use here — at the per-tool-call authorization boundary —
    # transparently re-issues the identity from the SAME kernel/stable key when it is
    # within the refresh-skew of expiry. Cheap, in-process, no external round-trip.
    # The mutable cell carries the renewed identity forward across calls; concurrent
    # refreshes are harmless (each re-issue is independently valid under the key).
    _identity_cell = {"identity": agent_identity}

    def _active_identity() -> Any:
        refresher = getattr(permissions_kernel, "refresh_identity_if_expiring", None)
        if not callable(refresher):
            return _identity_cell["identity"]
        refreshed = refresher(_identity_cell["identity"])
        if refreshed is not _identity_cell["identity"]:
            _identity_cell["identity"] = refreshed
        return _identity_cell["identity"]

    try:
        from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
    except ImportError as exc:
        raise RuntimeError(
            "Tool guard is enabled but the MCP approval wrapper is unavailable"
        ) from exc

    def _requires_approval(
        _ctx: Any, tool_def: Any, _tool_args: dict[str, Any]
    ) -> bool:
        """Compute the HITL bool for ``approval_required_func`` via :class:`PermissionPolicy`.

        A baseline ``allow *`` rule seeds the policy so ``context_policy`` (identity)
        can express its full allow/ask/deny range through the tighten-only merge —
        reproducing the pre-adoption behavior where the identity decision alone
        determined the outcome. ``default_verdict='deny'`` is the structural
        deny-by-default floor a caller reaches only by supplying a stricter/empty
        ``rules`` list (a deliberate lockdown lever this adapter now exposes).
        """
        name = getattr(tool_def, "name", "")

        # CONCEPT:AU-OS.safety.ontological-guardrail — real-time argument analysis, unchanged.
        # A malformed/unavailable policy graph fails closed by raising PermissionError
        # directly (propagates below, exactly as before this refactor).
        ontological_hit = check_ontological_guardrails(name, _tool_args, engine=engine)

        metadata = getattr(tool_def, "metadata", None)
        required_capability = getattr(tool_def, "required_capability", None)
        if required_capability is None and isinstance(metadata, dict):
            required_capability = metadata.get("required_capability")

        def _identity_context_policy(
            _ctx2: Any, _tool_name: str, _args2: dict[str, Any]
        ) -> Verdict:
            # CONCEPT:AU-OS.identity.identity-policy-check — mandatory identity policy,
            # adopted as the ``context_policy`` channel. A hard DENY is raised directly
            # (not returned as a Decision) so the model-facing/log-facing message stays
            # the exact, specific text it always was; ALLOW/REQUIRE_APPROVAL map onto
            # the Verdict the tighten-only merge composes with the other channels.
            try:
                decision = permissions_kernel.authorize_tool(
                    _active_identity(),
                    _tool_name,
                    required_capability=required_capability,
                )
                decision_value = getattr(decision, "value", decision)
                decision_name = str(decision_value).strip().lower().rsplit(".", 1)[-1]
                if decision_name == "deny":
                    # A hard authorization denial is not an approval request.
                    # Converting it to HITL would let an approver bypass RBAC.
                    raise PermissionError("Tool execution denied by identity policy")
                if decision_name == "require_approval":
                    return "ask"
                if decision_name == "allow":
                    return "allow"
                raise PermissionError("Tool authorization returned no valid decision")
            except PermissionError:
                raise
            except Exception as exc:
                raise PermissionError(
                    "Tool authorization unavailable; execution denied"
                ) from exc

        policy = PermissionPolicy(
            rules=[Rule("allow", tool="*")],
            default_verdict="deny",
            context_policy=_identity_context_policy,
        )
        decision = policy.decide(
            _ctx,
            name,
            _tool_args,
            ontological_guardrail=Decision(
                "ask",
                reason="ontological guardrail matched",
                source="ontological-guardrail",
            )
            if ontological_hit
            else None,
        )
        if decision.verdict == "deny":
            raise PermissionError(decision.reason)
        return decision.verdict == "ask"

    wrapped: list[Any] = []
    for ts in toolsets:
        if is_identity_governed_toolset(ts):
            wrapped.append(
                ApprovalRequiredToolset(
                    wrapped=ts,
                    approval_required_func=_requires_approval,
                )
            )
        else:
            wrapped.append(ts)

    return wrapped


def apply_tool_guard_approvals(agent: Any) -> None:
    """Apply requires_approval=True to all sensitive function tools on an agent.

    Iterates the agent's function toolset (the first entry in the public
    ``agent.toolsets`` property) and sets ``requires_approval=True`` on
    tools matching sensitive patterns.

    For MCP tools, use :func:`flag_mcp_tool_definitions` instead.

    Args:
        agent: The Pydantic AI Agent instance to modify.

    """
    flagged = 0

    try:
        from pydantic_ai.toolsets.function import FunctionToolset
    except ImportError:
        return

    for ts in agent.toolsets:
        if not isinstance(ts, FunctionToolset):
            continue
        if not hasattr(ts, "tools"):
            continue
        for tool_name, tool in ts.tools.items():
            sensitive = is_sensitive_tool(tool_name)
            orchestration = bool(re.search(r"run[-_]graph", tool_name.lower()))

            if sensitive and not orchestration:
                if not getattr(tool, "requires_approval", False):
                    logger.info(
                        f"Tool Guard: Flagging sensitive tool '{tool_name}' for approval."
                    )
                    tool.requires_approval = True
                    flagged += 1
            elif orchestration:
                if getattr(tool, "requires_approval", False):
                    logger.info(
                        f"Tool Guard: Removing sensitive flag from orchestration tool '{tool_name}'."
                    )
                tool.requires_approval = False
                logger.debug(
                    f"Tool Guard: Orchestration tool '{tool_name}' is trusted."
                )

    if flagged:
        logger.info(
            f"Tool Guard (native): Flagged {flagged} sensitive tools with requires_approval=True"
        )
