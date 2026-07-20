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
"""


import contextlib
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
                if ndata.get("type") == "SecurityPolicyNode":
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

    try:
        from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
    except ImportError as exc:
        raise RuntimeError(
            "Tool guard is enabled but the MCP approval wrapper is unavailable"
        ) from exc

    def _requires_approval(
        _ctx: Any, tool_def: Any, _tool_args: dict[str, Any]
    ) -> bool:
        name = getattr(tool_def, "name", "")

        # CONCEPT:AU-OS.safety.ontological-guardrail — Ontological Guardrails (real-time argument analysis)
        if check_ontological_guardrails(name, _tool_args, engine=engine):
            return True

        # CONCEPT:AU-OS.identity.identity-policy-check — mandatory identity policy.
        try:
            metadata = getattr(tool_def, "metadata", None)
            required_capability = getattr(tool_def, "required_capability", None)
            if required_capability is None and isinstance(metadata, dict):
                required_capability = metadata.get("required_capability")
            decision = permissions_kernel.authorize_tool(
                agent_identity,
                name,
                required_capability=required_capability,
            )
            decision_value = getattr(decision, "value", decision)
            decision_name = str(decision_value).strip().lower().rsplit(".", 1)[-1]
            if decision_name == "deny":
                # A hard authorization denial is not an approval request.
                # Converting it to HITL would let an approver bypass RBAC.
                raise PermissionError("Tool execution denied by identity policy")
            if decision_name == "require_approval":
                return True
            if decision_name == "allow":
                return False
            raise PermissionError("Tool authorization returned no valid decision")
        except PermissionError:
            raise
        except Exception as exc:
            raise PermissionError(
                "Tool authorization unavailable; execution denied"
            ) from exc

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
