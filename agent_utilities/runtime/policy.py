"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating — Bridge the developer-workspace mutating-action gate to the fleet
ActionPolicy (CONCEPT:AU-OS.deployment.fleet-lifecycle-control).

The workspace itself only knows a :data:`~.workspace.PolicyGate` — a ``str -> (allowed, reason)``
callable. This adapter wraps the real :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`
so the same fail-closed decision point that governs fleet mutations also governs in-sandbox
shell/file mutations when an operator opts in (by passing the gate to ``create_workspace`` /
``DevWorkspace``). With no gate supplied, the sandbox boundary alone applies.
"""

from __future__ import annotations

from typing import Any

from .workspace import PolicyGate


def action_policy_gate(
    policy: Any = None, *, target: str = "workspace", source: str = "swe_agent"
) -> PolicyGate:
    """Return a :data:`PolicyGate` backed by an :class:`ActionPolicy`.

    ``policy`` may be an existing ``ActionPolicy``. With ``None``, each call
    resolves the policy bound to the process engine that is active at that
    moment. Only an audited decision carries the receipt that authorizes an
    effect (5a4dd9a2f), and an engine-less policy cannot audit. So with no
    active engine every mutation fails closed. Previously the engine-less
    policy was pinned at gate construction, which denied every workspace
    action even when an engine was running. Each workspace mutating action
    name (``workspace.cmd`` / ``.write`` / ``.edit``) is resolved through
    ``decide`` and allowed only on an allowing tier.
    """
    from agent_utilities.orchestration.action_policy import ActionRequest

    def gate(action_name: str) -> tuple[bool, str]:
        active = policy if policy is not None else _process_policy()
        decision = active.decide(
            ActionRequest(kind=action_name, target=target, source=source)
        )
        return decision.allowed, decision.reason or decision.decision

    return gate


def _process_policy() -> Any:
    """The shared ActionPolicy bound to the currently active process engine."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.orchestration.action_policy import get_action_policy

    return get_action_policy(IntelligenceGraphEngine.get_active())
