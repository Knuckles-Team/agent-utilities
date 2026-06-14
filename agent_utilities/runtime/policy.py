"""CONCEPT:OS-5.33 — Bridge the developer-workspace mutating-action gate to the fleet
ActionPolicy (CONCEPT:OS-5.24).

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

    ``policy`` may be an existing ``ActionPolicy`` or ``None`` (the shared instance is fetched
    lazily). Each workspace mutating action name (``workspace.cmd`` / ``.write`` / ``.edit``) is
    resolved through ``policy.decide`` and allowed only on an allowing tier.
    """
    if policy is None:
        from agent_utilities.orchestration.action_policy import get_action_policy

        policy = get_action_policy()

    from agent_utilities.orchestration.action_policy import ActionRequest

    def gate(action_name: str) -> tuple[bool, str]:
        decision = policy.decide(
            ActionRequest(kind=action_name, target=target, source=source)
        )
        return decision.allowed, decision.reason or decision.decision

    return gate
