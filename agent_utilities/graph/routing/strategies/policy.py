"""Policy-driven routing (CONCEPT:AU-ORCH.adapter.kg-graph-materialization/1.5/1.7).

Migrated from the former ``graph/policy_driven_router.py``. Consolidates swarm
presets, subagent lifecycles, and learned agent routing into a single
parameterised engine driven by ``RoutingPolicy`` objects. A policy can be wrapped
as a :class:`~agent_utilities.graph.routing.strategy.RoutingStrategy` via
:class:`PolicyStrategy`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class RoutingPolicy(ABC):
    """Abstract base class for specific routing behaviours."""

    @abstractmethod
    def determine_route(self, context: dict[str, Any]) -> str:
        """Determine the next step or agent to route to."""
        raise NotImplementedError  # ABSTRACT-OK


class SwarmPresetPolicy(RoutingPolicy):
    """Routes based on a static YAML-driven DAG or swarm preset."""

    def __init__(self, preset_config: dict):
        self.config = preset_config

    def determine_route(self, context: dict[str, Any]) -> str:
        current = context.get("current_node", "START")
        transitions = self.config.get("transitions", {})
        return transitions.get(current, "END")


class LearnedAgentPolicy(RoutingPolicy):
    """Routes dynamically based on historical trace probabilities."""

    def __init__(self, learning_backend: Any):
        self.backend = learning_backend

    def determine_route(self, context: dict[str, Any]) -> str:
        task_embedding = context.get("task_embedding")
        if task_embedding and hasattr(self.backend, "predict_best_route"):
            return self.backend.predict_best_route(task_embedding)
        return "fallback_specialist"


class SubagentLifecyclePolicy(RoutingPolicy):
    """Routes based on the 4-tier interaction taxonomy (inline, fan-out, pool, team).

    ARPO (CONCEPT:AU-AHE.reward.this-is-read-back): also branches to ``fan_out`` at a high-entropy decision
    step even when task complexity is only moderate — uncertainty at a tool/decision
    boundary is where an extra rollout pays off. Bounded by ``ARPO_MAX_BRANCHES`` via
    the ``branch_count`` carried in the context so it cannot wedge the worker pool.
    """

    def determine_route(self, context: dict[str, Any]) -> str:
        complexity = context.get("task_complexity", 1)
        # ARPO entropy-gated branching — checked before the complexity tiers so a
        # genuinely uncertain step escalates from inline to a branched rollout.
        entropy = float(context.get("step_entropy", 0.0) or 0.0)
        if complexity > 1 and entropy > 0.0:
            from agent_utilities.graph.agent_step_po import should_branch

            if should_branch(entropy, branch_count=int(context.get("branch_count", 0))):
                return "fan_out"
        if complexity > 8:
            return "spawn_team"
        elif complexity > 4:
            return "fan_out"
        return "inline_tool"


class PolicyDrivenRouter:
    """Central routing engine that applies a configured (hot-swappable) policy."""

    def __init__(self, default_policy: RoutingPolicy):
        self.active_policy = default_policy

    def set_policy(self, policy: RoutingPolicy) -> None:
        """Hot-swap the routing policy at runtime."""
        self.active_policy = policy
        logger.info("Router switched to policy: %s", policy.__class__.__name__)

    def route(self, execution_context: dict[str, Any]) -> str:
        """Execute the current policy to determine the next destination."""
        destination = self.active_policy.determine_route(execution_context)
        logger.debug("Router resolved destination: %s", destination)
        return destination


class PolicyStrategy:
    """Adapt a synchronous :class:`RoutingPolicy` to the async RoutingStrategy API."""

    name = "policy"

    def __init__(self, policy: RoutingPolicy):
        self._policy = policy

    async def decide(self, ctx: Any) -> str | None:
        context = getattr(ctx, "execution_context", None)
        if context is None:
            state = getattr(ctx, "state", None)
            context = getattr(state, "execution_context", {}) if state else {}
        decision = self._policy.determine_route(context or {})
        return decision or None
