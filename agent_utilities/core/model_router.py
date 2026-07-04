#!/usr/bin/python
from __future__ import annotations

"""Adaptive local-LLM router (CONCEPT:AU-ORCH.routing.adaptive-role-routing).

Closes the economics loop the platform had the parts for but never connected: the
tiered model ladder (`ModelRegistry` light→medium→heavy→reasoning), the cost signal
(`ModelCostRate`, zero = local/free), and the confidence-gated selector
(`pick_for_task_adaptive`, built for CONCEPT:AU-ORCH.adapter.hot-cache-invalidation but with **no live caller**)
existed as three islands. This wires them: every role-routed model selection now
flows through a *learned* confidence so a role whose cheap/local model keeps
succeeding is routed DOWN the ladder (cheaper) and one that keeps failing is
escalated UP — and the routing improves from recorded outcomes.

The confidence is a per-route reward-EMA (same mechanism as `CapabilityIndex`),
fed by :func:`record_model_outcome` from the action-outcome loop (AHE-3.62). It is
process-local (a routing cache); durable cross-process learning rides on the
canonical reward EMA via ``graph_feedback action_outcome target_id=model_route:<role>``.
Default-on and degrades to the static `pick_for_role` when no registry / single
model is configured.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: route key -> reward EMA in [0,1] (0.5 neutral = give the cheap model a chance).
_ROUTE_REWARDS: dict[str, float] = {}
_ALPHA = 0.3
#: Below this many observations a route stays neutral (don't escalate on noise).
ROUTE_KEY_PREFIX = "model_route:"


def route_key(role: str | None = None, task_class: str | None = None) -> str:
    return f"{ROUTE_KEY_PREFIX}{task_class or role or 'default'}"


def route_confidence(key: str) -> float:
    """The learned confidence for a route (0.5 neutral when unseen)."""
    return _ROUTE_REWARDS.get(key, 0.5)


def record_model_outcome(
    key: str, *, success: bool | None = None, reward: float | None = None
) -> float:
    """Update a route's reward-EMA from an observed outcome (returns the new EMA)."""
    if not key.startswith(ROUTE_KEY_PREFIX):
        key = route_key(key)
    r = reward if reward is not None else (1.0 if success else 0.0)
    r = max(0.0, min(1.0, float(r)))
    prev = _ROUTE_REWARDS.get(key, 0.5)
    updated = (1 - _ALPHA) * prev + _ALPHA * r
    _ROUTE_REWARDS[key] = updated
    return updated


def reset_routes() -> None:
    """Clear the routing cache (tests)."""
    _ROUTE_REWARDS.clear()


def pick_adaptive(
    registry: Any,
    role: str,
    *,
    routing_percentile: float = 50.0,
) -> Any | None:
    """Pick a model for ``role`` adaptively from the learned route confidence.

    Resolves the role's base tier (`registry.resolve_role`), then routes via
    `registry.pick_for_task_adaptive` using the route's reward-EMA as the
    confidence — so a consistently-succeeding role drifts to a cheaper tier and a
    failing one escalates. Returns ``None`` (never raises) so the factory falls
    back to its default model when no registry/match exists.
    """
    try:
        if registry is None or not getattr(registry, "models", None):
            return None
        spec = registry.resolve_role(role)
        confidence = route_confidence(route_key(role))
        return registry.pick_for_task_adaptive(
            complexity=spec.tier,
            confidence_signal=confidence,
            routing_percentile=routing_percentile,
            required_tags=spec.tags,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("adaptive pick failed for role %r: %s", role, e)
        return None
