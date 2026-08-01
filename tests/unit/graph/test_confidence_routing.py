#!/usr/bin/python
from __future__ import annotations

"""Tests for CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Confidence-Gated Model Router."""


from agent_utilities.models.model_registry import (
    ModelCostRate,
    ModelDefinition,
    ModelRegistry,
)


def _make_registry() -> ModelRegistry:
    """Build a test registry with one model per tier."""
    return ModelRegistry(
        models=[
            ModelDefinition(
                id="fast",
                name="Fast Model",
                provider="openai",
                model_id="gpt-4o-mini",
                tier="light",
                tags=["code"],
                cost=ModelCostRate(input=0.15, output=0.60),
            ),
            ModelDefinition(
                id="balanced",
                name="Balanced Model",
                provider="openai",
                model_id="gpt-4o",
                tier="medium",
                tags=["code", "vision"],
                cost=ModelCostRate(input=2.50, output=10.00),
                is_default=True,
            ),
            ModelDefinition(
                id="power",
                name="Power Model",
                provider="anthropic",
                model_id="claude-3-5-sonnet",
                tier="heavy",
                tags=["code"],
                cost=ModelCostRate(input=3.00, output=15.00),
            ),
            ModelDefinition(
                id="thinker",
                name="Thinker Model",
                provider="anthropic",
                model_id="claude-3-opus",
                tier="reasoning",
                tags=["code"],
                cost=ModelCostRate(input=15.00, output=75.00),
            ),
        ]
    )


# ── Tier Helpers ──────────────────────────────────────────────────────


class TestTierHelpers:
    """Tests for _tier_down() and _tier_up()."""

    def test_tier_down_from_medium(self):
        assert ModelRegistry._tier_down("medium") == "light"

    def test_tier_down_from_light_clamped(self):
        assert ModelRegistry._tier_down("light") == "light"

    def test_tier_up_from_medium(self):
        assert ModelRegistry._tier_up("medium") == "heavy"

    def test_tier_up_from_reasoning_clamped(self):
        assert ModelRegistry._tier_up("reasoning") == "reasoning"

    def test_tier_down_from_heavy(self):
        assert ModelRegistry._tier_down("heavy") == "medium"

    def test_tier_up_from_light(self):
        assert ModelRegistry._tier_up("light") == "medium"


# ── Pick For Task Adaptive ────────────────────────────────────────────


class TestPickForTaskAdaptive:
    """Tests for ModelRegistry.pick_for_task_adaptive()."""

    def test_high_confidence_downgrades_tier(self):
        """High confidence → should pick a cheaper model."""
        reg = _make_registry()
        chosen = reg.pick_for_task_adaptive(
            complexity="medium",
            confidence_signal=0.9,
            routing_percentile=50.0,
        )
        # Medium with high confidence should downgrade to light
        assert chosen.tier == "light"
        assert chosen.id == "fast"

    def test_low_confidence_escalates_tier(self):
        """Low confidence → should pick a more capable model."""
        reg = _make_registry()
        chosen = reg.pick_for_task_adaptive(
            complexity="medium",
            confidence_signal=0.1,
            routing_percentile=50.0,
        )
        # Medium with low confidence should escalate to heavy
        assert chosen.tier == "heavy"
        assert chosen.id == "power"

    def test_neutral_confidence_keeps_tier(self):
        """Neutral confidence → should keep the original tier."""
        reg = _make_registry()
        chosen = reg.pick_for_task_adaptive(
            complexity="medium",
            confidence_signal=0.5,
            routing_percentile=50.0,
        )
        assert chosen.tier == "medium"
        assert chosen.id == "balanced"

    def test_aggressive_percentile_routes_more_to_cheap(self):
        """Low routing_percentile → even moderate confidence downgrades."""
        reg = _make_registry()
        # routing_percentile=30 means threshold=0.30
        chosen = reg.pick_for_task_adaptive(
            complexity="heavy",
            confidence_signal=0.5,
            routing_percentile=30.0,
        )
        # 0.5 > 0.30 → downgrade from heavy to medium
        assert chosen.tier == "medium"

    def test_conservative_percentile_rarely_downgrades(self):
        """High routing_percentile → only very high confidence downgrades."""
        reg = _make_registry()
        # routing_percentile=90 means threshold=0.90
        chosen = reg.pick_for_task_adaptive(
            complexity="heavy",
            confidence_signal=0.5,
            routing_percentile=90.0,
        )
        # 0.5 < 0.90 → no downgrade; 0.5 > 0.10 → no escalation either
        assert chosen.tier == "heavy"

    def test_downgrade_clamped_at_light(self):
        """Downgrading from light stays at light."""
        reg = _make_registry()
        chosen = reg.pick_for_task_adaptive(
            complexity="light",
            confidence_signal=0.95,
            routing_percentile=50.0,
        )
        assert chosen.tier == "light"

    def test_escalation_clamped_at_reasoning(self):
        """Escalating from reasoning stays at reasoning."""
        reg = _make_registry()
        chosen = reg.pick_for_task_adaptive(
            complexity="reasoning",
            confidence_signal=0.05,
            routing_percentile=50.0,
        )
        assert chosen.tier == "reasoning"

    def test_required_tags_respected(self):
        """Tags filter should still be applied after tier adjustment."""
        reg = _make_registry()
        # Only "balanced" and "fast" have "code" + vision tags won't match
        chosen = reg.pick_for_task_adaptive(
            complexity="medium",
            confidence_signal=0.9,
            routing_percentile=50.0,
            required_tags=["vision"],
        )
        # Only "balanced" has "vision" tag, so even with downgrade
        # it should fall back to balanced
        assert chosen.id == "balanced"

    def test_backward_compatible_with_pick_for_task(self):
        """pick_for_task_adaptive at neutral confidence matches pick_for_task."""
        reg = _make_registry()
        adaptive = reg.pick_for_task_adaptive(
            complexity="medium",
            confidence_signal=0.5,
            routing_percentile=50.0,
        )
        standard = reg.pick_for_task(complexity="medium")
        assert adaptive.id == standard.id


# ── Resource Optimizer Confidence Forwarding ──────────────────────────


class TestResourceOptimizerConfidence:
    """Tests for ResourceOptimizer.select_model_for_step() with confidence."""

    def test_confidence_forwarded_to_registry(self):
        from agent_utilities.core.resource_optimizer import (
            ResourceBudget,
            ResourceOptimizer,
        )

        reg = _make_registry()
        opt = ResourceOptimizer(
            budget=ResourceBudget(total_cost_budget_usd=10.0),
            model_registry=reg,
        )

        # High confidence should trigger adaptive routing
        result = opt.select_model_for_step(
            complexity="medium",
            confidence_signal=0.9,
            routing_percentile=50.0,
        )
        assert result is not None
        assert getattr(result, "tier", None) == "light"

    def test_none_confidence_uses_standard_routing(self):
        from agent_utilities.core.resource_optimizer import (
            ResourceBudget,
            ResourceOptimizer,
        )

        reg = _make_registry()
        opt = ResourceOptimizer(
            budget=ResourceBudget(total_cost_budget_usd=10.0),
            model_registry=reg,
        )

        result = opt.select_model_for_step(complexity="medium")
        assert result is not None
        assert getattr(result, "tier", None) == "medium"

    def test_budget_pressure_composes_with_confidence(self):
        """Budget pressure and confidence compose — budget sets initial tier,
        confidence can still adjust within the budget-allowed range."""
        from agent_utilities.core.resource_optimizer import (
            ResourceBudget,
            ResourceOptimizer,
        )

        reg = _make_registry()
        budget = ResourceBudget(total_cost_budget_usd=10.0)
        budget.cost_used_usd = 9.5  # Only 5% remaining → forces "light"
        opt = ResourceOptimizer(budget=budget, model_registry=reg)

        # Low confidence tries to escalate, but starts from budget-forced "light"
        # light + escalation = medium (not heavy, since budget already limited)
        result = opt.select_model_for_step(
            complexity="heavy",
            confidence_signal=0.1,
        )
        assert result is not None
        assert (
            getattr(result, "tier", None) == "medium"
        )  # Budget forced light, confidence escalated once


# ── GraphState Observability ──────────────────────────────────────────


class TestRoutingConfidenceLog:
    """Tests for GraphState.routing_confidence_log field."""

    def test_routing_confidence_log_exists(self):
        from agent_utilities.graph.state import GraphState

        state = GraphState(query="test")
        assert hasattr(state, "routing_confidence_log")
        assert state.routing_confidence_log == []

    def test_routing_confidence_log_appendable(self):
        from agent_utilities.graph.state import GraphState

        state = GraphState(query="test")
        state.routing_confidence_log.append(
            {
                "specialist_id": "gitlab_specialist",
                "confidence": 0.85,
                "original_tier": "medium",
                "routed_tier": "light",
                "timestamp": "2026-05-04T14:00:00Z",
            }
        )
        assert len(state.routing_confidence_log) == 1
        assert state.routing_confidence_log[0]["confidence"] == 0.85


# ── GraphDeps Routing Percentile ──────────────────────────────────────


class TestGraphDepsRoutingPercentile:
    """Tests for GraphDeps.routing_percentile field."""

    def test_default_routing_percentile(self):
        from agent_utilities.graph.state import GraphDeps

        deps = GraphDeps(
            tag_prompts={},
            tag_env_vars={},
            mcp_toolsets=[],
        )
        assert deps.routing_percentile == 50.0
