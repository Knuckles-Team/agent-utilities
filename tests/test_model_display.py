#!/usr/bin/python
"""Tests for CONCEPT:KG-2.17 — Model Display Optimization.

Validates display strategies, linearization, hinge collapse, adaptive
display, and complexity budget enforcement.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.model_display import ModelDisplayOptimizer
from agent_utilities.models.imodel import (
    DisplayComplexityBudget,
    DisplayStrategy,
    ModelDisplayNode,
)
from agent_utilities.models.knowledge_graph import RegistryNodeType


# ─── ModelDisplayOptimizer Tests ─────────────────────────────────────


class TestModelDisplayOptimizer:
    """Tests for the ModelDisplayOptimizer."""

    def test_linear_collapse_strategy(self):
        """LINEAR_COLLAPSE should produce y = ax + b form."""
        opt = ModelDisplayOptimizer()
        result = opt.optimize_display(
            "Feature A: 2.5, Feature B: -1.3, intercept: 0.7",
            features=["A", "B", "intercept"],
            strategy=DisplayStrategy.LINEAR_COLLAPSE,
        )
        assert "y =" in result
        assert "A" in result

    def test_symbolic_equation_strategy(self):
        """SYMBOLIC_EQUATION should condense to a single line."""
        opt = ModelDisplayOptimizer()
        result = opt.optimize_display(
            "y = 2.5*x0 + 1.3*x1 + 0.7\nRMSE: 0.45\nR2: 0.92",
            strategy=DisplayStrategy.SYMBOLIC_EQUATION,
        )
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_coefficient_summary_strategy(self):
        """COEFFICIENT_SUMMARY should rank by magnitude."""
        opt = ModelDisplayOptimizer()
        result = opt.optimize_display(
            "Weights: 0.1, 5.0, 2.3, -3.7",
            features=["small", "large", "medium", "negative"],
            strategy=DisplayStrategy.COEFFICIENT_SUMMARY,
        )
        assert "Coefficients" in result
        # 5.0 should appear first (largest magnitude)
        lines = result.split("\n")
        first_feature_line = lines[1] if len(lines) > 1 else ""
        assert "5.0" in first_feature_line or "large" in first_feature_line

    def test_adaptive_strategy_truncation(self):
        """ADAPTIVE should truncate long outputs with summary."""
        budget = DisplayComplexityBudget(max_lines=5)
        opt = ModelDisplayOptimizer(budget=budget)
        long_str = "\n".join([f"line {i}: value {i * 0.1}" for i in range(20)])
        result = opt.optimize_display(long_str, strategy=DisplayStrategy.ADAPTIVE)
        lines = result.strip().split("\n")
        assert len(lines) <= 5
        assert "more lines" in lines[-1]

    def test_piecewise_table_strategy(self):
        """PIECEWISE_TABLE should preserve model structure."""
        opt = ModelDisplayOptimizer()
        model_str = "Feature A:\n  knot=0.0 → 0.5\n  knot=1.0 → 0.8"
        result = opt.optimize_display(
            model_str, strategy=DisplayStrategy.PIECEWISE_TABLE,
        )
        assert len(result) > 0


# ─── Linearization Tests ────────────────────────────────────────────


class TestLinearization:
    """Tests for the linearize_feature method."""

    def test_perfect_linear_data(self):
        """Perfectly linear data should linearize with R²=1.0."""
        opt = ModelDisplayOptimizer()
        # y = 2x: coefficients at knots [0, 1, 2, 3]
        coeffs = [0.0, 2.0, 4.0, 6.0]
        knots = [0.0, 1.0, 2.0, 3.0]
        slope, r2 = opt.linearize_feature(coeffs, knots, r2_threshold=0.90)
        assert slope is not None
        assert abs(slope - 2.0) < 0.01
        assert r2 >= 0.99

    def test_nonlinear_data_rejected(self):
        """Highly nonlinear data should be rejected (R² below threshold)."""
        opt = ModelDisplayOptimizer()
        # Quadratic-ish: not linearizable
        coeffs = [0.0, 1.0, 4.0, 9.0, 16.0]
        knots = [0.0, 1.0, 2.0, 3.0, 4.0]
        slope, r2 = opt.linearize_feature(coeffs, knots, r2_threshold=0.99)
        assert slope is None
        assert r2 < 0.99

    def test_single_coefficient(self):
        """Single coefficient should return it directly."""
        opt = ModelDisplayOptimizer()
        slope, r2 = opt.linearize_feature([3.5], [0.0])
        assert slope == 3.5
        assert r2 == 1.0

    def test_constant_coefficients(self):
        """Constant coefficients (slope=0) should linearize."""
        opt = ModelDisplayOptimizer()
        coeffs = [2.0, 2.0, 2.0, 2.0]
        knots = [0.0, 1.0, 2.0, 3.0]
        slope, r2 = opt.linearize_feature(coeffs, knots, r2_threshold=0.90)
        # Slope should be ~0 with perfect R²
        assert slope is not None
        assert abs(slope) < 0.01

    def test_empty_coefficients(self):
        """Empty coefficients should return None."""
        opt = ModelDisplayOptimizer()
        slope, r2 = opt.linearize_feature([], [])
        assert slope is None


# ─── Hinge Collapse Tests ───────────────────────────────────────────


class TestHingeCollapse:
    """Tests for the collapse_hinges method."""

    def test_collapse_single_feature(self):
        """Single feature hinges should collapse to net coefficient."""
        opt = ModelDisplayOptimizer()
        hinges = [
            {"feature": "x", "knot": 0.0, "coefficient": 0.5, "direction": "positive"},
            {"feature": "x", "knot": 1.0, "coefficient": -0.3, "direction": "negative"},
        ]
        result = opt.collapse_hinges(hinges)
        assert len(result) == 1
        assert result[0]["feature"] == "x"
        assert result[0]["net_coefficient"] == pytest.approx(0.2)
        assert result[0]["n_hinges"] == 2

    def test_collapse_multiple_features(self):
        """Hinges from multiple features should group correctly."""
        opt = ModelDisplayOptimizer()
        hinges = [
            {"feature": "x", "knot": 0.0, "coefficient": 1.0},
            {"feature": "y", "knot": 0.5, "coefficient": 2.0},
            {"feature": "x", "knot": 1.0, "coefficient": 0.5},
        ]
        result = opt.collapse_hinges(hinges)
        assert len(result) == 2
        by_feature = {r["feature"]: r for r in result}
        assert by_feature["x"]["net_coefficient"] == pytest.approx(1.5)
        assert by_feature["y"]["net_coefficient"] == pytest.approx(2.0)

    def test_collapse_empty(self):
        """Empty hinges should return empty list."""
        opt = ModelDisplayOptimizer()
        assert opt.collapse_hinges([]) == []

    def test_knot_range_tracking(self):
        """Collapsed hinges should track knot range."""
        opt = ModelDisplayOptimizer()
        hinges = [
            {"feature": "x", "knot": 0.5, "coefficient": 1.0},
            {"feature": "x", "knot": 3.0, "coefficient": 2.0},
            {"feature": "x", "knot": 1.5, "coefficient": 0.5},
        ]
        result = opt.collapse_hinges(hinges)
        assert result[0]["knot_range"] == (0.5, 3.0)


# ─── Adaptive Display Tests ─────────────────────────────────────────


class TestAdaptiveDisplay:
    """Tests for the adaptive display (SmartAdditive pattern)."""

    def test_linear_features_displayed_inline(self):
        """Features with high R² should get linear display."""
        opt = ModelDisplayOptimizer()
        feature_data = [
            {
                "name": "temp",
                "coefficients": [0.0, 1.0, 2.0, 3.0],
                "knot_values": [0.0, 1.0, 2.0, 3.0],
            },
        ]
        result = opt.adaptive_display(feature_data, r2_threshold=0.90)
        assert "temp" in result
        assert "R²=" in result

    def test_nonlinear_features_show_table(self):
        """Features with low R² should show piecewise table."""
        opt = ModelDisplayOptimizer()
        feature_data = [
            {
                "name": "pressure",
                "coefficients": [0.0, 1.0, 100.0, 0.5],
                "knot_values": [0.0, 1.0, 2.0, 3.0],
            },
        ]
        result = opt.adaptive_display(feature_data, r2_threshold=0.99)
        assert "nonlinear" in result.lower() or "knot" in result

    def test_mixed_features(self):
        """Mix of linear and nonlinear features should display correctly."""
        opt = ModelDisplayOptimizer()
        feature_data = [
            {
                "name": "linear_feat",
                "coefficients": [0.0, 2.0, 4.0, 6.0],
                "knot_values": [0.0, 1.0, 2.0, 3.0],
            },
            {
                "name": "nonlinear_feat",
                "coefficients": [0.0, 1.0, 100.0, 0.5],
                "knot_values": [0.0, 1.0, 2.0, 3.0],
            },
        ]
        result = opt.adaptive_display(feature_data, r2_threshold=0.99)
        assert "linear_feat" in result
        assert "nonlinear_feat" in result


# ─── Budget Enforcement Tests ────────────────────────────────────────


class TestBudgetEnforcement:
    """Tests for display complexity budget enforcement."""

    def test_max_lines_enforced(self):
        """Output should not exceed max_lines."""
        budget = DisplayComplexityBudget(max_lines=3)
        opt = ModelDisplayOptimizer(budget=budget)
        long_str = "\n".join([f"line {i}" for i in range(50)])
        result = opt.optimize_display(long_str, strategy=DisplayStrategy.ADAPTIVE)
        lines = result.strip().split("\n")
        assert len(lines) <= 3

    def test_max_features_in_coefficient_summary(self):
        """Coefficient summary should respect max_features."""
        budget = DisplayComplexityBudget(max_features=3)
        opt = ModelDisplayOptimizer(budget=budget)
        model_str = "1.0 2.0 3.0 4.0 5.0 6.0 7.0"
        result = opt.optimize_display(
            model_str,
            features=["a", "b", "c", "d", "e", "f", "g"],
            strategy=DisplayStrategy.COEFFICIENT_SUMMARY,
        )
        # Should only show 3 features
        feature_lines = [l for l in result.split("\n") if l.strip().startswith(("a", "b", "c", "d", "e", "f", "g"))]
        assert len(feature_lines) <= 3


# ─── ModelDisplayNode Tests ──────────────────────────────────────────


class TestModelDisplayNode:
    """Tests for the ModelDisplayNode KG model."""

    def test_node_type(self):
        """Node type should be MODEL_DISPLAY."""
        node = ModelDisplayNode(id="test:1", name="Test")
        assert node.type == RegistryNodeType.MODEL_DISPLAY

    def test_full_construction(self):
        """Full construction with all fields."""
        node = ModelDisplayNode(
            id="display:abc123",
            name="Display: adaptive",
            display_type=DisplayStrategy.ADAPTIVE,
            display_content="y = 2.5*x0 + 1.3*x1",
            complexity_budget=256,
            r_squared_threshold=0.95,
            features_displayed=["x0", "x1"],
            features_hidden=["residual"],
            imodel_node_id="imodel:parent",
            token_count=12,
        )
        assert node.display_type == DisplayStrategy.ADAPTIVE
        assert len(node.features_displayed) == 2
        assert node.token_count == 12
