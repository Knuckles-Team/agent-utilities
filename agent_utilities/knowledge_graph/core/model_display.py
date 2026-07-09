#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-KG.compute.model-display-optimization — Model Display Optimization.

Implements the display-predict decoupling pattern from arXiv:2605.03808.
Optimizes model ``__str__()`` output for LLM consumption independently
of ``predict()`` logic.

Key patterns from the paper:
    - Pattern 1: Bounded display complexity (max tokens/features/knots)
    - Pattern 2: Display-predict decoupling (separate optimization axes)
    - Pattern 3: Reward hacking resistance (SmartAdditive adaptive display)

Integrates with:
    - CONCEPT:AU-KG.memory.tiered-memory-caching (ContextCompactor): Uses display budgets in prompts
    - CONCEPT:AU-KG.ingest.engineering-rules (SemanticSubsumption): Classifies display strategies
    - CONCEPT:AU-KG.compute.spectral-cluster-navigator (AnalogyEngine): Finds similar model displays

See docs/overview.md §CONCEPT:AU-KG.compute.model-display-optimization
"""


import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from ...models.imodel import (
    DisplayComplexityBudget,
    DisplayStrategy,
    ModelDisplayNode,
)
from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType

if TYPE_CHECKING:
    from .engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class ModelDisplayOptimizer:
    """Optimizes model ``__str__()`` for LLM consumption.

    CONCEPT:AU-KG.compute.model-display-optimization — Model Display Optimization

    Implements display-predict decoupling: the display representation
    of a model can be optimized for agent readability independently
    of its predictive logic. The optimizer applies strategy-specific
    transformations to reduce complexity while preserving simulability.

    Strategies:
        - ``LINEAR_COLLAPSE``: Reduce nonlinear effects to linear slopes
        - ``PIECEWISE_TABLE``: Full piecewise correction tables
        - ``SYMBOLIC_EQUATION``: Single-row symbolic formula
        - ``COEFFICIENT_SUMMARY``: Feature coefficients ranked by magnitude
        - ``ADAPTIVE``: Per-feature decision (SmartAdditive pattern)

    Args:
        budget: Display complexity budget.
        engine: Optional KG engine for persistence.
    """

    def __init__(
        self,
        budget: DisplayComplexityBudget | None = None,
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        self._budget = budget or DisplayComplexityBudget()
        self._engine = engine

    @property
    def budget(self) -> DisplayComplexityBudget:
        return self._budget

    def optimize_display(
        self,
        model_str: str,
        features: list[str] | None = None,
        strategy: DisplayStrategy = DisplayStrategy.ADAPTIVE,
    ) -> str:
        """Generate an optimized display for a model.

        Args:
            model_str: The raw ``__str__()`` output from the model.
            features: Feature names (for feature-level optimization).
            strategy: The display strategy to apply.

        Returns:
            Optimized display string.
        """
        if strategy == DisplayStrategy.LINEAR_COLLAPSE:
            return self._linear_collapse(model_str, features)
        elif strategy == DisplayStrategy.SYMBOLIC_EQUATION:
            return self._symbolic_equation(model_str, features)
        elif strategy == DisplayStrategy.COEFFICIENT_SUMMARY:
            return self._coefficient_summary(model_str, features)
        elif strategy == DisplayStrategy.PIECEWISE_TABLE:
            return self._piecewise_table(model_str, features)
        elif strategy == DisplayStrategy.ADAPTIVE:
            return self._adaptive_display(model_str, features)
        else:
            return self._enforce_budget(model_str)

    def linearize_feature(
        self,
        coefficients: list[float],
        knot_values: list[float],
        r2_threshold: float = 0.90,
    ) -> tuple[float | None, float]:
        """Collapse a nonlinear feature function to a linear coefficient.

        Given a piecewise-linear function defined by coefficients at
        knot points, fits a simple linear model. If R² exceeds the
        threshold, the linearization is accepted.

        Args:
            coefficients: Coefficient values at each knot.
            knot_values: Knot positions (x-values).
            r2_threshold: Minimum R² for accepting linearization.

        Returns:
            Tuple of (linear_slope or None if rejected, r2_value).
        """
        if len(coefficients) < 2 or len(knot_values) < 2:
            return (coefficients[0] if coefficients else None, 1.0)

        n = min(len(coefficients), len(knot_values))
        x_vals = knot_values[:n]
        y_vals = coefficients[:n]

        # Simple linear regression: y = mx + b
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)
        ss_xy = sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals, strict=False)
        )

        if ss_xx == 0:
            return (y_mean, 1.0)

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

        # Compute R²
        y_pred = [slope * x + intercept for x in x_vals]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_vals, y_pred, strict=False))
        ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        if r2 >= r2_threshold:
            return (slope, r2)
        return (None, r2)

    def collapse_hinges(
        self,
        hinge_terms: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reduce hinge basis terms to effective linear slopes.

        Each hinge term is a dict with:
            - ``feature``: Feature name
            - ``knot``: Hinge point
            - ``coefficient``: Coefficient value
            - ``direction``: 'positive' or 'negative' hinge

        Returns simplified terms where consecutive hinges are collapsed.
        """
        if not hinge_terms:
            return []

        # Group by feature
        by_feature: dict[str, list[dict[str, Any]]] = {}
        for term in hinge_terms:
            feat = term.get("feature", "unknown")
            by_feature.setdefault(feat, []).append(term)

        collapsed: list[dict[str, Any]] = []
        for feature, terms in by_feature.items():
            # Sort by knot position
            sorted_terms = sorted(terms, key=lambda t: t.get("knot", 0.0))
            # Sum all coefficients for a net linear effect
            net_coef = sum(t.get("coefficient", 0.0) for t in sorted_terms)
            collapsed.append(
                {
                    "feature": feature,
                    "net_coefficient": net_coef,
                    "n_hinges": len(sorted_terms),
                    "knot_range": (
                        sorted_terms[0].get("knot", 0.0),
                        sorted_terms[-1].get("knot", 0.0),
                    )
                    if sorted_terms
                    else (0.0, 0.0),
                }
            )

        return collapsed

    def adaptive_display(
        self,
        feature_data: list[dict[str, Any]],
        r2_threshold: float = 0.90,
    ) -> str:
        """Per-feature adaptive display (SmartAdditive pattern).

        For each feature, decides whether to show:
            - Linear coefficient (if linearization R² > threshold)
            - Full piecewise table (if nonlinear contribution is significant)

        Args:
            feature_data: List of feature dicts with 'name', 'coefficients',
                'knot_values' keys.
            r2_threshold: R² threshold for accepting linearization.

        Returns:
            Formatted display string.
        """
        lines = ["Prediction = intercept"]
        linear_features: list[str] = []
        nonlinear_features: list[str] = []

        for feat in feature_data:
            name = feat.get("name", "?")
            coefficients = feat.get("coefficients", [])
            knot_values = feat.get("knot_values", [])

            slope, r2 = self.linearize_feature(
                coefficients,
                knot_values,
                r2_threshold,
            )

            if slope is not None:
                sign = "+" if slope >= 0 else "-"
                lines.append(f"  {sign} {abs(slope):.4f} * {name}  (R²={r2:.3f})")
                linear_features.append(name)
            else:
                # Show piecewise table
                lines.append(f"  + f({name}):  [nonlinear, R²={r2:.3f}]")
                for i, (k, c) in enumerate(
                    zip(
                        knot_values[: self._budget.max_knots],
                        coefficients,
                        strict=False,
                    )
                ):
                    lines.append(f"      knot={k:.2f} → coef={c:.4f}")
                nonlinear_features.append(name)

        return "\n".join(lines[: self._budget.max_lines])

    def persist_display(
        self,
        imodel_node_id: str,
        display_content: str,
        strategy: DisplayStrategy,
        features_displayed: list[str] | None = None,
        features_hidden: list[str] | None = None,
    ) -> str | None:
        """Persist a ModelDisplayNode to the KG.

        Args:
            imodel_node_id: ID of the parent IModelNode.
            display_content: The optimized display string.
            strategy: Display strategy used.
            features_displayed: Features shown in display.
            features_hidden: Features hidden from display.

        Returns:
            ID of the persisted node, or None on failure.
        """
        if not self._engine:
            return None

        try:
            from .ogm import KGMapper

            ogm = KGMapper(self._engine)

            node_id = f"display:{uuid.uuid4().hex[:8]}"
            node = ModelDisplayNode(
                id=node_id,
                type=RegistryNodeType.MODEL_DISPLAY,
                name=f"Display: {strategy.value}",
                display_type=strategy,
                display_content=display_content,
                complexity_budget=self._budget.max_tokens,
                r_squared_threshold=0.90,
                features_displayed=features_displayed or [],
                features_hidden=features_hidden or [],
                imodel_node_id=imodel_node_id,
                token_count=len(display_content.split()),
                metadata={
                    "concept": "EG-KG.compute.compiled-semantic-reasoner",
                    "paper": "arXiv:2605.03808",
                },
            )
            ogm.upsert(node)
            ogm.upsert_edge(
                node_id,
                imodel_node_id,
                RegistryEdgeType.DISPLAY_OF,
                {"strategy": strategy.value},
            )
            return node_id
        except Exception as exc:
            logger.warning("Failed to persist ModelDisplayNode: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private strategy implementations
    # ------------------------------------------------------------------

    def _linear_collapse(
        self,
        model_str: str,
        features: list[str] | None,
    ) -> str:
        """Collapse model to linear equation form: y = ax₀ + bx₁ + c."""
        # Extract numeric coefficients from the string
        numbers = re.findall(r"[-+]?\d*\.?\d+", model_str)
        if not numbers:
            return self._enforce_budget(model_str)

        coeffs = [float(n) for n in numbers[: self._budget.max_features]]
        feat_names = features or [f"x{i}" for i in range(len(coeffs))]

        terms = []
        for i, coef in enumerate(coeffs):
            if i < len(feat_names):
                sign = "+" if coef >= 0 else "-"
                terms.append(
                    f"{sign} {abs(coef):.{self._budget.round_digits}f}·{feat_names[i]}"
                )

        return f"y = {' '.join(terms)}" if terms else model_str

    def _symbolic_equation(
        self,
        model_str: str,
        features: list[str] | None,
    ) -> str:
        """Condense model to a single symbolic equation line."""
        # Take only the first meaningful line
        lines = [line.strip() for line in model_str.strip().split("\n") if line.strip()]
        if not lines:
            return model_str
        equation = lines[0]
        return self._enforce_budget(equation)

    def _coefficient_summary(
        self,
        model_str: str,
        features: list[str] | None,
    ) -> str:
        """Extract and rank coefficients by magnitude."""
        numbers = re.findall(r"([-+]?\d*\.?\d+)", model_str)
        if not numbers:
            return self._enforce_budget(model_str)

        coeffs = [(abs(float(n)), float(n)) for n in numbers]
        coeffs.sort(reverse=True)

        feat_names = features or [f"feature_{i}" for i in range(len(coeffs))]
        lines = ["Coefficients (ranked by magnitude):"]
        for i, (abs_val, val) in enumerate(coeffs[: self._budget.max_features]):
            name = feat_names[i] if i < len(feat_names) else f"feature_{i}"
            lines.append(f"  {name}: {val:+.{self._budget.round_digits}f}")

        return "\n".join(lines[: self._budget.max_lines])

    def _piecewise_table(
        self,
        model_str: str,
        features: list[str] | None,
    ) -> str:
        """Format model as a piecewise correction table."""
        return self._enforce_budget(model_str)

    def _adaptive_display(
        self,
        model_str: str,
        features: list[str] | None,
    ) -> str:
        """Apply adaptive per-feature display strategy."""
        lines = model_str.strip().split("\n")
        if len(lines) <= self._budget.max_lines:
            return model_str
        # Truncate with summary
        kept = lines[: self._budget.max_lines - 1]
        kept.append(f"... ({len(lines) - len(kept)} more lines)")
        return "\n".join(kept)

    def _enforce_budget(self, content: str) -> str:
        """Enforce display complexity budget on content."""
        lines = content.split("\n")
        if len(lines) > self._budget.max_lines:
            lines = lines[: self._budget.max_lines - 1]
            lines.append(f"... (truncated to {self._budget.max_lines} lines)")
        result = "\n".join(lines)
        # Rough token check
        tokens = result.split()
        if len(tokens) > self._budget.max_tokens:
            tokens = tokens[: self._budget.max_tokens]
            result = " ".join(tokens) + " ..."
        return result
