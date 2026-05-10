#!/usr/bin/python
from __future__ import annotations

"""Agentic-iModels KG Node Models.

CONCEPT:AHE-3.15 — Agent-Interpretable Model Evolver
CONCEPT:AHE-3.16 — LLM-Graded Interpretability Tests
CONCEPT:KG-2.17  — Model Display Optimization

Pydantic models for persisting agent-interpretable ML model classes,
their interpretability test results, and optimized display representations
as first-class nodes in the Knowledge Graph.

Based on Microsoft Research's Agentic-iModels paper (arXiv:2605.03808):
Coding agents iteratively evolve scikit-learn-compatible regressors
optimized for both predictive accuracy and LLM readability via their
``__str__()`` output.

Integration points:
    - CONCEPT:AHE-3.2  (VariantPool): Tournament-selected model variants
    - CONCEPT:AHE-3.10 (RewardDecomposer): Accuracy as trajectory reward,
      interpretability as step reward
    - CONCEPT:KG-2.4   (EncPI): Vectorized model topology for cross-domain
      analogy matching
    - CONCEPT:KG-2.16  (SemanticSubsumption): Auto-classify evolved model
      types into OWL hierarchy

See docs/overview.md §CONCEPT:AHE-3.15.
"""


from enum import StrEnum

from pydantic import BaseModel, Field

from .knowledge_graph import RegistryNode, RegistryNodeType

# ---------------------------------------------------------------------------
# Display strategy enumeration (CONCEPT:KG-2.17)
# ---------------------------------------------------------------------------


class DisplayStrategy(StrEnum):
    """Display optimization strategy for model ``__str__()`` output.

    CONCEPT:KG-2.17 — Model Display Optimization

    Controls how the model's internal structure is rendered into a
    string representation optimized for LLM consumption. Based on
    the display-predict decoupling pattern from arXiv:2605.03808.
    """

    LINEAR_COLLAPSE = "linear_collapse"
    """Reduce all features to ``y = ax₀ + bx₁ + c`` form."""

    PIECEWISE_TABLE = "piecewise_table"
    """Show full piecewise corrections for nonlinear features."""

    SYMBOLIC_EQUATION = "symbolic_equation"
    """Single-row symbolic expression."""

    COEFFICIENT_SUMMARY = "coefficient_summary"
    """Feature coefficients with magnitude ranking."""

    ADAPTIVE = "adaptive"
    """Per-feature decision based on R² threshold (SmartAdditive pattern)."""


# ---------------------------------------------------------------------------
# Interpretability test category enumeration (CONCEPT:AHE-3.16)
# ---------------------------------------------------------------------------


class InterpretabilityTestCategory(StrEnum):
    """Categories of LLM-graded interpretability tests.

    CONCEPT:AHE-3.16 — LLM-Graded Interpretability Tests

    Six test categories following the protocol in arXiv:2605.03808.
    Each category exercises a different dimension of whether an LLM
    can reliably simulate the model's behavior from its ``__str__()``
    output alone.
    """

    FEATURE_ATTRIBUTION = "feature_attribution"
    """Most important feature, ranking, irrelevant detection, sign effects."""

    POINT_SIMULATION = "point_simulation"
    """Predict output for specific input vectors (1-20 features)."""

    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    """Direction of change, unit sensitivity, nonlinear thresholds."""

    COUNTERFACTUAL = "counterfactual"
    """What input value would produce target output Y?"""

    CONFIDENCE_CALIBRATION = "confidence_calibration"
    """Model uncertainty, prediction intervals."""

    DATA_ATTRIBUTION = "data_attribution"
    """Which training points influenced the prediction."""


# ---------------------------------------------------------------------------
# KG Node Models
# ---------------------------------------------------------------------------


class IModelNode(RegistryNode):
    """A KG node representing an evolved agent-interpretable model class.

    CONCEPT:AHE-3.15 — Agent-Interpretable Model Evolver

    Stores the full lifecycle metadata of an evolved scikit-learn-compatible
    model class: source code, fitted ``__str__()`` representation, dual-axis
    fitness scores (predictive accuracy + agent interpretability), and
    evolutionary lineage.

    The model's ``str_representation`` is the key artifact — downstream
    LLM agents can simulate predictions, feature effects, and counterfactuals
    solely from reading this string, without invoking the model or external
    tools.

    Evolutionary edges:
        - ``EVOLVED_MODEL``: Links this model to its parent (the model it
          was mutated from during the autoresearch loop).
        - ``PARETO_DOMINATES``: Links this model to any model it dominates
          on both accuracy and interpretability axes.

    Attributes:
        model_class_name: Python class name (e.g., ``'HingeEBM'``).
        str_representation: The ``__str__()`` output from a fitted instance.
        source_code: Full Python source of the model class.
        interpretability_score: Agent interpretability score (0.0–1.0),
            computed as the pass rate across 200 LLM-graded tests.
        predictive_rank: Normalized RMSE rank across datasets (0.0–1.0,
            lower is better).
        pareto_optimal: Whether this model is on the Pareto frontier
            (not dominated on both axes by any other model).
        display_strategy: The display optimization strategy used for
            the ``__str__()`` output.
        generation: Evolutionary generation counter (0 = base model).
        evolve_agent_id: ID of the coding agent that discovered this model.
        fit_datasets: List of dataset identifiers this model was evaluated on.
        sklearn_compatible: Whether it implements ``fit()``, ``predict()``,
            and ``__str__()``.
        feature_names: Feature names used during fitting.
        n_features: Number of features the model was trained on.
        accuracy_metrics: Dict of metric_name → value (e.g., rmse, r2).
    """

    type: RegistryNodeType = RegistryNodeType.IMODEL
    model_class_name: str = ""
    str_representation: str = ""
    source_code: str = ""
    interpretability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    predictive_rank: float = Field(default=1.0, ge=0.0, le=1.0)
    pareto_optimal: bool = False
    display_strategy: DisplayStrategy = DisplayStrategy.ADAPTIVE
    generation: int = 0
    evolve_agent_id: str = ""
    fit_datasets: list[str] = Field(default_factory=list)
    sklearn_compatible: bool = True
    feature_names: list[str] = Field(default_factory=list)
    n_features: int = 0
    accuracy_metrics: dict[str, float] = Field(default_factory=dict)


class InterpretabilityTestNode(RegistryNode):
    """A KG node representing an individual interpretability test result.

    CONCEPT:AHE-3.16 — LLM-Graded Interpretability Tests

    Records the result of a single quantitative interpretability test:
    whether an LLM agent could correctly answer a specific question about
    a model solely from its ``__str__()`` output. Test results are linked
    to their parent ``IModelNode`` via ``TESTED_INTERPRETABILITY`` edges.

    The 6 test categories (feature_attribution, point_simulation,
    sensitivity_analysis, counterfactual, confidence_calibration,
    data_attribution) follow the protocol in arXiv:2605.03808.

    Attributes:
        test_category: One of the six interpretability test categories.
        test_query: The quantitative question asked to the LLM.
        ground_truth: The correct numerical/categorical answer.
        llm_response: The LLM's actual response.
        passed: Whether the test passed (within numerical tolerance).
        tolerance: Numerical tolerance used for grading (e.g., 0.05).
        evaluator_model: Which LLM model evaluated the response.
        model_str_hash: SHA-256 hash of the model ``__str__()`` input.
        imodel_node_id: ID of the parent IModelNode.
    """

    type: RegistryNodeType = RegistryNodeType.INTERPRETABILITY_TEST
    test_category: InterpretabilityTestCategory = (
        InterpretabilityTestCategory.POINT_SIMULATION
    )
    test_query: str = ""
    ground_truth: str = ""
    llm_response: str = ""
    passed: bool = False
    tolerance: float = 0.05
    evaluator_model: str = ""
    model_str_hash: str = ""
    imodel_node_id: str = ""


class ModelDisplayNode(RegistryNode):
    """A KG node for an optimized display representation of a model.

    CONCEPT:KG-2.17 — Model Display Optimization

    Implements the paper's key insight: display optimization is a
    first-class design axis. A model's ``__str__()`` output can be
    optimized for LLM consumption independently of its ``predict()``
    logic. This node stores the optimized display along with the
    complexity budget and strategy metadata.

    The display-predict decoupling pattern enables:
        - **Linear collapse**: Reduce nonlinear effects to effective
          linear slopes when R² > threshold.
        - **Adaptive display**: Per-feature gating — linear when
          linearization R² passes, piecewise table otherwise.
        - **Bounded complexity**: Architectural caps on token count
          to fit within LLM context windows.

    Attributes:
        display_type: The display strategy used.
        display_content: The actual optimized string content.
        complexity_budget: Max tokens/characters allowed.
        r_squared_threshold: R² threshold for linearization.
        features_displayed: Features included in the display.
        features_hidden: Features hidden (e.g., residual correctors).
        imodel_node_id: ID of the parent IModelNode.
        token_count: Actual token count of the display content.
    """

    type: RegistryNodeType = RegistryNodeType.MODEL_DISPLAY
    display_type: DisplayStrategy = DisplayStrategy.ADAPTIVE
    display_content: str = ""
    complexity_budget: int = 512
    r_squared_threshold: float = 0.90
    features_displayed: list[str] = Field(default_factory=list)
    features_hidden: list[str] = Field(default_factory=list)
    imodel_node_id: str = ""
    token_count: int = 0


# ---------------------------------------------------------------------------
# Supporting models (not KG nodes, used in-memory by the harness)
# ---------------------------------------------------------------------------


class IModelCandidate(BaseModel):
    """A candidate model under evaluation in the autoresearch loop.

    CONCEPT:AHE-3.15 — Agent-Interpretable Model Evolver

    Wraps a scikit-learn-compatible model specification with dual-axis
    fitness scores. Used transiently during evolution rounds before
    the winner is persisted as an ``IModelNode``.

    Attributes:
        model_class_name: Python class name.
        source_code: Full Python source of the model class.
        str_output: The ``__str__()`` output from a fitted instance.
        rmse_scores: Per-dataset RMSE values.
        interpretability_score: Agent interpretability pass rate.
        predictive_rank: Normalized rank (0.0 = best, 1.0 = worst).
        generation: Evolutionary generation.
        parent_id: ID of the parent model in the evolutionary lineage.
        display_strategy: Display optimization strategy used.
        sklearn_compatible: Whether it implements fit/predict/__str__.
    """

    model_class_name: str
    source_code: str = ""
    str_output: str = ""
    rmse_scores: dict[str, float] = Field(default_factory=dict)
    interpretability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    predictive_rank: float = Field(default=1.0, ge=0.0, le=1.0)
    generation: int = 0
    parent_id: str = ""
    display_strategy: DisplayStrategy = DisplayStrategy.ADAPTIVE
    sklearn_compatible: bool = True


class DisplayComplexityBudget(BaseModel):
    """Controls display size for model ``__str__()`` output.

    CONCEPT:KG-2.17 — Model Display Optimization

    Implements Pattern 1 from arXiv:2605.03808: bounded display
    complexity. Architectural caps on the size and detail of the
    model string representation ensure it fits within LLM context
    windows and remains simulable.

    Attributes:
        max_tokens: Maximum token count for ``__str__()`` output.
        max_features: Maximum features to display.
        max_knots: Maximum quantile knots per feature.
        round_digits: Coefficient rounding precision.
        max_lines: Maximum lines in the display.
    """

    max_tokens: int = 512
    max_features: int = 20
    max_knots: int = 10
    round_digits: int = 4
    max_lines: int = 50


class ParetoPoint(BaseModel):
    """A single point on the Pareto frontier.

    Attributes:
        model_id: ID of the IModelNode.
        model_class_name: Python class name.
        predictive_rank: Normalized RMSE rank (lower = better).
        interpretability_score: Agent interpretability pass rate.
        generation: Evolutionary generation.
    """

    model_id: str
    model_class_name: str = ""
    predictive_rank: float = Field(default=1.0, ge=0.0, le=1.0)
    interpretability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    generation: int = 0
