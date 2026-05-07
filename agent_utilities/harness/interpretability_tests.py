#!/usr/bin/python
"""CONCEPT:AHE-3.16 — LLM-Graded Interpretability Tests.

Implements the 6-category interpretability testing protocol from
arXiv:2605.03808. Tests whether an LLM can correctly answer quantitative
questions about a model solely from its ``__str__()`` output.

Integrates with EvalRunner (CONCEPT:AHE-3.12) for LLM-as-Judge scoring
and EvaluationMonitor for trend tracking.

See docs/overview.md §CONCEPT:AHE-3.16.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from ..models.imodel import (
    InterpretabilityTestCategory,
    InterpretabilityTestNode,
)
from ..models.knowledge_graph import RegistryEdgeType, RegistryNodeType

if TYPE_CHECKING:
    from ..knowledge_graph.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test Case Definition
# ---------------------------------------------------------------------------


class InterpretabilityTestCase:
    """A single interpretability test case.

    Attributes:
        category: The test category.
        query: The quantitative question.
        ground_truth: The correct answer (string for flexible matching).
        tolerance: Numerical tolerance for grading (e.g., 0.05 = 5%).
    """

    def __init__(
        self,
        category: InterpretabilityTestCategory,
        query: str,
        ground_truth: str,
        tolerance: float = 0.05,
    ) -> None:
        self.category = category
        self.query = query
        self.ground_truth = ground_truth
        self.tolerance = tolerance


# ---------------------------------------------------------------------------
# Interpretability Grader
# ---------------------------------------------------------------------------


class InterpretabilityGrader:
    """LLM-based grading for interpretability test responses.

    CONCEPT:AHE-3.16 — LLM-Graded Interpretability Tests

    Uses the EvalRunner (CONCEPT:AHE-3.12) infrastructure for
    LLM-as-Judge scoring with numerical tolerance. Detects
    reward hacking when the model's ``__str__()`` directly
    contains the answer.

    Args:
        evaluator_model: Name of the LLM used for grading.
    """

    def __init__(self, evaluator_model: str = "default") -> None:
        self.evaluator_model = evaluator_model

    def grade(
        self,
        response: str,
        ground_truth: str,
        tolerance: float = 0.05,
    ) -> tuple[bool, str]:
        """Grade a single response against ground truth.

        Supports both numerical and categorical matching.

        Args:
            response: The LLM's response to the test question.
            ground_truth: The correct answer.
            tolerance: Relative tolerance for numerical answers.

        Returns:
            Tuple of (passed, reasoning).
        """
        response_clean = response.strip().lower()
        truth_clean = ground_truth.strip().lower()

        # Try numerical comparison first
        try:
            resp_val = float(response_clean.replace(",", ""))
            truth_val = float(truth_clean.replace(",", ""))

            if truth_val == 0.0:
                passed = abs(resp_val) <= tolerance
            else:
                relative_error = abs(resp_val - truth_val) / abs(truth_val)
                passed = relative_error <= tolerance

            return (
                passed,
                f"Numerical: response={resp_val:.4f}, truth={truth_val:.4f}, "
                f"tolerance={tolerance:.2%}, "
                f"error={abs(resp_val - truth_val) / max(abs(truth_val), 1e-10):.2%}",
            )
        except (ValueError, ZeroDivisionError):
            pass

        # Categorical/string comparison
        if response_clean == truth_clean:
            return True, "Exact string match"

        # Token overlap for partial matching
        resp_tokens = set(response_clean.split())
        truth_tokens = set(truth_clean.split())
        if truth_tokens and resp_tokens:
            overlap = len(resp_tokens & truth_tokens) / len(truth_tokens)
            if overlap >= 0.8:
                return True, f"Token overlap: {overlap:.0%}"

        return False, f"No match: '{response_clean}' vs '{truth_clean}'"

    def detect_reward_hacking(
        self,
        model_str: str,
        ground_truth: str,
    ) -> bool:
        """Detect if the model's __str__() directly contains the answer.

        Reward hacking occurs when a model's display representation
        is designed to directly recite test answers rather than
        expressing genuine structure. This is Pattern 3 from the paper.

        Args:
            model_str: The model's ``__str__()`` output.
            ground_truth: The test's correct answer.

        Returns:
            True if potential reward hacking detected.
        """
        truth_clean = ground_truth.strip()
        if len(truth_clean) < 3:
            return False
        return truth_clean in model_str


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class InterpretabilityTestSuite:
    """Manages the full interpretability test protocol.

    CONCEPT:AHE-3.16 — LLM-Graded Interpretability Tests

    Implements the 6-category, 200-test protocol from arXiv:2605.03808:
        1. Feature Attribution (32 tests)
        2. Point Simulation (43 tests)
        3. Sensitivity Analysis (32 tests)
        4. Counterfactual (32 tests)
        5. Confidence Calibration (32 tests)
        6. Data Attribution (29 tests)

    The suite generates test cases, runs them against a model's
    ``__str__()`` output, and computes the aggregate agent
    interpretability score (pass rate).

    Args:
        grader: The grader to use for test evaluation.
        engine: Optional KG engine for persisting test results.
    """

    # Default test distribution per category (200 total)
    DEFAULT_TEST_COUNTS: dict[InterpretabilityTestCategory, int] = {
        InterpretabilityTestCategory.FEATURE_ATTRIBUTION: 32,
        InterpretabilityTestCategory.POINT_SIMULATION: 43,
        InterpretabilityTestCategory.SENSITIVITY_ANALYSIS: 32,
        InterpretabilityTestCategory.COUNTERFACTUAL: 32,
        InterpretabilityTestCategory.CONFIDENCE_CALIBRATION: 32,
        InterpretabilityTestCategory.DATA_ATTRIBUTION: 29,
    }

    def __init__(
        self,
        grader: InterpretabilityGrader | None = None,
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        self._grader = grader or InterpretabilityGrader()
        self._engine = engine
        self._results: list[dict[str, Any]] = []

    @property
    def results(self) -> list[dict[str, Any]]:
        return list(self._results)

    def generate_feature_attribution_tests(
        self,
        feature_names: list[str],
        coefficients: list[float],
    ) -> list[InterpretabilityTestCase]:
        """Generate feature attribution test cases from known coefficients.

        Tests include:
            - Most important feature (largest absolute coefficient)
            - Feature ranking by importance
            - Sign of feature effects
            - Detection of irrelevant features (zero coefficient)

        Args:
            feature_names: List of feature names.
            coefficients: Corresponding coefficient values.

        Returns:
            List of test cases.
        """
        tests: list[InterpretabilityTestCase] = []
        if not feature_names or not coefficients:
            return tests

        paired = list(zip(feature_names, coefficients))
        abs_sorted = sorted(paired, key=lambda x: abs(x[1]), reverse=True)

        # Most important feature
        tests.append(
            InterpretabilityTestCase(
                category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                query="Which feature has the largest absolute effect on the prediction?",
                ground_truth=abs_sorted[0][0],
            )
        )

        # Feature ranking (top-3)
        top3 = [name for name, _ in abs_sorted[:3]]
        tests.append(
            InterpretabilityTestCase(
                category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                query="List the top 3 most important features in order.",
                ground_truth=", ".join(top3),
            )
        )

        # Sign effects
        for name, coef in paired[:5]:
            sign = "positive" if coef > 0 else "negative" if coef < 0 else "zero"
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                    query=f"What is the sign of the effect of feature '{name}'?",
                    ground_truth=sign,
                )
            )

        # Irrelevant features (zero coefficients)
        zero_features = [name for name, coef in paired if coef == 0.0]
        if zero_features:
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                    query="Which features have zero effect on the prediction?",
                    ground_truth=", ".join(zero_features),
                )
            )

        return tests

    def generate_point_simulation_tests(
        self,
        inputs: list[dict[str, float]],
        outputs: list[float],
    ) -> list[InterpretabilityTestCase]:
        """Generate point simulation tests from known input-output pairs.

        Args:
            inputs: List of input feature dicts.
            outputs: Corresponding predicted outputs.

        Returns:
            List of test cases.
        """
        tests: list[InterpretabilityTestCase] = []
        for inp, out in zip(inputs, outputs):
            features_str = ", ".join(f"{k}={v}" for k, v in inp.items())
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.POINT_SIMULATION,
                    query=f"What does the model predict for input: {features_str}?",
                    ground_truth=f"{out:.4f}",
                    tolerance=0.05,
                )
            )
        return tests

    def generate_sensitivity_tests(
        self,
        feature_names: list[str],
        sensitivities: dict[str, float],
    ) -> list[InterpretabilityTestCase]:
        """Generate sensitivity analysis tests.

        Args:
            feature_names: Feature names.
            sensitivities: Feature name → unit sensitivity value.

        Returns:
            List of test cases.
        """
        tests: list[InterpretabilityTestCase] = []
        for name in feature_names:
            if name not in sensitivities:
                continue
            sensitivity = sensitivities[name]
            direction = "increase" if sensitivity > 0 else "decrease"
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.SENSITIVITY_ANALYSIS,
                    query=(
                        f"If feature '{name}' increases by 1 unit, "
                        f"does the prediction increase or decrease?"
                    ),
                    ground_truth=direction,
                )
            )
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.SENSITIVITY_ANALYSIS,
                    query=(
                        f"By how much does the prediction change if "
                        f"feature '{name}' increases by 1 unit?"
                    ),
                    ground_truth=f"{sensitivity:.4f}",
                    tolerance=0.10,
                )
            )
        return tests

    def generate_counterfactual_tests(
        self,
        feature_names: list[str],
        counterfactuals: list[dict[str, Any]],
    ) -> list[InterpretabilityTestCase]:
        """Generate counterfactual test cases.

        Args:
            feature_names: Feature names.
            counterfactuals: List of dicts with 'target_output', 'feature',
                'required_value'.

        Returns:
            List of test cases.
        """
        tests: list[InterpretabilityTestCase] = []
        for cf in counterfactuals:
            tests.append(
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory.COUNTERFACTUAL,
                    query=(
                        f"What value of feature '{cf['feature']}' would produce "
                        f"a prediction of {cf['target_output']}?"
                    ),
                    ground_truth=f"{cf['required_value']:.4f}",
                    tolerance=0.10,
                )
            )
        return tests

    def run_test(
        self,
        model_str: str,
        test_case: InterpretabilityTestCase,
        llm_response: str,
    ) -> dict[str, Any]:
        """Run a single interpretability test and grade the response.

        Args:
            model_str: The model's ``__str__()`` output.
            test_case: The test case to evaluate.
            llm_response: The LLM's response to the test question.

        Returns:
            Dict with test results including passed, reasoning, etc.
        """
        passed, reasoning = self._grader.grade(
            llm_response,
            test_case.ground_truth,
            test_case.tolerance,
        )
        reward_hack = self._grader.detect_reward_hacking(
            model_str,
            test_case.ground_truth,
        )

        result = {
            "category": test_case.category.value,
            "query": test_case.query,
            "ground_truth": test_case.ground_truth,
            "llm_response": llm_response,
            "passed": passed and not reward_hack,
            "reasoning": reasoning,
            "reward_hacking_detected": reward_hack,
            "tolerance": test_case.tolerance,
            "timestamp": time.time(),
        }
        self._results.append(result)
        return result

    def run_suite(
        self,
        model_str: str,
        test_cases: list[InterpretabilityTestCase],
        llm_responses: list[str],
    ) -> dict[str, Any]:
        """Run a full suite of tests and compute aggregate score.

        Args:
            model_str: The model's ``__str__()`` output.
            test_cases: All test cases.
            llm_responses: Corresponding LLM responses.

        Returns:
            Dict with per-category and aggregate results.
        """
        if len(test_cases) != len(llm_responses):
            raise ValueError(
                f"Mismatch: {len(test_cases)} tests vs {len(llm_responses)} responses"
            )

        results = []
        for tc, resp in zip(test_cases, llm_responses):
            results.append(self.run_test(model_str, tc, resp))

        return self.compute_agent_interpretability_score(results)

    def compute_agent_interpretability_score(
        self,
        results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compute aggregate interpretability score from test results.

        Returns the overall pass rate and per-category breakdown.

        Args:
            results: Test results. Defaults to all recorded results.

        Returns:
            Dict with ``overall_score``, ``per_category``, ``total_tests``,
            ``total_passed``, ``reward_hacking_count``.
        """
        pool = results if results is not None else self._results
        if not pool:
            return {
                "overall_score": 0.0,
                "per_category": {},
                "total_tests": 0,
                "total_passed": 0,
                "reward_hacking_count": 0,
            }

        total = len(pool)
        passed = sum(1 for r in pool if r.get("passed", False))
        hacked = sum(1 for r in pool if r.get("reward_hacking_detected", False))

        # Per-category breakdown
        categories: dict[str, dict[str, int]] = {}
        for r in pool:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if r.get("passed", False):
                categories[cat]["passed"] += 1

        per_category = {
            cat: {
                "total": counts["total"],
                "passed": counts["passed"],
                "score": counts["passed"] / counts["total"] if counts["total"] else 0.0,
            }
            for cat, counts in categories.items()
        }

        return {
            "overall_score": passed / total if total else 0.0,
            "per_category": per_category,
            "total_tests": total,
            "total_passed": passed,
            "reward_hacking_count": hacked,
        }

    def persist_results_to_kg(
        self,
        imodel_node_id: str,
        results: list[dict[str, Any]] | None = None,
    ) -> int:
        """Persist test results as InterpretabilityTestNode in the KG.

        Args:
            imodel_node_id: ID of the parent IModelNode.
            results: Results to persist. Defaults to all recorded.

        Returns:
            Number of nodes persisted.
        """
        if not self._engine:
            return 0

        pool = results if results is not None else self._results
        persisted = 0

        try:
            from ..knowledge_graph.ogm import KGMapper

            ogm = KGMapper(self._engine)

            for r in pool:
                node_id = f"itest:{uuid.uuid4().hex[:8]}"
                cat_str = r.get("category", "point_simulation")
                try:
                    cat = InterpretabilityTestCategory(cat_str)
                except ValueError:
                    cat = InterpretabilityTestCategory.POINT_SIMULATION

                node = InterpretabilityTestNode(
                    id=node_id,
                    type=RegistryNodeType.INTERPRETABILITY_TEST,
                    name=f"InterpTest: {cat_str}",
                    test_category=cat,
                    test_query=r.get("query", ""),
                    ground_truth=r.get("ground_truth", ""),
                    llm_response=r.get("llm_response", ""),
                    passed=r.get("passed", False),
                    tolerance=r.get("tolerance", 0.05),
                    evaluator_model=self._grader.evaluator_model,
                    imodel_node_id=imodel_node_id,
                    metadata={"concept": "AHE-3.16", "paper": "arXiv:2605.03808"},
                )
                ogm.upsert(node)
                ogm.upsert_edge(
                    imodel_node_id,
                    node_id,
                    RegistryEdgeType.TESTED_INTERPRETABILITY,
                    {"category": cat_str},
                )
                persisted += 1
        except Exception as exc:
            logger.warning("Failed to persist test results: %s", exc)

        return persisted
