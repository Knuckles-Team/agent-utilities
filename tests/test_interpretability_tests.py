#!/usr/bin/python
"""Tests for CONCEPT:AHE-3.16 — LLM-Graded Interpretability Tests.

Validates the 6-category test protocol, grading logic, reward hacking
detection, and aggregate score computation.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.interpretability_tests import (
    InterpretabilityGrader,
    InterpretabilityTestCase,
    InterpretabilityTestSuite,
)
from agent_utilities.models.imodel import InterpretabilityTestCategory


# ─── InterpretabilityGrader Tests ────────────────────────────────────


class TestInterpretabilityGrader:
    """Tests for the LLM-based interpretability grader."""

    def test_exact_numerical_match(self):
        """Exact numerical match should pass."""
        grader = InterpretabilityGrader()
        passed, reason = grader.grade("2.5000", "2.5", tolerance=0.05)
        assert passed is True
        assert "Numerical" in reason

    def test_within_tolerance(self):
        """Response within tolerance should pass."""
        grader = InterpretabilityGrader()
        passed, _ = grader.grade("2.45", "2.5", tolerance=0.05)
        assert passed is True

    def test_outside_tolerance(self):
        """Response outside tolerance should fail."""
        grader = InterpretabilityGrader()
        passed, _ = grader.grade("3.0", "2.5", tolerance=0.05)
        assert passed is False

    def test_exact_string_match(self):
        """Exact string match should pass."""
        grader = InterpretabilityGrader()
        passed, reason = grader.grade("temperature", "temperature")
        assert passed is True
        assert "string match" in reason.lower()

    def test_token_overlap_match(self):
        """High token overlap should pass."""
        grader = InterpretabilityGrader()
        passed, reason = grader.grade(
            "the most important feature is temperature",
            "most important feature temperature",
        )
        assert passed is True
        assert "overlap" in reason.lower()

    def test_no_match(self):
        """Completely different response should fail."""
        grader = InterpretabilityGrader()
        passed, reason = grader.grade("humidity", "temperature")
        assert passed is False
        assert "No match" in reason

    def test_zero_ground_truth_tolerance(self):
        """Zero ground truth should use absolute tolerance."""
        grader = InterpretabilityGrader()
        passed, _ = grader.grade("0.01", "0.0", tolerance=0.05)
        assert passed is True

    def test_comma_in_number(self):
        """Numbers with commas should be handled."""
        grader = InterpretabilityGrader()
        passed, _ = grader.grade("1,234.5", "1234.5", tolerance=0.01)
        assert passed is True

    def test_reward_hacking_detected(self):
        """Reward hacking should be detected when answer is in model string."""
        grader = InterpretabilityGrader()
        model_str = "Model predicts: 42.567 for feature X"
        assert grader.detect_reward_hacking(model_str, "42.567") is True

    def test_no_reward_hacking(self):
        """No reward hacking when answer is NOT in model string."""
        grader = InterpretabilityGrader()
        model_str = "y = 2.5*x + 1.0"
        assert grader.detect_reward_hacking(model_str, "42.567") is False

    def test_short_answer_no_false_positive(self):
        """Short answers (<3 chars) should not trigger reward hacking."""
        grader = InterpretabilityGrader()
        model_str = "y = 2.5*x + 1.0"
        assert grader.detect_reward_hacking(model_str, "2") is False


# ─── Test Case Generation Tests ──────────────────────────────────────


class TestTestCaseGeneration:
    """Tests for interpretability test case generation."""

    def test_feature_attribution_tests(self):
        """Feature attribution tests should cover ranking and signs."""
        suite = InterpretabilityTestSuite()
        tests = suite.generate_feature_attribution_tests(
            feature_names=["temp", "pressure", "humidity"],
            coefficients=[0.8, -0.3, 0.0],
        )
        assert len(tests) >= 3  # At least most-important, ranking, signs
        # Most important feature test
        most_important = [t for t in tests if "largest absolute effect" in t.query]
        assert len(most_important) == 1
        assert most_important[0].ground_truth == "temp"

    def test_feature_attribution_zero_detection(self):
        """Zero-coefficient features should generate irrelevant detection test."""
        suite = InterpretabilityTestSuite()
        tests = suite.generate_feature_attribution_tests(
            feature_names=["a", "b", "c"],
            coefficients=[1.0, 0.0, 2.0],
        )
        zero_tests = [t for t in tests if "zero effect" in t.query]
        assert len(zero_tests) >= 1
        assert "b" in zero_tests[0].ground_truth

    def test_point_simulation_tests(self):
        """Point simulation tests should pair inputs with outputs."""
        suite = InterpretabilityTestSuite()
        tests = suite.generate_point_simulation_tests(
            inputs=[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
            outputs=[5.0, 11.0],
        )
        assert len(tests) == 2
        assert all(
            t.category == InterpretabilityTestCategory.POINT_SIMULATION for t in tests
        )
        assert "5.0000" in tests[0].ground_truth

    def test_sensitivity_tests(self):
        """Sensitivity tests should check direction and magnitude."""
        suite = InterpretabilityTestSuite()
        tests = suite.generate_sensitivity_tests(
            feature_names=["x1", "x2"],
            sensitivities={"x1": 0.5, "x2": -0.3},
        )
        assert len(tests) >= 2
        direction_tests = [t for t in tests if "increase or decrease" in t.query]
        assert len(direction_tests) >= 2

    def test_counterfactual_tests(self):
        """Counterfactual tests should ask for required input values."""
        suite = InterpretabilityTestSuite()
        tests = suite.generate_counterfactual_tests(
            feature_names=["x"],
            counterfactuals=[
                {
                    "target_output": 10.0,
                    "feature": "x",
                    "required_value": 4.5,
                }
            ],
        )
        assert len(tests) == 1
        assert tests[0].category == InterpretabilityTestCategory.COUNTERFACTUAL
        assert "4.5000" in tests[0].ground_truth

    def test_empty_inputs(self):
        """Empty inputs should return empty test lists."""
        suite = InterpretabilityTestSuite()
        assert suite.generate_feature_attribution_tests([], []) == []
        assert suite.generate_point_simulation_tests([], []) == []
        assert suite.generate_sensitivity_tests([], {}) == []
        assert suite.generate_counterfactual_tests([], []) == []


# ─── Test Suite Execution Tests ──────────────────────────────────────


class TestTestSuiteExecution:
    """Tests for running the full interpretability test suite."""

    def test_run_single_test_pass(self):
        """Running a single test with correct response should pass."""
        suite = InterpretabilityTestSuite()
        test = InterpretabilityTestCase(
            category=InterpretabilityTestCategory.POINT_SIMULATION,
            query="What does the model predict for x=1?",
            ground_truth="2.5",
            tolerance=0.05,
        )
        result = suite.run_test(
            model_str="LinearRegressor(features=['x'], fitted=True)",
            test_case=test,
            llm_response="2.5",
        )
        # "2.5" appears in model_str as "2.50" which is not an exact substring
        # match for the ground truth "2.5", so no reward hacking is detected
        assert result["passed"] is True
        assert result["category"] == "point_simulation"

    def test_run_single_test_fail(self):
        """Running a single test with wrong response should fail."""
        suite = InterpretabilityTestSuite()
        test = InterpretabilityTestCase(
            category=InterpretabilityTestCategory.POINT_SIMULATION,
            query="What does the model predict for x=1?",
            ground_truth="2.5",
            tolerance=0.05,
        )
        result = suite.run_test(
            model_str="y = 2.5*x + 0.0",
            test_case=test,
            llm_response="10.0",
        )
        assert result["passed"] is False

    def test_run_single_test_reward_hacking(self):
        """Test with reward hacking should fail even if answer is correct."""
        suite = InterpretabilityTestSuite()
        test = InterpretabilityTestCase(
            category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
            query="Which is the most important feature?",
            ground_truth="temperature",
        )
        result = suite.run_test(
            model_str="Answer: temperature is the answer",
            test_case=test,
            llm_response="temperature",
        )
        assert result["reward_hacking_detected"] is True
        assert result["passed"] is False  # Blocked by reward hacking

    def test_run_suite_length_mismatch(self):
        """Mismatched test/response lengths should raise ValueError."""
        suite = InterpretabilityTestSuite()
        tests = [
            InterpretabilityTestCase(
                category=InterpretabilityTestCategory.POINT_SIMULATION,
                query="Q1",
                ground_truth="1.0",
            )
        ]
        with pytest.raises(ValueError, match="Mismatch"):
            suite.run_suite("model_str", tests, ["resp1", "resp2"])

    def test_run_suite_aggregation(self):
        """Suite should correctly aggregate per-category scores."""
        suite = InterpretabilityTestSuite()
        tests = [
            InterpretabilityTestCase(
                category=InterpretabilityTestCategory.POINT_SIMULATION,
                query="Q1",
                ground_truth="1.0",
            ),
            InterpretabilityTestCase(
                category=InterpretabilityTestCategory.POINT_SIMULATION,
                query="Q2",
                ground_truth="2.0",
            ),
            InterpretabilityTestCase(
                category=InterpretabilityTestCategory.FEATURE_ATTRIBUTION,
                query="Q3",
                ground_truth="temp",
            ),
        ]
        responses = ["1.0", "999.0", "temp"]
        result = suite.run_suite("y = x", tests, responses)
        assert result["total_tests"] == 3
        assert result["total_passed"] == 2
        assert result["overall_score"] == pytest.approx(2.0 / 3.0)
        assert "point_simulation" in result["per_category"]
        assert result["per_category"]["point_simulation"]["score"] == pytest.approx(0.5)
        assert result["per_category"]["feature_attribution"]["score"] == pytest.approx(
            1.0
        )

    def test_compute_score_empty(self):
        """Score computation with no results should return 0."""
        suite = InterpretabilityTestSuite()
        result = suite.compute_agent_interpretability_score([])
        assert result["overall_score"] == 0.0
        assert result["total_tests"] == 0

    def test_default_test_counts(self):
        """Default test distribution should sum to 200."""
        total = sum(InterpretabilityTestSuite.DEFAULT_TEST_COUNTS.values())
        assert total == 200
