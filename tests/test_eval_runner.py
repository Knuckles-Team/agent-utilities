"""Tests for EvalRunner — Multi-Strategy Scoring (CONCEPT:AHE-3.1).

@pytest.mark.concept("AHE-3.12")
"""

import pytest

from agent_utilities.harness.continuous_evaluation_engine import (
    EvalResult,
    EvalRunner,
    EvalStrategy,
    EvaluationMonitor,
    TestCase,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> EvalRunner:
    """EvalRunner with default config."""
    return EvalRunner(pass_threshold=0.7)


@pytest.fixture
def runner_with_monitor() -> EvalRunner:
    """EvalRunner backed by an EvaluationMonitor."""
    monitor = EvaluationMonitor()
    return EvalRunner(monitor=monitor, pass_threshold=0.5)


@pytest.fixture
def simple_test_case() -> TestCase:
    return TestCase(
        id="tc-1",
        query="What is the capital of France?",
        expected_output="Paris",
        agent_name="geography_agent",
    )


@pytest.fixture
def exact_match_case() -> TestCase:
    return TestCase(
        id="tc-exact",
        query="What is 2+2?",
        expected_output="4",
        strategy=EvalStrategy.EXACT_MATCH,
    )


# ---------------------------------------------------------------------------
# TestCase model tests
# ---------------------------------------------------------------------------


class TestTestCaseModel:
    def test_create_with_defaults(self):
        tc = TestCase(query="q", expected_output="a")
        assert tc.strategy == EvalStrategy.COMPOSITE
        assert tc.tags == []

    def test_create_with_strategy(self):
        tc = TestCase(
            query="q",
            expected_output="a",
            strategy=EvalStrategy.EXACT_MATCH,
        )
        assert tc.strategy == EvalStrategy.EXACT_MATCH


# ---------------------------------------------------------------------------
# EvalResult model tests
# ---------------------------------------------------------------------------


class TestEvalResultModel:
    def test_defaults(self):
        r = EvalResult()
        assert r.final_score == 0.0
        assert r.passed is False
        assert r.timestamp > 0

    def test_score_bounds(self):
        r = EvalResult(final_score=0.85)
        assert r.final_score == 0.85


# ---------------------------------------------------------------------------
# Exact match strategy
# ---------------------------------------------------------------------------


class TestExactMatchStrategy:
    def test_perfect_match(self, runner, simple_test_case):
        result = runner.run_eval(
            simple_test_case,
            "Paris",
            strategy=EvalStrategy.EXACT_MATCH,
        )
        assert result.exact_match_score == 1.0
        assert result.passed is True

    def test_case_insensitive(self, runner, simple_test_case):
        result = runner.run_eval(
            simple_test_case,
            "paris",
            strategy=EvalStrategy.EXACT_MATCH,
        )
        assert result.exact_match_score == 1.0

    def test_punctuation_normalization(self, runner):
        tc = TestCase(query="q", expected_output="Hello, world!")
        result = runner.run_eval(
            tc,
            "hello world",
            strategy=EvalStrategy.EXACT_MATCH,
        )
        assert result.exact_match_score == 1.0

    def test_partial_overlap_jaccard(self, runner):
        tc = TestCase(query="q", expected_output="the quick brown fox")
        result = runner.run_eval(
            tc,
            "the slow brown fox",
            strategy=EvalStrategy.EXACT_MATCH,
        )
        # Jaccard: {the, brown, fox} / {the, quick, slow, brown, fox} = 3/5
        assert 0.5 < result.exact_match_score < 0.7

    def test_no_overlap(self, runner):
        tc = TestCase(query="q", expected_output="alpha beta")
        result = runner.run_eval(
            tc,
            "gamma delta",
            strategy=EvalStrategy.EXACT_MATCH,
        )
        assert result.exact_match_score == 0.0

    def test_empty_strings(self, runner):
        tc = TestCase(query="q", expected_output="")
        result = runner.run_eval(tc, "", strategy=EvalStrategy.EXACT_MATCH)
        assert result.exact_match_score == 1.0


# ---------------------------------------------------------------------------
# Semantic similarity strategy (falls back to token overlap w/o model)
# ---------------------------------------------------------------------------


class TestSemanticSimilarityStrategy:
    def test_fallback_to_token_overlap(self, runner, simple_test_case):
        """Without an embedding model, falls back to exact match."""
        result = runner.run_eval(
            simple_test_case,
            "Paris",
            strategy=EvalStrategy.SEMANTIC_SIMILARITY,
        )
        assert result.semantic_similarity_score == 1.0
        assert result.final_score == 1.0

    def test_partial_fallback(self, runner):
        tc = TestCase(query="q", expected_output="the quick brown fox")
        result = runner.run_eval(
            tc,
            "the slow brown fox",
            strategy=EvalStrategy.SEMANTIC_SIMILARITY,
        )
        assert 0.0 < result.semantic_similarity_score < 1.0


# ---------------------------------------------------------------------------
# LLM-as-Judge strategy (falls back w/o model)
# ---------------------------------------------------------------------------


class TestLLMJudgeStrategy:
    def test_prompt_construction(self):
        """Verify the prompt template formats correctly."""
        prompt = EvalRunner.LLM_JUDGE_PROMPT.format(
            query="What is AI?",
            expected="Artificial Intelligence",
            actual="Machine learning",
        )
        assert "What is AI?" in prompt
        assert "Artificial Intelligence" in prompt
        assert "Machine learning" in prompt
        assert '"score"' in prompt

    def test_fallback_without_model(self, runner, simple_test_case):
        """Falls back to semantic similarity when no LLM available."""
        result = runner.run_eval(
            simple_test_case,
            "Paris",
            strategy=EvalStrategy.LLM_JUDGE,
        )
        assert result.llm_judge_score > 0.0
        assert "fallback" in result.llm_judge_reasoning.lower()


# ---------------------------------------------------------------------------
# Composite strategy
# ---------------------------------------------------------------------------


class TestCompositeStrategy:
    def test_composite_combines_all(self, runner, simple_test_case):
        result = runner.run_eval(
            simple_test_case,
            "Paris",
            strategy=EvalStrategy.COMPOSITE,
        )
        assert result.exact_match_score > 0.0
        assert result.final_score > 0.0
        assert result.strategy == EvalStrategy.COMPOSITE

    def test_composite_weights(self):
        """Custom weights should affect final score."""
        runner = EvalRunner(
            exact_weight=1.0,
            semantic_weight=0.0,
            judge_weight=0.0,
        )
        tc = TestCase(query="q", expected_output="hello")
        result = runner.run_eval(tc, "hello", strategy=EvalStrategy.COMPOSITE)
        # Final score should be dominated by exact match
        assert result.final_score == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


class TestBatchEvaluation:
    def test_batch_success(self, runner):
        cases = [
            TestCase(id="b1", query="q1", expected_output="a1"),
            TestCase(id="b2", query="q2", expected_output="a2"),
        ]
        outputs = ["a1", "a2"]
        results = runner.run_batch(cases, outputs, strategy=EvalStrategy.EXACT_MATCH)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_batch_mismatch_raises(self, runner):
        cases = [TestCase(id="b1", query="q", expected_output="a")]
        with pytest.raises(ValueError, match="Mismatch"):
            runner.run_batch(cases, ["a", "b"])


# ---------------------------------------------------------------------------
# Results tracking & summary
# ---------------------------------------------------------------------------


class TestResultsTracking:
    def test_results_accumulate(self, runner, simple_test_case):
        runner.run_eval(simple_test_case, "Paris", strategy=EvalStrategy.EXACT_MATCH)
        runner.run_eval(simple_test_case, "London", strategy=EvalStrategy.EXACT_MATCH)
        assert len(runner.results) == 2

    def test_summary_empty(self, runner):
        s = runner.summary()
        assert s["total"] == 0
        assert s["pass_rate"] == 0.0

    def test_summary_with_results(self, runner, simple_test_case):
        runner.run_eval(simple_test_case, "Paris", strategy=EvalStrategy.EXACT_MATCH)
        s = runner.summary()
        assert s["total"] == 1
        assert s["passed"] == 1
        assert s["pass_rate"] == 1.0
        assert s["avg_score"] > 0.0


# ---------------------------------------------------------------------------
# Monitor integration
# ---------------------------------------------------------------------------


class TestMonitorIntegration:
    def test_feeds_into_monitor(self, runner_with_monitor):
        tc = TestCase(id="m1", query="q", expected_output="a")
        runner_with_monitor.run_eval(tc, "a", strategy=EvalStrategy.EXACT_MATCH)
        # Monitor should have one evaluation
        assert runner_with_monitor._monitor is not None
        summary = runner_with_monitor._monitor.summary()
        assert summary["total_evaluations"] == 1


# ---------------------------------------------------------------------------
# Pass/fail threshold
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_above_threshold_passes(self):
        runner = EvalRunner(pass_threshold=0.5)
        tc = TestCase(query="q", expected_output="hello world")
        result = runner.run_eval(tc, "hello world", strategy=EvalStrategy.EXACT_MATCH)
        assert result.passed is True

    def test_below_threshold_fails(self):
        runner = EvalRunner(pass_threshold=0.99)
        tc = TestCase(query="q", expected_output="hello world")
        result = runner.run_eval(tc, "goodbye", strategy=EvalStrategy.EXACT_MATCH)
        assert result.passed is False
