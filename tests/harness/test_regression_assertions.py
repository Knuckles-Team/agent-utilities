"""Tests for the plain-English assertion regression seam (CONCEPT:AU-AHE.evaluation.failure-analysis-loop)."""

from __future__ import annotations

from agent_utilities.harness.continuous_evaluation_engine import (
    EvalRunner,
    EvalStrategy,
)
from agent_utilities.harness.continuous_evaluation_engine import (
    TestCase as EvalTestCase,
)
from agent_utilities.harness.eval_corpus import EvalCorpus


def test_add_case_carries_assertion_to_loaded_case():
    corpus = EvalCorpus()
    corpus.add_case(
        query="re-run X",
        expected_output="ok",
        assertion="The response mentions success",
        tags=["regression"],
    )
    cases = corpus.load_cases()
    assert len(cases) == 1
    assert cases[0].assertion == "The response mentions success"


def test_assertion_judge_lexical_fallback_pass():
    # No model in the test env -> lexical fallback. Salient assertion words all
    # present in the output -> pass.
    score, reasoning = EvalRunner._assertion_judge(
        "authentication succeeded tokens issued",
        "did it work?",
        "authentication succeeded and tokens were issued",
    )
    assert score == 1.0
    assert "fallback" in reasoning


def test_assertion_judge_lexical_fallback_fail():
    score, _ = EvalRunner._assertion_judge(
        "authentication succeeded tokens issued",
        "did it work?",
        "completely unrelated content about weather",
    )
    assert score == 0.0


def test_run_eval_uses_assertion_when_present():
    runner = EvalRunner(pass_threshold=0.7)
    case = EvalTestCase(
        id="c1",
        query="did login work?",
        expected_output="irrelevant expected text",
        assertion="output confirms login succeeded",
    )
    # Output satisfies the assertion lexically even though it differs from
    # expected_output -> assertion path drives the pass.
    result = runner.run_eval(case, "login output confirms it succeeded")
    assert result.passed
    assert result.llm_judge_reasoning  # assertion judge populated reasoning


def test_assertion_strategy_explicitly_selected():
    runner = EvalRunner()
    case = EvalTestCase(
        id="c2",
        query="q",
        expected_output="the value must be cached",
        strategy=EvalStrategy.ASSERTION,
    )
    result = runner.run_eval(case, "the value must be cached now")
    assert result.final_score in (0.0, 1.0)


def test_lock_regression_cases_idempotent():
    """A verified remediation locks one assertion case per signature, once."""

    class _FakeEngine:
        def __init__(self):
            self.nodes = []

        def add_node(self, node_id, type=None, **props):  # noqa: A002
            self.nodes.append({"id": node_id, "type": type, **props})

    from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
        FailureAnalyzer,
    )

    analyzer = FailureAnalyzer.__new__(FailureAnalyzer)
    analyzer.engine = _FakeEngine()
    gaps = [
        {"signature": "sig1", "workflow": "wf-a", "name": "boom"},
        {"signature": "sig2", "workflow": "wf-b", "name": "crash"},
    ]
    analyzer._lock_regression_cases(gaps)
    analyzer._lock_regression_cases(gaps)  # second call must not duplicate
    locked = [n for n in analyzer.engine.nodes if n.get("type") == "eval_case"]
    assert len(locked) == 2
    assert all("assertion" in n and n["assertion"] for n in locked)
