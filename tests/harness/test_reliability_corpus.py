#!/usr/bin/python
"""Tests for the reliability seed corpus and EvalCorpus metadata passthrough.

CONCEPT:AHE-3.1
"""

import pytest

from agent_utilities.harness.eval_corpus import EvalCorpus
from agent_utilities.harness.reliability_corpus import (
    SEED_CASES,
    CorpusReport,
    run_reliability_corpus,
)

pytestmark = pytest.mark.concept("AHE-3.1")


# --- EvalCorpus metadata passthrough ---------------------------------------


def test_eval_corpus_carries_metadata():
    corpus = EvalCorpus()
    corpus.add_case(
        "q", "a", tags=["t"], metadata={"evidence": "some evidence", "gold_ids": ["x"]}
    )
    (case,) = corpus.load_cases()
    assert case.metadata["evidence"] == "some evidence"
    assert case.metadata["gold_ids"] == ["x"]


def test_eval_corpus_metadata_defaults_empty():
    corpus = EvalCorpus()
    corpus.add_case("q", "a")
    (case,) = corpus.load_cases()
    assert case.metadata == {}


# --- Seed corpus run -------------------------------------------------------


def test_seed_corpus_clean_all_match():
    report = run_reliability_corpus()
    assert isinstance(report, CorpusReport)
    assert report.total == len(SEED_CASES)
    assert report.match_rate == 1.0
    assert report.degraded is False
    assert all(c.matched for c in report.cases)


def test_seed_corpus_has_pass_and_fail_cases():
    # The corpus must contain both directions or it can't prove the scorers work.
    assert any(c.expect_pass for c in SEED_CASES)
    assert any(not c.expect_pass for c in SEED_CASES)


def test_seed_corpus_degrade_drops_below_floor():
    report = run_reliability_corpus(degrade=True)
    assert report.degraded is True
    # Clean "should pass" cases flip to failing under corruption.
    assert report.match_rate < 0.9
    flipped = [c for c in report.cases if c.expected_pass and not c.actual_pass]
    assert flipped, "degrade must flip at least one passing case"
    # The corruption is caught by the grounding / safety / injection scorers.
    caught = {s for c in flipped for s in c.failed_scorers}
    assert {"faithfulness", "safety_accuracy", "trap_injection"} & caught


def test_seed_corpus_failure_cases_attribute_scorers():
    report = run_reliability_corpus()
    fail_cases = [c for c in report.cases if not c.expected_pass]
    # Every adversarial case is correctly judged failing, with named offenders.
    for c in fail_cases:
        assert c.actual_pass is False
        assert c.failed_scorers
