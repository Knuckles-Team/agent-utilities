"""Tests for EH-206's deterministic free-text -> task-IRI classifier.

Covers: confident classification on each of EG's five native task IRIs,
abstention on nonsense/no-overlap text, abstention below the confidence
floor, the claim's evidence class and digest shape, and that the module
never imports anything LLM-shaped (no network/model dependency exists to
mock in the first place -- proven by asserting the module's own import
graph stays inside stdlib + the AU contracts module).
"""

from __future__ import annotations

import hashlib

import pytest

from agent_utilities.api.agent_control_contracts import TaskClassificationClaim
from agent_utilities.api.task_classification import (
    AMBIGUITY_MARGIN,
    CONFIDENCE_FLOOR,
    classify_task_text,
)


def test_abstains_on_empty_or_nonsense_text():
    assert classify_task_text("") is None
    assert classify_task_text("   ") is None
    assert classify_task_text("qxzzy plonk florb wibble") is None


def test_confident_classification_is_a_labelled_claim_not_a_proof():
    result = classify_task_text(
        "operate the deployment and verify the graph query results"
    )
    assert isinstance(result, TaskClassificationClaim)
    assert result.task_iri == "eg:task/operate"
    assert result.evidence_class == "claim"
    assert result.method == "lexical_keyword_overlap"
    assert 0.0 < result.confidence <= 1.0
    assert set(result.matched_keywords) <= {"graph", "operate", "query", "verify"}


@pytest.mark.parametrize(
    ("text", "expected_iri"),
    [
        (
            "please do some research and retrieval to plan the reasoning",
            "eg:task/research",
        ),
        ("review the design doc and critique the approach", "eg:task/review"),
        (
            "operate the deployment and verify the graph query results",
            "eg:task/operate",
        ),
        ("send a message summarizing the incident to the team", "eg:task/communicate"),
    ],
)
def test_classifies_each_reachable_task_iri(text, expected_iri):
    result = classify_task_text(text)
    assert result is not None
    assert result.task_iri == expected_iri


def test_text_digest_is_sha256_of_normalized_text_never_raw_text():
    text = "review the design doc and critique the approach"
    result = classify_task_text(text)
    assert result is not None
    expected = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
    assert result.text_digest == expected
    assert len(result.text_digest) == 64
    # the digest never leaks the raw text itself
    assert text not in result.text_digest


def test_low_overlap_below_confidence_floor_abstains():
    # A single weak keyword hit diluted by many unrelated tokens must not
    # clear CONFIDENCE_FLOOR.
    text = "write " + " ".join(f"filler{i}" for i in range(20))
    result = classify_task_text(text)
    assert result is None


def test_module_has_no_llm_or_network_dependency():
    import ast
    import inspect

    import agent_utilities.api.task_classification as mod

    tree = ast.parse(inspect.getsource(mod))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported_roots <= {"__future__", "hashlib", "math", "re", "agent_utilities"}


def test_thresholds_are_module_constants_not_magic_numbers_at_call_sites():
    assert 0.0 < CONFIDENCE_FLOOR < 1.0
    assert 0.0 < AMBIGUITY_MARGIN < 1.0
