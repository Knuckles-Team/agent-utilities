"""Visibility and fixture checks for the offline EH-652 comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.ontology.document_processing import (
    SectionTreeConfig,
    build_section_tree,
)
from scripts.compare_structured_retrieval import _case, evaluate


def test_hidden_section_is_pruned_before_both_rankers() -> None:
    text = "# Manual\n## Public path\nPublished.\n## Secret path\nPrivate.\n"
    roots = build_section_tree(text, config=SectionTreeConfig(thin=False))
    result = _case(
        "manual.md",
        text,
        roots,
        {
            "query": "secret path",
            "gold_path": ["Manual", "Public path"],
            "allowed_paths": [["Manual", "Public path"]],
        },
    )
    assert result.lexical_ids == [result.gold_id]
    assert result.toc_ids == [result.gold_id]
    assert result.bm25_ids == [result.gold_id]
    assert result.hybrid_ids == [result.gold_id]
    assert result.cited_ranges_valid


def test_hidden_gold_section_refuses_fixture() -> None:
    text = "# Manual\n## Public path\nPublished.\n## Secret path\nPrivate.\n"
    roots = build_section_tree(text, config=SectionTreeConfig(thin=False))
    with pytest.raises(ValueError, match="invalid visibility"):
        _case(
            "manual.md",
            text,
            roots,
            {
                "query": "secret path",
                "gold_path": ["Manual", "Secret path"],
                "allowed_paths": [["Manual", "Public path"]],
            },
        )


def test_checked_in_corpus_has_valid_gold_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (root / "tests/retrieval/fixtures/structured_retrieval_cases.json").read_text()
    )
    report = evaluate(root, fixture)
    assert report["cases"] == 7
    assert report["citations_valid"]
    assert report["toc_leaf"]["invalid_leaf_ids"] == 0


def test_heldout_ecosystem_snapshots_share_visibility_and_measure_updates() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (
            root / "tests/retrieval/fixtures/structured_retrieval_heldout.json"
        ).read_text()
    )
    report = evaluate(root, fixture)
    assert report["cases"] == 12
    assert report["citations_valid"]
    assert len(report["updates"]) == 3
    assert all(update["changed_leaf_count"] == 1 for update in report["updates"])
    assert all(update["bytes_added"] > 0 for update in report["updates"])
    for method in ("lexical", "toc_leaf", "bm25_body", "hybrid_lexical"):
        assert report[method]["invalid_leaf_ids"] == 0


def test_heldout_document_digest_must_match_the_pinned_source() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (
            root / "tests/retrieval/fixtures/structured_retrieval_heldout.json"
        ).read_text()
    )
    fixture["documents"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="document digest changed"):
        evaluate(root, fixture)
