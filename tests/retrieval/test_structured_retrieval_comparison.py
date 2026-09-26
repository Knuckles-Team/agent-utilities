"""Visibility and fixture checks for the offline EH-652 comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.ontology.document_processing import (
    SectionTreeConfig,
    build_section_tree,
)
from scripts.compare_structured_retrieval import _case, _update_cost, evaluate


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


def test_hidden_section_cannot_compete_with_two_visible_leaves() -> None:
    text = (
        "# Manual\n## Public overview\nGeneral instructions.\n"
        "## Public policy\nSecret access follows policy.\n"
        "## Hidden notes\nSecret secret secret access.\n"
    )
    roots = build_section_tree(text, config=SectionTreeConfig(thin=False))
    result = _case(
        "manual.md",
        text,
        roots,
        {
            "query": "secret access",
            "gold_path": ["Manual", "Public policy"],
            "gold_quote": "Secret access follows policy.",
            "allowed_paths": [
                ["Manual", "Public overview"],
                ["Manual", "Public policy"],
            ],
        },
    )
    assert len(result.allowed_ids) == 2
    for ids in (
        result.lexical_ids,
        result.toc_ids,
        result.bm25_ids,
        result.hybrid_ids,
    ):
        assert ids
        assert set(ids) <= set(result.allowed_ids)
    assert result.gold_quote_valid


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
    assert all(update["stable_leaf_ids_changed"] == 0 for update in report["updates"])
    assert all(update["bytes_added"] > 0 for update in report["updates"])
    for method in ("lexical", "toc_leaf", "bm25_body", "hybrid_lexical"):
        assert report[method]["invalid_leaf_ids"] == 0
        assert report[method]["citation_at_1"] == report[method]["recall_at_1"]
    assert all(row["gold_quote_valid"] for row in report["rows"])
    selector = report["heldout_selector"]
    assert selector["cases"] == 12
    assert selector["invalid_leaf_ids"] == 0
    assert selector["recall_at_1"] == 10 / 12
    assert selector["recall_at_3"] == 1.0
    assert set(selector["weights_by_fold"]) == {"0", "1", "2"}
    assert selector["training_ms"] >= 0
    assert selector["rank_p95_ms"] >= selector["rank_p50_ms"] >= 0
    cross = report["cross_document"]
    assert cross["cases"] == 12
    assert cross["toc_leaf"]["wrong_document_at_1"] == 3 / 12
    for method in ("toc_leaf", "bm25_body", "hybrid_lexical"):
        assert cross[method]["invalid_leaf_ids"] == 0


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


def test_gold_quote_must_be_in_the_cited_leaf() -> None:
    text = "# Manual\n## Public path\nAnswer is here.\n## Other path\nDecoy quote.\n"
    roots = build_section_tree(text, config=SectionTreeConfig(thin=False))
    with pytest.raises(ValueError, match="gold quote is outside cited leaf"):
        _case(
            "manual.md",
            text,
            roots,
            {
                "query": "answer",
                "gold_path": ["Manual", "Public path"],
                "gold_quote": "Decoy quote.",
            },
        )


def test_update_cost_refuses_embedded_heading() -> None:
    text = "# Manual\n## First\nBody.\n## Second\nOther.\n"
    roots = build_section_tree(text, config=SectionTreeConfig(thin=False))
    with pytest.raises(ValueError, match="non-heading body text"):
        _update_cost(
            text,
            roots,
            {
                "leaf_path": ["Manual", "First"],
                "append_text": "Body addition.\n## Injected section",
            },
        )


def test_cross_document_ranking_namespaces_ids_and_prunes_global_hidden_leaf(
    tmp_path: Path,
) -> None:
    (tmp_path / "alpha.md").write_text("# Manual\n## Shared\nAlpha answer.\n")
    (tmp_path / "beta.md").write_text(
        "# Manual\n## Shared\nBeta answer.\n## Private\nAlpha beta secret.\n"
    )
    fixture = {
        "documents": [
            {
                "path": "alpha.md",
                "cases": [{"query": "alpha answer", "gold_path": ["Manual", "Shared"]}],
            },
            {
                "path": "beta.md",
                "visible_paths": [["Manual", "Shared"]],
                "cases": [{"query": "beta answer", "gold_path": ["Manual", "Shared"]}],
            },
        ]
    }
    report = evaluate(tmp_path, fixture)["cross_document"]
    assert report["cases"] == 2
    assert report["bm25_body"]["recall_at_1"] == 1.0
    for row in report["rows"]:
        assert row["allowed_ids"] == ["0:0002", "1:0002"]
        assert row["gold_id"] in row["allowed_ids"]
        for method in ("toc_ids", "bm25_ids", "hybrid_ids"):
            assert set(row[method]) <= set(row["allowed_ids"])


def test_cross_document_ranking_excludes_other_tenant_documents(tmp_path: Path) -> None:
    (tmp_path / "alpha.md").write_text("# Manual\n## Answer\nAlpha answer.\n")
    (tmp_path / "beta.md").write_text("# Manual\n## Answer\nAlpha answer.\n")
    fixture = {
        "documents": [
            {
                "path": "alpha.md",
                "tenant": "alpha",
                "cases": [{"query": "alpha answer", "gold_path": ["Manual", "Answer"]}],
            },
            {
                "path": "beta.md",
                "tenant": "beta",
                "cases": [{"query": "alpha answer", "gold_path": ["Manual", "Answer"]}],
            },
        ]
    }
    report = evaluate(tmp_path, fixture)["cross_document"]
    assert [row["allowed_ids"] for row in report["rows"]] == [
        ["0:0002"],
        ["1:0002"],
    ]
    for row in report["rows"]:
        for method in ("toc_ids", "bm25_ids", "hybrid_ids"):
            assert row[method] == [row["gold_id"]]


def test_query_tenant_cannot_label_another_tenants_gold(tmp_path: Path) -> None:
    (tmp_path / "alpha.md").write_text("# Manual\n## Answer\nAlpha answer.\n")
    fixture = {
        "documents": [
            {
                "path": "alpha.md",
                "tenant": "alpha",
                "cases": [
                    {
                        "tenant": "beta",
                        "query": "alpha answer",
                        "gold_path": ["Manual", "Answer"],
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="query tenant does not own gold document"):
        evaluate(tmp_path, fixture)


def test_selector_fold_does_not_train_on_heldout_document(tmp_path: Path) -> None:
    (tmp_path / "alpha.md").write_text(
        "# A\n## One\nFirst answer.\n## Two\nSecond answer.\n"
    )
    (tmp_path / "beta.md").write_text(
        "# B\n## One\nFirst answer.\n## Two\nSecond answer.\n"
    )
    fixture = {
        "documents": [
            {
                "path": "alpha.md",
                "cases": [{"query": "first answer", "gold_path": ["A", "One"]}],
            },
            {
                "path": "beta.md",
                "cases": [{"query": "second answer", "gold_path": ["B", "Two"]}],
            },
        ]
    }
    before = evaluate(tmp_path, fixture)["heldout_selector"]["weights_by_fold"]
    fixture["documents"][0]["cases"][0]["gold_path"] = ["A", "Two"]
    after = evaluate(tmp_path, fixture)["heldout_selector"]["weights_by_fold"]
    assert before["0"] == after["0"]
