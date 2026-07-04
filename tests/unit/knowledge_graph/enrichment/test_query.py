"""'Which pytests need work' as a graph query (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 1).

End-to-end over the real in-memory backend: enrich → store → query. Skips if the
epistemic-graph daemon isn't reachable (same backend the other backend tests use).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.pipeline import EnrichmentPipeline
from agent_utilities.knowledge_graph.enrichment.query import (
    tests_needing_work as _tests_needing_work,
)


@pytest.fixture()
def backend():
    try:
        from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
            EpistemicGraphBackend,
        )

        return EpistemicGraphBackend()
    except Exception as e:  # pragma: no cover - env without daemon
        pytest.skip(f"epistemic-graph backend unavailable: {e}")


def _parse_fn(file_path, source):
    if file_path.endswith("test_good.py"):
        return {
            "nodes": [
                {
                    "node_id": "symbol:test_good",
                    "node_type": "SYMBOL",
                    "properties": {
                        "symbol_type": "Function",
                        "name": "test_good",
                        "line": "1",
                        "ast_hash": "g",
                        "file_path": file_path,
                        "is_test": "true",
                        "assert_count": "3",
                        "mock_count": "0",
                        "fixture_count": "1",
                        "marks": "",
                        "is_skipped": "false",
                        "calls": "",
                    },
                }
            ]
        }
    return {
        "nodes": [
            {
                "node_id": "symbol:test_bad",
                "node_type": "SYMBOL",
                "properties": {
                    "symbol_type": "Function",
                    "name": "test_bad",
                    "line": "1",
                    "ast_hash": "b",
                    "file_path": file_path,
                    "is_test": "true",
                    "assert_count": "0",
                    "mock_count": "5",
                    "fixture_count": "1",
                    "marks": "",
                    "is_skipped": "false",
                    "calls": "",
                },
            }
        ]
    }


def test_tests_needing_work_is_a_graph_query(backend, tmp_path):
    (tmp_path / "test_good.py").write_text("def test_good():\n    assert 1\n")
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    pass\n")

    pipe = EnrichmentPipeline(backend, _parse_fn)
    summary = pipe.enrich(tmp_path)
    assert summary.tests == 2
    assert summary.tests_needing_work == 1

    flagged = _tests_needing_work(backend)
    names = {r["name"] for r in flagged}
    assert "test_bad" in names
    assert "test_good" not in names

    bad = next(r for r in flagged if r["name"] == "test_bad")
    issue_codes = {i["code"] for i in bad["issues"]}
    assert "MockHeavyTest" in issue_codes and "AssertionFreeTest" in issue_codes
