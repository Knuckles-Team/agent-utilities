from __future__ import annotations

"""Tests for KG Eval Capture harness.

CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort
"""


from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.memory import EvaluationCapture


@pytest.fixture()
def mock_ke():
    ke = MagicMock()
    ke.ogm = MagicMock()

    # Store saved nodes locally to simulate DB
    saved_nodes = []

    def mock_save(node):
        saved_nodes.append(node)
        return node

    def mock_find(model_cls, properties=None):
        if properties and properties.get("evaluator") == "kg_capture":
            return saved_nodes
        return []

    ke.ogm.save.side_effect = mock_save
    ke.ogm.find.side_effect = mock_find
    ke.saved_nodes = saved_nodes
    return ke


@pytest.fixture()
def eval_db(mock_ke):
    return EvaluationCapture(knowledge_engine=mock_ke, enabled=True)


class TestEvalCaptureInit:
    def test_init_enabled(self, mock_ke):
        cap = EvaluationCapture(knowledge_engine=mock_ke, enabled=True)
        assert cap.enabled

    def test_disabled_by_default(self, mock_ke):
        cap = EvaluationCapture(knowledge_engine=mock_ke, enabled=False)
        assert not cap.enabled


class TestCapture:
    def test_capture_inserts(self, eval_db, mock_ke):
        eval_db.capture(
            query="test q",
            method="hybrid",
            result_node_ids=["a", "b"],
            scores=[0.9, 0.8],
            latency_ms=5.0,
            schema_pack="core",
        )
        assert eval_db.count() == 1
        assert len(mock_ke.saved_nodes) == 1
        assert mock_ke.saved_nodes[0].query == "test q"

    def test_capture_multiple(self, eval_db, mock_ke):
        for i in range(5):
            eval_db.capture(f"q{i}", "hybrid", [f"n{i}"], [0.9])
        assert eval_db.count() == 5

    def test_capture_noop_when_disabled(self, mock_ke):
        cap = EvaluationCapture(knowledge_engine=mock_ke, enabled=False)
        cap.capture("q", "hybrid", ["a"])
        assert cap.count() == 0


class TestReplay:
    def test_replay_identical(self, eval_db):
        eval_db.capture("q1", "hybrid", ["a", "b", "c"], [0.9, 0.8, 0.7])

        def search_fn(q):
            return [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        result = eval_db.replay(search_fn=search_fn)
        assert result.total_queries == 1
        assert result.mean_jaccard_at_k == 1.0
        assert result.top_1_stability == 1.0

    def test_replay_partial_match(self, eval_db):
        eval_db.capture("q1", "hybrid", ["a", "b", "c"], [0.9, 0.8, 0.7])

        def search_fn(q):
            return [{"id": "a"}, {"id": "x"}, {"id": "y"}]

        result = eval_db.replay(search_fn=search_fn)
        assert result.total_queries == 1
        # Jaccard({a,b,c}, {a,x,y}) = 1/5 = 0.2
        assert abs(result.mean_jaccard_at_k - 0.2) < 0.01

    def test_replay_flags_regressions(self, eval_db):
        eval_db.capture("q1", "hybrid", ["a", "b", "c"], [0.9, 0.8, 0.7])

        def search_fn(q):
            return [{"id": "x"}, {"id": "y"}, {"id": "z"}]

        result = eval_db.replay(search_fn=search_fn, regression_threshold=0.5)
        assert len(result.regressions) == 1
        assert result.regressions[0]["query"] == "q1"

    def test_replay_empty_db(self, eval_db):
        result = eval_db.replay(search_fn=lambda q: [])
        assert result.total_queries == 0
