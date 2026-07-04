from __future__ import annotations

"""Tests for CONCEPT:AU-AHE.evaluation.interpretability-tests: Model Synergy Tracker."""


from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.knowledge_graph import (
    SelfModelNode,  # type: ignore[attr-defined]
)


class TestSelfModelNodeSynergies:
    def test_model_synergies_default_empty(self):
        node = SelfModelNode(id="sm:test", name="Test")
        assert node.model_synergies == {}

    def test_model_synergies_set(self):
        node = SelfModelNode(
            id="sm:test",
            name="Test",
            model_synergies={"gpt-4o|claude-sonnet": 0.85, "gemini-2.5|llama-3": 0.72},
        )
        assert len(node.model_synergies) == 2

    def test_model_synergies_serialization(self):
        node = SelfModelNode(
            id="sm:test", name="Test", model_synergies={"heavy|light": 0.9}
        )
        data = node.model_dump()
        assert "model_synergies" in data
        restored = SelfModelNode.model_validate(data)
        assert restored.model_synergies == node.model_synergies

    def test_model_synergies_json_schema(self):
        schema = SelfModelNode.model_json_schema()
        assert "model_synergies" in schema["properties"]
        assert (
            "CONCEPT:AU-AHE.evaluation.interpretability-tests" in schema["properties"]["model_synergies"]["description"]
        )

    def test_synergy_key_format(self):
        models = ["claude-sonnet", "gpt-4o", "gemini-2.5"]
        key = "|".join(sorted(models))
        assert key == "claude-sonnet|gemini-2.5|gpt-4o"

    def test_ema_calculation(self):
        old_rate, alpha = 0.5, 0.3
        assert abs((alpha * 1.0 + (1 - alpha) * old_rate) - 0.65) < 0.001
        assert abs((alpha * 0.0 + (1 - alpha) * old_rate) - 0.35) < 0.001


class TestSelfModelSynergyTracking:
    def _engine(self):
        e = MagicMock()
        e.graph = GraphComputeEngine(backend_type="rust")
        e.backend = None
        return e

    def _session(self, error=None, routing_log=None):
        s = MagicMock()
        s.session_id = "test"
        s.routed_domain = "gitlab"
        s.error = error
        s.node_history = ["researcher"]
        s.task_list = MagicMock()
        s.task_list.tasks = []
        s.routing_confidence_log = routing_log or []
        return s

    def test_synergy_recorded_multi_model(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        sm = SelfModel(self._engine())
        sm.get_or_create()
        sm.update_after_session(
            self._session(
                routing_log=[
                    {"specialist_id": "r", "routed_tier": "light"},
                    {"specialist_id": "p", "routed_tier": "heavy"},
                ]
            )
        )
        updated = sm.get_current()
        assert updated is not None
        assert "heavy|light" in updated.model_synergies

    def test_no_synergy_single_model(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        sm = SelfModel(self._engine())
        sm.get_or_create()
        sm.update_after_session(
            self._session(
                routing_log=[
                    {"specialist_id": "r", "routed_tier": "medium"},
                    {"specialist_id": "p", "routed_tier": "medium"},
                ]
            )
        )
        updated = sm.get_current()
        assert updated is not None
        assert len(updated.model_synergies) == 0

    def test_synergies_carried_forward(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        sm = SelfModel(self._engine())
        initial = sm.get_or_create()
        initial.model_synergies = {"heavy|light": 0.8}
        sm.ogm.upsert(initial)
        snapshot = sm.create_snapshot(session_id="test")
        assert snapshot.model_synergies == {"heavy|light": 0.8}


class TestGetBestSynergies:
    def _engine(self):
        e = MagicMock()
        e.graph = GraphComputeEngine(backend_type="rust")
        e.backend = None
        return e

    def test_empty_synergies(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        sm = SelfModel(self._engine())
        sm.get_or_create()
        assert sm.get_best_synergies(["gpt-4o"]) == []

    def test_filters_by_available(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        sm = SelfModel(self._engine())
        initial = sm.get_or_create()
        initial.model_synergies = {"gpt-4o|claude": 0.9, "gemini|llama": 0.8}
        sm.ogm.upsert(initial)
        result = sm.get_best_synergies(["gpt-4o", "claude"])
        assert len(result) == 1
        assert result[0] == ("gpt-4o|claude", 0.9)

    def test_sorted_descending(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        sm = SelfModel(self._engine())
        initial = sm.get_or_create()
        initial.model_synergies = {"a|b": 0.6, "a|c": 0.9, "b|c": 0.75}
        sm.ogm.upsert(initial)
        result = sm.get_best_synergies(["a", "b", "c"], top_k=3)
        assert result[0][0] == "a|c"

    def test_top_k_limits(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        sm = SelfModel(self._engine())
        initial = sm.get_or_create()
        initial.model_synergies = {"a|b": 0.9, "a|c": 0.8, "b|c": 0.7}
        sm.ogm.upsert(initial)
        assert len(sm.get_best_synergies(["a", "b", "c"], top_k=1)) == 1

    def test_no_self_model(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        sm = SelfModel(self._engine())
        assert sm.get_best_synergies(["gpt-4o"]) == []


class TestExplainSelfSynergies:
    def test_includes_synergies(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        e = MagicMock()
        e.graph = GraphComputeEngine(backend_type="rust")
        e.backend = None
        sm = SelfModel(e)
        initial = sm.get_or_create()
        initial.model_synergies = {"heavy|light": 0.85}
        sm.ogm.upsert(initial)
        assert "Model Synergies" in sm.explain_self()

    def test_no_synergies_no_section(self):
        from agent_utilities.knowledge_graph.self_model import SelfModel

        e = MagicMock()
        e.graph = GraphComputeEngine(backend_type="rust")
        e.backend = None
        sm = SelfModel(e)
        sm.get_or_create()
        assert "Model Synergies" not in sm.explain_self()
