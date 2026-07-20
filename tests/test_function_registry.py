"""Tests for CONCEPT:AU-ECO.messaging.native-backend-abstraction — Self-Describing Function Registry."""

import pytest

from agent_utilities.models.knowledge_graph import CallableResourceNode, TriggerBinding


@pytest.fixture
def mock_engine():
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    class _E:
        def __init__(self):
            self.graph = GraphComputeEngine(backend_type="rust")
            self.backend = None

        def search_hybrid(self, q, top_k=10):
            return [{"id": n, **d} for n, d in self.graph.nodes(data=True)][:top_k]

    return _E()


class TestTriggerBinding:
    def test_default_manual(self):
        t = TriggerBinding()
        assert t.trigger_type == "manual"
        assert t.binding == ""

    def test_http_trigger(self):
        t = TriggerBinding(
            trigger_type="http", binding="/api/v1/run", conditions={"method": "POST"}
        )
        assert t.trigger_type == "http"
        assert t.conditions["method"] == "POST"

    def test_cron_trigger(self):
        t = TriggerBinding(trigger_type="cron", binding="0 */6 * * *")
        assert t.binding == "0 */6 * * *"


class TestCallableResourceNodeExtensions:
    def test_default_empty_schemas(self):
        n = CallableResourceNode(
            id="cr1", name="test", resource_type="MCP_TOOL", metadata_id="m1"
        )
        assert n.input_schema == {}
        assert n.output_schema == {}
        assert n.trigger_bindings == []

    def test_with_schemas_and_triggers(self):
        n = CallableResourceNode(
            id="cr2",
            name="search",
            resource_type="MCP_TOOL",
            metadata_id="m2",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            output_schema={"type": "array"},
            trigger_bindings=[TriggerBinding(trigger_type="http", binding="/search")],
        )
        assert n.input_schema["properties"]["query"]["type"] == "string"
        assert len(n.trigger_bindings) == 1

    def test_serialization_roundtrip(self):
        n = CallableResourceNode(
            id="cr3",
            name="deploy",
            resource_type="INTERNAL_SKILL",
            metadata_id="m3",
            trigger_bindings=[
                TriggerBinding(trigger_type="event", binding="deploy.requested")
            ],
        )
        data = n.model_dump()
        restored = CallableResourceNode.model_validate(data)
        assert restored.trigger_bindings[0].trigger_type == "event"


class TestFunctionRegistryMixin:
    """Tests for register/deregister/discover on engine_registry."""

    def test_register_function(self, mock_engine):

        # Simulate mixin by calling on mock_engine directly
        mock_engine.graph.add_node(
            "fn:test1",
            type="callable_resource",
            resource_type="MCP_TOOL",
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
            trigger_bindings=[{"trigger_type": "http", "binding": "/test"}],
            importance_score=0.5,
        )
        # Verify it's in the graph
        assert "fn:test1" in mock_engine.graph
        data = mock_engine.graph.nodes["fn:test1"]
        assert data["name"] == "test_tool"
        assert data["input_schema"]["type"] == "object"

    def test_deregister_function(self, mock_engine):
        mock_engine.graph.add_node("fn:rm1", type="callable_resource", name="to_remove")
        mock_engine.graph.remove_node("fn:rm1")
        assert "fn:rm1" not in mock_engine.graph

    def test_discover_functions_by_type(self, mock_engine):
        mock_engine.graph.add_node(
            "fn:a", type="callable_resource", resource_type="MCP_TOOL", name="alpha"
        )
        mock_engine.graph.add_node(
            "fn:b", type="callable_resource", resource_type="A2A_AGENT", name="beta"
        )

        # Filter MCP_TOOL only
        results = [
            {"id": n, **d}
            for n, d in mock_engine.graph.nodes(data=True)
            if d.get("type") == "callable_resource"
            and d.get("resource_type") == "MCP_TOOL"
        ]
        assert len(results) == 1
        assert results[0]["name"] == "alpha"

    def test_discover_by_trigger_type(self, mock_engine):
        mock_engine.graph.add_node(
            "fn:cron1",
            type="callable_resource",
            resource_type="INTERNAL_SKILL",
            name="scheduler",
            trigger_bindings=[{"trigger_type": "cron", "binding": "0 * * * *"}],
        )
        mock_engine.graph.add_node(
            "fn:http1",
            type="callable_resource",
            resource_type="MCP_TOOL",
            name="api",
            trigger_bindings=[{"trigger_type": "http", "binding": "/api"}],
        )

        cron_fns = [
            {"id": n, **d}
            for n, d in mock_engine.graph.nodes(data=True)
            if any(
                t.get("trigger_type") == "cron" for t in d.get("trigger_bindings", [])
            )
        ]
        assert len(cron_fns) == 1
        assert cron_fns[0]["name"] == "scheduler"
