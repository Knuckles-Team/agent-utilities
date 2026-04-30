"""Tests for expanded RLM trigger conditions and metadata-only root prompting.

CONCEPT:AU-007 — RLM Execution (Expanded Triggers & Whitepaper Alignment)
"""


import pytest

from agent_utilities.rlm.config import RLMConfig
from agent_utilities.rlm.repl import RecursionLimitError, RLMEnvironment


class TestRLMTriggerConditions:
    """Test the semantic trigger hierarchy in RLMConfig."""

    def test_should_trigger_global_override(self):
        config = RLMConfig(enabled=True)
        assert config.should_trigger() is True
        assert config.should_trigger(output_size=0) is True

    def test_should_trigger_large_output(self):
        config = RLMConfig(enabled=False, trigger_on_large_output=True)
        assert config.should_trigger(output_size=100) is False
        assert config.should_trigger(output_size=60_000) is True

    def test_should_trigger_ahe_distillation(self):
        config = RLMConfig(enabled=False, trigger_on_ahe_distillation=True)
        assert config.should_trigger(trace_count=100) is False
        assert config.should_trigger(trace_count=600) is True

    def test_should_trigger_kg_bulk(self):
        config = RLMConfig(enabled=False, trigger_on_kg_bulk_analysis=True)
        assert config.should_trigger(kg_node_count=500) is False
        assert config.should_trigger(kg_node_count=1500) is True

    def test_should_trigger_long_horizon(self):
        config = RLMConfig(enabled=False)
        assert config.should_trigger(requires_long_horizon=True) is True

    def test_should_not_trigger_when_disabled(self):
        config = RLMConfig(
            enabled=False,
            trigger_on_large_output=False,
            trigger_on_ahe_distillation=False,
            trigger_on_kg_bulk_analysis=False,
        )
        assert config.should_trigger(
            output_size=100_000,
            trace_count=10_000,
            kg_node_count=50_000,
        ) is False

    def test_custom_thresholds(self):
        config = RLMConfig(
            enabled=False,
            ahe_trace_threshold=100,
            kg_bulk_threshold=200,
            max_context_threshold=10_000,
        )
        assert config.should_trigger(trace_count=101) is True
        assert config.should_trigger(kg_node_count=201) is True
        assert config.should_trigger(output_size=10_001) is True


class TestRLMMetadataRoot:
    """Test whitepaper-aligned metadata-only root prompting."""

    def test_infer_context_type_json(self):
        assert RLMEnvironment._infer_context_type('{"key": "value"}') == "json"
        assert RLMEnvironment._infer_context_type('[1, 2, 3]') == "json"

    def test_infer_context_type_csv(self):
        csv_data = "name,age\nAlice,30\nBob,25"
        assert RLMEnvironment._infer_context_type(csv_data) == "csv"

    def test_infer_context_type_xml(self):
        assert RLMEnvironment._infer_context_type("<root><child/></root>") == "xml/html"

    def test_infer_context_type_text(self):
        assert RLMEnvironment._infer_context_type("Hello world") == "text"

    def test_build_context_metadata(self):
        env = RLMEnvironment(context='{"data": "value"}' * 1000)
        metadata = env._build_context_metadata()

        assert "CONTEXT METADATA:" in metadata
        assert "type: json" in metadata
        assert "length:" in metadata
        assert "ACCESS INSTRUCTIONS:" in metadata
        assert "context[start:end]" in metadata

    def test_build_stdout_metadata(self):
        env = RLMEnvironment(context="test")
        metadata = env._build_stdout_metadata("Hello stdout output", turn=0)

        assert "EXECUTION RESULT (turn 1):" in metadata
        assert "stdout_length:" in metadata
        assert "_stdout_1" in metadata
        assert env.vars["_stdout_1"] == "Hello stdout output"
        assert env.globals_dict["_stdout_1"] == "Hello stdout output"


class TestRLMOWLHelpers:
    """Test OWL/KG REPL helpers."""

    @pytest.mark.asyncio
    async def test_owl_query_no_engine(self):
        env = RLMEnvironment(context="test")
        result = await env.owl_query("SELECT ?s WHERE { ?s a ?o }")
        assert result[0]["error"] == "Knowledge engine not available"

    @pytest.mark.asyncio
    async def test_owl_query_no_bridge(self):
        from unittest.mock import MagicMock

        mock_deps = MagicMock()
        mock_deps.knowledge_engine = MagicMock()
        mock_deps.knowledge_engine.owl_bridge = None

        env = RLMEnvironment(context="test", graph_deps=mock_deps)
        result = await env.owl_query("SELECT ?s WHERE { ?s a ?o }")
        assert result[0]["error"] == "OWL bridge not configured"

    @pytest.mark.asyncio
    async def test_owl_query_delegates_to_bridge(self):
        from unittest.mock import MagicMock

        mock_bridge = MagicMock()
        mock_bridge.query_sparql.return_value = [{"id": "node_1", "type": "memory"}]

        mock_deps = MagicMock()
        mock_deps.knowledge_engine = MagicMock()
        mock_deps.knowledge_engine.owl_bridge = mock_bridge

        env = RLMEnvironment(context="test", graph_deps=mock_deps)
        result = await env.owl_query("SELECT ?s WHERE { ?s a au:Memory }")

        assert result == [{"id": "node_1", "type": "memory"}]
        mock_bridge.query_sparql.assert_called_once()

    @pytest.mark.asyncio
    async def test_kg_bulk_export_no_engine(self):
        env = RLMEnvironment(context="test")
        result = await env.kg_bulk_export("memory")
        assert result[0]["error"] == "Knowledge engine not available"

    @pytest.mark.asyncio
    async def test_kg_bulk_export_returns_nodes(self):
        from unittest.mock import MagicMock

        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.add_node("m1", type="memory", name="Test Memory 1")
        graph.add_node("m2", type="memory", name="Test Memory 2")
        graph.add_node("t1", type="tool", name="Test Tool")

        mock_deps = MagicMock()
        mock_deps.knowledge_engine = MagicMock()
        mock_deps.knowledge_engine.graph = graph

        env = RLMEnvironment(context="test", graph_deps=mock_deps)
        result = await env.kg_bulk_export("memory")

        assert len(result) == 2
        assert all(r["type"] == "memory" for r in result)

    @pytest.mark.asyncio
    async def test_kg_bulk_export_wildcard(self):
        from unittest.mock import MagicMock

        import networkx as nx

        graph = nx.MultiDiGraph()
        graph.add_node("m1", type="memory", name="Test")
        graph.add_node("t1", type="tool", name="Tool")

        mock_deps = MagicMock()
        mock_deps.knowledge_engine = MagicMock()
        mock_deps.knowledge_engine.graph = graph

        env = RLMEnvironment(context="test", graph_deps=mock_deps)
        result = await env.kg_bulk_export("*")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_kg_bulk_export_limit(self):
        from unittest.mock import MagicMock

        import networkx as nx

        graph = nx.MultiDiGraph()
        for i in range(100):
            graph.add_node(f"m{i}", type="memory", name=f"Memory {i}")

        mock_deps = MagicMock()
        mock_deps.knowledge_engine = MagicMock()
        mock_deps.knowledge_engine.graph = graph

        env = RLMEnvironment(context="test", graph_deps=mock_deps)
        result = await env.kg_bulk_export("memory", limit=10)

        assert len(result) == 10


# Keep existing tests
@pytest.mark.asyncio
async def test_rlm_environment_local_execution():
    config = RLMConfig(enabled=True, use_container=False)
    env = RLMEnvironment(context="Test data", config=config)

    code = """
import json
result = context + " works"
FINAL_VAR('out', result)
print("Debug: ran code")
"""
    vars, stdout = await env.execute(code)

    assert vars['out'] == "Test data works"
    assert "Debug: ran code" in stdout
    assert env.vars['out'] == "Test data works"

@pytest.mark.asyncio
async def test_rlm_environment_async_sub_calls():
    config = RLMConfig(enabled=True, async_enabled=True, use_container=False)
    # We patch run_full_rlm to avoid actual LLM calls
    env = RLMEnvironment(context="Parent", config=config)

    async def mock_run_full_rlm(self_instance, prompt):
        return f"Mocked {prompt} at depth {self_instance.depth}"

    import unittest.mock
    with unittest.mock.patch('agent_utilities.rlm.repl.RLMEnvironment.run_full_rlm', new=mock_run_full_rlm):
        code = """
calls = [
    {"prompt": "A", "context": "Data A"},
    {"prompt": "B", "context": "Data B"}
]
results = await run_parallel_sub_calls(calls)
FINAL_VAR('results', results)
"""
        vars, stdout = await env.execute(code)

        assert "results" in vars
        res = vars["results"]
        assert len(res) == 2
        assert res[0] == "Mocked A at depth 1"
        assert res[1] == "Mocked B at depth 1"

@pytest.mark.asyncio
async def test_rlm_recursion_limit():
    config = RLMConfig(enabled=True, max_depth=1)
    env = RLMEnvironment(context="Parent", config=config, depth=1)

    # Should throw exception if it tries to spawn depth 2 when max is 1
    with pytest.raises(RecursionLimitError):
        await env.rlm_query("Child query")
