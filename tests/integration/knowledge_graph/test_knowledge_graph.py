"""CONCEPT:AU-KG.query.object-graph-mapper"""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.pipeline import IntelligencePipeline
from agent_utilities.models.knowledge_graph import (
    PipelineConfig,
    RegistryEdgeType,
    RegistryNodeType,
)


def _bind_isolated_engine(compute: GraphComputeEngine) -> IntelligenceGraphEngine:
    """Build an ``IntelligenceGraphEngine`` bound to an already-isolated ``compute``.

    ``IntelligenceGraphEngine(db_path=...)`` builds its own backend via
    ``create_backend()``, which constructs a bare ``EpistemicGraphBackend()``.
    That backend resolves its OWN routing graph via ``resolve_routing_graph(None)``
    *before* asking ``GraphComputeEngine`` for one — so under the (autouse,
    test-suite-wide) ``isolate_graph_compute_engine`` fixture it lands on the
    ambient tenant's graph rather than this test's isolated graph (the
    fixture's redirect only catches a literal
    ``graph_name in (None, "__commons__", "__secrets__")``). Binding the
    backend directly to the already-isolated ``compute`` object (as returned
    by the ``mock_graph``/local ``graph`` fixtures here) sidesteps that
    divergent second resolution entirely, so ``graph`` and ``engine`` always
    see the same data.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    backend = object.__new__(EpistemicGraphBackend)
    backend._graph = compute
    backend.graph_name = compute.graph_name
    return IntelligenceGraphEngine(backend=backend)


@pytest.fixture
def mock_graph(tiny_engine):
    # CONCEPT:AU-KG.memory.provides-real-ephemeral-one — run against the REAL ephemeral engine. ``tiny_engine``
    # guarantees a live engine for the session; the autouse
    # ``isolate_graph_compute_engine`` fixture pins this GraphComputeEngine AND
    # the IntelligenceGraphEngine the dependent tests build to ONE per-test
    # tenant graph (and purges it on teardown), so the queries below see exactly
    # this data with no cross-test leakage — no SQLite, no mocks.
    graph = GraphComputeEngine(backend_type="rust")
    # Add an agent
    graph.add_node(
        "TestBot",
        node_type=RegistryNodeType.AGENT,
        name="TestBot",
        description="A test bot",
        agent_type="specialist",
    )
    # Add a tool
    graph.add_node(
        "tool:search",
        node_type=RegistryNodeType.TOOL,
        name="search",
        description="Search tool",
        mcp_server="TestBot",
    )
    # Link them
    graph.add_edge("TestBot", "tool:search", relationship=RegistryEdgeType.PROVIDES)
    return graph


@pytest.mark.asyncio
async def test_intelligence_pipeline_mock(tmp_path):
    config = PipelineConfig(
        workspace_path=str(tmp_path),
        persist_to_ladybug=False,
        enable_embeddings=False,
    )

    mock_agent = MagicMock()
    mock_agent.name = "TestBot"
    mock_agent.description = "desc"
    # AgentNode.agent_type is a strict Literal["specialist", "a2a"]; "prompt"
    # is not (and never was) one of them.
    mock_agent.agent_type = "specialist"
    mock_agent.system_prompt = "prompt"
    mock_agent.endpoint_url = None
    mock_agent.tool_count = 0

    mock_registry = MagicMock()
    mock_registry.agents = [mock_agent]
    mock_registry.tools = []

    with patch(
        "agent_utilities.core.config.get_discovery_registry",
        return_value=mock_registry,
    ):
        # IntelligencePipeline defaults to graph_name="__commons__", a sentinel
        # the isolate_graph_compute_engine fixture redirects -- but only
        # inside GraphComputeEngine.__init__. GraphComputeEngine.get_or_create()
        # (which IntelligencePipeline calls) separately compares its OWN
        # pre-redirect graph_name argument ("__commons__") against the
        # post-redirect root.graph_name, finds them unequal, and builds a
        # graph-scoped view PINNED to the literal "__commons__" string --
        # every call on it then fails with "A graph-scoped view cannot
        # retarget the verified GraphSession" because the ambient GraphSession
        # is scoped to the real isolated graph. Passing this test's own
        # already-isolated graph name explicitly sidesteps the mismatch: it
        # isn't a sentinel, so nothing gets redirected out from under it.
        isolated = GraphComputeEngine(backend_type="rust")
        pipeline = IntelligencePipeline(config, graph_name=isolated.graph_name)
        metadata = await pipeline.run()
        assert metadata.node_count > 0
        assert pipeline.graph.number_of_nodes() == metadata.node_count


@pytest.mark.asyncio
@pytest.mark.engine
async def test_intelligence_engine_queries(mock_graph):
    engine = _bind_isolated_engine(mock_graph)

    # Test tool to agent mapping
    agents = engine.find_agent_for_tool("search")
    assert "TestBot" in agents

    # Test agent to tools mapping
    tools = engine.get_agent_tools("TestBot")
    assert "search" in tools


@pytest.mark.engine
def test_intelligence_shortest_path(mock_graph):
    mock_graph.add_node(
        "T2", node_type=RegistryNodeType.TOOL, name="T2", mcp_server="TestBot"
    )
    mock_graph.add_edge("tool:search", "T2", relationship=RegistryEdgeType.DEPENDS_ON)

    engine = _bind_isolated_engine(mock_graph)
    path = engine.find_path("TestBot", "T2")
    assert path == ["TestBot", "tool:search", "T2"]


def _hashed_bow_embedding(text: str, dims: int = 64) -> list[float]:
    """A tiny deterministic bag-of-words embedding for hermetic search tests.

    No real embedding model is configured in this environment. Hashing each
    word into a fixed-size vector gives cosine similarity a real (if crude)
    lexical-overlap signal, so an exact-content query still scores as
    relevant -- unlike an all-zero/absent embedding, which the retrieval
    quality gate correctly rejects as noise.
    """
    import hashlib

    vec = [0.0] * dims
    for word in text.lower().split():
        idx = int(hashlib.sha256(word.encode()).hexdigest(), 16) % dims
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


@pytest.mark.asyncio
async def test_memory_operations(monkeypatch):
    fake_embed_model = MagicMock()
    fake_embed_model.get_text_embedding.side_effect = _hashed_bow_embedding
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model",
        lambda *args, **kwargs: fake_embed_model,
    )

    graph = GraphComputeEngine(backend_type="rust")
    engine = _bind_isolated_engine(graph)

    # Add
    content = "User prefers dark mode"
    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "testuuid"
        mem_id = engine.add_memory(content, name="PrefTest", category="preference")
        assert mem_id == "mem:testuuid"
        assert mem_id in graph
        assert graph.nodes[mem_id]["description"] == content

    # Search
    results = engine.search_memories("dark mode")
    assert len(results) == 1
    assert results[0]["id"] == mem_id

    # Update
    # MemoryNode's field is 'importance_score' (see models/knowledge_graph.py),
    # not 'importance'.
    engine.update_memory(mem_id, importance_score=0.9)
    assert graph.nodes[mem_id]["importance_score"] == 0.9

    # Delete
    engine.delete_memory(mem_id)
    assert mem_id not in graph
