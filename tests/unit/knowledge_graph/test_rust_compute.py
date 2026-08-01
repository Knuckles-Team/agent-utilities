"""CONCEPT:AU-KG.ingest.engineering-rules Rust epistemic-graph compute unit tests.
Verifies the optimized graph functions using the Tokio-based epistemic-graph daemon.
"""

import uuid

import pytest
from _test_engine import TEST_AGENT_ID, TEST_AUDIENCE, TEST_POLICY_VERSION, TEST_TENANT

from agent_utilities.knowledge_graph.core.engine_breaker import unwrap_client
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _session_for(graph_name: str) -> GraphSession:
    actor = ActorContext(
        actor_id=TEST_AGENT_ID,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:admin",),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=TEST_TENANT,
        scopes=frozenset({"kg:read", "kg:write", "kg:admin"}),
        graph=graph_name,
        policy_version=TEST_POLICY_VERSION,
        audience=TEST_AUDIENCE,
    )


@pytest.fixture
def engine():
    name = f"test_{uuid.uuid4().hex[:8]}"
    with use_session(_session_for(name)):
        g = GraphComputeEngine(backend_type="rust", graph_name=name)
        if g._client:
            try:
                # ``create_graph`` is not a client method (there is no such
                # RPC); durable registration is ``tenants.create``. This root
                # graph is usually already registered by __init__'s own
                # connect path, so a failure here (AlreadyExists or similar)
                # is expected and harmless — only ``.clear()`` below needs to
                # succeed.
                g._client.tenants.create(name)
            except Exception:
                pass
            g._client.clear()
        yield g


def test_rust_graph_compute_nodes_and_edges(engine):
    """Verify that the optimized Tokio-based epistemic_graph service can build and manage nodes and edges."""

    # Verify properties
    assert engine._mode in ["service", "embedded"]

    # Add nodes
    engine.add_node("A", {"label": "Agent"})
    engine.add_node("B", {"label": "Tool"})
    engine.add_node("C", {"label": "Knowledge"})

    # Verify node count
    assert engine.node_count() == 3

    # Add edges
    engine.add_edge("A", "B", {"weight": 1.5, "relationship": "connects"})
    engine.add_edge("B", "C", {"weight": 2.0, "relationship": "connects"})

    # Verify nodes and edges exist
    assert engine.has_node("A")
    assert engine.has_node("B")
    assert engine.has_node("C")
    assert engine.has_edge("A", "B")
    assert engine.has_edge("B", "C")
    assert not engine.has_edge("A", "C")

    # Remove edge
    engine.remove_edge("A", "B")
    assert not engine.has_edge("A", "B")

    # Remove node
    engine.remove_node("C")
    assert not engine.has_node("C")


def test_rust_topological_sort_and_cycles(engine):
    """Verify topological sorting and cycle detection on the optimized Tokio-based epistemic_graph service."""

    engine.add_node("A", {})
    engine.add_node("B", {})
    engine.add_node("C", {})

    engine.add_edge("A", "B", {"relationship": "connects"})
    engine.add_edge("B", "C", {"relationship": "connects"})

    # Verify topological order
    order = engine.topological_sort()
    assert order == ["A", "B", "C"]

    # Verify no cycle is found
    assert engine.find_cycle() is None

    # Introduce cycle C -> A
    engine.add_edge("C", "A", {"relationship": "connects"})

    # Cycle detection
    cycle = engine.find_cycle()
    assert cycle is not None
    assert len(cycle) > 1

    # Topological sort should now raise an error due to the cycle
    with pytest.raises(ValueError, match="Graph contains cycles"):
        engine.topological_sort()


def test_rust_shortest_path(engine):
    """Verify shortest path computations on the optimized Tokio-based epistemic_graph service."""

    engine.add_node("X", {})
    engine.add_node("Y", {})
    engine.add_node("Z", {})

    engine.add_edge("X", "Y", {"relationship": "connects"})
    engine.add_edge("Y", "Z", {"relationship": "connects"})

    # Get shortest path
    path = engine.get_shortest_path("X", "Z")
    assert path == ["X", "Y", "Z"]

    # Path for unconnected elements
    engine.add_node("W", {})
    assert engine.get_shortest_path("X", "W") is None


def test_rust_blast_radius(engine):
    """Verify blast radius computation on the optimized Tokio-based epistemic_graph service."""

    engine.add_node("A", {})
    engine.add_node("B", {})
    engine.add_node("C", {})
    engine.add_node("D", {})

    engine.add_edge("A", "B", {"relationship": "connects"})
    engine.add_edge("B", "C", {"relationship": "connects"})
    engine.add_edge("B", "D", {"relationship": "connects"})

    blast = engine.get_blast_radius("A", max_depth=2)
    # Expected outgoing neighbors from A: B (depth 1), then C and D (depth 2)
    assert len(blast) == 3
    node_ids = {node["id"] for node in blast}
    assert node_ids == {"B", "C", "D"}


def test_rust_repository_ast_parsing(engine):
    """Index logical source names and bytes without exposing a host path."""
    source = b"class MyAgent:\n    pass\ndef my_tool():\n    pass\n"

    result = engine.index_repository([("mod.py", source)])

    assert result["files_parsed"] == 1
    rendered_nodes = repr(result["nodes"])
    assert "MyAgent" in rendered_nodes
    assert "my_tool" in rendered_nodes


def test_rust_vf2_subgraph_matching(engine):
    """Verify subgraph isomorphism matching natively on the optimized Tokio-based epistemic_graph service."""
    import uuid

    # NOTE: "type" here is a plain, arbitrary VF2-matched *property* (the
    # engine's own vf2_subgraph_match compares it as an ordinary property
    # value, per epistemic-graph/tests/test_epistemic_graph.py::
    # test_vf2_subgraph_match) — NOT the graph's node-label system field, so
    # the repo-wide retired-bare-"type"-key -> "node_type" migration does not
    # apply to it. A prior sweep (6bb38954) mistakenly rewrote it to
    # "node_type", which the engine's VF2 matcher never looks at, so every
    # candidate pair silently failed its type-compatibility check (0
    # matches). GraphComputeEngine.add_node's own policy hard-rejects a
    # literal "type" property (reserved for "node_type"), so VF2's nodes are
    # built through the raw client's nodes/edges namespace instead — exactly
    # what the reference test does. Edge properties are empty, matching the
    # same reference test (VF2 only needs edge topology here, not edge
    # properties).
    raw_engine = unwrap_client(engine._client)
    raw_engine.nodes.add("A", {"type": "class"})
    raw_engine.nodes.add("B", {"type": "function"})
    raw_engine.nodes.add("C", {"type": "function"})
    raw_engine.edges.add("A", "B", {})
    raw_engine.edges.add("A", "C", {})

    pattern_name = f"test_pattern_{uuid.uuid4().hex[:8]}"
    with use_session(_session_for(pattern_name)):
        pattern = GraphComputeEngine.get_or_create(
            graph_name=pattern_name, backend_type="rust"
        )
        if pattern._client:
            try:
                # ``create_graph`` is not a client method (there is no such
                # RPC — it silently AttributeErrors and was swallowed by this
                # bare except, so the pattern graph was never actually
                # registered and the later cross-graph vf2_subgraph_match
                # RPC failed with "Graph ... not found"). Durable
                # registration is ``tenants.create``; unlike the root
                # ``engine`` graph (created by __init__'s own connect path)
                # this second named graph — reached via get_or_create()'s
                # scoped view over the SAME shared transport — is never
                # auto-registered, so this call must actually succeed.
                pattern._client.tenants.create(pattern_name)
            except Exception:
                pass
            pattern._client.clear()

        raw_pattern = unwrap_client(pattern._client)
        raw_pattern.nodes.add("P1", {"type": "class"})
        raw_pattern.nodes.add("P2", {"type": "function"})
        raw_pattern.edges.add("P1", "P2", {})

    matches = engine.vf2_subgraph_match(pattern)
    assert len(matches) == 2
    # Verify that pattern P1 matches A, P2 matches B or C
    assert matches[0]["P1"] == "A"
    assert matches[1]["P1"] == "A"
    matched_p2 = {m["P2"] for m in matches}
    assert matched_p2 == {"B", "C"}


def test_rust_reactive_state_ledger(engine):
    """Verify reactive state ledger serialization/deserialization and apply/replay."""
    import uuid

    engine.add_node("A", {"label": "Alpha"})
    engine.add_node("B", {"label": "Beta"})
    engine.add_edge("A", "B", {"weight": 1.0, "relationship": "connects"})

    # Capture ledger
    txs = engine.get_ledger()
    assert len(txs) >= 3

    # Replay onto another engine
    engine2_name = f"test_{uuid.uuid4().hex[:8]}"
    with use_session(_session_for(engine2_name)):
        # The process allows exactly one root transport (graph_compute.py:
        # "A process graph transport already exists; use
        # GraphComputeEngine.get_or_create()/for_graph()") — the `engine`
        # fixture already owns it, so a second named graph must be reached as
        # a scoped view via get_or_create(), not a fresh direct construction.
        engine2 = GraphComputeEngine.get_or_create(
            graph_name=engine2_name, backend_type="rust"
        )
        if engine2._client:
            try:
                # See test_rust_vf2_subgraph_matching: ``create_graph`` is not
                # a real client method; durable registration is
                # ``tenants.create``.
                engine2._client.tenants.create(engine2_name)
            except Exception:
                pass
            engine2._client.clear()

        engine2.apply_ledger(txs)
        assert engine2.node_count() == 2
        assert engine2.has_edge("A", "B")

    # Verify msgpack serialization
    js = engine.to_msgpack()
    engine3_name = f"test_{uuid.uuid4().hex[:8]}"
    with use_session(_session_for(engine3_name)):
        engine3 = GraphComputeEngine.get_or_create(
            graph_name=engine3_name, backend_type="rust"
        )
        if engine3._client:
            try:
                # See test_rust_vf2_subgraph_matching: ``create_graph`` is not
                # a real client method; durable registration is
                # ``tenants.create``.
                engine3._client.tenants.create(engine3_name)
            except Exception:
                pass
            engine3._client.clear()

        engine3.from_msgpack(js)
        assert engine3.node_count() == 2
        assert engine3.has_edge("A", "B")
