from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.models.knowledge_pack import (
    KnowledgePackBundle,
    KnowledgePackExporter,
    KnowledgePackImporter,
    generate_deterministic_id,
)


def test_deterministic_id():
    id1 = generate_deterministic_id("article", "https://example.com/paper")
    id2 = generate_deterministic_id("article", "https://example.com/paper")
    id3 = generate_deterministic_id("article", "https://example.com/other")
    id4 = generate_deterministic_id("software_project", "https://example.com/paper")

    assert id1 == id2
    assert id1 != id3
    assert id1 != id4
    assert id1.startswith("kp:")


def test_bundle_serialization():
    bundle = KnowledgePackBundle(
        name="test-pack",
        nodes=[{"type": "article", "id": "kp:123", "name": "Test Article"}],
        edges=[{"source": "kp:123", "target": "kp:456", "type": "contains"}],
    )

    d = bundle.to_dict()
    assert d["name"] == "test-pack"
    assert len(d["nodes"]) == 1

    b2 = KnowledgePackBundle.from_dict(d)
    assert b2.name == "test-pack"
    assert b2.nodes[0]["id"] == "kp:123"


def test_export_import(tmp_path):
    bundle = KnowledgePackBundle(
        name="test-pack",
        nodes=[{"type": "article", "id": "kp:123", "name": "Test Article"}],
        edges=[{"source": "kp:123", "target": "kp:456", "type": "contains"}],
    )

    yaml_path = tmp_path / "test.yaml"
    json_path = tmp_path / "test.json"

    KnowledgePackExporter.to_yaml(bundle, yaml_path)
    KnowledgePackExporter.to_json(bundle, json_path)

    loaded_yaml = KnowledgePackImporter.load(yaml_path)
    assert loaded_yaml.name == "test-pack"
    assert len(loaded_yaml.nodes) == 1

    loaded_json = KnowledgePackImporter.load(json_path)
    assert loaded_json.name == "test-pack"
    assert len(loaded_json.nodes) == 1


def test_seeding_idempotency():
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    # A bare EpistemicGraphBackend()/IntelligenceGraphEngine(db_path=...)
    # independently resolves its OWN tenant-routed default graph
    # (resolve_routing_graph(None) -> the shared "tenant__<tenant>____commons__"
    # graph), NOT the per-test isolated graph tests/conftest.py's
    # isolate_graph_compute_engine fixture provisions. Every test in a
    # multi-test run that took this bare path collided on that ONE shared
    # durable tenant graph -- durable lifecycle registrations against it raced
    # across tests and failed STALE_FENCE ("lifecycle batch ... is no longer
    # current"). Constructing the isolated GraphComputeEngine first and
    # rebinding the backend to it keeps this test on its own graph, matching
    # the idiom already established in test_kg_native_orchestration.py.
    compute = GraphComputeEngine(backend_type="rust")
    backend = EpistemicGraphBackend()
    backend._graph = compute
    engine = IntelligenceGraphEngine(backend=backend)

    bundle = KnowledgePackBundle(
        name="test-pack",
        nodes=[
            {"type": "article", "id": "kp:node1", "name": "Test Node 1"},
            {"type": "article", "id": "kp:node2", "name": "Test Node 2"},
        ],
        edges=[{"source": "kp:node1", "target": "kp:node2", "type": "mentions"}],
    )

    # Seed first time
    counts1 = KnowledgePackImporter.seed_into_kg(bundle, engine)
    assert counts1["nodes_seeded"] == 2
    assert counts1["edges_seeded"] == 1
    assert counts1["errors"] == 0

    # Seed second time (should be idempotent)
    counts2 = KnowledgePackImporter.seed_into_kg(bundle, engine)
    assert counts2["nodes_seeded"] == 2
    assert counts2["edges_seeded"] == 1
    assert counts2["errors"] == 0
