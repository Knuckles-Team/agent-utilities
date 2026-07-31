import json
import os
import tempfile

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.models.domains.infrastructure import (
    GPUAcceleratorNode,
    PlatformServiceNode,
    StorageArrayNode,
)
from agent_utilities.models.knowledge_graph import HostNode, RegistryNodeType

# Mock inventory data for testing
MOCK_INVENTORY = """
all:
  children:
    synthetic_fleet:
      hosts:
        storage-node-a:
          ansible_host: 192.0.2.10
          role: storage
          capacity_tb: 24
          storage_type: sas
        compute-node-b:
          ansible_host: 192.0.2.11
          role: compute
        analysis-node-a:
          ansible_host: 192.0.2.13
          role: compute_high
          cores: 32
          ram_gb: 256
        accelerator-node-a:
          ansible_host: 192.0.2.16
          role: gpu
          gpu: true
          vram_gb: 16
          gpu_vendor: nvidia
      vars:
        ansible_user: test_user
        ansible_ssh_private_key_file: /mock/path/id_rsa
"""


@pytest.fixture
def mock_inventory_file():
    """Create a temporary inventory.yaml file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        f.write(MOCK_INVENTORY)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_infrastructure_pydantic_models():
    """Verify that new infrastructure pydantic models are correct."""
    # Test HostNode
    h = HostNode(
        id="host:test", name="test", hostname="1.2.3.4", alias="test", user="u"
    )
    assert h.type == RegistryNodeType.HOST

    # Test PlatformServiceNode
    s = PlatformServiceNode(
        id="service:test",
        name="svc",
        endpoint="http://svc",
        labels={"requires_gpu": "true"},
    )
    assert s.type == RegistryNodeType.PLATFORM_SERVICE
    assert s.labels.get("requires_gpu") == "true"

    # Test GPUAcceleratorNode
    g = GPUAcceleratorNode(id="gpu:test", name="gpu", vram_gb=8.0, vendor="Nvidia")
    assert g.type == RegistryNodeType.GPU_ACCELERATOR

    # Test StorageArrayNode
    sa = StorageArrayNode(
        id="storage:test", name="storage", capacity_tb=24.0, storage_type="SAS"
    )
    assert sa.type == RegistryNodeType.STORAGE_ARRAY


def _create_engine():
    os.environ["AGENT_UTILITIES_TESTING"] = "true"
    return IntelligenceGraphEngine(db_path=":memory:")


def test_host_ingestion_and_sparql_matchmaking(mock_inventory_file):
    """Test full pipeline: ingestion, OWL bridge RDF promotion, and SPARQL matchmaking."""
    # Initialize high-performance graph engine
    engine = _create_engine()

    # Ingest hosts from mock inventory
    ingested = engine.ingest_hosts_from_inventory(inventory_path=mock_inventory_file)
    assert len(ingested) == 4
    assert any(node_id.startswith("host:pref_host_") for node_id in ingested)

    # Verify LPG relationships were correctly created. The engine stores the
    # edge's canonical property verbatim under "relationship" (no case
    # transform), so read edge data via get_edge_data rather than networkx
    # adjacency subscripting.
    def _host_role(node_id: str) -> str:
        # dict-valued properties (like HostNode.labels) are JSON-encoded for
        # storage (IntelligenceGraphEngine._serialize_node) and come back as a
        # JSON string, not a dict — see agent_utilities/knowledge_graph/core/
        # engine.py's _serialize_node.
        raw = engine.graph.nodes[node_id]["labels"]
        labels = json.loads(raw) if isinstance(raw, str) else raw
        return labels.get("role")

    storage_host = next(
        node_id for node_id in ingested if _host_role(node_id) == "storage"
    )
    storage_id = storage_host.replace("host:", "storage:", 1)
    assert engine.graph.has_edge(storage_host, storage_id)
    storage_edge = engine.graph.get_edge_data(storage_host, storage_id)[0]
    assert storage_edge["relationship"] == "attached_storage"

    accelerator_host = next(
        node_id for node_id in ingested if _host_role(node_id) == "gpu"
    )
    accelerator_id = accelerator_host.replace("host:", "gpu:", 1)
    assert engine.graph.has_edge(accelerator_host, accelerator_id)
    gpu_edge = engine.graph.get_edge_data(accelerator_host, accelerator_id)[0]
    assert gpu_edge["relationship"] == "has_accelerator"

    # Matchmaking runs SPARQL over the OWL backend, which needs owlready2
    # (an optional extra). Skip that portion when it isn't installed, matching
    # the importorskip convention used by the other OWL test modules.
    pytest.importorskip("owlready2")

    # generate_matchmaking_recommendations() only ever produces a recommendation
    # per PlatformService node found via SPARQL (`?service rdf:type
    # au:PlatformService`) — none exist yet (ingest_hosts_from_inventory only
    # creates Host/GPUAccelerator/StorageArray nodes), so this seeds the three
    # services the assertions below expect: one GPU-requiring, one
    # storage-requiring, one high-compute-requiring.
    for svc in (
        PlatformServiceNode(
            id="service:ollama",
            name="ollama-service",
            endpoint="http://ollama:11434",
            labels={"requires_gpu": "true"},
        ),
        PlatformServiceNode(
            id="service:postgres",
            name="postgres",
            endpoint="postgres://db:5432",
            labels={"requires_storage": "true"},
        ),
        PlatformServiceNode(
            id="service:reasoner",
            name="reasoner",
            endpoint="http://reasoner:8080",
            labels={"requires_high_compute": "true"},
        ),
    ):
        engine.graph.add_node(svc.id, **engine._serialize_node(svc))

    # Generate matchmaking recommendations
    recs = engine.generate_matchmaking_recommendations(
        inventory_path=mock_inventory_file
    )
    assert len(recs) > 0

    # Check that the model service matches the synthetic GPU-equipped node.
    ollama_rec = next(r for r in recs if r["service_name"] == "ollama-service")
    assert ollama_rec["best_host"] == accelerator_host
    assert ollama_rec["match_score"] >= 80.0
    assert any("GPU" in reason for reason in ollama_rec["rationale"])

    # Check that the database service matches the synthetic storage node.
    pg_rec = next(r for r in recs if r["service_name"] == "postgres")
    assert pg_rec["best_host"] == storage_host
    assert pg_rec["match_score"] >= 75.0
    assert any("storage" in reason.lower() for reason in pg_rec["rationale"])

    # Check that heavy-thinking reasoning worker matches with compute host
    reasoner_rec = next(r for r in recs if r["service_name"] == "reasoner")
    assert reasoner_rec["best_host"] in ingested
    assert reasoner_rec["match_score"] >= 80.0
    assert any("high-compute" in reason for reason in reasoner_rec["rationale"])
