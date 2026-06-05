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
    homelab:
      hosts:
        r510:
          ansible_host: 10.0.0.10
        r710:
          ansible_host: 10.0.0.11
        r820:
          ansible_host: 10.0.0.13
        gr1080:
          ansible_host: 10.0.0.16
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
    assert "host:r510" in ingested
    assert "host:gr1080" in ingested

    # Verify LPG relationships were correctly created. The graph engine
    # canonicalizes the relationship under ``rel_type`` (uppercased), so read
    # edge data via get_edge_data rather than networkx adjacency subscripting.
    assert engine.graph.has_edge("host:r510", "storage:r510")
    storage_edge = engine.graph.get_edge_data("host:r510", "storage:r510")[0]
    assert storage_edge["rel_type"] == "ATTACHED_STORAGE"

    assert engine.graph.has_edge("host:gr1080", "gpu:gr1080")
    gpu_edge = engine.graph.get_edge_data("host:gr1080", "gpu:gr1080")[0]
    assert gpu_edge["rel_type"] == "HAS_ACCELERATOR"

    # Matchmaking runs SPARQL over the OWL backend, which needs owlready2
    # (an optional extra). Skip that portion when it isn't installed, matching
    # the importorskip convention used by the other OWL test modules.
    pytest.importorskip("owlready2")

    # Generate matchmaking recommendations
    recs = engine.generate_matchmaking_recommendations(
        inventory_path=mock_inventory_file
    )
    assert len(recs) > 0

    # Check that ollama matches with GPU-equipped gr1080
    ollama_rec = next(r for r in recs if r["service_name"] == "ollama-service")
    assert ollama_rec["best_host"] == "gr1080"
    assert ollama_rec["match_score"] >= 80.0
    assert any("GPU" in reason for reason in ollama_rec["rationale"])

    # Check that postgres matches with Storage-equipped r510
    pg_rec = next(r for r in recs if r["service_name"] == "postgres")
    assert pg_rec["best_host"] == "r510"
    assert pg_rec["match_score"] >= 75.0
    assert any("storage" in reason.lower() for reason in pg_rec["rationale"])

    # Check that heavy-thinking reasoning worker matches with compute host
    reasoner_rec = next(r for r in recs if r["service_name"] == "reasoner")
    assert reasoner_rec["best_host"] == "r820"
    assert reasoner_rec["match_score"] >= 80.0
    assert any("high-compute" in reason for reason in reasoner_rec["rationale"])
