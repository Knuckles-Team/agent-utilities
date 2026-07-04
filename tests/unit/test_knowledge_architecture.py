#!/usr/bin/python
"""Tests for Knowledge Architecture gaps: SPARQL, ArchiMate, ADR.

CONCEPT:AU-KG.research.research-pipeline-runner — Context Graph Architecture

Tests covering:
- SPARQL read-only endpoint via rdflib materialization
- ArchiMate EA governance layer classification and view generation
- Architecture Decision Records (ADR) as first-class KG nodes
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_graph() -> GraphComputeEngine:
    """Create a sample KG graph with typed nodes and edges."""
    g = GraphComputeEngine(backend_type="rust")

    # Add nodes with types
    g.add_node("agent_1", type="agent", name="TestAgent", description="A test agent")
    g.add_node("tool_1", type="tool", name="CodeSearch", description="Search code")
    g.add_node("concept_1", type="concept", name="AU-KG.query.vendor-agnostic-traversal", description="Architecture")
    g.add_node("policy_1", type="policy", name="NoDelete", description="No deletions")
    g.add_node("server_1", type="server", name="MCPServer", description="MCP endpoint")
    g.add_node("goal_1", type="goal", name="Governance", description="Full governance")

    # Add edges
    g.add_edge("agent_1", "tool_1", type="provides")
    g.add_edge("agent_1", "concept_1", type="impacts_concept")
    g.add_edge("policy_1", "concept_1", type="motivated_by")

    return g


@pytest.fixture
def mock_owl_backend():
    """Create a mock OWL backend for OWLBridge."""
    mock = MagicMock()
    mock.query_sparql = None  # Force rdflib fallback
    del mock.query_sparql  # Remove the attribute entirely
    return mock


# ═══════════════════════════════════════════════════════════════════════════════
# SPARQL Tests (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSPARQL:
    """Tests for SPARQL read-only endpoint via rdflib."""

    def test_build_rdf_graph_from_lpg(self, sample_graph, mock_owl_backend):
        """Verifies node and edge materialization into rdflib graph."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

        bridge = OWLBridge(graph=sample_graph, owl_backend=mock_owl_backend)

        rdf_graph = bridge._build_rdf_graph()  # type: ignore[attr-defined]

        # Should have triples for nodes (type + properties) and edges
        triple_count = len(list(rdf_graph))
        assert triple_count > 0, "RDF graph should have triples"

        # Should have at least 6 type triples (one per node)
        type_triples = list(rdf_graph.triples((None, None, None)))
        assert len(type_triples) >= 6, f"Expected >= 6 triples, got {len(type_triples)}"

    def test_sparql_select_by_type(self, sample_graph, mock_owl_backend):
        """Test SELECT query filtering by type."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

        bridge = OWLBridge(graph=sample_graph, owl_backend=mock_owl_backend)

        results = bridge.query_sparql(
            "PREFIX au: <http://agent-utilities.dev/ontology#>\n"
            "SELECT ?s WHERE { ?s a au:Agent }"
        )

        assert len(results) >= 1, f"Expected at least 1 agent, got {results}"

    def test_sparql_with_namespace_prefix(self, sample_graph, mock_owl_backend):
        """Test that au: namespace prefix is auto-injected."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

        bridge = OWLBridge(graph=sample_graph, owl_backend=mock_owl_backend)

        # Query without explicit PREFIX — should be auto-injected
        results = bridge.query_sparql("SELECT ?s WHERE { ?s a au:Tool }")

        assert len(results) >= 1, "Should find at least 1 tool"

    def test_sparql_ask_query(self, sample_graph, mock_owl_backend):
        """Test ASK query returns a result."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

        bridge = OWLBridge(graph=sample_graph, owl_backend=mock_owl_backend)

        results = bridge.query_sparql("ASK { ?s a au:Agent }")

        assert isinstance(results, list)
        assert len(results) >= 1
        # Should not be an error response
        if results:
            assert "error" not in results[0]

    def test_sparql_cache_invalidation(self, mock_owl_backend):
        """The rdflib RDF-graph cache rebuilds when the LPG changes.

        This exercises the rdflib materialization (``_sparql_via_rdflib`` /
        ``_build_rdf_graph``) DIRECTLY: under the engine-native architecture
        (CONCEPT:AU-KG.compute.native-sparql-owl-shacl) ``query_sparql`` prefers the engine projection and only
        builds the rdflib cache on the no-engine path, so the cache invariant belongs
        to the rdflib materialization itself. A hermetic networkx LPG keeps it engine-
        independent and deterministic.
        """
        pytest.importorskip("rdflib")
        import networkx as nx

        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

        g = nx.MultiDiGraph()
        g.add_node("agent_1", type="agent", name="TestAgent")
        bridge = OWLBridge(graph=g, owl_backend=mock_owl_backend)

        # First materialization — builds the rdflib cache.
        bridge._sparql_via_rdflib("SELECT ?s WHERE { ?s a au:Agent }")  # type: ignore[attr-defined]
        hash1 = bridge._rdf_cache_hash  # type: ignore[attr-defined]

        # Mutate the LPG, then re-materialize — the cache key must change so the stale
        # RDF graph is rebuilt.
        g.add_node("agent_2", type="agent", name="NewAgent")
        bridge._sparql_via_rdflib("SELECT ?s WHERE { ?s a au:Agent }")  # type: ignore[attr-defined]
        hash2 = bridge._rdf_cache_hash  # type: ignore[attr-defined]

        assert hash2 != hash1, "Cache hash should change after graph modification"

    def test_sparql_empty_graph(self, mock_owl_backend):
        """Graceful handling of empty graph."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge

        empty_graph = GraphComputeEngine(backend_type="rust")
        bridge = OWLBridge(graph=empty_graph, owl_backend=mock_owl_backend)

        results = bridge.query_sparql("SELECT ?s WHERE { ?s a au:Agent }")

        assert isinstance(results, list)
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ArchiMate Tests (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestArchiMate:
    """Tests for ArchiMate EA governance layer."""

    def test_classify_agent_to_application_component(self):
        """Agent should map to ApplicationComponent in Application layer."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("agent")

        assert result.layer == ArchiMateLayerType.APPLICATION
        assert result.archimate_type == "ApplicationComponent"

    def test_classify_policy_to_business_rule(self):
        """Policy should map to BusinessRule in Business layer."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("policy")

        assert result.layer == ArchiMateLayerType.BUSINESS
        assert result.archimate_type == "BusinessRule"

    def test_classify_concept_to_capability(self):
        """Concept should map to Capability in Strategy layer."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("concept")

        assert result.layer == ArchiMateLayerType.STRATEGY
        assert result.archimate_type == "Capability"

    def test_classify_goal_to_motivation(self):
        """Goal should map to Goal in Motivation layer."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("goal")

        assert result.layer == ArchiMateLayerType.MOTIVATION
        assert result.archimate_type == "Goal"

    def test_archimate_layers_complete(self):
        """All 5 ArchiMate layers should have mappings."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
        )

        layer = ArchiMateLayer()
        all_layers = layer.get_all_layers()

        expected_layers = {
            "business",
            "application",
            "technology",
            "strategy",
            "motivation",
        }
        assert set(all_layers.keys()) == expected_layers

        for layer_name, members in all_layers.items():
            assert len(members) > 0, f"{layer_name} layer should have members"

    def test_unknown_type_returns_unclassified(self):
        """Unknown KG types should return UNCLASSIFIED."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("totally_unknown_type")

        assert result.layer == ArchiMateLayerType.UNCLASSIFIED
        assert result.archimate_type == "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# ADR Tests (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestArchitectureDecisionRecords:
    """Tests for ADR as first-class KG nodes."""

    def test_adr_model_validation(self):
        """ADR Pydantic model validates correctly."""
        from agent_utilities.models.knowledge_graph import (
            ArchitectureDecisionRecord,
            RegistryNodeType,
        )

        adr = ArchitectureDecisionRecord(
            id="adr-001",
            name="Use LadybugDB",
            title="Use LadybugDB for graph storage",
            status="accepted",
            context="Need a graph database that supports Cypher",
            decision="Use LadybugDB (Kuzu-based)",
            rationale="Native Cypher support, embedded, no external service",
            alternatives=["Neo4j", "FalkorDB", "SQLite+GraphComputeEngine"],
            consequences=["No native SPARQL", "WAL corruption risk"],
            authority="user",
            pillar="KG",
            impacted_concepts=["AU-KG.query.object-graph-mapper", "AU-KG.query.vendor-agnostic-traversal"],
        )

        assert adr.type == RegistryNodeType.ARCHITECTURE_DECISION  # type: ignore[attr-defined]
        assert adr.status == "accepted"
        assert len(adr.alternatives) == 3
        assert "AU-KG.query.vendor-agnostic-traversal" in adr.impacted_concepts

    def test_adr_status_transitions(self):
        """ADR status should accept all valid lifecycle values."""
        from agent_utilities.models.knowledge_graph import ArchitectureDecisionRecord

        for status in ["proposed", "accepted", "deprecated", "superseded"]:
            adr = ArchitectureDecisionRecord(
                id=f"adr-{status}",
                name=f"Test {status}",
                status=status,  # type: ignore[arg-type]
            )
            assert adr.status == status

    def test_adr_supersedes_chain(self):
        """ADR supersession should be trackable."""
        from agent_utilities.models.knowledge_graph import ArchitectureDecisionRecord

        adr_v1 = ArchitectureDecisionRecord(
            id="adr-v1",
            name="Original Decision",
            status="superseded",
            superseded_by="adr-v2",
        )

        adr_v2 = ArchitectureDecisionRecord(
            id="adr-v2",
            name="Updated Decision",
            status="superseded",
            superseded_by="adr-v3",
        )

        adr_v3 = ArchitectureDecisionRecord(
            id="adr-v3",
            name="Current Decision",
            status="accepted",
        )

        assert adr_v1.superseded_by == "adr-v2"
        assert adr_v2.superseded_by == "adr-v3"
        assert adr_v3.superseded_by == ""

    def test_adr_impacts_concepts(self):
        """ADR should track impacted concept IDs."""
        from agent_utilities.models.knowledge_graph import ArchitectureDecisionRecord

        adr = ArchitectureDecisionRecord(
            id="adr-sparql",
            name="Add SPARQL endpoint",
            impacted_concepts=["AU-KG.query.vendor-agnostic-traversal", "AU-KG.query.object-graph-mapper", "AU-ORCH.planning.recursion-nesting-depth"],
        )

        assert len(adr.impacted_concepts) == 3
        assert "AU-KG.query.vendor-agnostic-traversal" in adr.impacted_concepts

    def test_adr_from_evolution_cycle(self):
        """ADR created from evolution cycle has proper authority."""
        from agent_utilities.models.knowledge_graph import ArchitectureDecisionRecord

        adr = ArchitectureDecisionRecord(
            id="adr-evo-001",
            name="Adopt LCM Memory Architecture",
            context="Comparative analysis revealed memory compaction gap",
            decision="Implement Summary DAG from LCM paper",
            rationale="Native compaction with O(log n) lookup",
            authority="evolution_daemon",
            pillar="KG",
        )

        assert adr.authority == "evolution_daemon"
        assert adr.pillar == "KG"

    def test_adr_alternatives_tracked(self):
        """ADR alternatives should be a list of strings."""
        from agent_utilities.models.knowledge_graph import ArchitectureDecisionRecord

        adr = ArchitectureDecisionRecord(
            id="adr-backend",
            name="Choose graph backend",
            alternatives=["Neo4j", "FalkorDB", "Kuzu/LadybugDB"],
            decision="Use LadybugDB",
        )

        assert isinstance(adr.alternatives, list)
        assert "Neo4j" in adr.alternatives

    def test_adr_authority_field(self):
        """ADR authority field accepts various sources."""
        from agent_utilities.models.knowledge_graph import ArchitectureDecisionRecord

        for authority in [
            "user",
            "evolution_daemon",
            "policy:no-delete",
            "team:platform",
        ]:
            adr = ArchitectureDecisionRecord(
                id=f"adr-auth-{authority}",
                name="Test",
                authority=authority,
            )
            assert adr.authority == authority

    def test_adr_owl_promotion(self):
        """ADR node type should be in PROMOTABLE_NODE_TYPES."""
        from agent_utilities.knowledge_graph.core.owl_bridge import (
            PROMOTABLE_NODE_TYPES,
        )

        assert "architecture_decision" in PROMOTABLE_NODE_TYPES

    def test_adr_edge_types_promotable(self):
        """ADR edge types should be in PROMOTABLE_EDGE_TYPES."""
        from agent_utilities.knowledge_graph.core.owl_bridge import (
            PROMOTABLE_EDGE_TYPES,
        )

        assert "impacts_concept" in PROMOTABLE_EDGE_TYPES
        assert "alternatives_to" in PROMOTABLE_EDGE_TYPES
        assert "decided_by" in PROMOTABLE_EDGE_TYPES
        assert "supersedes" in PROMOTABLE_EDGE_TYPES

    def test_adr_registry_node_type_enum(self):
        """ADR should be in RegistryNodeType enum."""
        from agent_utilities.models.knowledge_graph import RegistryNodeType

        assert hasattr(RegistryNodeType, "ARCHITECTURE_DECISION")
        assert RegistryNodeType.ARCHITECTURE_DECISION == "architecture_decision"

    def test_adr_registry_edge_type_enum(self):
        """ADR edges should be in RegistryEdgeType enum."""
        from agent_utilities.models.knowledge_graph import RegistryEdgeType

        assert hasattr(RegistryEdgeType, "IMPACTS_CONCEPT")
        assert hasattr(RegistryEdgeType, "ALTERNATIVES_TO")


# ═══════════════════════════════════════════════════════════════════════════════
# SDD ArchiMate Mapping Tests (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSDDArchiMateMappings:
    """Tests for SDD type → ArchiMate 3.2 mappings."""

    def test_specification_maps_to_strategy_capability(self):
        """Specification → Strategy/Capability."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("specification")
        assert result.layer == ArchiMateLayerType.STRATEGY
        assert result.archimate_type == "Capability"

    def test_software_feature_maps_to_application_function(self):
        """SoftwareFeature → Application/ApplicationFunction."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("software_feature")
        assert result.layer == ArchiMateLayerType.APPLICATION
        assert result.archimate_type == "ApplicationFunction"

    def test_requirement_maps_to_motivation_requirement(self):
        """Requirement → Motivation/Requirement."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("requirement")
        assert result.layer == ArchiMateLayerType.MOTIVATION
        assert result.archimate_type == "Requirement"

    def test_user_story_maps_to_motivation_requirement(self):
        """UserStory → Motivation/Requirement."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("user_story")
        assert result.layer == ArchiMateLayerType.MOTIVATION
        assert result.archimate_type == "Requirement"

    def test_acceptance_criteria_maps_to_motivation(self):
        """AcceptanceCriteria → Motivation/Requirement."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("acceptance_criteria")
        assert result.layer == ArchiMateLayerType.MOTIVATION
        assert result.archimate_type == "Requirement"

    def test_software_component_maps_to_application_component(self):
        """SoftwareComponent → Application/ApplicationComponent."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("software_component")
        assert result.layer == ArchiMateLayerType.APPLICATION
        assert result.archimate_type == "ApplicationComponent"

    def test_api_contract_maps_to_application_interface(self):
        """APIContract → Application/ApplicationInterface."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("api_contract")
        assert result.layer == ArchiMateLayerType.APPLICATION
        assert result.archimate_type == "ApplicationInterface"

    def test_test_case_maps_to_application_function(self):
        """TestCase → Application/ApplicationFunction."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("test_case")
        assert result.layer == ArchiMateLayerType.APPLICATION
        assert result.archimate_type == "ApplicationFunction"

    def test_design_guideline_maps_to_motivation_principle(self):
        """DesignGuideline → Motivation/Principle."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("design_guideline")
        assert result.layer == ArchiMateLayerType.MOTIVATION
        assert result.archimate_type == "Principle"

    def test_compliance_constraint_maps_to_motivation_constraint(self):
        """ComplianceConstraint → Motivation/Constraint."""
        from agent_utilities.knowledge_graph.core.archimate_layer import (
            ArchiMateLayer,
            ArchiMateLayerType,
        )

        layer = ArchiMateLayer()
        result = layer.classify("compliance_constraint")
        assert result.layer == ArchiMateLayerType.MOTIVATION
        assert result.archimate_type == "Constraint"


# ═══════════════════════════════════════════════════════════════════════════════
# SHACL Validation Tests (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSHACLValidation:
    """Tests for SHACL governance validation."""

    def test_shacl_validator_init(self):
        """SHACLValidator should instantiate cleanly."""
        from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

        validator = SHACLValidator()
        assert validator is not None

    def test_shacl_validate_missing_shapes_file(self):
        """Should return conforms=True when shapes file doesn't exist."""
        from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

        validator = SHACLValidator()
        result = validator.validate(MagicMock(), "/nonexistent/shapes.ttl")

        assert result["conforms"] is True
        # Either "not found" (pyshacl present) or "not installed" (pyshacl missing)
        assert (
            "not found" in result["results_text"]
            or "not installed" in result["results_text"]
        )

    def test_shacl_parse_violations(self):
        """Should parse violation text into structured dicts."""
        from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

        text = (
            "Constraint Violation in AgentShape:\n"
            "  Severity: sh:Violation\n"
            "  Source Shape: :AgentShape\n"
            "  Focus Node: :agent_1\n"
            "  Message: Agent must have a name.\n"
            "  Result Path: :name\n"
        )

        violations = SHACLValidator._parse_violations(text)
        assert len(violations) == 1
        assert violations[0]["severity"] == "sh:Violation"
        assert violations[0]["focus_node"] == ":agent_1"
        assert "name" in violations[0]["message"]

    def test_shacl_validate_layered(self):
        """Layered validation combines results from multiple shapes."""
        from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

        validator = SHACLValidator()
        result = validator.validate_layered(
            MagicMock(),
            ["/nonexistent/a.ttl", "/nonexistent/b.ttl"],
        )

        assert result["conforms"] is True
        assert result["layers_checked"] == 2

    def test_shacl_governance_shapes_exist(self):
        """Governance shapes file should exist in shapes/ directory."""
        from pathlib import Path

        shapes_path = (
            Path(__file__).parent.parent.parent
            / "agent_utilities"
            / "knowledge_graph"
            / "shapes"
            / "governance.shapes.ttl"
        )
        assert shapes_path.exists(), f"Shapes file missing at {shapes_path}"

    def test_shacl_validate_kg_no_shapes(self):
        """validate_kg gracefully handles missing bridge._build_rdf_graph."""
        from agent_utilities.knowledge_graph.core.shacl_validator import SHACLValidator

        validator = SHACLValidator()

        mock_bridge = MagicMock()
        # _build_rdf_graph returns a mock graph
        import rdflib

        mock_bridge._build_rdf_graph.return_value = rdflib.Graph()

        result = validator.validate_kg(mock_bridge)
        # Should either succeed or note no violations
        assert isinstance(result, dict)
        assert "conforms" in result


# ═══════════════════════════════════════════════════════════════════════════════
# SPARQL HTTP Endpoint Tests (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSPARQLHTTPEndpoint:
    """Tests for W3C SPARQL Protocol HTTP endpoint."""

    def test_sparql_endpoint_init(self, sample_graph, mock_owl_backend):
        """SPARQLEndpoint should initialize with OWLBridge."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge
        from agent_utilities.knowledge_graph.core.sparql_http import SPARQLEndpoint

        bridge = OWLBridge(graph=sample_graph, owl_backend=mock_owl_backend)
        endpoint = SPARQLEndpoint(bridge)
        assert endpoint is not None

    def test_sparql_endpoint_execute(self, sample_graph, mock_owl_backend):
        """Execute should return W3C SPARQL Results JSON format."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge
        from agent_utilities.knowledge_graph.core.sparql_http import SPARQLEndpoint

        bridge = OWLBridge(graph=sample_graph, owl_backend=mock_owl_backend)
        endpoint = SPARQLEndpoint(bridge)

        result = endpoint.execute("SELECT ?s WHERE { ?s a au:Agent }")

        assert "head" in result
        assert "results" in result
        assert "bindings" in result["results"]

    def test_sparql_endpoint_handle_missing_query(self, sample_graph, mock_owl_backend):
        """Handle request with no query returns 400."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge
        from agent_utilities.knowledge_graph.core.sparql_http import SPARQLEndpoint

        bridge = OWLBridge(graph=sample_graph, owl_backend=mock_owl_backend)
        endpoint = SPARQLEndpoint(bridge)

        body, content_type, status = endpoint.handle_request(query=None)
        assert status == 400
        assert "error" in body

    def test_sparql_endpoint_handle_valid_query(self, sample_graph, mock_owl_backend):
        """Handle request with valid query returns 200 + correct content type."""
        from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge
        from agent_utilities.knowledge_graph.core.sparql_http import SPARQLEndpoint

        bridge = OWLBridge(graph=sample_graph, owl_backend=mock_owl_backend)
        endpoint = SPARQLEndpoint(bridge)

        body, content_type, status = endpoint.handle_request(
            query="SELECT ?s WHERE { ?s a au:Agent }"
        )
        assert status == 200
        assert content_type == "application/sparql-results+json"


# ═══════════════════════════════════════════════════════════════════════════════
# Ontology Publisher Tests (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOntologyPublisher:
    """Tests for ontology export and distribution."""

    def test_publisher_init(self):
        """OntologyPublisher should instantiate cleanly."""
        from agent_utilities.knowledge_graph.core.ontology_publisher import (
            OntologyPublisher,
        )

        publisher = OntologyPublisher()
        assert publisher is not None

    def test_export_ontology_to_file(self, tmp_path):
        """Export should write TTL file and return metadata."""
        import rdflib

        from agent_utilities.knowledge_graph.core.ontology_publisher import (
            OntologyPublisher,
        )

        g = rdflib.Graph()
        ns = rdflib.Namespace("http://test.dev/")
        g.add((ns.agent1, rdflib.RDF.type, ns.Agent))
        g.add((ns.agent1, ns.name, rdflib.Literal("TestAgent")))

        publisher = OntologyPublisher()
        output_path = tmp_path / "test_export.ttl"
        result = publisher.export_ontology(g, output_path)

        assert result["status"] == "success"
        assert result["triple_count"] == 2
        assert output_path.exists()

    def test_export_with_version_tag(self, tmp_path):
        """Export with version tag should append to filename."""
        import rdflib

        from agent_utilities.knowledge_graph.core.ontology_publisher import (
            OntologyPublisher,
        )

        g = rdflib.Graph()
        publisher = OntologyPublisher()
        result = publisher.export_ontology(
            g, tmp_path / "ontology.ttl", version_tag="v1.2.3"
        )

        assert result["status"] == "success"
        assert "v1.2.3" in result["path"]

    def test_push_to_stardog_without_pystardog(self):
        """Push to Stardog gracefully fails when pystardog is missing."""
        import rdflib

        from agent_utilities.knowledge_graph.core.ontology_publisher import (
            OntologyPublisher,
        )

        g = rdflib.Graph()
        publisher = OntologyPublisher()

        with patch.dict("sys.modules", {"stardog": None}):
            result = publisher.push_to_stardog(g)
            # Should return error (pystardog import will fail)
            assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Ontology Loader Tests (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOntologyLoader:
    """Tests for modular ontology import resolution."""

    def test_loader_init(self, tmp_path):
        """OntologyLoader should instantiate with custom cache dir."""
        from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader

        loader = OntologyLoader(cache_dir=tmp_path / "cache")
        assert loader is not None

    def test_uri_to_local_path_mapping(self, tmp_path):
        """URI mapping should resolve knuckles.team/kg/X to ontology_X.ttl."""
        from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader

        loader = OntologyLoader(cache_dir=tmp_path)

        path = loader._uri_to_local_path("http://knuckles.team/kg/enterprise", tmp_path)
        assert path is not None
        assert path.name == "ontology_enterprise.ttl"

    def test_uri_to_local_path_base(self, tmp_path):
        """Base URI should map to ontology.ttl."""
        from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader

        loader = OntologyLoader(cache_dir=tmp_path)

        path = loader._uri_to_local_path("http://knuckles.team/kg", tmp_path)
        assert path is not None
        assert path.name == "ontology.ttl"

    def test_cache_roundtrip(self, tmp_path):
        """Cache write and read should round-trip correctly."""
        from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader

        loader = OntologyLoader(cache_dir=tmp_path, cache_ttl_seconds=3600)

        test_content = "@prefix : <http://test.dev/> .\n:A a :B ."
        loader._write_cache("http://example.com/test.ttl", test_content)

        cached = loader._read_cache("http://example.com/test.ttl")
        assert cached == test_content

    def test_clear_cache(self, tmp_path):
        """Clear cache should remove all cached files."""
        from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader

        loader = OntologyLoader(cache_dir=tmp_path, cache_ttl_seconds=3600)

        # Write some cache files
        loader._write_cache("http://a.com/1.ttl", "test1")
        loader._write_cache("http://b.com/2.ttl", "test2")

        count = loader.clear_cache()
        assert count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Enterprise Ontology Module Tests (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnterpriseOntologyModules:
    """Tests for modular ontology organization."""

    def test_sdd_ontology_file_exists(self):
        """ontology_sdd.ttl should exist in knowledge_graph directory."""
        from pathlib import Path

        sdd_path = (
            Path(__file__).parent.parent.parent
            / "agent_utilities"
            / "knowledge_graph"
            / "ontology_sdd.ttl"
        )
        assert sdd_path.exists(), f"SDD ontology missing at {sdd_path}"

    def test_enterprise_ontology_file_exists(self):
        """ontology_enterprise.ttl should exist in knowledge_graph directory."""
        from pathlib import Path

        ent_path = (
            Path(__file__).parent.parent.parent
            / "agent_utilities"
            / "knowledge_graph"
            / "ontology_enterprise.ttl"
        )
        assert ent_path.exists(), f"Enterprise ontology missing at {ent_path}"

    def test_main_ontology_imports_modules(self):
        """Main ontology.ttl should declare owl:imports for enterprise and sdd."""
        from pathlib import Path

        main_path = (
            Path(__file__).parent.parent.parent
            / "agent_utilities"
            / "knowledge_graph"
            / "ontology.ttl"
        )
        content = main_path.read_text(encoding="utf-8")
        assert "owl:imports" in content
        assert "knuckles.team/kg/enterprise" in content
        assert "knuckles.team/kg/sdd" in content
