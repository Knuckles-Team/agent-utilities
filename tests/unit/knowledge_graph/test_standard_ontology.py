#!/usr/bin/python
"""Unit tests for standard ontology integration.

Covers the industry-standard ontology alignment (BFO, PROV-O, Schema.org,
Dublin Core, SKOS, OWL-Time, BIBO, FIBO) added to the knowledge graph.
Tests focus on:

* Ontology file parsing and namespace resolution (rdflib).
* BFO upper ontology alignment (continuant/occurrent split).
* New Pydantic node model construction and serialization.
* New edge type enum values and edge construction.
* Schema coverage for new table and relationship definitions.
* Inference engine standard rules (NX fallback).
* OWL bridge promotable type coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.models.knowledge_graph import (
    AccountNode,
    CreativeWorkNode,
    DatasetNode,
    DocumentNode,
    FinancialInstrumentNode,
    FinancialTransactionNode,
    MedicalEntityNode,
    ProcedureNode,
    RegistryEdge,
    RegistryEdgeType,
    RegistryNodeType,
    RegulationNode,
    SoftwareProjectNode,
)
from agent_utilities.models.schema_definition import SCHEMA

# Path to ontology.ttl
ONTOLOGY_PATH = Path(__file__).parent.parent.parent.parent / "agent_utilities" / "knowledge_graph" / "ontology.ttl"

ISO_TS = "2026-01-01T00:00:00Z"

# ---------------------------------------------------------------------------
# Standard Ontology Node Types — enum existence
# ---------------------------------------------------------------------------

STANDARD_NODE_ENUMS: tuple[RegistryNodeType, ...] = (
    RegistryNodeType.DOCUMENT,
    RegistryNodeType.CREATIVE_WORK,
    RegistryNodeType.DATASET,
    RegistryNodeType.SOFTWARE_PROJECT,
    RegistryNodeType.MEDICAL_ENTITY,
    RegistryNodeType.PROCEDURE,
    RegistryNodeType.REGULATION,
    RegistryNodeType.FINANCIAL_INSTRUMENT,
    RegistryNodeType.FINANCIAL_TRANSACTION,
    RegistryNodeType.ACCOUNT,
)

STANDARD_EDGE_ENUMS: tuple[RegistryEdgeType, ...] = (
    RegistryEdgeType.WAS_GENERATED_BY,
    RegistryEdgeType.WAS_DERIVED_FROM,
    RegistryEdgeType.WAS_ATTRIBUTED_TO,
    RegistryEdgeType.HAS_TEMPORAL_EXTENT,
    RegistryEdgeType.BROADER,
    RegistryEdgeType.NARROWER,
    RegistryEdgeType.RELATED_CONCEPT,
    RegistryEdgeType.EXACT_MATCH,
    RegistryEdgeType.CLOSE_MATCH,
    RegistryEdgeType.BROAD_MATCH,
    RegistryEdgeType.CREATOR,
    RegistryEdgeType.CITES_SOURCE,
    RegistryEdgeType.HAS_FINANCIAL_INSTRUMENT,
    RegistryEdgeType.EXECUTED_TRANSACTION,
)


# ---------------------------------------------------------------------------
# Ontology file parsing
# ---------------------------------------------------------------------------


@pytest.mark.concept("standard-ontology")
class TestOntologyFile:
    """Tests for the updated ontology.ttl file."""

    def test_ontology_file_exists(self) -> None:
        assert ONTOLOGY_PATH.exists(), f"Ontology file not found: {ONTOLOGY_PATH}"

    def test_ontology_loads_with_rdflib(self) -> None:
        """Verify the updated ontology.ttl parses without errors."""
        rdflib = pytest.importorskip("rdflib")
        g = rdflib.Graph()
        g.parse(str(ONTOLOGY_PATH), format="turtle")
        assert len(g) > 100, f"Expected >100 triples, got {len(g)}"

    def test_ontology_has_standard_namespaces(self) -> None:
        """Verify that all standard namespace prefixes are present."""
        content = ONTOLOGY_PATH.read_text()
        expected_prefixes = [
            "bfo:",
            "schema:",
            "dc:",
            "foaf:",
            "prov:",
            "skos:",
            "time:",
            "bibo:",
            "fibo-org:",
            "fibo-fi:",
        ]
        for prefix in expected_prefixes:
            assert prefix in content, f"Missing namespace prefix: {prefix}"

    def test_bfo_alignment_continuant_occurrent(self) -> None:
        """Verify BFO superclass assertions exist in the ontology."""
        content = ONTOLOGY_PATH.read_text()
        # Continuants
        assert "rdfs:subClassOf bfo:0000004" in content  # IndependentContinuant
        assert "rdfs:subClassOf bfo:0000031" in content  # GenericallyDependentContinuant
        assert "rdfs:subClassOf bfo:0000020" in content  # SpecificallyDependentContinuant
        # Occurrents
        assert "rdfs:subClassOf bfo:0000015" in content  # Process
        assert "rdfs:subClassOf bfo:0000008" in content  # TemporalRegion

    def test_prov_o_provenance_properties(self) -> None:
        """Verify PROV-O property alignments exist."""
        content = ONTOLOGY_PATH.read_text()
        assert ":wasGeneratedBy" in content
        assert ":wasDerivedFrom" in content
        assert ":wasAttributedTo" in content
        assert "prov:wasGeneratedBy" in content

    def test_skos_taxonomy_properties(self) -> None:
        """Verify full SKOS taxonomy property support."""
        content = ONTOLOGY_PATH.read_text()
        assert ":broader" in content
        assert ":narrower" in content
        assert ":related" in content
        assert ":exactMatch" in content
        assert ":closeMatch" in content
        assert ":broadMatch" in content
        assert ":prefLabel" in content
        assert ":altLabel" in content
        assert ":notation" in content

    def test_dublin_core_metadata_properties(self) -> None:
        """Verify Dublin Core datatype properties."""
        content = ONTOLOGY_PATH.read_text()
        assert ":title" in content
        assert ":subject" in content
        assert ":identifier" in content
        assert ":dateCreated" in content
        assert ":language" in content
        assert ":format" in content
        assert "dc:title" in content

    def test_finance_domain_classes(self) -> None:
        """Verify FIBO-aligned finance classes."""
        content = ONTOLOGY_PATH.read_text()
        assert ":FinancialInstrument" in content
        assert ":FinancialTransaction" in content
        assert ":Account" in content
        assert "fibo-fi:" in content


# ---------------------------------------------------------------------------
# New Pydantic node models — happy path construction
# ---------------------------------------------------------------------------


@pytest.mark.concept("standard-ontology")
class TestStandardNodeModels:
    """Test construction and serialization of new standard ontology node models."""

    def test_document_node_happy_path(self) -> None:
        n = DocumentNode(
            id="doc:paper-42",
            name="Research Paper",
            title="On the Integration of Ontologies",
            creator="Dr. Smith",
            date="2026-01-15",
            subject="Ontology Engineering",
            identifier="10.1234/onto.2026",
            format="application/pdf",
            language="en",
        )
        assert n.type is RegistryNodeType.DOCUMENT
        assert n.title == "On the Integration of Ontologies"
        assert n.identifier == "10.1234/onto.2026"

    def test_creative_work_node_happy_path(self) -> None:
        n = CreativeWorkNode(
            id="cw:manual-1",
            name="Agent Operations Manual",
            title="Agent Operations Manual v2",
            creator="Engineering Team",
            genre="technical",
        )
        assert n.type is RegistryNodeType.CREATIVE_WORK
        assert n.title == "Agent Operations Manual v2"

    def test_dataset_node_happy_path(self) -> None:
        n = DatasetNode(
            id="ds:covid-data",
            name="COVID-19 Dataset",
            distribution_url="https://data.example.com/covid19",
            temporal_coverage="2020-2023",
            record_count=1_000_000,
        )
        assert n.type is RegistryNodeType.DATASET
        assert n.record_count == 1_000_000

    def test_software_project_node_happy_path(self) -> None:
        n = SoftwareProjectNode(
            id="proj:agent-utils",
            name="agent-utilities",
            repo_url="https://github.com/knuckles/agent-utilities",
            language="Python",
            tech_stack=["python", "pydantic", "networkx"],
        )
        assert n.type is RegistryNodeType.SOFTWARE_PROJECT
        assert "pydantic" in n.tech_stack

    def test_medical_entity_node_happy_path(self) -> None:
        n = MedicalEntityNode(
            id="med:hypertension",
            name="Hypertension",
            entity_type="condition",
            icd_code="I10",
        )
        assert n.type is RegistryNodeType.MEDICAL_ENTITY
        assert n.icd_code == "I10"

    def test_procedure_node_happy_path(self) -> None:
        n = ProcedureNode(
            id="proc:oil-change",
            name="Engine Oil Change",
            steps=["Drain old oil", "Replace filter", "Add new oil"],
            required_tools=["wrench", "oil pan", "funnel"],
            safety_notes=["Ensure engine is cool"],
            estimated_duration="30m",
            category="automotive",
        )
        assert n.type is RegistryNodeType.PROCEDURE
        assert len(n.steps) == 3

    def test_regulation_node_happy_path(self) -> None:
        n = RegulationNode(
            id="reg:gdpr",
            name="GDPR",
            jurisdiction="EU",
            authority="European Commission",
            regulation_type="data_privacy",
            compliance_status="compliant",
        )
        assert n.type is RegistryNodeType.REGULATION
        assert n.regulation_type == "data_privacy"

    def test_financial_instrument_node_happy_path(self) -> None:
        n = FinancialInstrumentNode(
            id="fi:aapl",
            name="Apple Inc.",
            instrument_type="stock",
            ticker="AAPL",
            issuer="Apple Inc.",
            currency="USD",
            isin="US0378331005",
        )
        assert n.type is RegistryNodeType.FINANCIAL_INSTRUMENT
        assert n.ticker == "AAPL"

    def test_financial_transaction_node_happy_path(self) -> None:
        n = FinancialTransactionNode(
            id="ft:buy-001",
            name="Buy AAPL",
            transaction_type="buy",
            amount=15000.00,
            currency="USD",
            counterparty="NYSE",
            executed_at=ISO_TS,
        )
        assert n.type is RegistryNodeType.FINANCIAL_TRANSACTION
        assert n.amount == 15000.00

    def test_account_node_happy_path(self) -> None:
        n = AccountNode(
            id="acct:brokerage-1",
            name="Main Brokerage",
            account_type="brokerage",
            institution="Fidelity",
            currency="USD",
        )
        assert n.type is RegistryNodeType.ACCOUNT
        assert n.account_type == "brokerage"


# ---------------------------------------------------------------------------
# JSON round-trip for new models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def standard_sample_nodes() -> dict[type, object]:
    return {
        DocumentNode: DocumentNode(
            id="doc:a", name="A", title="T"
        ),
        CreativeWorkNode: CreativeWorkNode(
            id="cw:a", name="A", title="T"
        ),
        DatasetNode: DatasetNode(id="ds:a", name="A"),
        SoftwareProjectNode: SoftwareProjectNode(
            id="proj:a", name="A"
        ),
        MedicalEntityNode: MedicalEntityNode(
            id="med:a", name="A"
        ),
        ProcedureNode: ProcedureNode(id="proc:a", name="A"),
        RegulationNode: RegulationNode(id="reg:a", name="A"),
        FinancialInstrumentNode: FinancialInstrumentNode(
            id="fi:a", name="A"
        ),
        FinancialTransactionNode: FinancialTransactionNode(
            id="ft:a", name="A"
        ),
        AccountNode: AccountNode(id="acct:a", name="A"),
    }


@pytest.mark.parametrize(
    "cls",
    [
        DocumentNode,
        CreativeWorkNode,
        DatasetNode,
        SoftwareProjectNode,
        MedicalEntityNode,
        ProcedureNode,
        RegulationNode,
        FinancialInstrumentNode,
        FinancialTransactionNode,
        AccountNode,
    ],
)
def test_standard_node_json_round_trip(
    cls: type, standard_sample_nodes: dict[type, object]
) -> None:
    """model_dump_json → model_validate_json is identity for standard nodes."""
    original = standard_sample_nodes[cls]
    raw = original.model_dump_json()  # type: ignore[attr-defined]
    parsed = cls.model_validate_json(raw)  # type: ignore[attr-defined]
    assert parsed.type is original.type  # type: ignore[attr-defined]
    assert parsed.model_dump_json() == raw


# ---------------------------------------------------------------------------
# Edge construction for new standard edge types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge_type", STANDARD_EDGE_ENUMS)
def test_standard_edge_construction(edge_type: RegistryEdgeType) -> None:
    """RegistryEdge can be constructed for every new standard edge type."""
    edge = RegistryEdge(
        source="a:1",
        target="b:1",
        type=edge_type,
        weight=0.8,
    )
    assert edge.type is edge_type
    assert edge.type.value == edge_type.value
    assert edge.type.value.islower()


# ---------------------------------------------------------------------------
# Enum / Schema coverage
# ---------------------------------------------------------------------------


def test_all_standard_node_enum_members_present() -> None:
    """All standard ontology RegistryNodeType members resolve."""
    names = {m.name for m in RegistryNodeType}
    for expected in (
        "DOCUMENT",
        "CREATIVE_WORK",
        "DATASET",
        "SOFTWARE_PROJECT",
        "MEDICAL_ENTITY",
        "PROCEDURE",
        "REGULATION",
        "FINANCIAL_INSTRUMENT",
        "FINANCIAL_TRANSACTION",
        "ACCOUNT",
    ):
        assert expected in names, f"Missing node enum: {expected}"


def test_all_standard_edge_enum_members_present() -> None:
    """All standard ontology RegistryEdgeType members resolve."""
    names = {m.name for m in RegistryEdgeType}
    for expected in (
        "WAS_GENERATED_BY",
        "WAS_DERIVED_FROM",
        "WAS_ATTRIBUTED_TO",
        "HAS_TEMPORAL_EXTENT",
        "BROADER",
        "NARROWER",
        "RELATED_CONCEPT",
        "EXACT_MATCH",
        "CLOSE_MATCH",
        "BROAD_MATCH",
        "CREATOR",
        "CITES_SOURCE",
        "HAS_FINANCIAL_INSTRUMENT",
        "EXECUTED_TRANSACTION",
    ):
        assert expected in names, f"Missing edge enum: {expected}"


def test_every_standard_node_has_table_definition() -> None:
    """Every standard ontology node has a matching SCHEMA TableDefinition."""
    table_names = {t.name for t in SCHEMA.nodes}
    for expected in (
        "Document",
        "CreativeWork",
        "Dataset",
        "SoftwareProject",
        "MedicalEntity",
        "Procedure",
        "Regulation",
        "FinancialInstrument",
        "FinancialTransaction",
        "Account",
    ):
        assert expected in table_names, (
            f"TableDefinition missing for node label {expected!r}"
        )


def test_every_standard_edge_has_rel_definition() -> None:
    """Every standard ontology edge has a matching SCHEMA RelDefinition."""
    rel_types = {e.type for e in SCHEMA.edges}
    for expected in (
        "WAS_GENERATED_BY",
        "WAS_DERIVED_FROM",
        "WAS_ATTRIBUTED_TO",
        "HAS_TEMPORAL_EXTENT",
        "BROADER",
        "NARROWER",
        "RELATED_CONCEPT",
        "EXACT_MATCH",
        "CLOSE_MATCH",
        "BROAD_MATCH",
        "CREATOR",
        "CITES_SOURCE",
        "HAS_FINANCIAL_INSTRUMENT",
        "EXECUTED_TRANSACTION",
    ):
        assert expected in rel_types, (
            f"RelDefinition missing for edge type {expected!r}"
        )


def test_standard_table_definitions_include_base_columns() -> None:
    """Standard ontology tables include RegistryNode base columns."""
    base_cols = {
        "id",
        "type",
        "name",
        "description",
        "importance_score",
        "timestamp",
        "metadata",
        "is_permanent",
    }
    standard_names = {
        "Document",
        "CreativeWork",
        "Dataset",
        "SoftwareProject",
        "MedicalEntity",
        "Procedure",
        "Regulation",
        "FinancialInstrument",
        "FinancialTransaction",
        "Account",
    }
    for tbl in SCHEMA.nodes:
        if tbl.name in standard_names:
            missing = base_cols - set(tbl.columns.keys())
            assert not missing, (
                f"{tbl.name} missing base cols: {sorted(missing)}"
            )


# ---------------------------------------------------------------------------
# OWL Bridge promotable type coverage
# ---------------------------------------------------------------------------


def test_owl_bridge_promotes_standard_node_types() -> None:
    """New standard node types are in PROMOTABLE_NODE_TYPES."""
    from agent_utilities.knowledge_graph.owl_bridge import PROMOTABLE_NODE_TYPES

    for t in (
        "document",
        "creative_work",
        "dataset",
        "software_project",
        "medical_entity",
        "procedure",
        "regulation",
        "financial_instrument",
        "financial_transaction",
        "account",
    ):
        assert t in PROMOTABLE_NODE_TYPES, f"Missing promotable node type: {t}"


def test_owl_bridge_promotes_standard_edge_types() -> None:
    """New standard edge types are in PROMOTABLE_EDGE_TYPES."""
    from agent_utilities.knowledge_graph.owl_bridge import PROMOTABLE_EDGE_TYPES

    for t in (
        "was_generated_by",
        "was_derived_from",
        "was_attributed_to",
        "has_temporal_extent",
        "broader",
        "narrower",
        "related_concept",
        "exact_match",
        "close_match",
        "broad_match",
        "creator",
        "cites_source",
        "has_financial_instrument",
        "executed_transaction",
    ):
        assert t in PROMOTABLE_EDGE_TYPES, f"Missing promotable edge type: {t}"


# ---------------------------------------------------------------------------
# Inference Engine standard rules (NX fallback)
# ---------------------------------------------------------------------------


def test_standard_inference_skos_broader_transitivity() -> None:
    """SKOS broader transitivity rule produces expected NX inference."""
    import networkx as nx

    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.inference_engine import InferenceEngine

    g = nx.MultiDiGraph()
    g.add_node("concept:a", type="concept", name="A")
    g.add_node("concept:b", type="concept", name="B")
    g.add_node("concept:c", type="concept", name="C")
    g.add_edge("concept:a", "concept:b", type="broader")
    g.add_edge("concept:b", "concept:c", type="broader")

    engine = IntelligenceGraphEngine(g)
    inf = InferenceEngine(engine)
    result = inf.run_inference()

    assert result >= 1, "Expected at least 1 inference from SKOS broader transitivity"
    assert g.has_edge("concept:a", "concept:c"), (
        "Expected transitive broader edge a->c"
    )


def test_standard_inference_prov_derivation_transitivity() -> None:
    """PROV-O was_derived_from transitivity rule works."""
    import networkx as nx

    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.inference_engine import InferenceEngine

    g = nx.MultiDiGraph()
    g.add_node("doc:a", type="document", name="A")
    g.add_node("doc:b", type="document", name="B")
    g.add_node("doc:c", type="document", name="C")
    g.add_edge("doc:a", "doc:b", type="was_derived_from")
    g.add_edge("doc:b", "doc:c", type="was_derived_from")

    engine = IntelligenceGraphEngine(g)
    inf = InferenceEngine(engine)
    result = inf.run_inference()

    assert result >= 1
    assert g.has_edge("doc:a", "doc:c")


def test_standard_inference_temporal_phase_containment() -> None:
    """Temporal phase containment rule works."""
    import networkx as nx

    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.inference_engine import InferenceEngine

    g = nx.MultiDiGraph()
    g.add_node("event:1", type="event", name="E1")
    g.add_node("phase:q2", type="phase", name="Q2")
    g.add_node("phase:2026", type="phase", name="2026")
    g.add_edge("event:1", "phase:q2", type="occurred_during")
    g.add_edge("phase:q2", "phase:2026", type="part_of")

    engine = IntelligenceGraphEngine(g)
    inf = InferenceEngine(engine)
    result = inf.run_inference()

    assert result >= 1
    assert g.has_edge("event:1", "phase:2026")
