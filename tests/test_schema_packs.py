from __future__ import annotations

"""Tests for Schema Pack models and pre-built domain profiles.

CONCEPT:KG-2.2 — Schema Packs
"""


import pytest

from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from agent_utilities.models.schema_pack import (
    BacklinkBoostStrategy,
    SchemaPack,
    SchemaPackMode,
)

# --- SchemaPack Model Tests ---


class TestSchemaPackModel:
    """Test the base SchemaPack model."""

    def test_default_mode_is_additive(self):
        """Default mode should be ADDITIVE."""
        pack = SchemaPack(name="test")
        assert pack.mode == SchemaPackMode.ADDITIVE

    def test_default_backlink_strategy_is_global(self):
        """Default backlink boost strategy should be GLOBAL."""
        pack = SchemaPack(name="test")
        assert pack.backlink_boost_strategy == BacklinkBoostStrategy.GLOBAL

    def test_default_backlink_factor(self):
        """Default backlink boost factor should be 0.1."""
        pack = SchemaPack(name="test")
        assert pack.backlink_boost_factor == 0.1

    def test_core_node_types_are_protected(self):
        """CORE_NODE_TYPES should contain essential agent types."""
        assert RegistryNodeType.MEMORY in SchemaPack.CORE_NODE_TYPES
        assert RegistryNodeType.EPISODE in SchemaPack.CORE_NODE_TYPES
        assert RegistryNodeType.PERSON in SchemaPack.CORE_NODE_TYPES
        assert RegistryNodeType.CONCEPT in SchemaPack.CORE_NODE_TYPES
        assert RegistryNodeType.FACT in SchemaPack.CORE_NODE_TYPES
        assert RegistryNodeType.AGENT in SchemaPack.CORE_NODE_TYPES

    def test_core_edge_types_are_protected(self):
        """CORE_EDGE_TYPES should contain essential relationship types."""
        assert RegistryEdgeType.PROVIDES in SchemaPack.CORE_EDGE_TYPES
        assert RegistryEdgeType.DEPENDS_ON in SchemaPack.CORE_EDGE_TYPES
        assert RegistryEdgeType.MEMORY_OF in SchemaPack.CORE_EDGE_TYPES


class TestSchemaPackAdditive:
    """Test ADDITIVE mode behavior."""

    def test_additive_returns_all_node_types(self):
        """ADDITIVE mode should return ALL RegistryNodeType members."""
        pack = SchemaPack(
            name="test",
            mode=SchemaPackMode.ADDITIVE,
            node_types={RegistryNodeType.HYPOTHESIS},
        )
        active = pack.get_active_node_types()
        # Should contain every member of the enum
        for member in RegistryNodeType:
            assert member in active

    def test_additive_returns_all_edge_types(self):
        """ADDITIVE mode should return ALL RegistryEdgeType members."""
        pack = SchemaPack(
            name="test",
            mode=SchemaPackMode.ADDITIVE,
            edge_types={RegistryEdgeType.CITES_SOURCE},
        )
        active = pack.get_active_edge_types()
        for member in RegistryEdgeType:
            assert member in active

    def test_additive_is_node_type_active(self):
        """is_node_type_active should return True for any type in ADDITIVE mode."""
        pack = SchemaPack(name="test", mode=SchemaPackMode.ADDITIVE)
        assert pack.is_node_type_active(RegistryNodeType.FINANCIAL_INSTRUMENT)
        assert pack.is_node_type_active(RegistryNodeType.MEMORY)


class TestSchemaPackExclusive:
    """Test EXCLUSIVE mode behavior."""

    def test_exclusive_includes_core_types(self):
        """EXCLUSIVE mode should always include core types."""
        pack = SchemaPack(
            name="test",
            mode=SchemaPackMode.EXCLUSIVE,
            node_types={RegistryNodeType.HYPOTHESIS},
        )
        active = pack.get_active_node_types()
        for core_type in SchemaPack.CORE_NODE_TYPES:
            assert core_type in active

    def test_exclusive_includes_pack_types(self):
        """EXCLUSIVE mode should include the pack's declared types."""
        pack = SchemaPack(
            name="test",
            mode=SchemaPackMode.EXCLUSIVE,
            node_types={RegistryNodeType.DATASET, RegistryNodeType.DOCUMENT},
        )
        active = pack.get_active_node_types()
        assert RegistryNodeType.DATASET in active
        assert RegistryNodeType.DOCUMENT in active

    def test_exclusive_excludes_undeclared_types(self):
        """EXCLUSIVE mode should NOT include types not in core or pack."""
        pack = SchemaPack(
            name="test",
            mode=SchemaPackMode.EXCLUSIVE,
            node_types={RegistryNodeType.HYPOTHESIS},
        )
        active = pack.get_active_node_types()
        # FINANCIAL_INSTRUMENT is not in core or in this pack
        assert RegistryNodeType.FINANCIAL_INSTRUMENT not in active

    def test_exclusive_edge_filtering(self):
        """EXCLUSIVE mode should filter edges correctly."""
        pack = SchemaPack(
            name="test",
            mode=SchemaPackMode.EXCLUSIVE,
            edge_types={RegistryEdgeType.CITES_SOURCE},
        )
        active = pack.get_active_edge_types()
        assert RegistryEdgeType.CITES_SOURCE in active
        # Core edges should be present
        assert RegistryEdgeType.PROVIDES in active
        # Non-core, non-declared should be absent
        assert RegistryEdgeType.HAS_FINANCIAL_INSTRUMENT not in active

    def test_exclusive_is_edge_type_active(self):
        """is_edge_type_active should filter correctly in EXCLUSIVE mode."""
        pack = SchemaPack(
            name="test",
            mode=SchemaPackMode.EXCLUSIVE,
            edge_types=set(),
        )
        assert pack.is_edge_type_active(RegistryEdgeType.PROVIDES)  # core
        assert not pack.is_edge_type_active(RegistryEdgeType.EXECUTED_TRANSACTION)


class TestSchemaPackRetrievalBoosts:
    """Test retrieval boost configuration."""

    def test_get_boost_for_configured_edge(self):
        """Should return the configured boost value."""
        pack = SchemaPack(
            name="test",
            retrieval_boosts={"cites_source": 1.5, "supports_belief": 1.3},
        )
        assert pack.get_boost_for_edge("cites_source") == 1.5
        assert pack.get_boost_for_edge("supports_belief") == 1.3

    def test_get_boost_for_unconfigured_edge(self):
        """Should return 1.0 (neutral) for unconfigured edges."""
        pack = SchemaPack(name="test")
        assert pack.get_boost_for_edge("any_edge") == 1.0

    def test_backlink_factor_validation(self):
        """Factor should be between 0.0 and 1.0."""
        with pytest.raises(Exception):  # noqa: B017
            SchemaPack(name="test", backlink_boost_factor=1.5)
        with pytest.raises(Exception):  # noqa: B017
            SchemaPack(name="test", backlink_boost_factor=-0.1)


# --- Pre-Built Pack Tests ---


class TestPreBuiltPacks:
    """Test that pre-built packs instantiate with valid type references."""

    def test_core_pack(self):
        """Core pack should instantiate and be ADDITIVE."""
        from agent_utilities.models.schema_packs.core import CoreSchemaPack

        pack = CoreSchemaPack()
        assert pack.name == "core"
        assert pack.mode == SchemaPackMode.ADDITIVE

    def test_research_pack(self):
        """Research pack should activate research-specific types."""
        from agent_utilities.models.schema_packs.research import ResearchSchemaPack

        pack = ResearchSchemaPack()
        assert pack.name == "research-state"
        assert RegistryNodeType.HYPOTHESIS in pack.node_types
        assert RegistryNodeType.DATASET in pack.node_types
        assert RegistryEdgeType.CITES_SOURCE in pack.edge_types
        assert pack.backlink_boost_strategy == BacklinkBoostStrategy.CONTEXT_ONLY

    def test_biomedical_pack(self):
        """Biomedical pack should activate medical types."""
        from agent_utilities.models.schema_packs.biomedical import (
            BiomedicalSchemaPack,
        )

        pack = BiomedicalSchemaPack()
        assert pack.name == "biomedical"
        assert RegistryNodeType.MEDICAL_ENTITY in pack.node_types
        assert RegistryNodeType.PROCEDURE in pack.node_types

    def test_finance_pack(self):
        """Finance pack should activate financial types."""
        from agent_utilities.models.schema_packs.finance import FinanceSchemaPack

        pack = FinanceSchemaPack()
        assert pack.name == "finance"
        assert RegistryNodeType.FINANCIAL_INSTRUMENT in pack.node_types
        assert RegistryNodeType.ACCOUNT in pack.node_types
        assert RegistryEdgeType.HAS_FINANCIAL_INSTRUMENT in pack.edge_types

    def test_all_pack_node_types_are_valid(self):
        """All node types referenced by packs must be valid enum members."""
        from agent_utilities.models.schema_packs import (
            BiomedicalSchemaPack,
            CoreSchemaPack,
            FinanceSchemaPack,
            ResearchSchemaPack,
        )

        for pack_cls in [
            CoreSchemaPack,
            ResearchSchemaPack,
            BiomedicalSchemaPack,
            FinanceSchemaPack,
        ]:
            pack = pack_cls()
            for nt in pack.node_types:
                assert nt in RegistryNodeType, f"Invalid node type {nt} in {pack.name}"

    def test_all_pack_edge_types_are_valid(self):
        """All edge types referenced by packs must be valid enum members."""
        from agent_utilities.models.schema_packs import (
            BiomedicalSchemaPack,
            CoreSchemaPack,
            FinanceSchemaPack,
            ResearchSchemaPack,
        )

        for pack_cls in [
            CoreSchemaPack,
            ResearchSchemaPack,
            BiomedicalSchemaPack,
            FinanceSchemaPack,
        ]:
            pack = pack_cls()
            for et in pack.edge_types:
                assert et in RegistryEdgeType, f"Invalid edge type {et} in {pack.name}"


# --- Registry Tests ---


class TestSchemaPackRegistry:
    """Test the schema pack registry and factory."""

    def test_get_schema_pack_by_name(self):
        """get_schema_pack should return the correct pack."""
        from agent_utilities.models.schema_packs import get_schema_pack

        pack = get_schema_pack("core")
        assert pack.name == "core"

    def test_get_schema_pack_unknown_raises(self):
        """get_schema_pack should raise KeyError for unknown packs."""
        from agent_utilities.models.schema_packs import get_schema_pack

        with pytest.raises(KeyError, match="Unknown schema pack"):
            get_schema_pack("nonexistent-pack")

    def test_list_schema_packs(self):
        """list_schema_packs should return all registered packs."""
        from agent_utilities.models.schema_packs import list_schema_packs

        packs = list_schema_packs()
        names = {p["name"] for p in packs}
        assert "core" in names
        assert "research-state" in names
        assert "biomedical" in names
        assert "finance" in names

    def test_register_custom_pack(self):
        """register_schema_pack should allow custom pack registration."""
        from agent_utilities.models.schema_packs import (
            get_schema_pack,
            register_schema_pack,
        )

        class CustomPack(SchemaPack):
            name: str = "custom-test"
            description: str = "A custom test pack"

        register_schema_pack("custom-test", CustomPack)
        pack = get_schema_pack("custom-test")
        assert pack.name == "custom-test"


# --- SchemaPackNode Tests ---


class TestSchemaPackNode:
    """Test KG persistence model for schema packs."""

    def test_schema_pack_node_creation(self):
        """SchemaPackNode should instantiate with correct type."""
        from agent_utilities.models.knowledge_graph import SchemaPackNode

        node = SchemaPackNode(
            id="pack-001",
            name="research-state",
            pack_name="research-state",
            mode="additive",
            active_node_types=["hypothesis", "dataset"],
            active_edge_types=["cites_source"],
        )
        assert node.type == RegistryNodeType.SCHEMA_PACK
        assert node.pack_name == "research-state"
        assert node.mode == "additive"
        assert "hypothesis" in node.active_node_types
