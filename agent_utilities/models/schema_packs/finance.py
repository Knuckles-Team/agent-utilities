"""Finance Schema Pack — financial services domain profile.

Optimized for financial agents managing instruments, transactions,
accounts, regulations, and compliance. Aligned with FIBO ontology.
"""

from __future__ import annotations

from ..knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..schema_pack import BacklinkBoostStrategy, SchemaPack, SchemaPackMode


class FinanceSchemaPack(SchemaPack):
    """Finance domain pack: instruments, transactions, accounts, regulations."""

    name: str = "finance"
    description: str = (
        "Financial services profile. Activates financial instrument, "
        "transaction, account, and regulation types. Aligned with FIBO "
        "ontology for securities, compliance, and regulatory tracking."
    )
    mode: SchemaPackMode = SchemaPackMode.ADDITIVE
    node_types: set[RegistryNodeType] = {
        RegistryNodeType.FINANCIAL_INSTRUMENT,
        RegistryNodeType.FINANCIAL_TRANSACTION,
        RegistryNodeType.ACCOUNT,
        RegistryNodeType.REGULATION,
        RegistryNodeType.ORGANIZATION,
        RegistryNodeType.PERSON,
        RegistryNodeType.DOCUMENT,
        RegistryNodeType.DECISION,
        RegistryNodeType.EVIDENCE,
        RegistryNodeType.SYSTEM,
    }
    edge_types: set[RegistryEdgeType] = {
        RegistryEdgeType.HAS_FINANCIAL_INSTRUMENT,
        RegistryEdgeType.EXECUTED_TRANSACTION,
        RegistryEdgeType.BELONGS_TO_ORGANIZATION,
        RegistryEdgeType.DECIDED_BY,
        RegistryEdgeType.RESULTED_IN,
        RegistryEdgeType.OWNS_SYSTEM,
        RegistryEdgeType.DEPENDS_ON_SYSTEM,
        RegistryEdgeType.WAS_ATTRIBUTED_TO,
        RegistryEdgeType.CITES_SOURCE,
    }
    retrieval_boosts: dict[str, float] = {
        "has_financial_instrument": 1.4,
        "executed_transaction": 1.3,
        "belongs_to_organization": 1.2,
        "decided_by": 1.3,
    }
    backlink_boost_strategy: BacklinkBoostStrategy = BacklinkBoostStrategy.GLOBAL
    backlink_boost_factor: float = 0.12
