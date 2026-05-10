from __future__ import annotations

"""Core Schema Pack — the default full-ontology profile.

Activates all RegistryNodeType and RegistryEdgeType members.
This is the pack that loads when no explicit pack is configured,
preserving full backward compatibility.
"""


from ..knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..schema_pack import BacklinkBoostStrategy, SchemaPack, SchemaPackMode


class CoreSchemaPack(SchemaPack):
    """Default pack: all types active, global backlink boost enabled."""

    name: str = "core"
    description: str = (
        "Full ontology — all 90+ node types and 80+ edge types active. "
        "Default profile for general-purpose agent deployments."
    )
    mode: SchemaPackMode = SchemaPackMode.ADDITIVE
    node_types: set[RegistryNodeType] = set()  # ADDITIVE + empty = all types
    edge_types: set[RegistryEdgeType] = set()
    backlink_boost_strategy: BacklinkBoostStrategy = BacklinkBoostStrategy.GLOBAL
    backlink_boost_factor: float = 0.1
