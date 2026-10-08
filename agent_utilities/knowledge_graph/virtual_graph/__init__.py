"""Virtual graphs: materialize source metadata, read source data live.

See :mod:`.contracts` for the connection, metadata and mapping contracts,
:mod:`.ontology` for the ontology facts source selection reads, and
:mod:`.federation` for the cross-source report flow.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.virtual_graph.adapters import OperationAdapter
from agent_utilities.knowledge_graph.virtual_graph.contracts import (
    DiscoveredEntity,
    DiscoveredRelationship,
    MaterializationPolicy,
    MetadataContract,
    SourceConnection,
    VirtualMapping,
    metadata_triples,
)
from agent_utilities.knowledge_graph.virtual_graph.federation import (
    CrossSourceReport,
    VirtualCatalog,
    cross_source_report,
    select_sources,
)
from agent_utilities.knowledge_graph.virtual_graph.ontology import (
    TripleOntology,
    tbox_from_sparql,
)

__all__ = [
    "CrossSourceReport",
    "DiscoveredEntity",
    "DiscoveredRelationship",
    "MaterializationPolicy",
    "MetadataContract",
    "OperationAdapter",
    "SourceConnection",
    "TripleOntology",
    "VirtualCatalog",
    "VirtualMapping",
    "cross_source_report",
    "metadata_triples",
    "select_sources",
    "tbox_from_sparql",
]
