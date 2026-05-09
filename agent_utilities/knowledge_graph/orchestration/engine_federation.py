#!/usr/bin/python
"""Federation Mixin for the Unified Intelligence Graph Engine.

This module provides support for registering external ontologies (e.g. via SPARQL
endpoints) and ingesting metadata stubs from external knowledge graphs (like LeanIX)
that lack semantic web capabilities.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FederationMixin:
    """Mixin for external ontology federation and external KG metadata ingestion.

    CONCEPT:ORCH-1.2 — External Integration & SDLC Entities
    """

    def register_external_ontology(self, uri: str, endpoint: str | None = None) -> None:
        """Register an external ontology with the graph engine.

        If `endpoint` is provided, the engine or OWL bridge can use it for federated
        SPARQL queries (e.g., using SPARQL 1.1 SERVICE clauses). If only `uri` is provided,
        it registers the namespace prefix so the reasoner is aware of it.
        """
        if not hasattr(self, "_external_ontologies"):
            self._external_ontologies = {}

        self._external_ontologies[uri] = endpoint
        logger.info(
            "Registered external ontology URI: %s %s",
            uri,
            f"(endpoint: {endpoint})" if endpoint else "",
        )

        # We also ingest a reference node into the graph so agents can query
        # what external ontologies are currently mapped.
        node_id = f"OntologyReference_{hash(uri)}"
        self.add_node(  # type: ignore[attr-defined]
            node_id=node_id,
            node_type="external_graph_reference",
            properties={
                "externalUri": uri,
                "sourceUrl": endpoint,
                "platform": "sparql",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def get_registered_ontologies(self) -> dict[str, str | None]:
        """Get the mapping of registered external ontologies and their endpoints."""
        if not hasattr(self, "_external_ontologies"):
            return {}
        return self._external_ontologies

    def ingest_external_entity_stub(
        self,
        internal_node_id: str,
        external_id: str,
        external_uri: str,
        platform: str,
        name: str | None = None,
    ) -> str:
        """Ingest a high-level metadata stub from an external KG (e.g. LeanIX).

        Creates an `external_entity` node and links it to the specified `internal_node_id`
        via `mapped_to_external` to create a bridge between the internal structural graph
        and the external metadata graph.

        Returns the ID of the created external stub node.
        """
        stub_id = f"ExternalEntity_{platform}_{external_id}"

        self.add_node(  # type: ignore[attr-defined]
            node_id=stub_id,
            node_type="external_entity",
            properties={
                "externalSystemId": external_id,
                "externalUri": external_uri,
                "platform": platform,
                "name": name or f"External Entity {external_id}",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        # Link internal node to this external stub
        self.link_nodes(  # type: ignore[attr-defined]
            source_id=internal_node_id, target_id=stub_id, rel_type="mapped_to_external"
        )

        logger.debug(
            "Bridged internal node %s to external %s entity %s",
            internal_node_id,
            platform,
            external_id,
        )
        return stub_id
