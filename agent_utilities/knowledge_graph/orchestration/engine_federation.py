#!/usr/bin/python
from __future__ import annotations

"""Federation Mixin for the Unified Intelligence Graph Engine.

This module provides support for registering external ontologies (e.g. via SPARQL
endpoints) and ingesting metadata stubs from external knowledge graphs (like EARs)
that lack semantic web capabilities.
"""


import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import requests

from ..backends import create_backend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FederationMixin:
    """Mixin for external ontology federation and external KG metadata ingestion.

    CONCEPT:KG-2.1 — External Graph Federation
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
            node_type="ExternalGraphReference",
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
        """Ingest a high-level metadata stub from an external KG (e.g. EAR).

        Creates an `ExternalEntity` node and links it to the specified `internal_node_id`
        via `mapped_to_external` to create a bridge between the internal structural graph
        and the external metadata graph.

        Returns the ID of the created external stub node.
        """
        stub_id = f"ExternalEntity_{platform}_{external_id}"

        self.add_node(  # type: ignore[attr-defined]
            node_id=stub_id,
            node_type="ExternalEntity",
            properties={
                "externalSystemId": external_id,
                "externalUri": external_uri,
                "platform": platform,
                "name": name or f"External Entity {external_id}",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        # Link internal node to this external reference
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

    def execute_federated_query(
        self, reference_id: str, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query against an external graph reference.

        Args:
            reference_id: The ID of the ExternalGraphReferenceNode in the local graph.
            query: The SPARQL or Cypher query string.
            parameters: Optional query parameters (mostly for Cypher/LPG).

        Returns:
            A list of dictionary records.
        """
        # 0. REST virtual sources are served by invoking their extractor (the
        # `query` string carries no SPARQL endpoint; an optional `node_type`
        # parameter filters to one canonical type).
        if reference_id in getattr(self, "_rest_sources", {}):
            node_type = (parameters or {}).get("node_type")
            return self.query_rest_source(reference_id, node_type=node_type)

        # 1. Retrieve the endpoint details from the local graph
        if not hasattr(self, "backend") or not self.backend:  # type: ignore[attr-defined]
            # Fallback to local memory graph if no persistent backend
            node_data = self.graph.nodes.get(reference_id)  # type: ignore[attr-defined]
            if not node_data:
                raise ValueError(
                    f"External graph reference {reference_id} not found in local graph."
                )
            endpoint_url = node_data.get("endpoint_url") or node_data.get("sourceUrl")
            graph_type = node_data.get("graph_type") or node_data.get("platform")
        else:
            res = self.backend.execute(  # type: ignore[attr-defined]
                "MATCH (n) WHERE n.id = $id RETURN n.endpoint_url as url, n.graph_type as type, n.sourceUrl as surl, n.platform as plat",
                {"id": reference_id},
            )
            if not res:
                raise ValueError(
                    f"External graph reference {reference_id} not found in persistent graph."
                )
            row = res[0]
            endpoint_url = row.get("url") or row.get("surl")
            graph_type = row.get("type") or row.get("plat")

        if not endpoint_url:
            raise ValueError(
                f"No endpoint URL configured for reference {reference_id}."
            )

        graph_type = str(graph_type).lower()

        # 2. Route to the appropriate executor
        if "sparql" in graph_type:
            return self.execute_federated_sparql(endpoint=endpoint_url, query=query)
        else:
            return self.execute_federated_lpg(
                endpoint=endpoint_url, query=query, parameters=parameters
            )

    # ── REST virtualization (query-time, extractor-backed) ───────────────────
    #
    # Camunda/ServiceNow/ERPNext speak REST/JSON, not SPARQL, so true Ontop-style
    # virtual-SPARQL is out of scope. Instead we virtualize by invoking the
    # *existing* self-registering extractor on demand (TTL-cached) and returning
    # its materialized records — no duplicate mapping code, no extra dependency.
    # Limitation: reasoning applies only over the fetched slice; for full
    # cross-source reasoning, materialize via the ingestion pipeline instead.

    def register_rest_source(
        self,
        reference_id: str,
        extractor_category: str,
        client: Any,
        *,
        ttl_seconds: float = 60.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register a REST-backed virtual source keyed to an existing extractor.

        ``extractor_category`` is an enrichment registry key (e.g. ``"camunda"``,
        ``"servicenow"``, ``"erpnext"``, ``"leanix"``); ``client`` is the
        duck-typed API client that extractor consumes. Fetches are cached for
        ``ttl_seconds`` to bound query-time latency.
        """
        if not hasattr(self, "_rest_sources"):
            self._rest_sources: dict[str, dict[str, Any]] = {}
        self._rest_sources[reference_id] = {
            "category": extractor_category,
            "client": client,
            "config": dict(config or {}),
            "ttl_seconds": float(ttl_seconds),
            "cache": None,  # tuple(monotonic_ts, batch)
        }
        # Make the virtual source discoverable like SPARQL references.
        self.add_node(  # type: ignore[attr-defined]
            node_id=reference_id,
            node_type="ExternalGraphReference",
            properties={
                "platform": "rest",
                "extractorCategory": extractor_category,
                "ttlSeconds": ttl_seconds,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        logger.info(
            "Registered REST virtual source %s (extractor=%s, ttl=%ss)",
            reference_id,
            extractor_category,
            ttl_seconds,
        )

    def _fetch_rest_batch(self, reference_id: str) -> Any:
        """Fetch (or return cached) ExtractionBatch for a REST virtual source."""
        from ..enrichment.registry import discover_extractors, get_source

        src = getattr(self, "_rest_sources", {}).get(reference_id)
        if src is None:
            raise ValueError(f"No REST virtual source registered as {reference_id}.")

        cache = src.get("cache")
        if cache is not None:
            ts, batch = cache
            if (time.monotonic() - ts) < src["ttl_seconds"]:
                return batch

        extractor = get_source(src["category"])
        if extractor is None:
            discover_extractors()  # lazy-load extractor modules, then retry
            extractor = get_source(src["category"])
        if extractor is None:
            raise ValueError(f"Unknown extractor category {src['category']!r}.")

        config = {"client": src["client"], **src["config"]}
        batch = extractor.extract(config)
        src["cache"] = (time.monotonic(), batch)
        return batch

    def query_rest_source(
        self, reference_id: str, node_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Return materialized records from a REST virtual source (TTL-cached).

        Each record is ``{"id", "type", **props}``. Pass ``node_type`` to filter
        to one canonical type (e.g. ``"Incident"``, ``"BusinessProcess"``).
        """
        batch = self._fetch_rest_batch(reference_id)
        records: list[dict[str, Any]] = []
        for node in batch.nodes:
            if node_type is not None and node.type != node_type:
                continue
            records.append({"id": node.id, "type": node.type, **dict(node.props)})
        return records

    def query_rest_union(
        self,
        reference_id: str,
        local_records: list[dict[str, Any]],
        node_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Union freshly-fetched REST records with locally materialized ones.

        De-duplicated by record ``id`` (local records take precedence).
        """
        merged: dict[str, dict[str, Any]] = {}
        for rec in self.query_rest_source(reference_id, node_type=node_type):
            rid = rec.get("id")
            if rid:
                merged[rid] = rec
        for rec in local_records:  # local wins on id collision
            rid = rec.get("id")
            if rid:
                merged[rid] = rec
        return list(merged.values())

    def execute_federated_sparql(
        self, endpoint: str, query: str
    ) -> list[dict[str, Any]]:
        """Execute a SPARQL query against an external HTTP endpoint using the `requests` library."""
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        logger.info("Executing federated SPARQL query against %s", endpoint)

        try:
            response = requests.post(
                endpoint, data={"query": query}, headers=headers, timeout=30
            )
            response.raise_for_status()

            data = response.json()
            # Flatten SPARQL XML/JSON bindings to a simple dict list
            results = []
            if "results" in data and "bindings" in data["results"]:
                for binding in data["results"]["bindings"]:
                    row = {k: v["value"] for k, v in binding.items()}
                    results.append(row)
            return results
        except Exception as e:
            logger.error("Federated SPARQL query failed: %s", e)
            raise RuntimeError(f"Federated SPARQL execution failed: {e}") from e

    def execute_federated_lpg(
        self, endpoint: str, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher/LPG query against an external endpoint.

        Uses the `create_backend` factory to abstract away the specific
        LPG driver (Neo4j, FalkorDB, PostgreSQL, etc.).
        """
        logger.info("Executing federated LPG query against %s", endpoint)

        try:
            # create_backend parses the URI schema dynamically
            ext_backend = create_backend(uri=endpoint)
            if not ext_backend:
                raise RuntimeError(
                    f"Failed to create backend for endpoint {endpoint}. "
                    "Ensure appropriate driver (e.g. agent-utilities[neo4j]) is installed."
                )

            # Use the abstracted GraphBackend execution method
            results = ext_backend.execute(query, parameters or {})
            return results
        except Exception as e:
            logger.error("Federated LPG query failed: %s", e)
            raise RuntimeError(f"Federated LPG execution failed: {e}") from e
