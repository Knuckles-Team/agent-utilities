#!/usr/bin/python
from __future__ import annotations

"""Apache Fuseki SPARQL Backend.

CONCEPT:KG-2.7 — Vendor-Agnostic Graph Backend Abstraction

Remote SPARQL backend for Apache Fuseki via HTTP. Suitable for
production multi-agent deployments where a shared, persistent
RDF store is required.

Fuseki provides full SPARQL 1.1 Query + Update over HTTP, with
TDB2 persistence, SHACL validation, and optional full-text search.

Environment Variables:
    GRAPH_FUSEKI_URL: Fuseki server URL (default: http://localhost:3030).
    GRAPH_FUSEKI_DATASET: Dataset name (default: agent_kg).
    GRAPH_FUSEKI_USER: Optional HTTP Basic auth username.
    GRAPH_FUSEKI_PASSWORD: Optional HTTP Basic auth password.
"""

import asyncio
import logging
import math
import os
from typing import Any

from agent_utilities.knowledge_graph.core.event_backend import (
    TOPIC_MUTATIONS,
    EventBackend,
)

from .base import SparqlAdapter

logger = logging.getLogger(__name__)

_NS = "http://agent-utilities.dev/kg#"


class JenaFusekiBackend(SparqlAdapter):
    """Apache Fuseki SPARQL 1.1 backend via HTTP.

    All queries are sent to Fuseki's standard SPARQL endpoints:
        - ``/query`` for SELECT, ASK, CONSTRUCT
        - ``/update`` for INSERT, DELETE
        - ``/data`` for direct graph store protocol access

    Features:
        - Full SPARQL 1.1 Query + Update
        - TDB2-backed persistence (server-side)
        - Cypher-to-SPARQL transpilation for compatibility
        - HTTP connection pooling via httpx
        - Automatic dataset creation if missing
    """

    def __init__(
        self,
        jena_fuseki_url: str | None = None,
        dataset: str | None = None,
        username: str | None = None,
        password: str | None = None,
        event_backend: EventBackend | None = None,
    ) -> None:
        try:
            import httpx

            self._httpx = httpx
        except ImportError as e:
            raise ImportError(
                "JenaFusekiBackend requires httpx. "
                "Install with: pip install 'agent-utilities[jena_fuseki]'"
            ) from e

        self._base_url = (
            jena_fuseki_url
            or os.environ.get("GRAPH_FUSEKI_URL")
            or "http://localhost:3030"
        ).rstrip("/")
        self._dataset = dataset or os.environ.get("GRAPH_FUSEKI_DATASET") or "agent_kg"
        self._username = username or os.environ.get("GRAPH_FUSEKI_USER")
        self._password = password or os.environ.get("GRAPH_FUSEKI_PASSWORD")
        self._event_backend = event_backend

        # Build auth tuple if credentials provided
        self._auth = None
        if self._username and self._password:
            self._auth = (self._username, self._password)

        # Connection pool
        self._client = httpx.Client(
            base_url=self._base_url,
            auth=self._auth,
            timeout=30.0,
            headers={"Accept": "application/sparql-results+json"},
        )

        self._embeddings: dict[str, list[float]] = {}

        logger.info(
            "JenaFusekiBackend initialized: url=%s, dataset=%s",
            self._base_url,
            self._dataset,
        )

    @property
    def _query_url(self) -> str:
        return f"/{self._dataset}/query"

    @property
    def _update_url(self) -> str:
        return f"/{self._dataset}/update"

    @property
    def _data_url(self) -> str:
        return f"/{self._dataset}/data"

    # ------------------------------------------------------------------
    # SparqlAdapter ABC Implementation
    # ------------------------------------------------------------------

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SPARQL query/update against Fuseki.

        This is the RDF/SPARQL tier — SPARQL is its native query language. Cypher
        (labeled-property-graph) queries are NOT supported here: there is no
        Cypher→SPARQL transpiler (the previous ``CypherToSPARQL`` import was dead
        code that raised on every Cypher query). Use a SPARQL query, or an LPG
        backend for Cypher. The LPG↔OWL bridge answers SPARQL over any LPG store
        via ``OWLBridge.query_sparql`` — this backend is for a real triplestore.
        """
        stripped = query.strip().upper()
        if stripped.startswith(
            ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE", "PREFIX", "INSERT", "DELETE")
        ):
            return self.execute_sparql(query)

        logger.debug(
            "JenaFusekiBackend received a non-SPARQL (Cypher?) query; unsupported "
            "on the RDF/SPARQL tier: %s",
            query[:120],
        )
        return []

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute a query over a batch of parameters."""
        results: list[dict[str, Any]] = []
        for params in batch:
            results.extend(self.execute(query, params))
        return results

    def create_schema(self) -> None:
        """Ensure the Fuseki dataset exists.

        Creates the dataset via Fuseki's admin API if it doesn't exist.
        Requires admin permissions on the Fuseki server.
        """
        try:
            resp = self._client.get(f"/$/datasets/{self._dataset}")
            if resp.status_code == 200:
                logger.debug("Fuseki dataset '%s' already exists", self._dataset)
                return
        except Exception:
            pass

        # Try to create the dataset
        try:
            resp = self._client.post(
                "/$/datasets",
                data={
                    "dbName": self._dataset,
                    "dbType": "tdb2",
                },
            )
            if resp.status_code in (200, 201):
                logger.info("Created Fuseki dataset: %s", self._dataset)
            else:
                logger.warning(
                    "Failed to create Fuseki dataset %s: %s %s",
                    self._dataset,
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as e:
            logger.warning("Could not create Fuseki dataset: %s", e)

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store embedding vector (client-side; Fuseki has no native vector index)."""
        self._embeddings[node_id] = embedding

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Brute-force cosine similarity over locally cached embeddings."""
        if not self._embeddings:
            return []

        scored = []
        for node_id, emb in self._embeddings.items():
            sim = self._cosine_similarity(query_embedding, emb)
            scored.append((node_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for node_id, score in scored[:n_results]:
            results.append({"id": node_id, "score": score})
        return results

    def prune(self, criteria: dict[str, Any]) -> None:
        """Prune nodes matching criteria via SPARQL DELETE."""
        min_importance = criteria.get("min_importance", 0.0)
        if min_importance > 0:
            self.execute_sparql(
                f"""
                PREFIX au: <{_NS}>
                DELETE {{ ?s ?p ?o }}
                WHERE {{
                    ?s au:importance ?imp .
                    FILTER(xsd:float(?imp) < {min_importance})
                    ?s ?p ?o .
                }}
            """
            )

    def close(self) -> None:
        """Close the HTTP client."""
        try:
            self._client.close()
        except Exception:
            pass
        self._embeddings.clear()

    # ------------------------------------------------------------------
    # SPARQL Capability
    # ------------------------------------------------------------------

    def execute_sparql_query(
        self, query: str, timeout_ms: int = 30_000
    ) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT, ASK, or CONSTRUCT query."""
        try:
            resp = self._client.post(
                self._query_url,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout_ms / 1000,
            )

            if resp.status_code != 200:
                return [
                    {"error": f"Query failed: {resp.status_code} {resp.text[:200]}"}
                ]

            data = resp.json()

            # ASK result
            if "boolean" in data:
                return [{"result": data["boolean"]}]

            # SELECT result
            bindings = data.get("results", {}).get("bindings", [])
            results = []
            for binding in bindings:
                row = {}
                for var, val_obj in binding.items():
                    row[var] = val_obj.get("value", "")
                results.append(row)
            return results

        except Exception as e:
            logger.error("Fuseki SPARQL query failed: %s", e)
            return [{"error": str(e)}]

    def execute_sparql_update(self, update: str, timeout_ms: int = 30_000) -> None:
        """Execute a SPARQL INSERT or DELETE update."""
        try:
            resp = self._client.post(
                self._update_url,
                data={"update": update},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=timeout_ms / 1000,
            )
            if resp.status_code not in (200, 204):
                raise RuntimeError(
                    f"Update failed: {resp.status_code} {resp.text[:200]}"
                )

            # Emit event to backbone if available
            if self._event_backend:
                stripped = update.strip().upper()
                event_type = (
                    "TRIPLE_INSERT"
                    if stripped.startswith("INSERT")
                    else "TRIPLE_DELETE"
                )
                if stripped.startswith("SPARQL"):
                    event_type = "SPARQL_UPDATE"

                event_payload = {
                    "event_type": event_type,
                    "query": update,
                    "source": "jena_fuseki_backend",
                }

                # We use asyncio.create_task to fire-and-forget the publish,
                # assuming there's a running event loop (e.g., Tokio or Asyncio).
                try:
                    asyncio.create_task(
                        self._event_backend.publish(TOPIC_MUTATIONS, event_payload)
                    )
                except RuntimeError:
                    # No running event loop
                    pass

        except Exception as e:
            logger.error("Fuseki SPARQL update failed: %s", e)
            raise

    def upload_graph(self, ttl_content: str, graph_uri: str | None = None) -> None:
        """Upload a full Turtle (.ttl) graph representation."""
        params = {}
        if graph_uri:
            params["graph"] = graph_uri

        resp = self._client.post(
            self._data_url,
            content=ttl_content,
            headers={"Content-Type": "text/turtle"},
            params=params,
        )
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(
                f"Failed to upload graph: {resp.status_code} {resp.text[:200]}"
            )

    def download_graph(self, graph_uri: str | None = None) -> str:
        """Download the full graph as a Turtle (.ttl) string."""
        params = {}
        if graph_uri:
            params["graph"] = graph_uri

        resp = self._client.get(
            self._data_url, headers={"Accept": "text/turtle"}, params=params
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to download graph: {resp.status_code} {resp.text[:200]}"
            )
        return resp.text

    # ------------------------------------------------------------------
    # LPG Convenience Methods
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, properties: dict[str, Any] | None = None) -> None:
        """Add a node as RDF triples via SPARQL INSERT."""
        properties = properties or {}
        node_type = properties.get("type", "Concept")

        triples = [f"<{_NS}{node_id}> <{_NS}type> <{_NS}{node_type}> ."]
        for key, value in properties.items():
            if key == "type":
                continue
            escaped = str(value).replace('"', '\\"')
            triples.append(f'<{_NS}{node_id}> <{_NS}{key}> "{escaped}" .')

        self.execute_sparql(f"INSERT DATA {{ {' '.join(triples)} }}")

    def add_edge(
        self, source_id: str, target_id: str, properties: dict[str, Any] | None = None
    ) -> None:
        """Add an edge as RDF triples via SPARQL INSERT."""
        properties = properties or {}
        edge_type = properties.get("type", "relatedTo")

        triples = [f"<{_NS}{source_id}> <{_NS}{edge_type}> <{_NS}{target_id}> ."]
        self.execute_sparql(f"INSERT DATA {{ {' '.join(triples)} }}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)
