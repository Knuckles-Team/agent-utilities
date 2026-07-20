import hashlib
import logging
from typing import Any
from urllib.parse import urlsplit

from agent_utilities.core.config import config
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from agent_utilities.protocols.source_connectors.http_safety import safe_get_json

logger = logging.getLogger(__name__)

# Basic ontology mapper from external RDF types to our internal RegistryNodeType
DEFAULT_MAPPING = {
    "http://schema.org/Person": RegistryNodeType.EMPLOYEE,
    "http://schema.org/Organization": RegistryNodeType.BUSINESS_DIVISION,
    "http://www.wikidata.org/entity/Q5": RegistryNodeType.EMPLOYEE,  # Wikidata Human
    "http://www.wikidata.org/entity/Q43229": RegistryNodeType.BUSINESS_DIVISION,  # Wikidata Organization
}


def _sparql_iri(value: object) -> str:
    rendered = str(value or "")
    parsed = urlsplit(rendered)
    if (
        not 1 <= len(rendered) <= 2_048
        or parsed.scheme not in {"http", "https", "urn"}
        or any(character.isspace() or character in '<>"{}|^`\\' for character in rendered)
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("SPARQL IRI is invalid")
    return f"<{rendered}>"


class FederatedSparqlIngestor:
    """
    Federated SPARQL Ingestion Client.
    Acts as a bridge to pull Semantic Web triples from external authoritative
    endpoints and map them directly into our local operational Epistemic Graph.
    """

    def __init__(
        self,
        endpoints: list[str],
        engine: GraphComputeEngine | None = None,
        mapping_config: dict[str, str] | None = None,
    ):
        self.endpoints = endpoints
        self.engine = engine or GraphComputeEngine.get_or_create()
        self.mapping = mapping_config or DEFAULT_MAPPING

    def query_endpoint(self, endpoint: str, query: str) -> list[dict[str, Any]]:
        """Executes a standard SPARQL query against an external HTTP endpoint."""
        try:
            data = safe_get_json(
                endpoint,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30,
                max_bytes=10 * 1024 * 1024,
                max_redirects=0,
                allowed_private_hosts=config.source_http_allowed_private_hosts,
                tls_service="sparql-source",
            )
            results = data.get("results", {}) if isinstance(data, dict) else {}
            bindings = results.get("bindings", []) if isinstance(results, dict) else []
            return bindings if isinstance(bindings, list) else []
        except Exception as exc:
            logger.error("SPARQL source query failed: error_type=%s", type(exc).__name__)
            return []

    def ingest_entities(self, limit: int = 100) -> int:
        """
        Federated ingestion of mapped entities.
        Queries each endpoint for instances of known mapped classes, translates them
        to our native RegistryNode schema, and adds provenance tracking edges.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be an integer between 1 and 10000")
        total_ingested = 0
        for endpoint in self.endpoints:
            endpoint_ref = (
                "source:sparql:"
                + hashlib.sha256(str(endpoint).encode("utf-8")).hexdigest()[:32]
            )
            # Ensure the endpoint exists as a PROVENANCE_AGENT in the local graph
            self.engine.add_node(
                node_id=endpoint_ref,
                type=RegistryNodeType.PROVENANCE_AGENT,
                name="SPARQL source",
            )

            for rdf_class, target_node_type in self.mapping.items():
                # Fetch subjects of the target class
                query = f"""
                SELECT ?subject ?label WHERE {{
                    ?subject a {_sparql_iri(rdf_class)} .
                    OPTIONAL {{ ?subject <http://www.w3.org/2000/01/rdf-schema#label> ?label }}
                    FILTER(LANG(?label) = "en" || !BOUND(?label))
                }} LIMIT {limit}
                """
                bindings = self.query_endpoint(endpoint, query)

                for b in bindings:
                    subj_uri = b.get("subject", {}).get("value")
                    if not subj_uri:
                        continue

                    label = b.get("label", {}).get("value", subj_uri.split("/")[-1])

                    # Store in our native graph using operational source of truth semantics
                    self.engine.add_node(
                        node_id=subj_uri,
                        type=target_node_type,
                        name=label,
                        uri=subj_uri,
                        ingested_from=endpoint_ref,
                    )

                    # Add provenance edge (Operational constraint mapping)
                    self.engine.add_edge(
                        source_id=subj_uri,
                        target_id=endpoint_ref,
                        type=RegistryEdgeType.WAS_GENERATED_BY,
                        method="sparql_federation",
                    )
                    total_ingested += 1

        return total_ingested
