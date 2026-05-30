import logging
from typing import Any

import requests

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType

logger = logging.getLogger(__name__)

# Basic ontology mapper from external RDF types to our internal RegistryNodeType
DEFAULT_MAPPING = {
    "http://schema.org/Person": RegistryNodeType.EMPLOYEE,
    "http://schema.org/Organization": RegistryNodeType.BUSINESS_DIVISION,
    "http://www.wikidata.org/entity/Q5": RegistryNodeType.EMPLOYEE,  # Wikidata Human
    "http://www.wikidata.org/entity/Q43229": RegistryNodeType.BUSINESS_DIVISION,  # Wikidata Organization
}


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
        self.engine = engine or GraphComputeEngine()
        self.mapping = mapping_config or DEFAULT_MAPPING

    def query_endpoint(self, endpoint: str, query: str) -> list[dict[str, Any]]:
        """Executes a standard SPARQL query against an external HTTP endpoint."""
        try:
            resp = requests.get(
                endpoint,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", {}).get("bindings", [])
        except Exception as e:
            logger.error("Failed to query SPARQL endpoint %s: %s", endpoint, e)
            return []

    def ingest_entities(self, limit: int = 100) -> int:
        """
        Federated ingestion of mapped entities.
        Queries each endpoint for instances of known mapped classes, translates them
        to our native RegistryNode schema, and adds provenance tracking edges.
        """
        total_ingested = 0
        for endpoint in self.endpoints:
            # Ensure the endpoint exists as a PROVENANCE_AGENT in the local graph
            self.engine.add_node(
                id=endpoint,
                type=RegistryNodeType.PROVENANCE_AGENT,
                name=f"SPARQL Endpoint: {endpoint}",
            )

            for rdf_class, target_node_type in self.mapping.items():
                # Fetch subjects of the target class
                query = f"""
                SELECT ?subject ?label WHERE {{
                    ?subject a <{rdf_class}> .
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
                        id=subj_uri,
                        type=target_node_type,
                        name=label,
                        uri=subj_uri,
                        ingested_from=endpoint,
                    )

                    # Add provenance edge (Operational constraint mapping)
                    self.engine.add_edge(
                        source_id=subj_uri,
                        target_id=endpoint,
                        type=RegistryEdgeType.WAS_GENERATED_BY,
                        method="sparql_federation",
                    )
                    total_ingested += 1

        return total_ingested
