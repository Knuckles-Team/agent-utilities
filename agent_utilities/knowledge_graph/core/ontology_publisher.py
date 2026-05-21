#!/usr/bin/python
"""Ontology Publisher — Export & Push to External Triplestores.

CONCEPT:KG-2.6 — Ontology Distribution

Exports the materialized RDF ontology and pushes it to external
triplestores (Stardog, Apache Jena Fuseki) for enterprise-wide
consumption via SPARQL federation.

Supports:
- Local TTL/RDF-XML export
- Stardog push via pystardog
- Apache Jena Fuseki push via REST API
- Versioned publishing with timestamps
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OntologyPublisher:
    """Export and distribute ontologies to enterprise triplestores.

    CONCEPT:KG-2.6 — Enterprise Ontology Distribution

    This class enables agent-utilities to serve as both the authoritative
    ontology source and a consumer — pushing evolved ontologies back to
    centralized infrastructure (Stardog, Fuseki) for enterprise-wide
    consumption.

    Example::

        publisher = OntologyPublisher()
        # Export locally
        publisher.export_ontology(rdf_graph, "/tmp/ontology.ttl")

        # Push to Stardog
        publisher.push_to_stardog(rdf_graph, endpoint="http://stardog:5820")

        # Push to Fuseki
        publisher.push_to_fuseki(rdf_graph, endpoint="http://fuseki:3030")
    """

    def export_ontology(
        self,
        rdf_graph: Any,
        output_path: str | Path,
        fmt: str = "turtle",
        version_tag: str | None = None,
    ) -> dict[str, Any]:
        """Serialize RDF graph to a local file.

        Args:
            rdf_graph: An rdflib.Graph to serialize.
            output_path: Filesystem path for output.
            fmt: Serialization format (turtle, xml, n3, ntriples, json-ld).
            version_tag: Optional version tag appended to filename.

        Returns:
            Dict with export metadata (path, triple_count, timestamp).
        """
        path = Path(output_path)

        if version_tag:
            stem = path.stem
            suffix = path.suffix
            path = path.parent / f"{stem}_{version_tag}{suffix}"

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = rdf_graph.serialize(format=fmt)
            if isinstance(data, bytes):
                path.write_bytes(data)
            else:
                path.write_text(data, encoding="utf-8")

            triple_count = len(rdf_graph)
            logger.info(
                "Exported %d triples to %s (format: %s)", triple_count, path, fmt
            )

            return {
                "status": "success",
                "path": str(path),
                "triple_count": triple_count,
                "format": fmt,
                "timestamp": datetime.now(UTC).isoformat(),
                "version_tag": version_tag,
            }
        except Exception as e:
            logger.error("Ontology export failed: %s", e)
            return {"status": "error", "error": str(e)}

    def push_to_stardog(
        self,
        rdf_graph: Any,
        endpoint: str | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        named_graph: str | None = None,
    ) -> dict[str, Any]:
        """Push ontology to a Stardog triplestore.

        Args:
            rdf_graph: An rdflib.Graph to push.
            endpoint: Stardog server URL (default: env STARDOG_ENDPOINT).
            database: Database name (default: env STARDOG_DATABASE).
            username: Auth username (default: env STARDOG_USER).
            password: Auth password (default: env STARDOG_PASSWORD).
            named_graph: Optional named graph URI for the upload.

        Returns:
            Dict with push status and metadata.
        """
        import os

        try:
            import stardog
        except ImportError:
            return {
                "status": "error",
                "error": "pystardog not installed. Install with: pip install pystardog",
            }

        endpoint = endpoint or os.environ.get(
            "STARDOG_ENDPOINT", "http://localhost:5820"
        )
        database = database or os.environ.get("STARDOG_DATABASE", "agent_kg")
        username = username or os.environ.get("STARDOG_USER", "admin")
        password = password or os.environ.get("STARDOG_PASSWORD", "admin")

        conn_details = {
            "endpoint": endpoint,
            "username": username,
            "password": password,
        }

        try:
            # Serialize to turtle for upload
            ttl_data = rdf_graph.serialize(format="turtle")
            if isinstance(ttl_data, str):
                ttl_data = ttl_data.encode("utf-8")

            conn = stardog.Connection(database, **conn_details)
            try:
                conn.begin()
                content = stardog.content.Raw(ttl_data, content_type="text/turtle")
                if named_graph:
                    conn.add(content, graph_uri=named_graph)
                else:
                    conn.add(content)
                conn.commit()

                triple_count = len(rdf_graph)
                logger.info(
                    "Pushed %d triples to Stardog %s/%s",
                    triple_count,
                    endpoint,
                    database,
                )

                return {
                    "status": "success",
                    "endpoint": endpoint,
                    "database": database,
                    "triple_count": triple_count,
                    "named_graph": named_graph,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

        except Exception as e:
            logger.error("Stardog push failed: %s", e)
            return {"status": "error", "error": str(e)}

    def push_to_fuseki(
        self,
        rdf_graph: Any,
        endpoint: str | None = None,
        dataset: str = "agent_kg",
        named_graph: str | None = None,
    ) -> dict[str, Any]:
        """Push ontology to Apache Jena Fuseki via REST API.

        Args:
            rdf_graph: An rdflib.Graph to push.
            endpoint: Fuseki server URL (default: http://localhost:3030).
            dataset: Dataset name.
            named_graph: Optional named graph URI.

        Returns:
            Dict with push status and metadata.
        """
        import os

        try:
            import requests
        except ImportError:
            return {
                "status": "error",
                "error": "requests not installed.",
            }

        endpoint = endpoint or os.environ.get(
            "FUSEKI_ENDPOINT", "http://localhost:3030"
        )

        # Fuseki Graph Store Protocol endpoint
        url = f"{endpoint}/{dataset}/data"
        params = {}
        if named_graph:
            params["graph"] = named_graph
        else:
            params["default"] = ""

        try:
            ttl_data = rdf_graph.serialize(format="turtle")
            if isinstance(ttl_data, str):
                ttl_data = ttl_data.encode("utf-8")

            response = requests.put(
                url,
                data=ttl_data,
                params=params,
                headers={"Content-Type": "text/turtle"},
                timeout=30,
            )
            response.raise_for_status()

            triple_count = len(rdf_graph)
            logger.info(
                "Pushed %d triples to Fuseki %s/%s",
                triple_count,
                endpoint,
                dataset,
            )

            return {
                "status": "success",
                "endpoint": endpoint,
                "dataset": dataset,
                "triple_count": triple_count,
                "named_graph": named_graph,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.error("Fuseki push failed: %s", e)
            return {"status": "error", "error": str(e)}
