#!/usr/bin/python
"""Stardog OWL Backend.

Full implementation using pystardog for remote Stardog server reasoning.
Stardog provides built-in OWL reasoning — queries with ``reasoning=True``
automatically apply the ontology's inference rules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .base import OWLBackend

logger = logging.getLogger(__name__)

# Reuse the same mappings from the owlready2 backend
from .owlready2_backend import _EDGE_TYPE_TO_OWL_PROP, _NODE_TYPE_TO_OWL_CLASS

_NAMESPACE = "http://knuckles.team/kg#"


class StardogBackend(OWLBackend):
    """Stardog OWL backend via pystardog.

    Environment Variables:
        STARDOG_ENDPOINT: Stardog server URL (default: http://localhost:5820).
        STARDOG_DATABASE: Database name (default: agent_kg).
        STARDOG_USER: Username (default: admin).
        STARDOG_PASSWORD: Password (default: admin).
    """

    def __init__(
        self,
        endpoint: str | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        try:
            import stardog  # noqa: F401

            self._stardog = stardog
        except ImportError as e:
            raise ImportError(
                "Stardog backend requires pystardog package. "
                "Install with: pip install pystardog"
            ) from e

        self._endpoint = endpoint or os.environ.get(
            "STARDOG_ENDPOINT", "http://localhost:5820"
        )
        self._database = database or os.environ.get("STARDOG_DATABASE", "agent_kg")
        self._username = username or os.environ.get("STARDOG_USER", "admin")
        self._password = password or os.environ.get("STARDOG_PASSWORD", "admin")

        self._conn_details = {
            "endpoint": self._endpoint,
            "username": self._username,
            "password": self._password,
        }

        self._inferences: list[dict[str, Any]] = []
        self._ontology_loaded = False

        logger.info(
            "Initialized Stardog backend: endpoint=%s, database=%s",
            self._endpoint,
            self._database,
        )

    def _ensure_database(self) -> None:
        """Create the Stardog database if it doesn't exist."""
        with self._stardog.Admin(**self._conn_details) as admin:
            existing = [db.name for db in admin.databases()]
            if self._database not in existing:
                admin.new_database(
                    self._database,
                    {
                        "search.enabled": True,
                        "reasoning.type": "SL",  # Schema + RDFS reasoning
                    },
                )
                logger.info("Created Stardog database: %s", self._database)

    def _connection(self):
        """Create a new Stardog connection (caller must manage lifecycle)."""
        return self._stardog.Connection(self._database, **self._conn_details)

    def load_ontology(self, ontology_path: str) -> None:
        """Upload ontology file to Stardog database."""
        path = Path(ontology_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

        self._ensure_database()

        conn = self._connection()
        try:
            conn.begin()
            content = self._stardog.content.File(str(path))
            conn.add(content)
            conn.commit()
            self._ontology_loaded = True
            logger.info("Loaded ontology into Stardog from %s", ontology_path)
        except Exception as e:
            conn.rollback()
            logger.error("Failed to load ontology into Stardog: %s", e)
            raise
        finally:
            conn.close()

    def promote(self, stable_nodes: list[dict[str, Any]]) -> int:
        """Insert LPG nodes as RDF individuals via SPARQL UPDATE."""
        if not self._ontology_loaded:
            logger.warning("No ontology loaded; skipping promotion")
            return 0

        triples = []
        for node in stable_nodes:
            node_type = node.get("type", "")
            owl_class = _NODE_TYPE_TO_OWL_CLASS.get(node_type)
            if not owl_class:
                continue

            safe_id = node.get("id", "").replace(":", "_").replace("/", "_")
            if not safe_id:
                continue

            triples.append(f"<{_NAMESPACE}{safe_id}> a <{_NAMESPACE}{owl_class}> .")

            # Add datatype properties
            if "importance_score" in node:
                triples.append(
                    f"<{_NAMESPACE}{safe_id}> <{_NAMESPACE}importance> "
                    f'"{node["importance_score"]}"^^<http://www.w3.org/2001/XMLSchema#float> .'
                )
            if "confidence" in node:
                triples.append(
                    f"<{_NAMESPACE}{safe_id}> <{_NAMESPACE}confidence> "
                    f'"{node["confidence"]}"^^<http://www.w3.org/2001/XMLSchema#float> .'
                )

        if not triples:
            return 0

        # Batch insert via SPARQL UPDATE
        insert_query = f"INSERT DATA {{ {' '.join(triples)} }}"

        conn = self._connection()
        try:
            conn.begin()
            conn.update(insert_query)
            conn.commit()
            count = len([t for t in triples if " a " in t])
            logger.info("Promoted %d nodes to Stardog", count)
            return count
        except Exception as e:
            conn.rollback()
            logger.error("Stardog promotion failed: %s", e)
            return 0
        finally:
            conn.close()

    def promote_edges(self, edges: list[dict[str, Any]]) -> int:
        """Insert LPG edges as RDF property assertions."""
        if not self._ontology_loaded:
            return 0

        triples = []
        for edge in edges:
            prop_name = _EDGE_TYPE_TO_OWL_PROP.get(edge.get("type", ""))
            if not prop_name:
                continue

            src = edge.get("source", "").replace(":", "_").replace("/", "_")
            tgt = edge.get("target", "").replace(":", "_").replace("/", "_")
            if not src or not tgt:
                continue

            triples.append(
                f"<{_NAMESPACE}{src}> <{_NAMESPACE}{prop_name}> <{_NAMESPACE}{tgt}> ."
            )

        if not triples:
            return 0

        insert_query = f"INSERT DATA {{ {' '.join(triples)} }}"
        conn = self._connection()
        try:
            conn.begin()
            conn.update(insert_query)
            conn.commit()
            logger.info("Promoted %d edges to Stardog", len(triples))
            return len(triples)
        except Exception as e:
            conn.rollback()
            logger.error("Stardog edge promotion failed: %s", e)
            return 0
        finally:
            conn.close()

    def reason(self) -> list[dict[str, Any]]:
        """Query Stardog with reasoning enabled and diff against base facts."""
        if not self._ontology_loaded:
            return []

        conn = self._connection()
        try:
            # Query without reasoning
            base_results = set()
            base_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
            for row in (
                conn.select(base_query, reasoning=False)
                .get("results", {})
                .get("bindings", [])
            ):
                s = row.get("s", {}).get("value", "")
                p = row.get("p", {}).get("value", "")
                o = row.get("o", {}).get("value", "")
                base_results.add((s, p, o))

            # Query with reasoning
            inferred_results = set()
            for row in (
                conn.select(base_query, reasoning=True)
                .get("results", {})
                .get("bindings", [])
            ):
                s = row.get("s", {}).get("value", "")
                p = row.get("p", {}).get("value", "")
                o = row.get("o", {}).get("value", "")
                inferred_results.add((s, p, o))

            # Diff
            new_facts = inferred_results - base_results
            self._inferences = [
                {
                    "subject": s.split("#")[-1] if "#" in s else s.split("/")[-1],
                    "predicate": p.split("#")[-1] if "#" in p else p.split("/")[-1],
                    "object": o.split("#")[-1] if "#" in o else o.split("/")[-1],
                    "inference_type": "stardog_reasoning",
                }
                for s, p, o in new_facts
                # Filter out internal OWL/RDF triples
                if _NAMESPACE in s or not s.startswith("http://www.w3.org")
            ]

            logger.info(
                "Stardog reasoning found %d new inferences", len(self._inferences)
            )
            return self._inferences
        except Exception as e:
            logger.error("Stardog reasoning failed: %s", e)
            return []
        finally:
            conn.close()

    def get_inferences(self) -> list[dict[str, Any]]:
        """Return cached inferences from last reasoning run."""
        return self._inferences

    def export_rdf(self, output_path: str, fmt: str = "turtle") -> None:
        """Export database contents to RDF file."""
        conn = self._connection()
        try:
            fmt_map = {
                "turtle": "text/turtle",
                "ttl": "text/turtle",
                "xml": "application/rdf+xml",
                "ntriples": "application/n-triples",
            }
            content_type = fmt_map.get(fmt.lower(), "text/turtle")
            data = conn.export(content_type=content_type)
            Path(output_path).write_bytes(
                data if isinstance(data, bytes) else data.encode("utf-8")
            )
            logger.info("Exported Stardog RDF to %s", output_path)
        except Exception as e:
            logger.error("Stardog RDF export failed: %s", e)
        finally:
            conn.close()

    def clear(self) -> None:
        """Remove all ABox individuals (preserve TBox)."""
        conn = self._connection()
        try:
            # Delete individuals (instances of our classes), not the ontology itself
            delete_query = f"""
            DELETE {{ ?s ?p ?o }}
            WHERE {{
                ?s a ?type .
                ?type a <http://www.w3.org/2002/07/owl#Class> .
                FILTER(STRSTARTS(STR(?type), "{_NAMESPACE}"))
                ?s ?p ?o .
            }}
            """
            conn.begin()
            conn.update(delete_query)
            conn.commit()
            self._inferences = []
            logger.info("Cleared Stardog ABox individuals")
        except Exception as e:
            conn.rollback()
            logger.error("Stardog clear failed: %s", e)
        finally:
            conn.close()

    def close(self) -> None:
        """Release resources (no persistent connection to close)."""
        self._inferences = []

    def get_stats(self) -> dict[str, int]:
        """Query Stardog for ontology statistics."""
        conn = self._connection()
        try:
            individuals = conn.select(
                f"SELECT (COUNT(DISTINCT ?s) AS ?c) WHERE {{ ?s a ?t . FILTER(STRSTARTS(STR(?t), '{_NAMESPACE}')) }}",
                reasoning=False,
            )
            ind_count = int(
                individuals.get("results", {})
                .get("bindings", [{}])[0]
                .get("c", {})
                .get("value", "0")
            )

            return {
                "individuals": ind_count,
                "classes": len(_NODE_TYPE_TO_OWL_CLASS),
                "properties": len(_EDGE_TYPE_TO_OWL_PROP),
            }
        except Exception as e:
            logger.error("Stardog stats query failed: %s", e)
            return {"individuals": 0, "classes": 0, "properties": 0}
        finally:
            conn.close()
