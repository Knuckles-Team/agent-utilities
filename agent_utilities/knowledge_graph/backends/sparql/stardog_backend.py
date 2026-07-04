#!/usr/bin/python
from __future__ import annotations

"""Stardog SPARQL data backend.

CONCEPT:AU-KG.query.vendor-agnostic-traversal — Vendor-Agnostic Graph Backend Abstraction.

A first-class :class:`SparqlAdapter` for Stardog, so the KG can **push, pull, and
query instance data** (not just the ontology) against a Stardog triplestore — as a
standalone backend, a fan-out **mirror** (CONCEPT:AU-KG.backend.mirror-health-repair), or an ad-hoc connection
(CONCEPT:AU-KG.backend.multi-connection-registry). This is distinct from
``backends/owl/stardog_backend.py::StardogBackend``, which is the OWL **reasoning**
backend (TBox + inference); the two compose — reason over the schema, store/serve
the data here.

Why a Cypher→SPARQL write translation lives here
------------------------------------------------
The execution plane has **no SPARQL routing**: every write
(``IntelligenceGraphEngine._upsert_node`` / ``._upsert_edge`` /
``ingest_external_batch``, the cross-backend ``copy_graph`` reconcile, and the
fan-out mirror replay) reaches a backend by calling ``backend.execute(<cypher>)`` /
``execute_batch(<cypher>, batch)``. A SPARQL backend that ignores Cypher therefore
receives **nothing** — so to mirror or backfill real KG data into Stardog we
translate the engine's **finite, owned** MERGE shapes into SPARQL INSERT/DELETE.
This is the same shape-coupling the fan-out backend already relies on
(``_EDGE_MERGE_RE`` in ``fanout_backend.py``): we parse only the structural bits
(label / rel-type) and take the data from ``params``/``batch`` — never from the
Cypher text. Unrecognised queries (ad-hoc reads) return ``[]`` — reads are served by
the authority store in a tiered/fan-out deployment, with SPARQL the native query
language here.

Instance data is partitioned into ``urn:source:<system>`` named graphs by
:mod:`.source_partition`, so each external source (LeanIX, ServiceNow, …) is
pushable / queryable / clearable as a slice.

Environment Variables (shared with the OWL backend):
    STARDOG_ENDPOINT: Stardog server URL (default: http://localhost:5820).
    STARDOG_DATABASE: Database name (default: agent_kg).
    STARDOG_USER: Username (default: admin).
    STARDOG_PASSWORD: Password (default: admin).
"""

import json
import logging
import math
import re
from typing import Any
from urllib.parse import quote

from agent_utilities.core.config import setting

from .base import SparqlAdapter
from .source_partition import graph_uri_for

logger = logging.getLogger(__name__)

# SPARQL-tier namespace — identical to JenaFusekiBackend so a mixed Fuseki/Stardog
# deployment yields byte-identical IRIs for the same node/edge.
_NS = "http://agent-utilities.dev/kg#"

# IRI-illegal characters get percent-encoded; sub-delims and ``:`` / ``/`` (valid in
# an IRI) are preserved so readable ids like ``app:123`` stay readable.
_IRI_SAFE = ":/#?@!$&'()*+,;=-._~"

# --- engine Cypher shapes we translate (structural match only) ---
_NODE_MERGE_RE = re.compile(r"MERGE\s*\(\s*n:(?P<label>\w+)\s*\{\s*id:\s*\$id\s*\}")
_EDGE_MERGE_RE = re.compile(r"MERGE\s*\(s\)-\[r:(?P<rel>\w+)\]->\(t\)")
_LABEL_LOOKUP_RE = re.compile(
    r"MATCH\s*\(n\)\s*WHERE\s*n\.id\s*=\s*\$id\s*RETURN\s*labels?\(n\)"
)
_EMBED_SET_RE = re.compile(r"SET\s*n\.embedding\s*=\s*\$emb")
_UNWIND_NODE_RE = re.compile(r"MERGE\s*\(n:(?P<label>\w+)\s*\{\s*id:\s*row\.id\s*\}")
_UNWIND_EDGE_RE = re.compile(r"MERGE\s*\(s\)-\[r:(?P<rel>\w+)")


class StardogSparqlBackend(SparqlAdapter):
    """Stardog SPARQL 1.1 data backend via pystardog.

    Full SPARQL query/update, Turtle graph upload/download, full-fidelity
    ``add_node``/``add_edge``, and Cypher→SPARQL write translation so the engine's
    write path (and the fan-out mirror) lands real data here.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        try:
            import stardog

            self._stardog = stardog
        except ImportError as e:
            raise ImportError(
                "StardogSparqlBackend requires the pystardog package. "
                "Install with: pip install pystardog"
            ) from e

        self._endpoint = endpoint or setting(
            "STARDOG_ENDPOINT", "http://localhost:5820"
        )
        self._database = database or setting("STARDOG_DATABASE", "agent_kg")
        self._username = username or setting("STARDOG_USER", "admin")
        self._password = password or setting("STARDOG_PASSWORD", "admin")
        self._conn_details = {
            "endpoint": self._endpoint,
            "username": self._username,
            "password": self._password,
        }
        # Client-side embedding cache (Stardog has no native vector index here).
        self._embeddings: dict[str, list[float]] = {}
        logger.info(
            "StardogSparqlBackend initialized: endpoint=%s, database=%s",
            self._endpoint,
            self._database,
        )

    # ------------------------------------------------------------------
    # Connection / lifecycle
    # ------------------------------------------------------------------
    def _connection(self):
        return self._stardog.Connection(self._database, **self._conn_details)

    def create_schema(self) -> None:
        """Ensure the Stardog database exists (idempotent)."""
        try:
            with self._stardog.Admin(**self._conn_details) as admin:
                existing = [db.name for db in admin.databases()]
                if self._database not in existing:
                    admin.new_database(
                        self._database, {"search.enabled": True, "reasoning.type": "SL"}
                    )
                    logger.info("Created Stardog database: %s", self._database)
        except Exception as e:  # noqa: BLE001 — best-effort; server may forbid admin
            logger.debug("Stardog ensure-database skipped: %s", e)

    def close(self) -> None:
        self._embeddings.clear()

    # ------------------------------------------------------------------
    # SparqlAdapter ABC — native SPARQL
    # ------------------------------------------------------------------
    def execute_sparql_query(
        self, query: str, timeout_ms: int = 30_000
    ) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT/ASK/CONSTRUCT and return flat binding rows."""
        conn = self._connection()
        try:
            data = conn.select(query)
            if isinstance(data, dict) and "boolean" in data:
                return [{"result": data["boolean"]}]
            rows: list[dict[str, Any]] = []
            for binding in data.get("results", {}).get("bindings", []):
                rows.append({var: v.get("value", "") for var, v in binding.items()})
            return rows
        except Exception as e:  # noqa: BLE001
            logger.error("Stardog SPARQL query failed: %s", e)
            return [{"error": str(e)}]
        finally:
            conn.close()

    def execute_sparql_update(self, update: str, timeout_ms: int = 30_000) -> None:
        """Execute a SPARQL INSERT/DELETE update inside a transaction."""
        conn = self._connection()
        try:
            conn.begin()
            conn.update(update)
            conn.commit()
        except Exception as e:  # noqa: BLE001
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            logger.error("Stardog SPARQL update failed: %s", e)
            raise
        finally:
            conn.close()

    def upload_graph(self, ttl_content: str, graph_uri: str | None = None) -> None:
        """Upload a Turtle graph into the named graph (or the default graph)."""
        conn = self._connection()
        try:
            conn.begin()
            content = self._stardog.content.Raw(
                ttl_content.encode("utf-8"), content_type="text/turtle"
            )
            if graph_uri:
                conn.add(content, graph_uri=graph_uri)
            else:
                conn.add(content)
            conn.commit()
        except Exception as e:  # noqa: BLE001
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            logger.error("Stardog graph upload failed: %s", e)
            raise
        finally:
            conn.close()

    def download_graph(self, graph_uri: str | None = None) -> str:
        """Export the named graph (or the whole DB) as Turtle."""
        conn = self._connection()
        try:
            kwargs: dict[str, Any] = {"content_type": "text/turtle"}
            if graph_uri:
                kwargs["graph_uri"] = graph_uri
            data = conn.export(**kwargs)
            return data.decode("utf-8") if isinstance(data, bytes) else str(data)
        except Exception as e:  # noqa: BLE001
            logger.error("Stardog graph download failed: %s", e)
            return ""
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # GraphBackend ABC — write via Cypher translation, read via SPARQL
    # ------------------------------------------------------------------
    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Route a query: native SPARQL passes through; the engine's Cypher MERGE
        shapes translate to SPARQL; everything else is a no-op read (``[]``)."""
        params = params or {}
        stripped = query.strip()
        upper = stripped.upper()

        # Native SPARQL passthrough.
        if upper.startswith(
            ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE", "PREFIX")
        ) or upper.startswith(("INSERT", "DELETE", "LOAD", "CLEAR", "DROP", "CREATE")):
            return self.execute_sparql(query, timeout_ms=30_000)

        # Engine label lookup → answer from the store.
        if _LABEL_LOOKUP_RE.search(stripped):
            return self._label_of(params.get("id"))

        # Engine node upsert.
        m = _NODE_MERGE_RE.search(stripped)
        if m:
            self._upsert_node_triples(m.group("label"), params)
            return []

        # Engine edge upsert (re-embed handled here too).
        e = _EDGE_MERGE_RE.search(stripped)
        if e:
            edge_props = {k: v for k, v in params.items() if k not in ("sid", "tid")}
            self._upsert_edge_triple(
                e.group("rel"),
                params.get("sid"),
                params.get("tid"),
                props=edge_props,
            )
            return []
        if _EMBED_SET_RE.search(stripped):
            nid, emb = params.get("id"), params.get("emb")
            if nid is not None and isinstance(emb, list):
                self.add_embedding(str(nid), emb)
            return []

        logger.debug(
            "StardogSparqlBackend: unsupported non-SPARQL query: %s", query[:120]
        )
        return []

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate the engine's UNWIND-batch node/edge MERGE into SPARQL."""
        node = _UNWIND_NODE_RE.search(query or "")
        if node:
            for row in batch or []:
                self._upsert_node_triples(node.group("label"), dict(row))
            return []
        edge = _UNWIND_EDGE_RE.search(query or "")
        if edge:
            for row in batch or []:
                rel = str(row.get("type") or edge.group("rel") or "RELATED_TO")
                self._upsert_edge_triple(
                    rel, row.get("source"), row.get("target"), props=dict(row)
                )
            return []
        # Fall back to per-row execute for any other shape.
        out: list[dict[str, Any]] = []
        for row in batch or []:
            out.extend(self.execute(query, row))
        return out

    # ------------------------------------------------------------------
    # LPG convenience — full-fidelity node/edge writes
    # ------------------------------------------------------------------
    def add_node(self, node_id: str, properties: dict[str, Any] | None = None) -> None:
        """Add/update a node and ALL its properties as RDF triples."""
        props = dict(properties or {})
        label = str(props.get("type") or "Node")
        self._upsert_node_triples(label, {"id": node_id, **props})

    def add_edge(
        self, source_id: str, target_id: str, properties: dict[str, Any] | None = None
    ) -> None:
        """Add an edge as a direct RDF triple ``source <rel> target``."""
        props = dict(properties or {})
        rel = str(props.get("type") or "relatedTo")
        self._upsert_edge_triple(rel, source_id, target_id, props=props)

    # ------------------------------------------------------------------
    # Embeddings (client-side; Stardog has no native vector index here)
    # ------------------------------------------------------------------
    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        self._embeddings[node_id] = list(embedding)

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        if not self._embeddings:
            return []
        scored = [
            (nid, self._cosine(query_embedding, emb))
            for nid, emb in self._embeddings.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"id": nid, "score": s} for nid, s in scored[:n_results]]

    def prune(self, criteria: dict[str, Any]) -> None:
        """Delete nodes below a minimum importance (matches JenaFusekiBackend)."""
        min_importance = float(criteria.get("min_importance", 0.0) or 0.0)
        if min_importance <= 0:
            return
        self.execute_sparql_update(
            f"DELETE {{ ?s ?p ?o }} WHERE {{ ?s <{_NS}importance> ?imp . "
            f"FILTER(xsd:float(?imp) < {min_importance}) ?s ?p ?o . }}"
        )

    def edge_count(self) -> int | None:
        """Total object-property triples between our nodes (drift counting)."""
        rows = self.execute_sparql_query(
            f"SELECT (COUNT(*) AS ?c) WHERE {{ ?s ?p ?o . "
            f'FILTER(STRSTARTS(STR(?o), "{_NS}")) FILTER(?p != rdf:type) }}'
        )
        try:
            return int(rows[0]["c"]) if rows and "c" in rows[0] else None
        except (ValueError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Internal — SPARQL emission
    # ------------------------------------------------------------------
    def _iri(self, value: Any) -> str:
        return f"<{_NS}{quote(str(value), safe=_IRI_SAFE)}>"

    @staticmethod
    def _literal(value: Any) -> str:
        if isinstance(value, bool):
            return f'"{str(value).lower()}"^^<http://www.w3.org/2001/XMLSchema#boolean>'
        if isinstance(value, int):
            return f'"{value}"^^<http://www.w3.org/2001/XMLSchema#integer>'
        if isinstance(value, float):
            return f'"{value}"^^<http://www.w3.org/2001/XMLSchema#double>'
        if isinstance(value, dict | list):
            value = json.dumps(value, default=str)
        text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{text}"'

    @staticmethod
    def _graph_open(graph_uri: str | None) -> str:
        return f"GRAPH <{graph_uri}> {{ " if graph_uri else ""

    @staticmethod
    def _graph_close(graph_uri: str | None) -> str:
        return " }" if graph_uri else ""

    def _upsert_node_triples(self, label: str, params: dict[str, Any]) -> None:
        """Emit an idempotent node upsert: type assertion + per-property replace,
        routed into the source's named graph."""
        node_id = params.get("id")
        if node_id is None:
            return
        s = self._iri(node_id)
        g = graph_uri_for(params)
        go, gc = self._graph_open(g), self._graph_close(g)
        ops: list[str] = [f"INSERT DATA {{ {go}{s} a {self._iri(label)} .{gc} }}"]
        for key, val in params.items():
            if key in ("id", "type", "embedding") or val is None:
                continue
            p = self._iri(key)
            lit = self._literal(val)
            ops.append(
                f"DELETE {{ {go}{s} {p} ?o .{gc} }} "
                f"INSERT {{ {go}{s} {p} {lit} .{gc} }} "
                f"WHERE {{ OPTIONAL {{ {go}{s} {p} ?o .{gc} }} }}"
            )
        self.execute_sparql_update(" ;\n".join(ops))

    def _upsert_edge_triple(
        self,
        rel: str,
        source_id: Any,
        target_id: Any,
        props: dict[str, Any] | None = None,
    ) -> None:
        """Emit an edge as a direct triple ``s <rel> t`` in the source's named graph."""
        if source_id is None or target_id is None:
            return
        g = graph_uri_for(props or {})
        go, gc = self._graph_open(g), self._graph_close(g)
        triple = f"{self._iri(source_id)} {self._iri(rel)} {self._iri(target_id)} ."
        self.execute_sparql_update(f"INSERT DATA {{ {go}{triple}{gc} }}")

    def _label_of(self, node_id: Any) -> list[dict[str, Any]]:
        """Answer the engine's ``label(n)`` lookup from the stored type triple."""
        if node_id is None:
            return []
        s = self._iri(node_id)
        rows = self.execute_sparql_query(
            f"SELECT ?t WHERE {{ {{ {s} a ?t }} UNION {{ GRAPH ?g {{ {s} a ?t }} }} "
            f'FILTER(STRSTARTS(STR(?t), "{_NS}")) }} LIMIT 1'
        )
        if rows and rows[0].get("t"):
            return [{"lbl": str(rows[0]["t"]).split("#")[-1]}]
        return []

    @staticmethod
    def _cosine(v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        m1 = math.sqrt(sum(a * a for a in v1))
        m2 = math.sqrt(sum(a * a for a in v2))
        return dot / (m1 * m2) if m1 and m2 else 0.0
