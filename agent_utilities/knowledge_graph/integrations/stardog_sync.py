#!/usr/bin/python
from __future__ import annotations

"""Stardog instance-data sync — explicit push / pull, partitioned by source.

CONCEPT:AU-KG.query.vendor-agnostic-traversal — Vendor-Agnostic Graph Backend Abstraction.

On-demand counterpart to the live fan-out mirror: push the KG's instance data
(nodes + edges) into a Stardog triplestore, or pull it back. Data is partitioned
into ``urn:source:<system>`` named graphs (CONCEPT via :mod:`..backends.sparql.source_partition`)
so an operator can push, query, or re-ingest one source's slice (LeanIX,
ServiceNow, …) on its own.

* **push** reads nodes/edges with the proven backend-agnostic iterators from
  :mod:`..migration` (the same readers the fan-out reconcile uses) and writes them
  through the Stardog backend's full-fidelity ``add_node`` / ``add_edge`` — which
  route each triple into the right named graph.
* **pull** queries Stardog (optionally one named graph) and re-materialises the
  individuals + their properties + object-property edges back into the live KG.
"""

import logging
from typing import Any

from ..backends.sparql.source_partition import graph_uri_for_source, source_of

logger = logging.getLogger(__name__)

_NS = "http://agent-utilities.dev/kg#"


def push_to_stardog(
    source_engine: Any,
    backend: Any,
    *,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Push KG nodes + edges into ``backend`` (a Stardog SPARQL backend).

    Args:
        source_engine: the KG engine / facade (anything ``migration`` can read).
        backend: a ``StardogSparqlBackend`` (or any backend with full-fidelity
            ``add_node``/``add_edge``).
        sources: optional allow-list of source systems (``["leanix", …]``). When
            given, only nodes/edges tagged with one of those sources are pushed;
            otherwise the whole graph is pushed.

    Returns counts of nodes/edges written, broken down by named graph.
    """
    from ..migration import _iter_source_edges, _iter_source_nodes

    allow = {s.strip().lower() for s in sources} if sources else None
    by_graph: dict[str, dict[str, int]] = {}
    nodes = edges = 0

    def _bump(graph: str, kind: str) -> None:
        by_graph.setdefault(graph, {"nodes": 0, "edges": 0})[kind] += 1

    for node_id, _label, props in _iter_source_nodes(source_engine):
        src = source_of(props)
        if allow is not None and src not in allow:
            continue
        try:
            backend.add_node(node_id, props)
            nodes += 1
            _bump(graph_uri_for_source(src) if src else "default", "nodes")
        except Exception as exc:  # noqa: BLE001
            logger.debug("push_to_stardog: node %s failed: %s", node_id, exc)

    for src_id, tgt_id, rel, props in _iter_source_edges(source_engine):
        src = source_of(props)
        if allow is not None and src not in allow:
            continue
        try:
            backend.add_edge(src_id, tgt_id, {"type": rel, **props})
            edges += 1
            _bump(graph_uri_for_source(src) if src else "default", "edges")
        except Exception as exc:  # noqa: BLE001
            logger.debug("push_to_stardog: edge %s->%s failed: %s", src_id, tgt_id, exc)

    logger.info(
        "push_to_stardog: %d nodes, %d edges across %d graph(s)",
        nodes,
        edges,
        len(by_graph),
    )
    return {"status": "ok", "nodes": nodes, "edges": edges, "graphs": by_graph}


def pull_from_stardog(
    backend: Any,
    target_engine: Any,
    *,
    graph_uri: str | None = None,
    source: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Pull instance data from Stardog back into the live KG.

    Materialises each individual (rdf:type → node label), its literal properties,
    and its object-property edges into ``target_engine`` via ``add_node`` /
    ``link_nodes``.

    Args:
        backend: a ``StardogSparqlBackend``.
        target_engine: the KG engine / facade to ingest into.
        graph_uri: restrict to one named graph; mutually informs ``source``.
        source: restrict to ``urn:source:<source>`` (convenience over ``graph_uri``).
        limit: max individuals to pull.
    """
    if source and not graph_uri:
        graph_uri = graph_uri_for_source(source)
    g_open = f"GRAPH <{graph_uri}> {{ " if graph_uri else ""
    g_close = " }" if graph_uri else ""

    # 1. Individuals + their type.
    rows = backend.execute_sparql_query(
        f"SELECT ?s ?t WHERE {{ {g_open}?s a ?t .{g_close} "
        f'FILTER(STRSTARTS(STR(?t), "{_NS}")) }} LIMIT {limit}'
    )
    nodes = 0
    edges = 0
    for row in rows:
        s = row.get("s")
        t = row.get("t")
        if not s:
            continue
        node_type = str(t).split("#")[-1] if t else "Node"
        node_id = str(s).split("#")[-1] if s.startswith(_NS) else str(s)
        # 2. Literal properties for this individual.
        props: dict[str, Any] = {}
        for pr in backend.execute_sparql_query(
            f"SELECT ?p ?o WHERE {{ {g_open}<{s}> ?p ?o .{g_close} "
            f"FILTER(isLiteral(?o)) FILTER(?p != rdf:type) }}"
        ):
            p, o = pr.get("p"), pr.get("o")
            if p and p.startswith(_NS):
                props[p.split("#")[-1]] = o
        try:
            target_engine.add_node(node_id, node_type, props)
            nodes += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("pull_from_stardog: node %s failed: %s", node_id, exc)

    # 3. Object-property edges between our individuals.
    for er in backend.execute_sparql_query(
        f"SELECT ?s ?p ?o WHERE {{ {g_open}?s ?p ?o .{g_close} "
        f'FILTER(isIRI(?o)) FILTER(STRSTARTS(STR(?o), "{_NS}")) '
        f"FILTER(?p != rdf:type) }} LIMIT {limit}"
    ):
        s, p, o = er.get("s"), er.get("p"), er.get("o")
        if not (s and p and o):
            continue
        sid = str(s).split("#")[-1] if s.startswith(_NS) else str(s)
        oid = str(o).split("#")[-1] if o.startswith(_NS) else str(o)
        rel = str(p).split("#")[-1]
        try:
            target_engine.link_nodes(sid, oid, rel)
            edges += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("pull_from_stardog: edge %s->%s failed: %s", sid, oid, exc)

    logger.info("pull_from_stardog: ingested %d nodes, %d edges", nodes, edges)
    return {"status": "ok", "nodes": nodes, "edges": edges, "graph": graph_uri}
