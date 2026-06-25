"""Code-scoped graph analytics over the resolved ``:Code`` subgraph.

CONCEPT:KG-2.210 — god nodes, communities and "surprising connections" scoped to
the code call/inheritance graph, and CONCEPT:KG-2.213 — the architecture report
synthesized from them.

Assimilated from Graphify's in-package analytics (``god_nodes`` = degree ranking,
Leiden/Louvain communities, edge-betweenness "surprising connections", import
cycles) — but run over our durable, type/scope-resolved engine call graph instead
of a one-shot NetworkX notebook. We exceed it by:

* reusing the Rust engine's ephemeral Louvain (``community_detect_ephemeral``,
  KG-2.58) so detection runs on a throwaway ``:Code`` projection with **no tenant
  load and no persistence churn**;
* carrying the resolver's per-edge ``confidence`` through every view (Graphify's
  EXTRACTED/INFERRED/AMBIGUOUS analog);
* producing a regenerable, ``file:line``-cited report node rather than a static
  ``GRAPH_REPORT.md`` snapshot.

The module is deliberately dependency-light (pure-Python degree + Tarjan SCC) so it
runs on the serving plane, which has dropped the heavy graph wheels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

# Relationship types that make up the code call/inheritance subgraph. Mirrors the
# set the resolver binds (lowercase canonical) — kept aligned with the ``call_graph``
# / ``graph_code_nav`` actions.
_CODE_EDGE_TYPES = (
    "calls",
    "inherits",
    "realizes",
    "depends_on",
)

# Hard cap on projected edges so a fleet-wide call graph can't blow up memory on the
# serving plane. Scope with ``scope`` (a file_path/source_system substring) to focus.
_MAX_EDGES = 20_000


def _confidence_bucket(value: Any) -> str:
    """Bucket a raw edge confidence into Graphify's three-tier vocabulary.

    Numeric scores follow Graphify's defaults (EXTRACTED=1.0, INFERRED=0.5,
    AMBIGUOUS=0.2); string strategies pass through upper-cased.
    """
    if value is None:
        return "EXTRACTED"  # resolver default for a directly-bound edge
    if isinstance(value, (int, float)):
        if value >= 0.9:
            return "EXTRACTED"
        if value >= 0.4:
            return "INFERRED"
        return "AMBIGUOUS"
    return str(value).upper()


def load_code_subgraph(
    engine: Any, scope: str = "", *, max_edges: int = _MAX_EDGES
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Project the ``:Code`` call/inheritance subgraph out of the engine.

    Returns ``(nodes, edges)`` where ``nodes`` maps id → display metadata and each
    edge is ``{src, dst, rel, confidence}``. ``scope`` (optional) restricts to nodes
    whose ``file_path`` or ``source_system`` contains the substring, so a single repo
    or service can be analysed in isolation (Graphify analyses one tree at a time).
    """
    backend = getattr(engine, "backend", None)
    if backend is None:
        raise RuntimeError("no graph backend available")

    rel_list = ", ".join(f"'{t}'" for t in _CODE_EDGE_TYPES)
    where = [f"type(r) IN [{rel_list}]"]
    params: dict[str, Any] = {}
    if scope:
        where.append(
            "(s.file_path CONTAINS $scope OR s.source_system CONTAINS $scope) "
            "AND (t.file_path CONTAINS $scope OR t.source_system CONTAINS $scope)"
        )
        params["scope"] = scope
    cypher = (
        "MATCH (s:Code)-[r]->(t:Code) "
        f"WHERE {' AND '.join(where)} "
        "RETURN s.id AS src, s.name AS src_name, s.file_path AS src_file, "
        "s.language AS src_lang, s.kind_detail AS src_kind, "
        "t.id AS dst, t.name AS dst_name, t.file_path AS dst_file, "
        "t.language AS dst_lang, t.kind_detail AS dst_kind, "
        "type(r) AS rel, r.confidence AS confidence "
        f"LIMIT {int(max_edges)}"
    )
    rows = backend.execute(cypher, params) or []

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def _remember(nid: Any, name: Any, file_path: Any, lang: Any, kind: Any) -> None:
        if not nid or nid in nodes:
            return
        nodes[nid] = {
            "id": nid,
            "label": name or str(nid).rsplit(":", 1)[-1],
            "file_path": file_path,
            "language": lang,
            "kind": kind,
        }

    for r in rows:
        src, dst = r.get("src"), r.get("dst")
        if not src or not dst:
            continue
        _remember(src, r.get("src_name"), r.get("src_file"), r.get("src_lang"), r.get("src_kind"))
        _remember(dst, r.get("dst_name"), r.get("dst_file"), r.get("dst_lang"), r.get("dst_kind"))
        edges.append(
            {
                "src": src,
                "dst": dst,
                "rel": r.get("rel"),
                "confidence": r.get("confidence"),
            }
        )
    return nodes, edges


def _degrees(node_ids: set[str], edges: list[dict[str, Any]]) -> dict[str, int]:
    """Undirected degree per node (Graphify's ``god_nodes`` metric)."""
    deg: dict[str, int] = dict.fromkeys(node_ids, 0)
    for e in edges:
        deg[e["src"]] = deg.get(e["src"], 0) + 1
        deg[e["dst"]] = deg.get(e["dst"], 0) + 1
    return deg


def detect_communities(
    engine: Any,
    node_ids: list[str],
    edges: list[dict[str, Any]],
    resolution: float = 1.0,
) -> dict[str, int]:
    """Map node → community id via the engine's ephemeral Louvain (KG-2.58).

    Falls back to weakly-connected components (pure-Python union-find) if the engine
    op is unavailable, so the analytic degrades gracefully offline.
    """
    edge_pairs = [(e["src"], e["dst"]) for e in edges]
    try:
        communities = engine.graph_compute.community_detect_ephemeral(
            node_ids, edge_pairs, resolution
        )
        if communities:
            return {nid: i for i, com in enumerate(communities) for nid in com}
    except Exception:
        pass
    # Fallback: union-find weakly-connected components.
    parent: dict[str, str] = {n: n for n in node_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edge_pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    roots = {n: find(n) for n in node_ids}
    order = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {n: order[r] for n, r in roots.items()}


def surprising_connections(
    edges: list[dict[str, Any]],
    node_comm: dict[str, int],
    degrees: dict[str, int],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Cross-community edges, ranked by combined endpoint degree (Graphify analog).

    Graphify ranks these by edge-betweenness; combined endpoint degree is a cheap,
    deterministic proxy that surfaces the same "bridge between clusters" edges
    without a full Brandes pass over the projection.
    """
    out: list[dict[str, Any]] = []
    for e in edges:
        ca, cb = node_comm.get(e["src"]), node_comm.get(e["dst"])
        if ca is None or cb is None or ca == cb:
            continue
        out.append(
            {
                "src": e["src"],
                "dst": e["dst"],
                "rel": e["rel"],
                "confidence": e["confidence"],
                "from_community": ca,
                "to_community": cb,
                "bridge_score": degrees.get(e["src"], 0) + degrees.get(e["dst"], 0),
            }
        )
    out.sort(key=lambda x: x["bridge_score"], reverse=True)
    return out[:top_n]


def import_cycles(
    node_ids: list[str], edges: list[dict[str, Any]], limit: int = 20
) -> list[list[str]]:
    """Strongly-connected components of size > 1 = dependency cycles (Tarjan)."""
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        adj[e["src"]].append(e["dst"])

    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[list[str]] = []

    import sys

    sys.setrecursionlimit(max(10_000, len(node_ids) * 4 + 1000))

    def strongconnect(v: str) -> None:
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, ()):  # noqa: SIM118
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                result.append(comp)

    for v in node_ids:
        if v not in index:
            strongconnect(v)
        if len(result) >= limit:
            break
    return result[:limit]


def build_code_metrics(
    engine: Any, scope: str = "", top_k: int = 10, render_limit: int = 250
) -> dict[str, Any]:
    """The Graphify-style structural analysis of the ``:Code`` subgraph.

    Returns summary counts (file types / relations / confidence), god nodes (degree),
    communities, and surprising connections — the durable, KG-native equivalent of
    the tutorial's NetworkX analysis cell. Also returns a bounded ``graph`` render
    payload (top ``render_limit`` nodes by degree + the edges among them) so a single
    call drives the force-directed canvas (size=degree, color=community, KG-2.214).
    """
    nodes, edges = load_code_subgraph(engine, scope)
    if not nodes:
        return {
            "status": "empty",
            "scope": scope or None,
            "message": "no :Code subgraph found (ingest a repo first, or widen scope)",
        }

    node_ids = list(nodes)
    degrees = _degrees(set(node_ids), edges)
    node_comm = detect_communities(engine, node_ids, edges)

    # Summary distributions (mirror the tutorial's Counters).
    langs = Counter(n.get("language") or "?" for n in nodes.values())
    kinds = Counter(n.get("kind") or "?" for n in nodes.values())
    rels = Counter(e.get("rel") or "?" for e in edges)
    conf = Counter(_confidence_bucket(e.get("confidence")) for e in edges)

    def _node_view(nid: str) -> dict[str, Any]:
        meta = nodes.get(nid, {"id": nid, "label": nid})
        return {
            "id": nid,
            "label": meta.get("label"),
            "file_path": meta.get("file_path"),
            "degree": degrees.get(nid, 0),
            "community": node_comm.get(nid),
        }

    gods = sorted(node_ids, key=lambda n: degrees.get(n, 0), reverse=True)[: max(1, top_k)]

    # Group community membership for display (cap members so the payload stays lean).
    by_comm: dict[int, list[str]] = defaultdict(list)
    for nid, cid in node_comm.items():
        by_comm[cid].append(nodes.get(nid, {}).get("label") or nid)
    communities = [
        {"id": cid, "size": len(members), "members": sorted(members)[:25]}
        for cid, members in sorted(by_comm.items(), key=lambda kv: len(kv[1]), reverse=True)
    ]

    # Bounded render payload (CONCEPT:KG-2.214): the top nodes by degree + edges
    # among them, so the force-directed canvas stays responsive on large graphs.
    render_ids = set(
        sorted(node_ids, key=lambda n: degrees.get(n, 0), reverse=True)[: max(1, render_limit)]
    )
    render = {
        "nodes": [_node_view(n) for n in render_ids],
        "edges": [
            {
                "source": e["src"],
                "target": e["dst"],
                "rel": e["rel"],
                "confidence": e["confidence"],
            }
            for e in edges
            if e["src"] in render_ids and e["dst"] in render_ids
        ],
        "truncated": len(node_ids) > len(render_ids),
    }

    return {
        "status": "ok",
        "scope": scope or None,
        "nodes": len(nodes),
        "edges": len(edges),
        "by_language": dict(langs),
        "by_kind": dict(kinds),
        "by_relation": dict(rels),
        "by_confidence": dict(conf),
        "god_nodes": [_node_view(n) for n in gods],
        "community_count": len(communities),
        "communities": communities,
        "surprising_connections": surprising_connections(
            edges, node_comm, degrees, top_n=max(1, top_k)
        ),
        "graph": render,
    }


def build_arch_report(
    engine: Any, scope: str = "", top_k: int = 10
) -> dict[str, Any]:
    """CONCEPT:KG-2.213 — a regenerable architecture report (GRAPH_REPORT.md analog).

    Composes :func:`build_code_metrics` with import-cycle detection into a Markdown
    document plus the structured metrics. The caller persists it as a node.
    """
    metrics = build_code_metrics(engine, scope, top_k)
    if metrics.get("status") != "ok":
        return {"status": metrics.get("status", "empty"), "metrics": metrics, "markdown": ""}

    _, edges = load_code_subgraph(engine, scope)
    node_ids = list({e["src"] for e in edges} | {e["dst"] for e in edges})
    cycles = import_cycles(node_ids, edges)

    scope_label = scope or "entire ingested code graph"
    lines: list[str] = [
        f"# Architecture Report — {scope_label}",
        "",
        "## Summary",
        f"- Code symbols: **{metrics['nodes']}**, relationships: **{metrics['edges']}**",
        f"- Languages: {metrics['by_language']}",
        f"- Relationships: {metrics['by_relation']}",
        f"- Confidence: {metrics['by_confidence']}",
        f"- Communities: **{metrics['community_count']}**",
        "",
        "## God Nodes (highest-degree hubs)",
    ]
    for g in metrics["god_nodes"]:
        loc = f" — `{g['file_path']}`" if g.get("file_path") else ""
        lines.append(
            f"- **{g['label']}** (degree={g['degree']}, community={g['community']}){loc}"
        )
    lines += ["", "## Community Hubs"]
    for c in metrics["communities"][: max(1, top_k)]:
        preview = ", ".join(c["members"][:8])
        lines.append(f"- Community {c['id']} ({c['size']} symbols): {preview}")
    lines += ["", "## Surprising Connections (cross-community bridges)"]
    if metrics["surprising_connections"]:
        for s in metrics["surprising_connections"]:
            lines.append(
                f"- `{s['src']}` —[{s['rel']}]→ `{s['dst']}` "
                f"(communities {s['from_community']}→{s['to_community']}, "
                f"bridge={s['bridge_score']})"
            )
    else:
        lines.append("- none (graph is a single cohesive community)")
    lines += ["", "## Dependency Cycles"]
    if cycles:
        for cyc in cycles:
            lines.append(f"- cycle of {len(cyc)}: {' → '.join(cyc[:8])}")
    else:
        lines.append("- none detected ✅")

    return {
        "status": "ok",
        "scope": scope or None,
        "metrics": metrics,
        "cycles": cycles,
        "markdown": "\n".join(lines),
    }
