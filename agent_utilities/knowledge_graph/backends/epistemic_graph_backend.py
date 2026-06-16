from __future__ import annotations

"""Pure In-Memory Graph Backend (CONCEPT:OS-5.0).

Zero-dependency, zero-disk backend using GraphComputeEngine (Rust/epistemic-graph).
Ideal for testing, edge devices, ephemeral containers,
and as the default lightweight backend.

Implements the full GraphBackend ABC with optional
JSON serialization for persistence.
"""


import json
import logging
import re
from typing import Any

from .base import GraphBackend

logger = logging.getLogger(__name__)

# A variable-length relationship pattern ``-[*lo..hi]-`` / ``-[*]->`` etc. These
# are bounded multi-hop traversals the L1 engine resolves via a BFS over its
# native neighbour ops. (CONCEPT:KG-2.7 P1 — L1 native traversal.)
_VAR_LEN_RE = re.compile(r"\[\s*[A-Za-z_]*\s*:?\s*\w*\s*\*")

# Write-DDL keyword detection as *whole words*. A substring scan misroutes a READ
# whose alias/property merely contains the letters — ``RETURN n.x AS created`` is
# not a CREATE, ``n.merge_status`` is not a MERGE — sending it to the legacy
# reader and silently returning []. Match on word boundaries instead. (KG-2.63)
_WRITE_KW_RE = {
    kw: re.compile(rf"\b{kw}\b")
    for kw in ("CREATE", "MERGE", "DELETE", "SET", "REMOVE")
}


def _has_kw(qu: str, kw: str) -> bool:
    """True if ``kw`` appears as a whole Cypher clause keyword in ``qu`` (upper)."""
    return bool(_WRITE_KW_RE[kw].search(qu))


class EpistemicGraphBackend(GraphBackend):
    """Pure in-memory graph backend using GraphComputeEngine (Rust-native).

    This is the lightest-weight backend: zero disk, zero external
    dependencies beyond the compiled graph engine. All data lives in
    process memory and is lost on shutdown unless explicitly saved
    via ``save_to_json()``.

    Use cases:
        - Unit testing (fast, deterministic)
        - Edge compute (minimal footprint)
        - Ephemeral containers (no persistent storage needed)
        - Development/prototyping
    """

    @property
    def cypher_support(self) -> str:
        """No Cypher engine: only the bounded operational subset the orchestration
        engine emits is interpreted directly (CONCEPT:KG-2.63)."""
        return "subset"

    def __init__(self) -> None:
        from ..core.graph_compute import GraphComputeEngine

        self._graph = GraphComputeEngine(backend_type="rust")
        self._embeddings: dict[str, list[float]] = {}
        self._node_counter = 0
        logger.info(
            "EpistemicGraphBackend initialized (GraphComputeEngine, pure in-memory)"
        )

    @property
    def graph(self) -> Any:
        """Direct access to the underlying GraphComputeEngine."""
        return self._graph

    # --- GraphBackend ABC Implementation ---

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query against the in-memory graph.

        This backend has no Cypher engine, so it interprets the *operational
        subset* the orchestration engine relies on directly over the
        ``GraphComputeEngine`` node store. Supported for single-node
        ``MATCH`` patterns (no relationships):

          - label filter ``(v:Label ...)`` (matched against ``node_type``,
            ``label``, or ``labels``)
          - inline ``{prop: $param | 'literal'}`` and ``WHERE`` equality,
            ``IN [...]``, ``CONTAINS``, ``IS [NOT] NULL`` filters
          - ``SET v.prop = $param | 'literal'`` property mutation (upsert
            merge — critical for Task status transitions)
          - ``DETACH DELETE v``
          - ``RETURN`` projection: bare ``v`` (full node), ``v.prop AS alias``,
            and ``count(v) AS alias``; honours ``LIMIT``

        Anything outside this subset (relationship traversal, MERGE/CREATE,
        unrecognised shapes) falls back to the legacy id/label/all behaviour.
        """
        params = params or {}
        q = (query or "").strip()
        qu = q.upper()

        # Node upsert: ``MERGE (n:Label {id: $id}) SET n.k = $props_k, ...`` is the
        # persistence path used by the graph-writer daemon and sync phase. Without
        # this, ingested nodes never land in the in-memory store. (CONCEPT:KG-2.0)
        if qu.startswith("MERGE") and "->" not in q and "<-" not in q:
            handled, result = self._exec_merge_node(q, params)
            if handled:
                return result

        # Relationship upsert: ``MATCH (a),(b) WHERE a.id=$x AND b.id=$y
        # MERGE (a)-[:REL]->(b)``. Resolve both endpoints by id (O(1)) and add
        # the edge directly — otherwise this falls to the full-scan legacy reader
        # AND never creates the L1 edge. (CONCEPT:KG-2.8 ingestion throughput)
        if "->" in q and _has_kw(qu, "MERGE") and qu.startswith("MATCH"):
            handled, result = self._exec_rel_merge(q, params)
            if handled:
                return result

        # Relationship-pattern READ — single-hop outbound ``->``, inbound ``<-``,
        # or bounded variable-length ``-[*lo..hi]-``. The L1 engine resolves these
        # natively over its neighbour/BFS ops, so they no longer have to fall back
        # to L3. Critically, if no traversal interpreter matches we return ``[]``
        # — NOT the whole graph — so a tiered caller can defer to L3 instead of
        # silently receiving every node (the old legacy full-scan footgun).
        # (CONCEPT:KG-2.7 P1 — L1 native traversal.)
        if (
            qu.startswith("MATCH")
            and ("->" in q or "<-" in q or _VAR_LEN_RE.search(q))
            and not _has_kw(qu, "MERGE")
            and not _has_kw(qu, "CREATE")
            and not _has_kw(qu, "DELETE")
        ):
            if _VAR_LEN_RE.search(q):
                handled, result = self._exec_var_length_match(q, params)
                if handled:
                    return result
            handled, result = self._exec_rel_match(q, params)
            if handled:
                return result
            # Edge existence/count/property reads (count(r), r.prop, r) anchored by
            # WHERE or inline ids — makes written edges readable from L1.
            handled, result = self._exec_rel_aggregate(q, params)
            if handled:
                return result
            # LABEL+WHERE-anchored traversal (the graph_code_nav shapes): resolve
            # the anchor by scan, then walk. (CONCEPT:KG-2.9g)
            handled, result = self._exec_where_anchored_traversal(q, params)
            if handled:
                return result
            logger.debug(
                "epistemic_graph backend: unhandled relationship read; "
                "returning [] (no full-graph fallback): %s",
                q[:160],
            )
            return []

        # Single-node MATCH patterns are interpreted directly; relationship
        # traversals and other write-DDL fall through to the legacy reader.
        if (
            qu.startswith("MATCH")
            and "->" not in q
            and "<-" not in q
            and not _has_kw(qu, "MERGE")
            and not _has_kw(qu, "CREATE")
        ):
            handled, result = self._exec_node_match(q, params)
            if handled:
                return result

        return self._legacy_execute(params)

    def _exec_rel_match(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret an id-anchored single-hop traversal read, in either direction:

          - outbound ``MATCH (a {id:$x})-[:REL]->(b:Label) RETURN b...`` → successors
          - inbound  ``MATCH (a {id:$x})<-[:REL]-(b:Label) RETURN b...`` → predecessors

        Resolves the anchor ``a`` by id, walks ``REL`` neighbours in the matched
        direction, filters targets ``b`` by label, and projects on ``b``. Returns
        ``(False, [])`` for any shape outside this subset so the caller can fall
        back. (CONCEPT:KG-2.7 P1 — L1 native traversal.)
        """
        anchor_re = (
            r"MATCH\s*\(\s*(\w+)\s*(?::\w+)?\s*\{\s*id\s*:\s*"
            r"(\$\w+|'[^']*'|\"[^\"]*\")\s*\}\s*\)"
        )
        tgt_re = r"\(\s*(\w+)\s*(?::(\w+))?\s*\)"
        # Anchor id accepts either a ``$param`` placeholder or an inline quoted
        # literal (``{id:'foo'}``) — interactive callers commonly write the literal.
        out = re.search(
            anchor_re + r"\s*-\s*\[\s*\w*\s*:?\s*(\w+)?[^\]]*\]\s*->\s*" + tgt_re,
            q,
            re.I,
        )
        if out:
            direction, m = "out", out
        else:
            inb = re.search(
                anchor_re + r"\s*<-\s*\[\s*\w*\s*:?\s*(\w+)?[^\]]*\]\s*-\s*" + tgt_re,
                q,
                re.I,
            )
            if not inb:
                return False, []
            direction, m = "in", inb
        _src_var, id_token, rel, tgt_var, tgt_label = m.groups()

        # Only the simple "anchor by id, project the target" shape is supported.
        if re.search(r"\bSET\b|\bDETACH\b", q, re.I):
            return False, []

        anchor_id = self._coerce_literal(id_token, params)
        if anchor_id is None or not self._graph.has_node(anchor_id):
            return True, []

        rel_upper = rel.upper() if rel else None
        step = (
            self._graph.get_successors
            if direction == "out"
            else self._graph.get_predecessors
        )
        matched: list[tuple[str, dict[str, Any]]] = []
        for tgt in step(anchor_id):
            # Edge ordering follows the arrow direction: outbound is (anchor→tgt),
            # inbound is (tgt→anchor).
            if direction == "out":
                edge_props = self._graph._get_edge_properties(anchor_id, tgt) or {}
            else:
                edge_props = self._graph._get_edge_properties(tgt, anchor_id) or {}
            if rel_upper:
                edge_rel = str(
                    edge_props.get("rel_type") or edge_props.get("type") or ""
                ).upper()
                if edge_rel != rel_upper:
                    continue
            data = self._graph._get_node_properties(tgt) or {}
            if tgt_label and not self._label_match(data, tgt_label):
                continue
            matched.append((tgt, data))

        # Honour ``ORDER BY <tgt>.<prop> [DESC]`` over the projected targets,
        # since neighbour order is not guaranteed to be meaningful.
        ob = re.search(
            rf"\bORDER\s+BY\s+{tgt_var}\.(\w+)\s*(DESC|ASC)?",
            q,
            re.I,
        )
        if ob:
            order_prop = ob.group(1)
            reverse = bool(ob.group(2)) and ob.group(2).upper() == "DESC"

            def _sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, Any]:
                val = item[1].get(order_prop)
                # None sorts last; keep numeric/string ordering stable otherwise.
                return (1, "") if val is None else (0, val)

            matched.sort(key=_sort_key, reverse=reverse)

        return True, self._project(q, tgt_var, matched, params)

    def _exec_rel_aggregate(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Edge existence / count / property read, anchored by WHERE or inline ids.

        Handles the shapes the conformance contract and ``query_cypher`` emit that
        ``_exec_rel_match`` (target-projection only) does not:

          MATCH (s[:L])-[r:REL]->(t[:L]) WHERE s.id=$s AND t.id=$t
            RETURN count(r) AS c | r.<prop> AS alias | r

        Resolves endpoints by id, matches the edge (and ``rel_type`` if given), and
        projects a count, an edge property, or the full edge-property map. This is
        what makes edges *readable* from the L1 backend (they were write-only before
        — present in the compute graph but not returned by ``backend.execute``).
        """
        pat = re.search(
            r"\(\s*(\w*)[^)]*\)\s*-\s*\[\s*(\w*)\s*:?\s*(\w+)?[^\]]*\]\s*->\s*"
            r"\(\s*(\w*)[^)]*\)",
            q,
            re.I,
        )
        if not pat:
            return False, []
        s_var, r_var, rel, t_var = (
            pat.group(1),
            pat.group(2),
            pat.group(3),
            pat.group(4),
        )

        idmap: dict[str, Any] = {}
        for mv in re.finditer(r"(\w+)\.id\s*=\s*(\$\w+|'[^']*'|\"[^\"]*\")", q, re.I):
            idmap[mv.group(1)] = self._coerce_literal(mv.group(2), params)
        for mv in re.finditer(
            r"\(\s*(\w+)\s*(?::\w+)?\s*\{\s*id\s*:\s*(\$\w+|'[^']*'|\"[^\"]*\")\s*\}",
            q,
            re.I,
        ):
            idmap[mv.group(1)] = self._coerce_literal(mv.group(2), params)
        s_id, t_id = idmap.get(s_var), idmap.get(t_var)
        rel_upper = rel.upper() if rel else None

        def _edge_between(a: Any, b: Any) -> dict[str, Any] | None:
            ep = self._graph._get_edge_properties(a, b) or {}
            if not ep and not self._graph.has_edge(a, b):
                return None
            if rel_upper:
                er = str(ep.get("rel_type") or ep.get("type") or "").upper()
                if er != rel_upper:
                    return None
            return ep

        matches: list[dict[str, Any]] = []
        if s_id is not None and t_id is not None:
            ep = _edge_between(s_id, t_id)
            if ep is not None:
                matches.append(ep)
        elif s_id is not None and self._graph.has_node(s_id):
            for tgt in self._graph.get_successors(s_id):
                ep = _edge_between(s_id, tgt)
                if ep is not None:
                    matches.append(ep)
        elif t_id is not None and self._graph.has_node(t_id):
            for src in self._graph.get_predecessors(t_id):
                ep = _edge_between(src, t_id)
                if ep is not None:
                    matches.append(ep)
        else:
            # No id anchor → global edge read. Only *edge-centric* projections are
            # served here (``count(r)`` or the edges/edge-props themselves): a bare
            # ``count(r)`` is answered in O(1) via the engine's edge count; a
            # rel-type-filtered count or an ``r``/``r.prop`` projection enumerates
            # the engine's native triple export. An unanchored *node* projection
            # (e.g. ``RETURN b``) still defers (``return False``) rather than
            # scanning every node — the deliberate L1 guard. This is what makes
            # ``MATCH ()-[r]->() RETURN count(r)`` readable from the L1 backend.
            # (CONCEPT:KG-2.7 P1 — L1 native traversal.)
            wants_count = re.search(r"RETURN\s+count\s*\(", q, re.I) is not None
            wants_edge = bool(r_var) and re.search(rf"RETURN\s+{r_var}(\b|\.)", q, re.I)
            if not (wants_count or wants_edge):
                return False, []
            edge_count = getattr(self._graph, "edge_count", None)
            get_triples = getattr(self._graph, "get_triples", None)
            if wants_count and not rel_upper and callable(edge_count):
                alias_m = re.search(r"count\s*\(\s*\w*\s*\)\s*(?:AS\s+(\w+))?", q, re.I)
                alias = (alias_m.group(1) if alias_m else None) or "count"
                return True, [{alias: edge_count()}]
            if not callable(get_triples):
                return False, []
            for triple in get_triples():
                if len(triple) != 3:
                    continue
                src, rel_t, tgt = triple
                if rel_upper and str(rel_t).upper() != rel_upper:
                    continue
                ep = self._graph._get_edge_properties(src, tgt) or {}
                if rel_upper and not (ep.get("rel_type") or ep.get("type")):
                    ep = {**ep, "rel_type": rel_t}
                matches.append(ep)

        cnt = re.search(r"RETURN\s+count\s*\(\s*\w+\s*\)\s*(?:AS\s+(\w+))?", q, re.I)
        if cnt:
            return True, [{cnt.group(1) or "count": len(matches)}]
        if r_var:
            propm = re.search(rf"RETURN\s+{r_var}\.(\w+)\s*(?:AS\s+(\w+))?", q, re.I)
            if propm:
                key, alias = propm.group(1), (propm.group(2) or propm.group(1))
                return True, [{alias: ep.get(key)} for ep in matches]
            if re.search(rf"RETURN\s+{r_var}\b", q, re.I):
                return True, [{r_var: ep} for ep in matches]
        return False, []

    def _exec_where_anchored_traversal(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret a LABEL+WHERE-anchored traversal (not id-anchored):

          MATCH (l:La)-[:REL]->(r:Lb)        WHERE <anchor>.<p> = $x  RETURN <free>…
          MATCH (l:La)-[:REL*lo..hi]->(r:Lb) WHERE <anchor>.<p> = $x  RETURN <free>…

        This is the shape ``graph_code_nav`` emits (find_references / trace_call_graph
        / impact_of_change): the *anchor* node is pinned by a WHERE property (e.g.
        ``def.name = $symbol``), and the *free* node is projected. We resolve the
        anchor ids via a label+WHERE scan (the engine's labeled fetch), then walk
        ``calls`` edges in the direction implied by which side is anchored —
        single-hop (rel-type filtered) or bounded k-hop. (CONCEPT:KG-2.9g)
        """
        import re

        if re.search(r"\bSET\b|\bDETACH\b|\bDELETE\b|\bMERGE\b", q, re.I):
            return False, []
        node = r"\(\s*(\w+)\s*(?::(\w+))?\s*\)"
        rel = (
            r"\s*(<-|-)\s*\[\s*\w*\s*:?\s*(\w+)?\s*"
            r"(\*\s*(\d*)\s*\.\.?\s*(\d*))?\s*[^\]]*\]\s*(->|-)\s*"
        )
        m = re.search(node + rel + node, q, re.I)
        if not m:
            return False, []
        (
            lvar,
            llabel,
            larrow,
            rel_type,
            varlen,
            lo_s,
            hi_s,
            rarrow,
            rvar,
            rlabel,
        ) = m.groups()

        mw = re.search(r"\bWHERE\b(.+?)(?:\bRETURN\b|$)", q, re.I | re.S)
        if not mw:
            return False, []
        # Anchor = the node carrying the WHERE condition; free = the other.
        if re.search(rf"\b{lvar}\.\w", mw.group(1)):
            anchor_var, free_var = lvar, rvar
            anchor_label, free_label = llabel, rlabel
        elif re.search(rf"\b{rvar}\.\w", mw.group(1)):
            anchor_var, free_var = rvar, lvar
            anchor_label, free_label = rlabel, llabel
        else:
            return False, []

        groups = self._parse_where_or(mw.group(1), anchor_var, params)
        if not groups:
            return False, []

        # Resolve anchor node ids: labeled fetch + WHERE filter.
        anchor_ids: list[str] = []
        try:
            rows = self._graph.get_nodes_by_label(anchor_label or "", 0)
        except Exception:  # noqa: BLE001
            rows = self._graph._get_all_nodes_with_properties()
        for nid, data in rows or []:
            data = data or {}
            if anchor_label and not self._label_match(data, anchor_label):
                continue
            if self._eval_groups(nid, data, groups):
                anchor_ids.append(nid)

        # Direction from the anchor's perspective. Pattern is ``(l)-[…]->(r)``
        # (left→right) when ``rarrow == '->'``. Anchored-on-source ⇒ walk out
        # (its callees); anchored-on-target ⇒ walk in (its callers).
        ltr = rarrow == "->" and larrow != "<-"
        if anchor_var == lvar:
            direction = "out" if ltr else "in"
        else:
            direction = "in" if ltr else "out"

        lo = int(lo_s) if lo_s else 1
        hi = int(hi_s) if hi_s else (5 if varlen else 1)
        if not varlen:
            lo = hi = 1
        rel_upper = rel_type.upper() if rel_type else None

        matched: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for aid in anchor_ids:
            if varlen:
                hops = self._khop(aid, lo, hi, direction)
            else:
                step = (
                    self._graph.get_successors
                    if direction == "out"
                    else self._graph.get_predecessors
                )
                hops = []
                for nb in step(aid):
                    if rel_upper:
                        ep = (
                            self._graph._get_edge_properties(aid, nb)
                            if direction == "out"
                            else self._graph._get_edge_properties(nb, aid)
                        ) or {}
                        er = str(ep.get("rel_type") or ep.get("type") or "").upper()
                        if er != rel_upper:
                            continue
                    hops.append(nb)
            for nid in hops:
                if nid in seen:
                    continue
                seen.add(nid)
                data = self._graph._get_node_properties(nid) or {}
                if free_label and not self._label_match(data, free_label):
                    continue
                matched.append((nid, data))

        return True, self._project(q, free_var, matched, params)

    def _exec_var_length_match(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret an id-anchored bounded variable-length traversal read:

        ``MATCH (n)-[*1..3]-(t {id:$x}) RETURN n`` and its mirror (anchor on the
        left) and directed ``->`` / ``<-`` variants. Resolves the anchor by id,
        runs a bounded BFS over the engine's native neighbour ops to ``hi`` hops,
        filters the free node by label, and projects on it. Returns ``(False, [])``
        for shapes outside this subset. (CONCEPT:KG-2.7 P1 — L1 native traversal.)
        """
        if re.search(r"\bSET\b|\bDETACH\b|\bDELETE\b", q, re.I):
            return False, []
        node = r"\(\s*(\w+)\s*(?::(\w+))?\s*(\{[^}]*\})?\s*\)"
        rel = r"\s*(<-|-)\s*\[[^\]]*\*\s*(\d*)\s*(\.\.)?\s*(\d*)[^\]]*\]\s*(->|-)\s*"
        m = re.search(node + rel + node, q, re.I)
        if not m:
            return False, []
        (
            lvar,
            llabel,
            linline,
            larrow,
            lo_s,
            dotdot,
            hi_s,
            rarrow,
            rvar,
            rlabel,
            rinline,
        ) = m.groups()

        def _id_from_inline(inline: str | None) -> Any:
            if not inline:
                return None
            im = re.search(r"id\s*:\s*(\$\w+|'[^']*'|\"[^\"]*\")", inline, re.I)
            return self._coerce_literal(im.group(1), params) if im else None

        left_id, right_id = _id_from_inline(linline), _id_from_inline(rinline)
        if left_id is not None and self._graph.has_node(left_id):
            anchor_id, anchor_on_left = left_id, True
            free_var, free_label = rvar, rlabel
        elif right_id is not None and self._graph.has_node(right_id):
            anchor_id, anchor_on_left = right_id, False
            free_var, free_label = lvar, llabel
        else:
            # No inline-id anchor → defer so a WHERE/label-anchored traversal
            # (``…-[:calls*1..n]->(t) WHERE t.name=$s``) can resolve the anchor by
            # scan instead of being swallowed as "no rows". (CONCEPT:KG-2.9g)
            return False, []

        lo = int(lo_s) if lo_s else 1
        if hi_s:
            hi = int(hi_s)
        elif dotdot:
            hi = max(lo, 5)  # open upper bound → safe cap
        else:
            hi = lo
        if hi < lo:
            hi = lo

        # Direction from the anchor's perspective. ``rarrow == '->'`` is outbound,
        # ``larrow == '<-'`` is inbound; otherwise undirected.
        if rarrow == "->" and larrow != "<-":
            direction = "out" if anchor_on_left else "in"
        elif larrow == "<-" and rarrow != "->":
            direction = "in" if anchor_on_left else "out"
        else:
            direction = "both"

        matched: list[tuple[str, dict[str, Any]]] = []
        for nid in self._khop(anchor_id, lo, hi, direction):
            data = self._graph._get_node_properties(nid) or {}
            if free_label and not self._label_match(data, free_label):
                continue
            matched.append((nid, data))
        return True, self._project(q, free_var, matched, params)

    def _khop(self, anchor: str, lo: int, hi: int, direction: str) -> list[str]:
        """Bounded BFS from ``anchor``: node ids first reached in ``[lo, hi]`` hops.

        ``direction``: 'out' (successors), 'in' (predecessors), 'both' (neighbours).
        Each frontier expansion is one engine round-trip per node; bounded by
        ``hi`` so an L1 traversal stays a small, fast working set.
        """
        if direction == "out":
            step = self._graph.get_successors
        elif direction == "in":
            step = self._graph.get_predecessors
        else:
            step = self._graph.get_neighbors
        visited = {anchor}
        frontier = [anchor]
        out: list[str] = []
        for depth in range(1, hi + 1):
            nxt: list[str] = []
            for node in frontier:
                for nb in step(node):
                    if nb not in visited:
                        visited.add(nb)
                        nxt.append(nb)
                        if depth >= lo:
                            out.append(nb)
            frontier = nxt
            if not frontier:
                break
        return out

    def _exec_rel_merge(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret ``MATCH (a),(b) ... MERGE (a)-[:REL]->(b)`` as an O(1) edge add."""
        import re

        mm = re.search(
            r"MERGE\s*\(\s*(\w+)\s*\)\s*-\s*\[\s*(\w*)\s*:?\s*(\w+)?[^\]]*\]\s*->\s*\(\s*(\w+)\s*\)",
            q,
            re.I,
        )
        if not mm:
            return False, []
        src_var, rel_var, rel, tgt_var = (
            mm.group(1),
            mm.group(2),
            (mm.group(3) or "RELATED"),
            mm.group(4),
        )

        # Resolve each var's id from WHERE (``v.id = $param``) or inline
        # (``(v:Label {id: $param})``).
        idmap: dict[str, Any] = {}
        for mv in re.finditer(r"(\w+)\.id\s*=\s*\$(\w+)", q, re.I):
            idmap[mv.group(1)] = params.get(mv.group(2))
        for mv in re.finditer(
            r"\(\s*(\w+)\s*(?::\w+)?\s*\{\s*id\s*:\s*\$(\w+)\s*\}", q, re.I
        ):
            idmap[mv.group(1)] = params.get(mv.group(2))

        src_id, tgt_id = idmap.get(src_var), idmap.get(tgt_var)
        if not src_id or not tgt_id:
            return False, []  # unrecognised shape → defer to legacy

        # Persist edge properties (confidence, source, bitemporal stamps, …) from
        # the SET clause and inline relationship props — not just rel_type — so the
        # durable L1 edge carries the same data as the compute edge (KG-2.7 parity).
        edge_props: dict[str, Any] = {"rel_type": rel}
        if rel_var:
            for sm in re.finditer(
                rf"\b{rel_var}\.(\w+)\s*=\s*(\$\w+|'[^']*'|\"[^\"]*\"|-?[\d.]+)",
                q,
                re.I,
            ):
                edge_props[sm.group(1)] = self._coerce_literal(sm.group(2), params)
        try:
            self._graph.add_edge(src_id, tgt_id, edge_props)
        except Exception:  # noqa: BLE001
            pass
        return True, []

    def _exec_merge_node(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret ``MERGE (n:Label {id: $id}) [SET ...]`` as an upsert."""
        import re

        m = re.search(
            r"MERGE\s*\(\s*(\w+)\s*:(\w+)\s*\{\s*id\s*:\s*\$(\w+)\s*\}\s*\)", q, re.I
        )
        if not m:
            return False, []
        var, label, id_param = m.group(1), m.group(2), m.group(3)
        nid = params.get(id_param)
        if nid is None:
            return True, []

        existing = (
            self._graph._get_node_properties(nid) if self._graph.has_node(nid) else {}
        )
        merged = dict(existing or {})
        merged["node_type"] = label
        # The engine's label index (``get_nodes_by_label``) keys off ``label`` /
        # ``type`` / ``labels`` — NOT ``node_type``. Stamp the MERGE label so a
        # later ``MATCH (n:Label) WHERE …`` scan finds the node. (CONCEPT:KG-2.9g)
        merged["label"] = label

        ms = re.search(r"\bSET\b(.+?)$", q, re.I | re.S)
        if ms:
            for frag in self._split_top_level(ms.group(1)):
                if "=" not in frag:
                    continue
                lhs, rhs = frag.split("=", 1)
                prop = lhs.strip()
                prop = prop[len(var) + 1 :] if prop.startswith(var + ".") else prop
                # Bulk-write templates backtick-quote keys (``n.`name` = $name``);
                # strip them so the stored property is ``name``, not `` `name` ``.
                prop = prop.strip().strip("`")
                merged[prop] = self._coerce_literal(rhs, params)

        self._graph.add_node(nid, merged)
        return True, []

    # --- Operational Cypher subset interpreter ---------------------------

    def _legacy_execute(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Original best-effort reader: id lookup / label-param / return-all."""
        if "id" in params:
            node_id = params["id"]
            if self._graph.has_node(node_id):
                data = self._graph._get_node_properties(node_id)
                data["id"] = node_id
                return [data]
            return []

        if "label" in params:
            label = params["label"]
            results = []
            for nid in self._graph._get_all_nodes():
                data = self._graph._get_node_properties(nid)
                if data.get("label") == label:
                    entry = dict(data)
                    entry["id"] = nid
                    results.append(entry)
            return results

        # CONCEPT:ORCH-1.40 (hardening) — NEVER silently return the entire graph for an
        # unparsed query. That over-match was the `graph_context list` "garbage" bug and a
        # latent correctness/cost hazard for every caller. Default to empty; require an
        # explicit opt-in (KG_ALLOW_FULL_SCAN=true) for a rare deliberate full enumeration.
        # Fresh AgentConfig() (not the import-time singleton) so a runtime
        # override is honored on this rare, deliberate full-scan path — same
        # live-env contract as MergePolicy.from_env.
        from agent_utilities.core.config import AgentConfig

        if AgentConfig().kg_allow_full_scan:
            results = []
            for nid in self._graph._get_all_nodes():
                data = self._graph._get_node_properties(nid)
                entry = dict(data)
                entry["id"] = nid
                results.append(entry)
            return results
        logger.warning(
            "epistemic_graph backend: unscoped query (no id/label, no parseable WHERE) — "
            "refusing full-graph scan, returning []. Set KG_ALLOW_FULL_SCAN=true to override."
        )
        return []

    @staticmethod
    def _label_match(data: dict[str, Any], label: str) -> bool:
        # ``type`` is the system's canonical label key (the Rust node store
        # normalises node_type→type on read-back; see graph_compute
        # ``props.get("type", props.get("node_type", ...))``). Check all
        # conventions so a label filter matches regardless of writer path.
        # Compare case-insensitively: schema labels are PascalCase (``Prompt``)
        # while the node-type enum values are lowercase (``prompt``).
        target = label.lower()
        if target in (
            str(data.get("type", "")).lower(),
            str(data.get("node_type", "")).lower(),
            str(data.get("label", "")).lower(),
        ):
            return True
        labels = data.get("labels")
        return isinstance(labels, list | tuple) and any(
            str(lbl).lower() == target for lbl in labels
        )

    @staticmethod
    def _coerce_literal(raw: str, params: dict[str, Any]) -> Any:
        """Resolve a Cypher value token to a Python value."""
        raw = raw.strip()
        if raw.startswith("$"):
            return params.get(raw[1:])
        if (raw.startswith("'") and raw.endswith("'")) or (
            raw.startswith('"') and raw.endswith('"')
        ):
            return raw[1:-1]
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        if raw.lower() in ("null", "current_timestamp()"):
            import datetime

            if raw.lower() == "null":
                return None
            return datetime.datetime.now(datetime.UTC).isoformat()
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw

    def _node_value(self, nid: str, data: dict[str, Any], prop: str) -> Any:
        return nid if prop == "id" else data.get(prop)

    def _exec_node_match(
        self, q: str, params: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Interpret a single-node MATCH. Returns (handled, rows)."""
        import re

        m = re.search(r"MATCH\s*\(\s*(\w+)\s*(?::(\w+))?\s*(\{[^}]*\})?\s*\)", q, re.I)
        if not m:
            return False, []
        var, label, inline = m.group(1), m.group(2), m.group(3)
        qu = q.upper()

        inline_conds: list[tuple[str, str, Any]] = []
        if inline:
            for pair in inline.strip("{}").split(","):
                if ":" not in pair:
                    continue
                k, v = pair.split(":", 1)
                inline_conds.append((k.strip(), "=", self._coerce_literal(v, params)))

        # WHERE is parsed into DNF — a list of AND-groups that are OR-combined (a row
        # matches if ANY group matches). An inline ``{prop:..}`` pattern ANDs into
        # every group. A genuinely unsupported shape returns None and is logged
        # loudly rather than silently returning [] — the old behaviour masqueraded
        # "I can't parse this" as "no rows", a debugging footgun. (CONCEPT:KG-2.0)
        where_groups: list[list[tuple[str, str, Any]]] = [inline_conds]
        mw = re.search(
            r"\bWHERE\b(.+?)(?:\bSET\b|\bRETURN\b|\bDETACH\b|\bDELETE\b|$)",
            q,
            re.I | re.S,
        )
        if mw:
            groups = self._parse_where_or(mw.group(1), var, params)
            if groups is None:
                logger.warning(
                    "epistemic_graph backend: unsupported WHERE shape, deferring to "
                    "legacy reader (may under-match): %s",
                    mw.group(1).strip()[:200],
                )
                return False, []  # unsupported WHERE → defer to legacy
            where_groups = [inline_conds + g for g in groups]

        # Gather matching nodes. Fast path: a single AND-group with an ``id = <value>``
        # equality resolves one node directly (O(1)) instead of scanning the whole
        # graph (O(N)). Ingestion issues many id-keyed MATCH/SET upserts, so without
        # this every write is O(N) → ingestion is O(N²) and degrades as the graph
        # grows. A disjunction (OR) can't use the fast path → full scan.
        # (CONCEPT:KG-2.8 ingestion throughput)
        matched: list[tuple[str, dict[str, Any]]] = []
        id_val = None
        if len(where_groups) == 1:
            id_val = next(
                (
                    v
                    for (p, op, v) in where_groups[0]
                    if p == "id" and op == "=" and v is not None
                ),
                None,
            )

        # Fast path: a bare aggregate count (``MATCH (n) RETURN count(n)``) with
        # no label/WHERE/SET/DELETE needs only the node *count* — never per-node
        # properties. The general scan below issues one ``_get_node_properties``
        # UDS round-trip per node, so an unfiltered count is O(N) round-trips and
        # gets slower as the graph grows (a full ``count(*)`` was taking minutes
        # on an accumulated graph). Resolve it from the id list directly.
        # (CONCEPT:KG-2.8 ingestion throughput)
        cnt_m = re.search(r"\bRETURN\b\s+count\s*\(\s*\*?\s*\w*\s*\)", q, re.I)
        if (
            id_val is None
            and not any(where_groups)
            and not label
            and cnt_m
            and not _has_kw(qu, "SET")
            and not _has_kw(qu, "DELETE")
        ):
            cnt = len(self._graph._get_all_nodes())
            alias_m = re.search(r"count\s*\([^)]*\)\s+as\s+(\w+)", q, re.I)
            alias = alias_m.group(1) if alias_m else "count"
            return True, [{alias: cnt}]

        if id_val is not None:
            if self._graph.has_node(id_val):
                data = self._graph._get_node_properties(id_val) or {}
                if (not label or self._label_match(data, label)) and self._eval_groups(
                    id_val, data, where_groups
                ):
                    matched.append((id_val, data))
        else:
            # Engine-side labeled fetch (CONCEPT:KG-2.51): a label-scoped MATCH
            # fetches only that label's nodes — never the whole graph — and pushes
            # the LIMIT down for a pure read (no WHERE/SET/DELETE could drop a
            # match and under-return). A bare (label-less) MATCH still does the
            # single-round-trip full scan (guarded elsewhere). Before this, every
            # label scan materialized all ~80K nodes' properties over the wire.
            rows: Any
            if label:
                lim_m = re.search(r"\bLIMIT\s+(\$?\w+)", q, re.I)
                limit_n = 0
                if lim_m:
                    tok = lim_m.group(1)
                    try:
                        limit_n = (
                            int(params.get(tok[1:], 0))
                            if tok.startswith("$")
                            else int(tok)
                        )
                    except (TypeError, ValueError):
                        limit_n = 0
                pure_read = (
                    limit_n > 0
                    and not any(where_groups)
                    and not _has_kw(qu, "SET")
                    and not _has_kw(qu, "DELETE")
                )
                try:
                    rows = self._graph.get_nodes_by_label(
                        label, limit_n if pure_read else 0
                    )
                except Exception:  # noqa: BLE001 — fall back to the legacy full scan
                    rows = self._graph._get_all_nodes_with_properties()
            else:
                # Full-graph scan: fetch all nodes WITH their properties in a single
                # round-trip (one ``_get_node_properties`` per node is an N+1 that
                # cost ~45s on a 40K-node graph). (CONCEPT:KG-2.8)
                rows = self._graph._get_all_nodes_with_properties()
            for nid, data in rows:
                data = data or {}
                if label and not self._label_match(data, label):
                    continue
                if not self._eval_groups(nid, data, where_groups):
                    continue
                matched.append((nid, data))

        if "DETACH DELETE" in qu or re.search(rf"\bDELETE\s+{var}\b", q, re.I):
            for nid, _ in matched:
                self._graph.remove_node(nid)
                self._embeddings.pop(nid, None)
            return True, []

        ms = re.search(r"\bSET\b(.+?)(?:\bRETURN\b|$)", q, re.I | re.S)
        if ms:
            assigns: dict[str, Any] = {}
            for frag in self._split_top_level(ms.group(1)):
                if "=" not in frag:
                    continue
                lhs, rhs = frag.split("=", 1)
                prop = lhs.strip()
                prop = prop[len(var) + 1 :] if prop.startswith(var + ".") else prop
                assigns[prop] = self._coerce_literal(rhs, params)
            for nid, data in matched:
                merged = dict(data)
                merged.update(assigns)
                self._graph.add_node(nid, merged)
                data.update(assigns)

        return True, self._project(q, var, matched, params)

    def _parse_where(
        self, text: str, var: str, params: dict[str, Any]
    ) -> list[tuple[str, str, Any]] | None:
        """Parse a conjunctive WHERE clause. None ⇒ unsupported shape."""
        import re

        conds: list[tuple[str, str, Any]] = []
        for clause in re.split(r"\bAND\b", text, flags=re.I):
            clause = clause.strip()
            if not clause:
                continue
            m_in = re.match(rf"{var}\.(\w+)\s+IN\s+\[(.*?)\]", clause, re.I | re.S)
            m_null = re.match(rf"{var}\.(\w+)\s+IS\s+(NOT\s+)?NULL", clause, re.I)
            m_has = re.match(rf"{var}\.(\w+)\s+CONTAINS\s+(.+)", clause, re.I)
            m_eq = re.match(rf"{var}\.(\w+)\s*=\s*(.+)", clause, re.I)
            if m_in:
                vals = [
                    self._coerce_literal(t, params)
                    for t in m_in.group(2).split(",")
                    if t.strip()
                ]
                conds.append((m_in.group(1), "IN", vals))
            elif m_null:
                conds.append(
                    (m_null.group(1), "NOTNULL" if m_null.group(2) else "ISNULL", None)
                )
            elif m_has:
                conds.append(
                    (
                        m_has.group(1),
                        "CONTAINS",
                        self._coerce_literal(m_has.group(2), params),
                    )
                )
            elif m_eq:
                conds.append(
                    (m_eq.group(1), "=", self._coerce_literal(m_eq.group(2), params))
                )
            else:
                return None
        return conds

    def _parse_where_or(
        self, text: str, var: str, params: dict[str, Any]
    ) -> list[list[tuple[str, str, Any]]] | None:
        """Parse a WHERE clause into DNF — OR of AND-groups.

        Splits on top-level ``OR`` and parses each disjunct with the conjunctive
        ``_parse_where``. Returns a list of AND-groups (a row matches if ANY group
        matches), or ``None`` if any disjunct is an unsupported shape. With no
        top-level ``OR`` this returns a single group, identical to the prior
        AND-only behaviour.
        """
        groups: list[list[tuple[str, str, Any]]] = []
        for part in self._split_or(text):
            parsed = self._parse_where(part, var, params)
            if parsed is None:
                return None
            groups.append(parsed)
        return groups or None

    @staticmethod
    def _split_or(text: str) -> list[str]:
        """Split on top-level ``OR`` (case-insensitive), respecting (), [], {}, quotes."""
        import re

        out: list[str] = []
        buf: list[str] = []
        depth = 0
        quote: str | None = None
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
                i += 1
            elif ch in "'\"":
                quote = ch
                buf.append(ch)
                i += 1
            elif ch in "([{":
                depth += 1
                buf.append(ch)
                i += 1
            elif ch in ")]}":
                depth -= 1
                buf.append(ch)
                i += 1
            elif depth == 0 and (m := re.match(r"\s+OR\s+", text[i:], re.I)):
                out.append("".join(buf))
                buf = []
                i += m.end()
            else:
                buf.append(ch)
                i += 1
        if buf:
            out.append("".join(buf))
        return [s for s in out if s.strip()]

    def _eval_groups(
        self,
        nid: str,
        data: dict[str, Any],
        groups: list[list[tuple[str, str, Any]]],
    ) -> bool:
        """DNF evaluation: the row matches if ANY AND-group matches."""
        return any(self._eval_conds(nid, data, g) for g in groups)

    def _eval_conds(
        self, nid: str, data: dict[str, Any], conds: list[tuple[str, str, Any]]
    ) -> bool:
        for prop, op, val in conds:
            actual = self._node_value(nid, data, prop)
            if op == "=":
                if actual != val:
                    return False
            elif op == "IN":
                if actual not in val:
                    return False
            elif op == "CONTAINS":
                if actual is None or val is None or str(val) not in str(actual):
                    return False
            elif op == "ISNULL":
                if actual is not None:
                    return False
            elif op == "NOTNULL":
                if actual is None:
                    return False
        return True

    @staticmethod
    def _split_top_level(text: str) -> list[str]:
        """Split on commas not nested inside brackets/quotes."""
        out, buf, depth, quote = [], [], 0, None
        for ch in text:
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
                buf.append(ch)
            elif ch in "[{(":
                depth += 1
                buf.append(ch)
            elif ch in "]})":
                depth -= 1
                buf.append(ch)
            elif ch == "," and depth == 0:
                out.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
        return [s for s in out if s.strip()]

    def _project(
        self,
        q: str,
        var: str,
        matched: list[tuple[str, dict[str, Any]]],
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        import re

        m_ret = re.search(r"\bRETURN\b(.+?)$", q, re.I | re.S)
        if not m_ret:
            return []
        ret = m_ret.group(1)

        limit = None
        m_lim = re.search(r"\bLIMIT\s+(\$?\w+)", ret, re.I)
        if m_lim:
            tok = m_lim.group(1)
            limit = params.get(tok[1:]) if tok.startswith("$") else int(tok)
            ret = ret[: m_lim.start()]
        ret = re.sub(r"\bORDER\s+BY\b.*$", "", ret, flags=re.I | re.S)

        # ``RETURN DISTINCT ...`` — dedupe the projected rows (KG-2.63).
        distinct = False
        m_dist = re.match(r"\s*DISTINCT\b", ret, re.I)
        if m_dist:
            distinct = True
            ret = ret[m_dist.end() :]

        items = self._split_top_level(ret)

        def _project_item(it: str, nid: str, data: dict[str, Any]) -> dict[str, Any]:
            """Project a single non-aggregate RETURN item to its column(s)."""
            it = it.strip()
            cols: dict[str, Any] = {}
            m_alias = re.match(rf"{var}\.(\w+)\s+as\s+(\w+)", it, re.I) or re.match(
                rf"{var}\.(\w+)", it, re.I
            )
            if m_alias and "." in it:
                prop = m_alias.group(1)
                value = self._node_value(nid, data, prop)
                if m_alias.lastindex and m_alias.lastindex >= 2:
                    cols[m_alias.group(2)] = value
                else:
                    # No explicit ``AS`` alias: expose both the standard Cypher
                    # column name (``var.prop``) and the bare prop name so callers
                    # using either convention resolve it.
                    cols[f"{var}.{prop}"] = value
                    cols.setdefault(prop, value)
            elif it == var or re.match(rf"{var}\s+as\s+\w+", it, re.I):
                # Bare ``RETURN v`` → a single column named ``v`` holding the full
                # node dict (Cypher semantics; callers read ``res["v"]``).
                full = dict(data)
                full["id"] = nid
                alias_m = re.match(rf"{var}\s+as\s+(\w+)", it, re.I)
                cols[alias_m.group(1) if alias_m else var] = full
            else:
                cols[it] = None
            return cols

        # Cypher has no explicit GROUP BY: aggregates (count/sum/avg/min/max)
        # collapse over the grouping keys, which are exactly the *non-aggregate*
        # return items. Partition the items so ``RETURN n.kind, count(*)`` groups
        # by ``n.kind`` instead of collapsing to one total. (KG-2.63)
        _agg_re = re.compile(
            r"\s*(count|sum|avg|min|max)\s*\(\s*(\*|[\w.]*)\s*\)\s*(?:as\s+(\w+))?",
            re.I,
        )
        aggregates = [(it, _agg_re.match(it.strip())) for it in items]
        agg_specs = [m for _it, m in aggregates if m]
        group_items = [it for it, m in aggregates if not m]

        if agg_specs:

            def _agg_value(
                spec: re.Match[str], bucket: list[tuple[str, dict[str, Any]]]
            ) -> tuple[str, Any]:
                fn = spec.group(1).lower()
                arg = spec.group(2)
                alias = spec.group(3) or (fn if arg in ("", "*") else f"{fn}({arg})")
                if fn == "count":
                    return alias, len(bucket)
                prop = arg.split(".", 1)[1] if "." in arg else arg
                vals = [
                    v
                    for nid2, data2 in bucket
                    if isinstance(
                        (v := self._node_value(nid2, data2, prop)), int | float
                    )
                ]
                if not vals:
                    return alias, None
                agg = {
                    "sum": sum(vals),
                    "avg": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
                return alias, agg[fn]

            def _agg_row(
                gcols: dict[str, Any], bucket: list[tuple[str, dict[str, Any]]]
            ) -> dict[str, Any]:
                row = dict(gcols)
                for spec in agg_specs:
                    alias, val = _agg_value(spec, bucket)
                    row[alias] = val
                return row

            if not group_items:
                return [_agg_row({}, matched)]

            buckets: dict[str, tuple[dict[str, Any], list]] = {}
            order: list[str] = []
            for nid, data in matched:
                gcols: dict[str, Any] = {}
                for it in group_items:
                    gcols.update(_project_item(it, nid, data))
                key = json.dumps(gcols, sort_keys=True, default=str)
                if key not in buckets:
                    buckets[key] = (gcols, [])
                    order.append(key)
                buckets[key][1].append((nid, data))
            grouped = [_agg_row(*buckets[k]) for k in order]
            return grouped[:limit] if limit is not None else grouped

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for nid, data in matched:
            row: dict[str, Any] = {}
            for it in items:
                row.update(_project_item(it, nid, data))
            if distinct:
                key = json.dumps(row, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
            rows.append(row)

        if limit is not None:
            rows = rows[:limit]
        return rows

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute batch operations.

        The bulk-write callers (``ingest_external_batch``, ``write_batch``) emit the
        Neo4j/AGE idiom ``UNWIND $batch AS row MERGE (n:Label {id: row.id}) SET
        n.`k` = row.`k` …``. This adapter has no UNWIND engine, so translate the
        template ONCE into the per-row ``$param`` shape the MERGE-node/MERGE-rel
        interpreters already handle (``row.`k```/``row.k`` → ``$k``) and run it per
        row. Without this the whole batch silently no-ops (the raw ``row.id``
        template matches no interpreter) — ingested code/entities never land.
        (CONCEPT:KG-2.9g)
        """
        per_row = self._unwind_to_per_row(query)
        results = []
        for params in batch:
            results.extend(self.execute(per_row, params))
        return results

    @staticmethod
    def _unwind_to_per_row(query: str) -> str:
        """Strip an ``UNWIND $batch AS row`` header and rewrite ``row.<k>`` /
        ``row.`<k>``` references to ``$<k>`` so each row runs as a normal
        parameterized statement. A non-UNWIND query is returned unchanged."""
        import re

        q = (query or "").strip()
        m = re.match(r"(?is)^UNWIND\s+\$batch\s+AS\s+row\b(.*)$", q)
        if not m:
            return query
        body = m.group(1).strip()
        body = re.sub(r"row\.`([^`]+)`", r"$\1", body)
        body = re.sub(r"row\.(\w+)", r"$\1", body)
        return body

    def create_schema(self) -> None:
        """Initialize schema metadata representation."""
        # Simple schema metadata tracking for in-memory backend
        self._schema_created = True

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store an embedding for a node — in the engine HNSW AND the local cache.

        The engine index (CONCEPT:KG-2.0) is what makes ``semantic_search`` O(log N)
        and survives across processes; before this, embeddings only lived in the
        per-process ``_embeddings`` dict, so on the served/restarted graph the
        index was empty and retrieval fell back to an O(N) full-graph scan.
        """
        self._embeddings[node_id] = embedding  # write-through cache (single-proc/tests)
        try:
            self._graph.add_embedding(node_id, embedding)
        except Exception as e:  # noqa: BLE001 — engine index is best-effort
            logger.debug(
                "engine add_embedding failed for %s (cache kept): %s", node_id, e
            )

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        """Vector search — engine HNSW first (O(log N)), local cosine as fallback."""
        # Preferred: the engine's native HNSW. Scales and works after a restart.
        try:
            hits = self._graph.semantic_search(query_embedding, n_results)
            results: list[dict[str, Any]] = []
            for item in hits or []:
                if isinstance(item, list | tuple) and len(item) >= 2:
                    node_id, score = str(item[0]), float(item[1])
                elif isinstance(item, dict):
                    node_id, score = (
                        str(item.get("id", "")),
                        float(item.get("_similarity", item.get("score", 0.0)) or 0.0),
                    )
                else:
                    continue
                if not node_id:
                    continue
                data = self._graph._get_node_properties(node_id) or {}
                data["id"] = node_id
                data["_similarity"] = score
                results.append(data)
            if results:
                return results
        except Exception as e:  # noqa: BLE001 — fall back to local cosine
            logger.debug("engine semantic_search failed, using local cache: %s", e)

        # Fallback: in-process cosine over the local cache (single-proc / tests).
        if not self._embeddings:
            return []

        import numpy as np

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scores: list[tuple[str, float]] = []
        for node_id, emb in self._embeddings.items():
            emb_vec = np.array(emb)
            emb_norm = np.linalg.norm(emb_vec)
            if emb_norm == 0:
                continue
            similarity = float(np.dot(query_vec, emb_vec) / (query_norm * emb_norm))
            scores.append((node_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for node_id, score in scores[:n_results]:
            if self._graph.has_node(node_id):
                data = self._graph._get_node_properties(node_id)
                data["id"] = node_id
                data["_similarity"] = score
                results.append(data)

        return results

    def hydrate_engine_embeddings(self, batch_log_every: int = 5000) -> int:
        """One-time backfill: index existing node ``embedding`` properties into the
        engine HNSW. Embeddings have long been stored as node properties but never
        registered in the index, so legacy graphs need a single pass to make
        ``semantic_search`` fast. Reads from the graph (no re-embedding); a single
        full scan is acceptable for a one-shot migration. Returns the count indexed.
        """
        count = 0
        for nid, props in self._graph._get_all_nodes_with_properties():
            emb = (props or {}).get("embedding")
            if not emb:
                continue
            try:
                self._graph.add_embedding(nid, list(emb))
                count += 1
                if count % batch_log_every == 0:
                    logger.info("hydrate_engine_embeddings: indexed %d so far", count)
            except Exception as e:  # noqa: BLE001 — best-effort per node
                logger.debug("hydrate add_embedding failed for %s: %s", nid, e)
        logger.info("hydrate_engine_embeddings: indexed %d embeddings into HNSW", count)
        return count

    def prune(self, criteria: dict[str, Any]) -> None:
        """Prune nodes matching criteria."""
        to_remove = []
        for nid in self._graph._get_all_nodes():
            data = self._graph._get_node_properties(nid)
            match = all(data.get(k) == v for k, v in criteria.items())
            if match:
                to_remove.append(nid)

        for nid in to_remove:
            self._graph.remove_node(nid)
            self._embeddings.pop(nid, None)

        logger.info("Pruned %d nodes", len(to_remove))

    def close(self) -> None:
        """Reset the in-memory graph."""
        from ..core.graph_compute import GraphComputeEngine

        self._graph = GraphComputeEngine(backend_type="rust")
        self._embeddings.clear()

    # --- Extended API ---

    def health_check(self) -> bool:
        """Always healthy for in-memory backend."""
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        return {
            "backend": "memory",
            "nodes": self._graph.node_count(),
            "edges": self._graph.edge_count(),
            "embeddings": len(self._embeddings),
        }

    # --- Node/Edge Operations ---

    def add_node(self, node_id: str, label: str = "", **properties: Any) -> None:
        """Add a node to the graph."""
        props = {"label": label, **properties}
        self._graph.add_node(node_id, props)
        self._node_counter += 1

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str = "",
        **properties: Any,
    ) -> None:
        """Add an edge between two nodes."""
        props = {"rel_type": rel_type, **properties}
        self._graph.add_edge(source, target, props)

    def get_node_properties(self, node_id: str) -> dict[str, Any] | None:
        """Return a node's current properties, or ``None`` if it doesn't exist.

        Cheap in-memory read used by the Company Brain write-path guard for
        field-level survivorship (CONCEPT:KG-2.6). Returns ``{}`` for a node that
        exists with no properties; ``None`` lets the guard fall back to
        node-level arbitration.
        """
        try:
            if not self._graph.has_node(node_id):
                return None
            props = self._graph._get_node_properties(node_id)
            return dict(props) if isinstance(props, dict) else {}
        except Exception:  # pragma: no cover - read best-effort
            return None

    # --- Persistence ---

    def save_to_json(self, path: str) -> None:
        """Serialize the graph to a JSON file."""
        graph_json = self._graph.to_json()
        data = json.loads(graph_json)
        data["_embeddings"] = {k: v for k, v in self._embeddings.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Graph saved to %s", path)

    def load_from_json(self, path: str) -> None:
        """Deserialize the graph from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        embeddings = data.pop("_embeddings", {})
        self._graph.from_json(json.dumps(data))
        self._embeddings = embeddings
        logger.info(
            "Graph loaded from %s (%d nodes, %d edges)",
            path,
            self._graph.node_count(),
            self._graph.edge_count(),
        )
