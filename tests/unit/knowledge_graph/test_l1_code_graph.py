"""L1 backend support for the resolved code graph (CONCEPT:AU-KG.backend.declared-columns-so-schema).

Exercises EpistemicGraphBackend against an injected mutable fake compute graph
(no running engine): UNWIND-batch writes actually PERSIST, the MERGE label is
findable by ``get_nodes_by_label`` (label+WHERE reads), and the graph_code_nav
shapes — WHERE-anchored single-hop (find_references) and bounded var-length
(trace_call_graph / impact_of_change) — resolve the anchor by scan and walk.
"""

import re

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.mcp.tools.query_tools import build_code_nav_query


class MutableFakeGraph:
    """Writable directed graph supporting the ops the L1 interpreter calls."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.succ: dict[str, list[str]] = {}
        self.pred: dict[str, list[str]] = {}
        self.edges: dict[tuple[str, str], dict] = {}

    def has_node(self, nid):
        return nid in self.nodes

    def add_node(self, nid, properties=None, **kw):
        p = dict(properties or {})
        p.update(kw)
        self.nodes[nid] = p

    def add_edge(self, src, tgt, properties=None, **kw):
        p = dict(properties or {})
        p.update(kw)
        if tgt not in self.succ.setdefault(src, []):
            self.succ[src].append(tgt)
        if src not in self.pred.setdefault(tgt, []):
            self.pred[tgt].append(src)
        self.edges[(src, tgt)] = p

    def get_successors(self, nid):
        return list(self.succ.get(nid, []))

    def get_predecessors(self, nid):
        return list(self.pred.get(nid, []))

    def get_neighbors(self, nid):
        return list(
            dict.fromkeys(self.get_successors(nid) + self.get_predecessors(nid))
        )

    def has_edge(self, src, tgt):
        return (src, tgt) in self.edges

    def _get_node_properties(self, nid):
        return dict(self.nodes.get(nid, {}))

    def _get_edge_properties(self, src, tgt):
        return dict(self.edges.get((src, tgt), {}))

    def _get_all_nodes(self):
        return list(self.nodes)

    def _get_all_nodes_with_properties(self):
        return [(n, dict(p)) for n, p in self.nodes.items()]

    def get_nodes_by_label(self, label, limit=0):
        out = []
        for n, p in self.nodes.items():
            labels = p.get("labels") if isinstance(p.get("labels"), list) else []
            if (
                label in (p.get("label"), p.get("type"), p.get("node_type"))
                or label in labels
            ):
                out.append((n, dict(p)))
        return out[:limit] if limit else out

    def _project_rows(self, rows, ret):
        """Project (node_id, props) pairs through a RETURN clause's column list."""
        out = []
        for nid, props in rows:
            row = {}
            for item in ret.split(","):
                item = item.strip()
                mi = re.match(r"(\w+)\.(\w+)\s+AS\s+(\w+)", item, re.I)
                if mi:
                    _, prop, alias = mi.groups()
                    row[alias] = props.get(prop)
                    continue
                mi = re.match(r"(\w+)\s+AS\s+(\w+)", item, re.I)
                if mi:
                    _, alias = mi.groups()
                    row[alias] = nid
            out.append(row)
        return out

    def _reachable(self, seeds, *, forward, depth):
        """BFS over successors (forward) or predecessors (not forward), up to
        ``depth`` hops, excluding the seeds themselves — the fake's stand-in for
        the engine's bounded ``[:calls*1..depth]`` variable-length traversal."""
        step = self.get_successors if forward else self.get_predecessors
        frontier = set(seeds)
        found: dict[str, None] = {}
        for _ in range(depth):
            nxt: set[str] = set()
            for nid in frontier:
                for neighbor in step(nid):
                    if neighbor not in found and neighbor not in seeds:
                        found[neighbor] = None
                        nxt.add(neighbor)
            if not nxt:
                break
            frontier = nxt
        return list(found)

    def query_cypher(self, query):
        """Minimal stand-in for the native engine's Cypher executor, covering the
        shapes ``build_code_nav_query`` emits (CONCEPT:AU-P0-2):

        * ``find_definition``: ``MATCH (v:Label) WHERE v.prop = 'literal' RETURN ...``
        * ``find_references``: ``MATCH (caller:Code)-[:calls]->(def:Code) WHERE
          def.prop = 'literal' RETURN caller...`` (one-hop predecessors of the
          WHERE-anchored node)
        * ``trace_call_graph``/``impact_of_change``: the same shape with a bounded
          ``[:calls*1..N]`` variable-length edge, walking successors (forward,
          anchor on the LEFT var) or predecessors (anchor on the RIGHT var).

        Real callers hit the real engine; this fake exists so the test doesn't
        need a live one.
        """
        m = re.match(
            r"MATCH\s*\((\w+):(\w+)\)-\[:(\w+)(?:\*1\.\.(\d+))?\]->\((\w+):(\w+)\)\s*"
            r"WHERE\s*(\w+)\.(\w+)\s*=\s*'([^']*)'\s*"
            r"RETURN\s+(?:DISTINCT\s+)?(.+?)(?:\s+LIMIT\s+(\d+))?$",
            query,
            re.I,
        )
        if m:
            (
                var1,
                _label1,
                _rel,
                depth_str,
                var2,
                label2,
                where_var,
                wprop,
                wval,
                ret,
                limit,
            ) = m.groups()
            depth = int(depth_str) if depth_str else 1
            anchor_label = label2 if where_var == var2 else _label1
            anchors = {
                nid
                for nid, props in self.get_nodes_by_label(anchor_label, 0)
                if props.get(wprop) == wval
            }
            # where_var == var2 (the RIGHT-hand node, e.g. `def`/`t`) → return
            # var1 (upstream: predecessors). where_var == var1 (e.g. `s`) →
            # return var2 (downstream: successors).
            forward = where_var == var1
            result_ids = self._reachable(anchors, forward=forward, depth=depth)
            rows = [(nid, self._get_node_properties(nid)) for nid in result_ids]
            if limit:
                rows = rows[: int(limit)]
            return self._project_rows(rows, ret)

        m = re.match(
            r"MATCH\s*\((\w+):(\w+)\)\s*WHERE\s*(\w+)\.(\w+)\s*=\s*'([^']*)'\s*"
            r"RETURN\s+(.+?)(?:\s+LIMIT\s+(\d+))?$",
            query,
            re.I,
        )
        if not m:
            raise NotImplementedError(
                f"MutableFakeGraph.query_cypher: unsupported test shape: {query!r}"
            )
        _var, label, _wvar, wprop, wval, ret, limit = m.groups()
        rows = [
            (nid, props)
            for nid, props in self.get_nodes_by_label(label, 0)
            if props.get(wprop) == wval
        ]
        if limit:
            rows = rows[: int(limit)]
        return self._project_rows(rows, ret)


def _backend():
    b = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    b._graph = MutableFakeGraph()
    b._embeddings = {}
    b._node_counter = 0
    return b


def _names(rows):
    return sorted(r.get("name") for r in rows if r.get("name"))


def test_unwind_to_per_row_translation():
    """The UNWIND-to-per-row Python mutation compiler is retired on
    ``EpistemicGraphBackend`` (CONCEPT:AU-P0-2): batch writes must go through
    native ChangeEnvelope ingestion instead of a second Python mutation
    compiler — see ``execute_batch``'s docstring, and the same invariant is
    independently enforced by ``tests/unit/test_native_cypher_routing.py``
    (``retired`` tuple, which lists ``_unwind_to_per_row`` among the helper
    names that must never reappear in ``execute``/``execute_read``/
    ``execute_write``/``execute_batch``'s source). ``_unwind_to_per_row``
    now lives only on ``LadybugBackend`` (a different, contrib backend that
    still executes raw UNWIND batches itself)."""
    assert not hasattr(EpistemicGraphBackend, "_unwind_to_per_row")

    b = _backend()
    with pytest.raises(RuntimeError, match="ChangeEnvelope"):
        b.execute_batch(
            "UNWIND $batch AS row MERGE (n:Code {id: row.id}) SET n.`name` = row.`name`",
            [{"id": "x", "name": "x"}],
        )


def _seed(b):
    # call chain: top -> mid -> leaf (all :Code). ``execute_batch`` (raw Cypher
    # UNWIND) is deliberately rejected by EpistemicGraphBackend now — batch
    # writes must go through native ChangeEnvelope ingestion, not a second
    # Python mutation compiler (see execute_batch's docstring). This helper only
    # sets up fixture state for the *read*-side tests below, so it writes
    # directly to the injected fake graph instead, reproducing exactly what the
    # old UNWIND MERGE achieved (a `Code`-labeled node per row, `calls` edges).
    for row in (
        {"id": "top", "name": "top"},
        {"id": "mid", "name": "mid"},
        {"id": "leaf", "name": "leaf"},
    ):
        b._graph.add_node(row["id"], properties={**row, "label": "Code"})
    for source, target in (("top", "mid"), ("mid", "leaf")):
        b._graph.add_edge(source, target, properties={"rel_type": "calls"})


def test_batch_write_persists_and_is_label_findable():
    b = _backend()
    _seed(b)
    # Writes landed AND carry a label the engine index can find.
    assert b._graph.has_node("mid")
    assert {n for n, _ in b._graph.get_nodes_by_label("Code")} == {"top", "mid", "leaf"}


def test_find_definition_label_where():
    b = _backend()
    _seed(b)
    cy, p = build_code_nav_query(action="find_definition", symbol="mid")
    assert _names(b.execute(cy, p)) == ["mid"]


def test_find_references_where_anchored_single_hop():
    b = _backend()
    _seed(b)
    cy, p = build_code_nav_query(action="find_references", symbol="mid")
    assert _names(b.execute(cy, p)) == ["top"]  # top calls mid


def test_trace_call_graph_where_anchored_varlen():
    b = _backend()
    _seed(b)
    cy, p = build_code_nav_query(action="trace_call_graph", symbol="top", depth=3)
    assert _names(b.execute(cy, p)) == ["leaf", "mid"]  # transitive callees


def test_impact_of_change_where_anchored_varlen():
    b = _backend()
    _seed(b)
    cy, p = build_code_nav_query(action="impact_of_change", symbol="leaf", depth=3)
    assert _names(b.execute(cy, p)) == [
        "mid",
        "top",
    ]  # transitive callers (blast radius)
