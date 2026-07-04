"""L1 backend support for the resolved code graph (CONCEPT:AU-KG.backend.declared-columns-so-schema).

Exercises EpistemicGraphBackend against an injected mutable fake compute graph
(no running engine): UNWIND-batch writes actually PERSIST, the MERGE label is
findable by ``get_nodes_by_label`` (label+WHERE reads), and the graph_code_nav
shapes — WHERE-anchored single-hop (find_references) and bounded var-length
(trace_call_graph / impact_of_change) — resolve the anchor by scan and walk.
"""

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


def _backend():
    b = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    b._graph = MutableFakeGraph()
    b._embeddings = {}
    b._node_counter = 0
    return b


def _names(rows):
    return sorted(r.get("name") for r in rows if r.get("name"))


def test_unwind_to_per_row_translation():
    out = EpistemicGraphBackend._unwind_to_per_row(
        "UNWIND $batch AS row MERGE (n:Code {id: row.id}) SET n.`name` = row.`name`"
    )
    assert out == "MERGE (n:Code {id: $id}) SET n.`name` = $name"
    # Non-UNWIND passes through unchanged.
    assert (
        EpistemicGraphBackend._unwind_to_per_row("MATCH (n) RETURN n")
        == "MATCH (n) RETURN n"
    )


def _seed(b):
    # call chain: top -> mid -> leaf (all :Code), via the exact UNWIND idiom the
    # bulk-writer emits.
    b.execute_batch(
        "UNWIND $batch AS row MERGE (n:Code {id: row.id}) SET n.`name` = row.`name`",
        [
            {"id": "top", "name": "top"},
            {"id": "mid", "name": "mid"},
            {"id": "leaf", "name": "leaf"},
        ],
    )
    b.execute_batch(
        "UNWIND $batch AS row MATCH (s {id: row.source}) MATCH (t {id: row.target}) MERGE (s)-[r:calls]->(t)",
        [{"source": "top", "target": "mid"}, {"source": "mid", "target": "leaf"}],
    )


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
