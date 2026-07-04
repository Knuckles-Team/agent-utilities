"""CONCEPT:EG-KG.storage.nonblocking-checkpoint/2.16 — buffered backend batches per-node writes into bulk RPCs (#1)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.pipeline import _BatchedBackend


class _BulkBackend:
    """Backend whose wrapped ``_graph`` exposes a bulk op (like the real engine)."""

    def __init__(self):
        self.bulk_calls: list[list[dict]] = []
        self.per_node = 0
        self.per_edge = 0

        outer = self

        class _Graph:
            def bulk_mutate(self, ops):
                outer.bulk_calls.append(ops)

        self._graph = _Graph()

    def add_node(self, node_id, **props):
        self.per_node += 1

    def add_edge(self, source, target, **props):
        self.per_edge += 1


def test_writes_are_flushed_as_bulk_ops_nodes_before_edges():
    be = _BulkBackend()
    bw = _BatchedBackend(be, batch_size=1000)
    for i in range(5):
        bw.add_node(f"code:{i}", type="Code", language="java")
    bw.add_edge("code:0", "code:1", rel_type="CALLS")
    # Nothing flushed below batch_size yet.
    assert be.bulk_calls == []
    bw.flush()

    # One bulk call for nodes, one for edges; per-item path NOT used.
    assert be.per_node == 0 and be.per_edge == 0
    assert len(be.bulk_calls) == 2
    node_ops, edge_ops = be.bulk_calls[0], be.bulk_calls[1]
    assert all(o["op"] == "add_node" for o in node_ops) and len(node_ops) == 5
    assert node_ops[0]["properties"]["language"] == "java"
    assert all(o["op"] == "add_edge" for o in edge_ops) and len(edge_ops) == 1
    assert edge_ops[0]["source"] == "code:0"


def test_auto_flush_at_batch_size():
    be = _BulkBackend()
    bw = _BatchedBackend(be, batch_size=2)
    bw.add_node("a")
    bw.add_node("b")  # hits batch_size → auto-flush
    assert len(be.bulk_calls) == 1
    assert len(be.bulk_calls[0]) == 2


def test_falls_back_to_per_item_without_bulk_path():
    class _Plain:
        def __init__(self):
            self.nodes = 0
            self.edges = 0

        def add_node(self, node_id, **props):
            self.nodes += 1

        def add_edge(self, source, target, **props):
            self.edges += 1

    be = _Plain()
    bw = _BatchedBackend(be)  # no _graph → no bulk path
    bw.add_node("a")
    bw.add_edge("a", "b")
    bw.flush()
    assert be.nodes == 1 and be.edges == 1


def test_reads_delegate_to_wrapped_backend():
    class _R:
        def get_node_properties(self, nid):
            return {"id": nid}

    bw = _BatchedBackend(_R())
    assert bw.get_node_properties("x") == {"id": "x"}
