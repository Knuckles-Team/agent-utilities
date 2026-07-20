#!/usr/bin/python
from __future__ import annotations

"""Round-trip test: distill a subgraph to a pack, then import it back.

Validates that ``import_skill_graph_pack`` reconstructs the original node ids,
types, body text, and edges recorded in ``kg_manifest.json`` (CONCEPT:AU-AHE.optimization.physical-distillation-engine).
"""

import msgpack

from agent_utilities.knowledge_graph.distillation import (
    SkillGraphDistiller,
    import_skill_graph_pack,
)
from agent_utilities.knowledge_graph.distillation.skill_graph_importer import (
    _extract_body,
)


# Reuse the fake async client shape from the distiller test.
class _Nodes:
    def __init__(self, nodes, adj):
        self._nodes, self._adj = nodes, adj

    async def neighbors(self, nid):
        return list(self._adj.get(nid, []))

    async def properties(self, nid):
        return self._nodes.get(nid)


class _Edges:
    def __init__(self, edges):
        self._edges = edges

    async def list(self):
        return [(s, d, list(msgpack.packb(p))) for (s, d, p) in self._edges]


class _Graph:
    async def community_detection(self, resolution=1.0):
        return []

    async def semantic_search(self, emb, n_results=5):
        return []


class FakeClient:
    def __init__(self, nodes, edges):
        adj = {}
        for s, d, _ in edges:
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)
        self.nodes = _Nodes(nodes, {k: sorted(v) for k, v in adj.items()})
        self.edges = _Edges(edges)
        self.graph = _Graph()

    async def close(self):
        pass


# Minimal recipient engine capturing backend writes.
class FakeBackend:
    def __init__(self):
        self.nodes: dict = {}
        self.edges: list = []

    def add_node(self, nid, **props):
        self.nodes[nid] = props

    def add_edge(self, src, dst, **props):
        self.edges.append((src, dst, props))


class FakeEngine:
    def __init__(self):
        self.backend = FakeBackend()


def test_extract_body_strips_frontmatter_heading_and_related():
    md = (
        "---\ntitle: X\nkg_node_id: c:1\n---\n\n"
        "# X\n\nThe real body text.\n\n## Related\n- [Y](y.md)\n"
    )
    assert _extract_body(md) == "The real body text."


async def test_distill_then_import_round_trips(tmp_path):
    nodes = {
        "concept:sn": {
            "type": "Concept",
            "name": "ServiceNow",
            "summary": "ITSM platform.",
        },
        "concept:inc": {
            "type": "Concept",
            "name": "Incident",
            "summary": "Unplanned interruption.",
        },
    }
    edges = [("concept:sn", "concept:inc", {"rel_type": "RELATES_TO"})]

    # 1) Distill to a pack.
    distiller = SkillGraphDistiller(FakeClient(nodes, edges), graph_name="__t__")
    manifest = await distiller.distill(
        seed="concept:sn", depth=2, out_dir=str(tmp_path)
    )
    assert manifest["stats"]["files"] >= 2

    # 2) Import the pack into a fresh recipient engine.
    engine = FakeEngine()
    stats = import_skill_graph_pack(engine, str(tmp_path))

    # Original node ids + types reconstructed; Concept body → summary.
    assert "concept:sn" in engine.backend.nodes
    assert engine.backend.nodes["concept:sn"]["type"] == "Concept"
    assert "ITSM platform." in engine.backend.nodes["concept:sn"]["summary"]
    # The RELATES_TO edge survived the round trip.
    assert any(
        s == "concept:sn" and d == "concept:inc" and p.get("rel_type") == "RELATES_TO"
        for s, d, p in engine.backend.edges
    )
    assert stats["nodes"] >= 2 and stats["edges"] >= 1
