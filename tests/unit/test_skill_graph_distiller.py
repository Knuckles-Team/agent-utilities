#!/usr/bin/python
from __future__ import annotations

"""Unit tests for the KG→skill-graph distiller (CONCEPT:AU-AHE.optimization.physical-distillation-engine, read side).

Uses an in-memory fake async client that mimics the ``EpistemicGraphClient``
namespace surface (``nodes``/``edges``/``graph``), so the distiller logic is
exercised without a live epistemic-graph daemon. A live round-trip belongs in a
``live``-marked test.
"""

import json

import msgpack

from agent_utilities.knowledge_graph.distillation import SkillGraphDistiller

# ── fake client mimicking the namespaced async client ─────────────────────


class _Nodes:
    def __init__(self, nodes: dict, adj: dict) -> None:
        self._nodes = nodes
        self._adj = adj

    async def neighbors(self, node_id):
        return list(self._adj.get(node_id, []))

    async def properties(self, node_id):
        return self._nodes.get(node_id)


class _Edges:
    def __init__(self, edges: list) -> None:
        self._edges = edges

    async def list(self):
        # Mirror the real client: third element is a msgpack blob.
        return [(s, d, list(msgpack.packb(props))) for (s, d, props) in self._edges]


class _Graph:
    def __init__(self, communities, seed_hits) -> None:
        self._communities = communities
        self._seed_hits = seed_hits

    async def community_detection(self, resolution=1.0):
        return self._communities

    async def semantic_search(self, emb, n_results=5):
        return self._seed_hits[:n_results]


def _make_subgraph_fn(nodes, edges):
    """Mimics the engine's batched GetSubgraph (decoded nodes + induced edges)."""

    async def get_subgraph(node_ids):
        idset = set(node_ids)
        return {
            "nodes": [
                {"id": nid, "properties": nodes[nid]}
                for nid in node_ids
                if nid in nodes
            ],
            "edges": [
                {"source": s, "target": d, "properties": p}
                for (s, d, p) in edges
                if s in idset and d in idset
            ],
        }

    return get_subgraph


class FakeClient:
    def __init__(self, nodes, edges, communities=None, seed_hits=None, batched=False):
        adj: dict = {}
        for s, d, _ in edges:
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)
        self.nodes = _Nodes(nodes, {k: sorted(v) for k, v in adj.items()})
        self.edges = _Edges(edges)
        self.graph = _Graph(communities or [], seed_hits or [])
        # When batched, expose the one-round-trip GetSubgraph; otherwise the
        # distiller falls back to per-node reads (both paths covered by tests).
        if batched:
            self.graph.get_subgraph = _make_subgraph_fn(nodes, edges)

    async def close(self):
        pass


def _fixture_graph():
    nodes = {
        "concept:servicenow": {
            "type": "Concept",
            "name": "ServiceNow",
            "summary": "ServiceNow is an ITSM platform.",
        },
        "doc:1": {"type": "Document", "name": "Incident Management"},  # body-less
        "ideablock:a": {
            "type": "idea_block",
            "name": "Creating an incident",
            "trusted_answer": "To create an incident, open the form...",
        },
        "ideablock:b": {
            "type": "idea_block",
            "name": "Closing an incident",
            "trusted_answer": "To close an incident, set state to Closed...",
        },
        "concept:incident": {
            "type": "Concept",
            "name": "Incident",
            "summary": "An incident is an unplanned interruption.",
        },
        "concept:unrelated": {
            "type": "Concept",
            "name": "Unrelated",
            "summary": "Out of scope.",
        },
    }
    edges = [
        ("doc:1", "concept:servicenow", {"rel_type": "MENTIONS"}),
        ("doc:1", "ideablock:a", {"rel_type": "CONTAINS"}),
        ("doc:1", "ideablock:b", {"rel_type": "CONTAINS"}),
        ("ideablock:a", "concept:incident", {"rel_type": "MENTIONS"}),
        ("ideablock:b", "concept:incident", {"rel_type": "MENTIONS"}),
        ("concept:servicenow", "concept:incident", {"rel_type": "RELATES_TO"}),
        # An edge to an out-of-selection node — must NOT appear in the manifest.
        ("concept:incident", "concept:unrelated", {"rel_type": "RELATES_TO"}),
    ]
    communities = [
        ["ideablock:a", "concept:incident"],
        ["ideablock:b", "doc:1", "concept:servicenow"],
        ["concept:unrelated"],
    ]
    return nodes, edges, communities


async def test_distill_seed_builds_reference_tree_and_manifest(tmp_path):
    nodes, edges, communities = _fixture_graph()
    client = FakeClient(nodes, edges, communities)
    distiller = SkillGraphDistiller(client, graph_name="__test__")

    manifest = await distiller.distill(
        seed="concept:servicenow", depth=2, out_dir=str(tmp_path)
    )

    ref = tmp_path / "reference"
    assert ref.is_dir()

    # Manifest shape + provenance.
    raw = json.loads((tmp_path / "kg_manifest.json").read_text())
    assert raw["schema"] == "skill-graph-kg-manifest/v1"
    assert raw["selector"]["seed"] == "concept:servicenow"
    assert raw["graph_name"] == "__test__"

    # The out-of-scope node is depth-2 reachable, but its edge to itself only
    # matters if selected. With depth=2 from servicenow it IS reachable; assert
    # the unrelated->* edges that leave the selection are excluded.
    selected = set(
        raw["clusters"] and [n for ms in raw["clusters"].values() for n in ms]
    )
    for e in raw["edges"]:
        assert e["src"] in selected and e["dst"] in selected

    # Body-bearing nodes become files; the body-less Document does not.
    files = {p["file"] for p in raw["nodes"] if p["file"]}
    assert any("Creating-an-incident" in f or "ideablock" in f.lower() for f in files)
    doc_entry = next(n for n in raw["nodes"] if n["id"] == "doc:1")
    assert doc_entry["file"] is None  # container node, no empty file

    # Every written file exists on disk and carries the kg_node_id frontmatter.
    for f in files:
        fp = tmp_path / f
        assert fp.exists(), f
        assert "kg_node_id:" in fp.read_text()

    # At least one file has an inline cross-link "Related" section.
    assert any("## Related" in (tmp_path / f).read_text() for f in files)

    assert manifest["stats"]["files"] >= 3


async def test_batched_get_subgraph_path_matches_fallback(tmp_path):
    # Same fixture, but the engine exposes the batched GetSubgraph. Output must
    # match the per-node fallback: files written, edges restricted to selection.
    nodes, edges, communities = _fixture_graph()
    client = FakeClient(nodes, edges, communities, batched=True)
    distiller = SkillGraphDistiller(client, graph_name="__test__")

    manifest = await distiller.distill(
        seed="concept:servicenow", depth=2, out_dir=str(tmp_path)
    )
    raw = json.loads((tmp_path / "kg_manifest.json").read_text())
    selected = {n for ms in raw["clusters"].values() for n in ms}
    assert manifest["stats"]["files"] >= 3
    assert raw["edges"], "batched path should still surface induced edges"
    for e in raw["edges"]:
        assert e["src"] in selected and e["dst"] in selected
    # The cross-link section proves edges flowed through the batched read.
    assert any(
        "## Related" in (tmp_path / n["file"]).read_text()
        for n in raw["nodes"]
        if n["file"]
    )


async def test_flat_taxonomy_when_no_communities(tmp_path):
    nodes, edges, _ = _fixture_graph()
    # No community structure → single flat reference/ (no subfolders).
    client = FakeClient(nodes, edges, communities=[])
    distiller = SkillGraphDistiller(client, graph_name="__test__")

    manifest = await distiller.distill(
        seed="concept:servicenow", depth=2, out_dir=str(tmp_path)
    )
    ref = tmp_path / "reference"
    # All materialised files sit directly under reference/ (flat).
    subdirs = [p for p in ref.iterdir() if p.is_dir()]
    assert not subdirs, f"expected flat tree, found subdirs: {subdirs}"
    assert manifest["stats"]["files"] >= 3


async def test_part_of_chunks_are_covered_by_parent_document(tmp_path):
    # Standardized ingestion: a Document with full content + verbatim chunk
    # children (IdeaBlock --PART_OF--> Document). The distiller must emit the
    # Document, not the Document AND its chunks.
    nodes = {
        "doc:guide": {
            "type": "Document",
            "name": "Setup Guide",
            "content": "Full verbatim setup guide body...",
        },
        "doc:guide:chunk:0": {
            "type": "idea_block",
            "name": "Setup Guide §1",
            "trusted_answer": "Full verbatim setup guide body...",
        },
        "doc:guide:chunk:1": {
            "type": "idea_block",
            "name": "Setup Guide §2",
            "trusted_answer": "...more body...",
        },
    }
    edges = [
        ("doc:guide:chunk:0", "doc:guide", {"rel_type": "PART_OF"}),
        ("doc:guide:chunk:1", "doc:guide", {"rel_type": "PART_OF"}),
    ]
    client = FakeClient(nodes, edges, communities=[])
    distiller = SkillGraphDistiller(client, graph_name="__test__")
    manifest = await distiller.distill(seed="doc:guide", depth=2, out_dir=str(tmp_path))

    # Exactly one file (the Document); both chunks recorded but not written.
    assert manifest["stats"]["files"] == 1
    written = [n for n in manifest["nodes"] if n["file"]]
    assert len(written) == 1 and written[0]["id"] == "doc:guide"
    chunks = [n for n in manifest["nodes"] if n["id"].endswith(("chunk:0", "chunk:1"))]
    assert all(c["file"] is None and c.get("covered_by_parent") for c in chunks)


async def test_distill_workflow_orders_steps_by_precedes(tmp_path):
    import re

    nodes = {
        "proc:a": {
            "type": "Procedure",
            "name": "Gather inputs",
            "description": "Collect the request inputs.",
        },
        "proc:b": {
            "type": "Procedure",
            "name": "Validate",
            "description": "Validate the inputs.",
        },
        "proc:c": {
            "type": "Procedure",
            "name": "Submit",
            "description": "Submit the change.",
        },
    }
    edges = [
        ("proc:a", "proc:b", {"rel_type": "PRECEDES"}),
        ("proc:b", "proc:c", {"rel_type": "PRECEDES"}),
    ]
    distiller = SkillGraphDistiller(FakeClient(nodes, edges), graph_name="__t__")
    result = await distiller.distill_workflow(
        seed="proc:a", depth=2, out_dir=str(tmp_path)
    )

    assert result["steps"] == 3
    md = (tmp_path / "SKILL.md").read_text()

    # Validator-compatible step headers, topologically ordered with depends_on.
    pattern = re.compile(
        r"^###\s+Step\s+(\d+):\s*([a-zA-Z0-9_-]+)(?:\s*\[depends_on:\s*([^\]]+)\])?",
        re.MULTILINE,
    )
    steps = pattern.findall(md)
    assert [int(n) for n, _, _ in steps] == [0, 1, 2]
    tokens = [t for _, t, _ in steps]
    assert tokens == ["Gather_inputs", "Validate", "Submit"]
    # Each step depends on its predecessor's step number.
    assert steps[0][2] == ""
    assert "Step 0" in steps[1][2]
    assert "Step 1" in steps[2][2]
    assert "Expected:" in md


async def test_unknown_seed_emits_no_files_but_valid_manifest(tmp_path):
    # A seed with no properties and no neighbours yields a body-less, file-less
    # but still well-formed manifest (deterministic empty-ish output).
    client = FakeClient({}, [], communities=[])
    distiller = SkillGraphDistiller(client, graph_name="__test__")
    manifest = await distiller.distill(seed="nope:404", depth=2, out_dir=str(tmp_path))
    assert manifest["stats"]["files"] == 0
    assert manifest["schema"] == "skill-graph-kg-manifest/v1"
    assert (tmp_path / "kg_manifest.json").exists()
    assert (tmp_path / "reference").is_dir()
