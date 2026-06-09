#!/usr/bin/python
from __future__ import annotations

"""Unit tests for the KG→skill-graph distiller (CONCEPT:AHE-3.9, read side).

Uses an in-memory fake async client that mimics the ``EpistemicGraphClient``
namespace surface (``nodes``/``edges``/``graph``), so the distiller logic is
exercised without a live epistemic-graph daemon. A live round-trip belongs in a
``live``-marked test.
"""

import json
from pathlib import Path

import msgpack
import pytest

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
        return [
            (s, d, list(msgpack.packb(props))) for (s, d, props) in self._edges
        ]


class _Graph:
    def __init__(self, communities, seed_hits) -> None:
        self._communities = communities
        self._seed_hits = seed_hits

    async def community_detection(self, resolution=1.0):
        return self._communities

    async def semantic_search(self, emb, n_results=5):
        return self._seed_hits[:n_results]


class FakeClient:
    def __init__(self, nodes, edges, communities=None, seed_hits=None) -> None:
        adj: dict = {}
        for s, d, _ in edges:
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)
        self.nodes = _Nodes(nodes, {k: sorted(v) for k, v in adj.items()})
        self.edges = _Edges(edges)
        self.graph = _Graph(communities or [], seed_hits or [])

    async def close(self):
        pass


def _fixture_graph():
    nodes = {
        "concept:servicenow": {"type": "Concept", "name": "ServiceNow",
                                "summary": "ServiceNow is an ITSM platform."},
        "doc:1": {"type": "Document", "name": "Incident Management"},  # body-less
        "ideablock:a": {"type": "idea_block", "name": "Creating an incident",
                         "trusted_answer": "To create an incident, open the form..."},
        "ideablock:b": {"type": "idea_block", "name": "Closing an incident",
                         "trusted_answer": "To close an incident, set state to Closed..."},
        "concept:incident": {"type": "Concept", "name": "Incident",
                              "summary": "An incident is an unplanned interruption."},
        "concept:unrelated": {"type": "Concept", "name": "Unrelated",
                              "summary": "Out of scope."},
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
    selected = set(raw["clusters"] and [n for ms in raw["clusters"].values() for n in ms])
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
        "doc:guide": {"type": "Document", "name": "Setup Guide",
                       "content": "Full verbatim setup guide body..."},
        "doc:guide:chunk:0": {"type": "idea_block", "name": "Setup Guide §1",
                               "trusted_answer": "Full verbatim setup guide body..."},
        "doc:guide:chunk:1": {"type": "idea_block", "name": "Setup Guide §2",
                               "trusted_answer": "...more body..."},
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
