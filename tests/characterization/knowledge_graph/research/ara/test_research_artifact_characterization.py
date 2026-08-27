"""Characterization tests for ``ResearchArtifact.to_graph_payload`` and
``ResearchArtifact.from_extracted`` (CX-AU-09).

CCN at time of writing: ``to_graph_payload`` 14, ``from_extracted`` 11
(``agent_utilities/knowledge_graph/research/ara/artifact.py``). These tests pin
the OBSERVED, black-box behaviour of both public methods before any
decomposition -- exact node/edge shapes, exact ordering, and the fallback
rules for empty/optional fields -- so a later refactor can be checked
byte-for-byte against real (not aspirational) behaviour.

Per the two-commit discipline, this file must be added and pass GREEN against
the UNMODIFIED ``artifact.py`` before any refactor commit, and must not
change during the refactor commit that follows.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara.artifact import (
    Claim,
    CodeSpec,
    Evidence,
    ExplorationNode,
    ResearchArtifact,
)

# ─────────────────────────────────────────────────────────────────────────
# from_extracted
# ─────────────────────────────────────────────────────────────────────────


def test_from_extracted_wires_every_claim_to_all_evidence_and_code() -> None:
    art = ResearchArtifact.from_extracted(
        "art-1",
        "Title One",
        claims=["Claim Alpha", "Claim Beta"],
        evidence=["ev text 0", "ev text 1"],
        code_specs=["code text 0"],
    )

    assert [e.id for e in art.evidence] == ["evidence:art-1:0", "evidence:art-1:1"]
    assert [e.content for e in art.evidence] == ["ev text 0", "ev text 1"]
    assert [c.id for c in art.code_specs] == ["code_spec:art-1:0"]
    assert [c.summary for c in art.code_specs] == ["code text 0"]

    assert len(art.claims) == 2
    assert art.claims[0].id == "claim:art-1:claim-alpha:0"
    assert art.claims[0].statement == "Claim Alpha"
    assert art.claims[1].id == "claim:art-1:claim-beta:1"
    # every claim is wired to ALL evidence/code (conservative full binding)
    for cl in art.claims:
        assert cl.evidence_ids == ["evidence:art-1:0", "evidence:art-1:1"]
        assert cl.code_spec_ids == ["code_spec:art-1:0"]


def test_from_extracted_defaults_on_empty_and_missing_optionals() -> None:
    art = ResearchArtifact.from_extracted("art-2", "Title Two")

    assert art.claims == []
    assert art.evidence == []
    assert art.code_specs == []
    assert art.summary == ""
    assert art.authors == []
    assert art.source_url == ""
    # OBSERVED: source_ref defaults to f"article:{article_id}" when not passed
    assert art.source_ref == "article:art-2"


def test_from_extracted_explicit_source_ref_wins_over_default() -> None:
    art = ResearchArtifact.from_extracted(
        "art-3",
        "Title Three",
        source_ref="paper:external-3",
        authors=["A. Author"],
        source_url="https://example.test/paper",
    )
    assert art.source_ref == "paper:external-3"
    assert art.authors == ["A. Author"]
    assert art.source_url == "https://example.test/paper"


def test_from_extracted_claim_id_slug_strips_and_truncates() -> None:
    # OBSERVED: _slug lowercases, replaces non-alnum runs with single '-', and
    # truncates to 40 chars for the claim id segment.
    long_statement = "A" * 60
    art = ResearchArtifact.from_extracted(
        "art-4", "Title Four", claims=["Weird!!  Punctuation??", long_statement]
    )
    assert art.claims[0].id == "claim:art-4:weird-punctuation:0"
    assert art.claims[1].id == f"claim:art-4:{'a' * 40}:1"


# ─────────────────────────────────────────────────────────────────────────
# to_graph_payload
# ─────────────────────────────────────────────────────────────────────────


def _artifact_with_all_layers() -> ResearchArtifact:
    return ResearchArtifact(
        article_id="art-5",
        title="Full Artifact",
        summary="A summary",
        authors=["Author A"],
        source_url="https://example.test/5",
        source_ref="raw:paper-5",
        timestamp="2026-01-01T00:00:00Z",
        evidence=[
            Evidence(id="ev-1", content="observed output", source_ref="raw:ev-1"),
            Evidence(id="ev-2", content="", kind="log"),  # empty content, no source_ref
        ],
        code_specs=[
            CodeSpec(id="cs-1", summary="does a thing", symbol="do_thing"),
        ],
        claims=[
            Claim(
                id="claim-1",
                statement="X improves Y",
                confidence=0.75,
                evidence_ids=["ev-1"],
                code_spec_ids=["cs-1"],
            ),
            Claim(id="claim-2", statement="", evidence_ids=[], code_spec_ids=[]),
        ],
        exploration=[
            ExplorationNode(
                id="ex-1", kind="pivot", text="pivoted here", parent_id="ex-0"
            ),
            ExplorationNode(id="ex-2", kind="dead_end", text="", parent_id="ex-1"),
            ExplorationNode(
                id="ex-3", kind="decision", text="chose X", parent_id="ex-2"
            ),
            ExplorationNode(id="ex-4", kind="result", text="final"),  # no parent_id
        ],
    )


def test_to_graph_payload_artifact_node_shape() -> None:
    art = _artifact_with_all_layers()
    nodes, _edges = art.to_graph_payload()
    assert nodes[0] == {
        "id": "research_artifact:art-5",
        "type": "research_artifact",
        "properties": {
            "name": "Full Artifact",
            "title": "Full Artifact",
            "summary": "A summary",
            "authors": ["Author A"],
            "source_url": "https://example.test/5",
            "timestamp": "2026-01-01T00:00:00Z",
        },
    }


def test_to_graph_payload_provenance_edge_only_when_source_ref_set() -> None:
    art = _artifact_with_all_layers()
    _nodes, edges = art.to_graph_payload()
    assert {
        "source": "research_artifact:art-5",
        "target": "raw:paper-5",
        "type": "was_derived_from",
    } in edges

    art_no_ref = ResearchArtifact(article_id="art-6", title="No provenance")
    _nodes2, edges2 = art_no_ref.to_graph_payload()
    assert all(
        e["type"] != "was_derived_from" or e["source"] != "research_artifact:art-6"
        for e in edges2
    )


def test_to_graph_payload_evidence_nodes_and_conditional_provenance_edge() -> None:
    art = _artifact_with_all_layers()
    nodes, edges = art.to_graph_payload()
    ev1 = next(n for n in nodes if n["id"] == "ev-1")
    assert ev1 == {
        "id": "ev-1",
        "type": "evidence",
        "properties": {
            "name": "observed output",
            "content": "observed output",
            "kind": "observation",
            "timestamp": "2026-01-01T00:00:00Z",
        },
    }
    # ev-1 has a source_ref -> provenance edge emitted
    assert {"source": "ev-1", "target": "raw:ev-1", "type": "was_derived_from"} in edges

    ev2 = next(n for n in nodes if n["id"] == "ev-2")
    # OBSERVED: empty content falls back to the evidence id for "name"
    assert ev2["properties"]["name"] == "ev-2"
    assert ev2["properties"]["kind"] == "log"
    # ev-2 has NO source_ref -> no provenance edge for ev-2
    assert all(e["source"] != "ev-2" for e in edges)


def test_to_graph_payload_code_spec_node_shape() -> None:
    art = _artifact_with_all_layers()
    nodes, _edges = art.to_graph_payload()
    cs1 = next(n for n in nodes if n["id"] == "cs-1")
    assert cs1 == {
        "id": "cs-1",
        "type": "code_spec",
        "properties": {
            "name": "do_thing",
            "summary": "does a thing",
            "language": "",
            "symbol": "do_thing",
            "path": "",
        },
    }


def test_to_graph_payload_claim_nodes_contains_edge_and_fallback_name() -> None:
    art = _artifact_with_all_layers()
    nodes, edges = art.to_graph_payload()
    claim1 = next(n for n in nodes if n["id"] == "claim-1")
    assert claim1["properties"]["confidence"] == 0.75
    assert claim1["properties"]["statement"] == "X improves Y"
    assert claim1["properties"]["name"] == "X improves Y"

    claim2 = next(n for n in nodes if n["id"] == "claim-2")
    # OBSERVED: empty statement falls back to the claim id for "name"
    assert claim2["properties"]["name"] == "claim-2"

    assert {
        "source": "research_artifact:art-5",
        "target": "claim-1",
        "type": "contains",
    } in edges
    assert {
        "source": "research_artifact:art-5",
        "target": "claim-2",
        "type": "contains",
    } in edges
    assert {"source": "claim-1", "target": "ev-1", "type": "grounded_in"} in edges
    assert {"source": "claim-1", "target": "cs-1", "type": "implemented_by"} in edges
    # claim-2 has no evidence/code ids -> no grounded_in/implemented_by edges from it
    assert all(
        not (
            e["source"] == "claim-2" and e["type"] in ("grounded_in", "implemented_by")
        )
        for e in edges
    )


def test_to_graph_payload_exploration_dag_edge_kinds() -> None:
    art = _artifact_with_all_layers()
    nodes, edges = art.to_graph_payload()

    ex1 = next(n for n in nodes if n["id"] == "ex-1")
    assert ex1["properties"]["exploration_kind"] == "pivot"
    assert ex1["properties"]["name"] == "pivoted here"
    # pivot -> pivoted_from
    assert {"source": "ex-1", "target": "ex-0", "type": "pivoted_from"} in edges

    ex2 = next(n for n in nodes if n["id"] == "ex-2")
    # OBSERVED: empty text falls back to the exploration node id for "name"
    assert ex2["properties"]["name"] == "ex-2"
    # dead_end -> reached_dead_end
    assert {"source": "ex-2", "target": "ex-1", "type": "reached_dead_end"} in edges

    # OBSERVED: any other kind (here "decision") with a parent_id falls back to
    # was_derived_from -- not omitted.
    assert {"source": "ex-3", "target": "ex-2", "type": "was_derived_from"} in edges

    # ex-4 has no parent_id -> no DAG edge from it at all
    assert all(e["source"] != "ex-4" for e in edges)

    # every exploration node still gets a contains edge from the artifact
    for ex_id in ("ex-1", "ex-2", "ex-3", "ex-4"):
        assert {
            "source": "research_artifact:art-5",
            "target": ex_id,
            "type": "contains",
        } in edges


def test_to_graph_payload_empty_artifact_has_only_the_root_node() -> None:
    art = ResearchArtifact(article_id="empty-1", title="Empty")
    nodes, edges = art.to_graph_payload()
    assert len(nodes) == 1
    assert nodes[0]["id"] == "research_artifact:empty-1"
    assert edges == []
