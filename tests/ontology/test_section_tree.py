#!/usr/bin/python
"""Tests for the per-document section tree (CONCEPT:AU-KG.retrieval.section-tree).

Deterministic + offline: markdown is passed inline (no PDF/OCR), no LLM (the
default heuristic summarizer is used), and persistence is exercised against an
in-memory writer mirroring the backend contract — no daemon required.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.ontology.document_processing import (
    HAS_SECTION_EDGE,
    HAS_SUBSECTION_EDGE,
    SECTION_NODE_TYPE,
    SECTION_OF_EDGE,
    DocumentProcessor,
    SectionTreeConfig,
    build_section_tree,
    build_section_tree_from_pages,
    iter_sections,
    rebuild_section_tree,
    section_nodes_and_edges,
    verify_section_tree,
)

MD = """# Title
Opening overview paragraph describing the whole document at a high level.

## Installation
Install the package with pip and configure the database connection settings.

### Requirements
Python 3.11 or newer is required for the runtime.

## Retrieval
Vectorless reasoning-tree navigation avoids the embedder recall ceiling on long documents.

## Billing
Payment, invoices, refunds and subscription tiers are described in this section here.
"""


class _FakeWriter:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, label="", **properties):
        self.nodes[node_id] = {"label": label, **properties}

    def add_edge(self, source, target, rel_type="", **properties):
        self.edges.append((source, target, rel_type, properties))


class _FakeFacade:
    def __init__(self, writer):
        self.store = writer


def test_markdown_tree_structure_and_ids():
    roots = build_section_tree(MD, config=SectionTreeConfig(thin=False))
    assert len(roots) == 1
    title = roots[0]
    assert title.title == "Title"
    # Pre-order DFS ids are contiguous zero-padded.
    ids = [s.node_id for s in iter_sections(roots)]
    assert ids == ["0001", "0002", "0003", "0004", "0005"]
    # Nesting: Requirements under Installation; Retrieval/Billing under Title.
    assert [c.title for c in title.children] == [
        "Installation",
        "Retrieval",
        "Billing",
    ]
    install = title.children[0]
    assert [c.title for c in install.children] == ["Requirements"]


def test_char_ranges_slice_the_source():
    roots = build_section_tree(MD, config=SectionTreeConfig(thin=False))
    for node in iter_sections(roots):
        span = MD[node.char_start : node.char_end]
        # Each node's own span begins at its heading line.
        assert node.title in span


def test_char_starts_are_monotonic_preorder():
    roots = build_section_tree(MD, config=SectionTreeConfig(thin=False))
    starts = [s.char_start for s in iter_sections(roots)]
    assert starts == sorted(starts)


def test_thinning_collapses_small_sections():
    thin = build_section_tree(
        MD, config=SectionTreeConfig(thin=True, min_node_tokens=10_000)
    )
    # A huge threshold collapses everything into the single root.
    assert len(iter_sections(thin)) == 1


def test_no_headings_yields_single_root():
    roots = build_section_tree("Just a flat paragraph with no headings at all.")
    assert len(roots) == 1
    assert roots[0].char_start == 0
    assert roots[0].children == []


def test_summaries_are_populated_when_requested():
    roots = build_section_tree(MD, config=SectionTreeConfig(thin=False, summarize=True))
    assert all(s.summary for s in iter_sections(roots))


def test_nodes_edges_roundtrip():
    roots = build_section_tree(MD, config=SectionTreeConfig(thin=False))
    nodes, edges = section_nodes_and_edges("doc:1", roots)
    assert all(n["type"] == SECTION_NODE_TYPE for n in nodes)
    n_sections = len(iter_sections(roots))
    assert sum(1 for e in edges if e["type"] == HAS_SECTION_EDGE) == n_sections
    assert sum(1 for e in edges if e["type"] == SECTION_OF_EDGE) == n_sections
    # HAS_SUBSECTION for every non-root node.
    assert sum(1 for e in edges if e["type"] == HAS_SUBSECTION_EDGE) == n_sections - 1

    rebuilt = rebuild_section_tree(nodes)
    assert len(iter_sections(rebuilt)) == n_sections
    assert [c.title for c in rebuilt[0].children] == [
        "Installation",
        "Retrieval",
        "Billing",
    ]


def test_verify_passes_for_markdown_tree():
    roots = build_section_tree(MD, config=SectionTreeConfig(thin=False))
    report = verify_section_tree(MD, roots)
    assert report["mismatched"] == 0
    assert report["verified"] == report["checked"]


def test_verify_repairs_a_bad_range():
    roots = build_section_tree(MD, config=SectionTreeConfig(thin=False))
    # Corrupt one node's span so its title is no longer inside it.
    bad = roots[0].children[-1]  # "Billing"
    bad.char_start = 0
    bad.char_end = 5
    report = verify_section_tree(MD, roots, fix=True)
    assert report["repaired"] >= 1
    assert bad.title in MD[bad.char_start : bad.char_end]


def test_pdf_page_path_fallback_one_section_per_page():
    pages = ["Alpha content on page one.", "Beta content on page two."]
    roots = build_section_tree_from_pages(pages, llm_fn=None)
    secs = iter_sections(roots)
    assert len(secs) == 2
    assert secs[0].page_start == 1 and secs[1].page_start == 2


def test_processor_builds_and_persists_section_slice():
    writer = _FakeWriter()
    proc = DocumentProcessor(
        _FakeFacade(writer), embed_fn=lambda texts: [None] * len(texts)
    )
    result = proc.process(MD, document_id="doc:test", section_tree=True, persist=True)
    assert result.section_nodes
    # Section nodes landed in the writer.
    section_ids = [nid for nid in writer.nodes if "::section::" in nid]
    assert len(section_ids) == len(result.section_nodes)
    # HAS_SECTION edges from the document to its sections were written.
    assert any(rt == HAS_SECTION_EDGE for _, _, rt, _ in writer.edges)


def test_section_tree_off_by_default():
    writer = _FakeWriter()
    proc = DocumentProcessor(
        _FakeFacade(writer), embed_fn=lambda texts: [None] * len(texts)
    )
    result = proc.process(MD, document_id="doc:test")
    assert result.section_nodes == []
    assert not any("::section::" in nid for nid in writer.nodes)
