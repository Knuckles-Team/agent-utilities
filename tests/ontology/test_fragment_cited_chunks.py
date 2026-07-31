#!/usr/bin/python
"""Chunk <-> Fragment citation wiring (CONCEPT:AU-KG.retrieval.fragment-cited-chunk).

The evidence spine (``evidence_spine.py``, track 1 of the universal-ingestion
program) gives every document a stable, addressable ``Fragment`` per
structural unit. This closes track 8's chunking requirement: the retrieval
unit (``DocumentChunk``, fixed-size/overlapping/embedded) must be able to
cite the evidence unit (``Fragment``) its span overlaps — "a chunk that
cannot cite a fragment is not citable" (reports/program/universal-ingestion.md).

Self-contained: text is passed inline, embeddings are injected (no network),
and ``persist=False`` so the pipeline never reaches the live write path (no
backend/marking-store dependency — mirrors the existing offline tests in
``test_document_processing.py``).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.ontology.document_processing import (
    ChunkingConfig,
    DocumentChunk,
    DocumentProcessor,
    _fragment_ids_for_span,
)

_FILE_DOC_TEXT = (
    "# Title Heading\n\n"
    "Paragraph one has some words in it for testing purposes here and more words.\n\n"
    "## Sub Heading\n\n"
    "Paragraph two continues with more words to fill out the chunk nicely here.\n\n"
    "| Col A | Col B |\n"
    "|-------|-------|\n"
    "| 1 | 2 |\n"
)

# A short markdown document with multiple structural units (heading +
# several paragraphs) so fragment_markdown produces more than one Fragment,
# and chunk_size is small enough to force multiple chunks.
_DOC_TEXT = (
    "# Title\n\n"
    "Paragraph one has some words in it for testing purposes here.\n\n"
    "Paragraph two continues with more words to fill out the chunk nicely.\n\n"
    "Paragraph three keeps going so we get multiple chunks in the output.\n"
)


def _fake_embed(texts):
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _process(**kwargs):
    proc = DocumentProcessor(
        graph=None,
        embed_fn=_fake_embed,
        chunking=ChunkingConfig(chunk_size=80, overlap=10),
    )
    return proc.process(_DOC_TEXT, source="memory://fragment-citation", persist=False, **kwargs)


# --- _fragment_ids_for_span (pure function) --------------------------------


class _Frag:
    def __init__(self, fragment_id, char_start, char_end):
        self.fragment_id = fragment_id
        self.char_start = char_start
        self.char_end = char_end


def test_fragment_ids_for_span_overlap():
    frags = [_Frag("f1", 0, 10), _Frag("f2", 10, 20), _Frag("f3", 20, 30)]
    assert _fragment_ids_for_span(5, 15, frags) == ["f1", "f2"]


def test_fragment_ids_for_span_no_overlap():
    frags = [_Frag("f1", 0, 10)]
    assert _fragment_ids_for_span(20, 30, frags) == []


def test_fragment_ids_for_span_ignores_unlocated_fragments():
    frags = [_Frag("f1", -1, -1)]
    assert _fragment_ids_for_span(0, 100, frags) == []


# --- DocumentChunk defaults --------------------------------------------------


def test_document_chunk_defaults_to_uncited_and_unversioned():
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        position=0,
        text="hi",
        char_start=0,
        char_end=2,
        content_hash="deadbeef",
        word_count=1,
    )
    assert chunk.fragment_ids == []
    assert chunk.embedding_version == ""


# --- end-to-end: process() wires real Fragment citations onto real chunks --


def test_every_chunk_cites_at_least_one_fragment():
    result = _process()
    assert result.chunk_count >= 1
    assert len(result.fragments) >= 1
    for chunk_node in result.chunk_nodes:
        assert chunk_node["fragment_ids"], (
            f"chunk {chunk_node['id']!r} has no fragment citation — every "
            "chunk must resolve to real evidence"
        )


def test_cited_fragment_ids_resolve_to_real_fragments():
    """The mandatory-citation proof point: a chunk's fragment_ids are not
    just non-empty — they resolve to REAL Fragment objects in the same
    result, each carrying the artifact_id/content_hash a reader needs to
    verify the citation."""
    result = _process()
    real_ids = {f.fragment_id for f in result.fragments}
    for chunk_node in result.chunk_nodes:
        for fid in chunk_node["fragment_ids"]:
            assert fid in real_ids, f"chunk cites {fid!r}, which is not a real fragment"


def test_chunk_embedding_version_is_tagged_when_embedded(monkeypatch):
    """When an embedding version CAN be resolved, every embedded chunk is
    tagged with it (CONCEPT:AU-KG.retrieval.embedding-version-identity) — this
    environment may have no default embedding model configured at all, so the
    resolver is pinned to prove the wiring rather than depend on ambient
    config."""
    from agent_utilities.knowledge_graph.retrieval import embedding_versioning

    monkeypatch.setattr(
        embedding_versioning,
        "resolve_current_embedding_version",
        lambda: embedding_versioning.EmbeddingVersion(
            provider="openai", model="test-embed-v1"
        ),
    )
    result = _process()
    for chunk_node in result.chunk_nodes:
        assert "embedding" in chunk_node
        assert chunk_node.get("embedding_version") == "openai:test-embed-v1"


def test_chunk_embedding_version_empty_when_unresolvable():
    """No default embedding model configured -> chunks are still embedded
    (the injected embed_fn doesn't care) but carry no version claim, rather
    than raising or fabricating one."""
    result = _process()
    for chunk_node in result.chunk_nodes:
        assert "embedding" in chunk_node
        assert chunk_node.get("embedding_version") in (None, "")


# --- D-ES-1 regression: file-sourced markdown must NOT be structure-flattened
# before fragmenting --------------------------------------------------------


def test_file_sourced_markdown_preserves_structure_for_fragmenting(tmp_path):
    """Regression for D-ES-1 (found during the evidence-spine lane's own work):
    ``DocumentProcessor._read_file`` used to route a real file through
    ``KBDocumentParser.parse_file()``, which additionally ran the extracted
    text through ``_chunk_text`` — a word-count splitter (``text.split()``
    then ``" ".join(...)``) that collapses EVERY whitespace boundary (blank
    lines between paragraphs, heading/list/table line breaks) before
    ``DocumentProcessor`` ever saw the text. Both this processor's own
    chunker AND the evidence-spine fragmenter then ran on that already-
    flattened text, so a real markdown FILE's headings/paragraphs/table
    degraded to a single fragment for the whole document — citations still
    "resolved", but at garbage granularity, defeating the structural
    addressing evidence_spine exists for.

    The fix reads the KB parser's per-format reader directly (still verbatim
    for md/txt/html — genuinely-extracted text for pdf/docx/epub, which have
    no "verbatim" original to preserve) instead of its post-chunked output.
    """
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(_FILE_DOC_TEXT, encoding="utf-8")

    proc = DocumentProcessor(
        graph=None,
        embed_fn=_fake_embed,
        chunking=ChunkingConfig(chunk_size=80, overlap=10),
    )
    result = proc.process(str(doc_path), persist=False)

    kinds = {f.kind for f in result.fragments}
    # A structure-flattened extraction would yield ONE fragment (kind
    # "paragraph" or "document"), with no distinguishable heading/table units.
    assert "heading" in kinds, (
        f"headings were not preserved through file extraction — got kinds {kinds}"
    )
    assert "table" in kinds or "table_row" in kinds, (
        f"the markdown table was not preserved through file extraction — got kinds {kinds}"
    )
    assert len(result.fragments) > 2, (
        "file-sourced markdown degraded to whole-document granularity "
        f"({len(result.fragments)} fragment(s))"
    )
    # And every chunk still cites real fragments post-fix.
    real_ids = {f.fragment_id for f in result.fragments}
    for chunk_node in result.chunk_nodes:
        assert chunk_node["fragment_ids"]
        for fid in chunk_node["fragment_ids"]:
            assert fid in real_ids
