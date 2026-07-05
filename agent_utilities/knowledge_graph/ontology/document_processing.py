#!/usr/bin/python
from __future__ import annotations

"""Document Processing → Ontology — media/text → Chunk objects linked to a Document.

CONCEPT:AU-KG.ingest.chunk-overlap-stage — Document Processing Pipeline.

Palantir Foundry doc matched: *ontology / document-processing* — the
media-set → text-extraction/OCR → chunk-with-overlap → explode → embed →
*materialize Chunk objects linked to the source Document* pipeline. In Foundry a
document (a media-set item) is run through an extraction transform, the extracted
text is split into overlapping chunks, each chunk row is "exploded" into its own
object instance, embedded, and written back as a first-class ``Chunk`` object type
linked (``HAS_CHUNK`` / ``CHUNK_OF``) to the parent ``Document`` object. Downstream
RAG / Functions-on-Objects then operate on the Chunk objects (semantic search over
the per-chunk embedding, provenance back to the source document + position).

This module ports that pipeline into agent-utilities as a real, end-to-end
processor — never a stub:

  - :class:`ChunkingConfig` — configurable ``chunk_size`` / ``overlap`` /
    separator-priority splitting, with real recursive separator-priority chunking
    (paragraph → line → sentence → word → hard-cut) that produces deterministic,
    unique chunk ids and **monotonically increasing character positions** with the
    requested overlap.
  - :class:`DocumentChunk` / :class:`ProcessedDocument` — the materialized
    ontology objects: one ``Document`` node and one ``Chunk`` node per chunk,
    each carrying ``position`` / ``char_start`` / ``char_end`` provenance and its
    own embedding for semantic search, joined by ``HAS_CHUNK`` (Document→Chunk)
    and ``CHUNK_OF`` (Chunk→Document) edges.
  - :class:`DocumentProcessor` — ``process(document)`` taking a path, raw bytes,
    or already-extracted text. Text extraction reuses the KB stack
    (``kb/parser.py`` :class:`KBDocumentParser` for files, and
    ``enrichment.extractors.document.read_document_text`` for the lightweight
    single-file path); PDFs use ``pypdf``/``pdfminer`` when importable and degrade
    to a *clear, explicit* error path otherwise (never a silent empty stub).
    Embeddings come from :func:`create_embedding_model` (768-dim default).
    Materialization writes through the **live graph write path** (the facade
    store's ``add_node`` / ``add_edge``, exactly as ``ingestion/engine.py`` does)
    and returns the ``{document_node, chunk_nodes, edges}`` structure even when
    offline (no backend) so the pipeline is testable + usable without a daemon.

Reuses the existing fabric — nothing reinvented:
  - Text extraction: ``kb/parser.py`` (:class:`KBDocumentParser`) and
    ``enrichment.extractors.document`` (``read_document_text`` / ``detect_doc_type``
    / ``extract_metadata``).
  - Embeddings: ``core.embedding_utilities.create_embedding_model`` (the single
    ``EmbeddingFactory``) via ``get_text_embedding_batch`` — same call the
    enrichment ``make_embed_fn`` uses.
  - Write path: the facade store backend ``add_node(node_id, label, **props)`` /
    ``add_edge(source, target, rel_type, **props)`` contract.
  - Ontology property types: the ``embedding`` / ``vector`` :class:`PropertyType`
    (KG-2.47, 768-dim default) is the declared type of the per-chunk vector — the
    integrator can promote the Chunk object type's ``embedding`` property to it.
"""

import concurrent.futures
import hashlib
import logging
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent_utilities.core._env import setting
from agent_utilities.core.config import config

logger = logging.getLogger(__name__)

# Default per-chunk embedding dimensionality — driven by the unified XDG config
# (config.kg_embedding_dim) so it tracks the configured embedding model; ties to
# create_embedding_model() and the ontology ``embedding`` PropertyType
# (CONCEPT:AU-KG.ingest.chunk-overlap-stage). 768 is only a last-resort fallback.
DEFAULT_EMBEDDING_DIM = int(config.kg_embedding_dim or "768")

# Ontology object/link type names. ``Document`` already exists as a first-class
# node label in the ingestion fabric; ``Chunk`` is the per-chunk object this
# pipeline materializes. Edge labels mirror the Palantir HAS_CHUNK / CHUNK_OF
# parent↔child link pair.
DOCUMENT_NODE_TYPE = "Document"
CHUNK_NODE_TYPE = "Chunk"
HAS_CHUNK_EDGE = "HAS_CHUNK"
CHUNK_OF_EDGE = "CHUNK_OF"

# Section-tree object/link type names (CONCEPT:AU-KG.retrieval.section-tree). The
# per-document reasoning tree materializes one ``Section`` node per heading/TOC
# entry, linked to the parent Document (HAS_SECTION / SECTION_OF) and to nested
# child sections (HAS_SUBSECTION) so the tree can be reconstructed from the graph.
SECTION_NODE_TYPE = "Section"
HAS_SECTION_EDGE = "HAS_SECTION"
SECTION_OF_EDGE = "SECTION_OF"
HAS_SUBSECTION_EDGE = "HAS_SUBSECTION"

# Markdown-link → graph edge (CONCEPT:AU-KG.ingest.broken-link-tolerance). A
# document's inline ``[label](target)`` links become ``LINKS_TO`` edges; a target
# not otherwise present becomes a ``dangling`` placeholder node so the edge
# survives (OKF SPEC §5: "Consumers MUST tolerate broken links … not-yet-written
# knowledge"). Image links (``![...]``) and pure in-page anchors (``#frag``) are
# excluded — they are not concept references.
LINKS_TO_EDGE = "LINKS_TO"
_MD_LINK_RE = re.compile(r"(?<!\!)\[([^\]]+)\]\(\s*<?([^)\s>]+)>?[^)]*\)")


def extract_markdown_links(text: str) -> list[tuple[str, str]]:
    """Return ``(label, target)`` pairs for markdown links in *text*.

    CONCEPT:AU-KG.ingest.broken-link-tolerance. Skips images and bare in-page
    anchors; deduplicates on the target while keeping first-seen order.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for m in _MD_LINK_RE.finditer(text or ""):
        label, target = m.group(1).strip(), m.group(2).strip()
        if not target or target.startswith("#") or target in seen:
            continue
        seen.add(target)
        out.append((label, target))
    return out


def _link_target_id(target: str) -> str:
    """Stable node id for a link target (bundle path or URL, anchor stripped)."""
    base = target.split("#", 1)[0].strip()
    if base.lower().endswith((".md", ".markdown")):
        base = base.rsplit(".", 1)[0]
    return f"doc::{base}"


# Separator priority for recursive chunking — highest-semantic boundary first,
# matching the Foundry/LangChain "recursive character" splitter ordering.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ChunkingConfig(BaseModel):
    """Configuration for separator-priority text chunking with overlap.

    CONCEPT:AU-KG.ingest.chunk-overlap-stage — the chunk-with-overlap stage of the document pipeline.

    Attributes:
        chunk_size: Target maximum chunk length in characters.
        overlap: Number of characters of trailing context repeated at the start
            of the next chunk (the sliding-window overlap). Must be < chunk_size.
        separators: Ordered separator priority — the splitter tries to break on
            the first separator that yields pieces under ``chunk_size`` before
            falling back to a finer-grained one (paragraph → line → sentence →
            word → hard character cut).
        strip_whitespace: Trim leading/trailing whitespace on each emitted chunk.
        min_chunk_chars: Drop trailing fragments shorter than this (after strip)
            unless they are the only chunk — avoids dust chunks.
    """

    chunk_size: int = 800
    overlap: int = 120
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    strip_whitespace: bool = True
    min_chunk_chars: int = 1

    @field_validator("overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info: Any) -> int:
        size = info.data.get("chunk_size", 800)
        if v < 0:
            raise ValueError("overlap must be non-negative")
        if v >= size:
            raise ValueError(f"overlap ({v}) must be < chunk_size ({size})")
        return v

    @field_validator("chunk_size")
    @classmethod
    def _size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v


class ChunkSpan(BaseModel):
    """A raw chunk + its character span in the source text (pre-materialization).

    CONCEPT:AU-KG.ingest.chunk-overlap-stage — the intermediate "exploded" row before it becomes a Chunk
    object. ``char_start`` is monotonically non-decreasing across the sequence and
    successive spans overlap by (up to) the configured overlap.
    """

    index: int
    text: str
    char_start: int
    char_end: int


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[ChunkSpan]:
    """Split ``text`` into overlapping :class:`ChunkSpan`s by separator priority.

    CONCEPT:AU-KG.ingest.chunk-overlap-stage — real recursive separator-priority chunking. The text is
    first segmented on the highest-priority separator that produces sub-``chunk_size``
    pieces; pieces are then *packed* greedily into windows of up to ``chunk_size``
    characters, and each new window repeats the trailing ``overlap`` characters of
    the previous window so retrieval keeps cross-boundary context. Character spans
    are tracked against the original string so ``char_start`` is monotonic and the
    overlap is observable as ``span[i+1].char_start <= span[i].char_end``.

    Guarantees (asserted by the unit test):
      - non-empty input yields ≥ 1 span;
      - ``index`` is 0..N-1 contiguous;
      - ``char_start`` is non-decreasing;
      - consecutive spans overlap by ≤ ``overlap`` chars (and > 0 when the source
        spans multiple chunks and overlap > 0).
    """
    cfg = config or ChunkingConfig()
    if not text:
        return []

    # 1) Segment into atoms on the best separator (highest priority that yields
    #    pieces all under chunk_size, else finest). Track each atom's offset.
    atoms = _segment(text, cfg)

    # 2) Greedily pack atoms into <= chunk_size windows, then add overlap tails.
    spans: list[ChunkSpan] = []
    cur_start = atoms[0][1] if atoms else 0
    cur_end = cur_start
    for atom_text, a_start, a_end in atoms:
        # Would this atom overflow the current window? Emit and start a new one.
        if cur_end > cur_start and (a_end - cur_start) > cfg.chunk_size:
            spans.append(_emit(text, cur_start, cur_end, cfg))
            # New window begins with overlap from the just-emitted window.
            cur_start = max(cur_end - cfg.overlap, cur_start)
            # Re-anchor to the atom start if overlap pushed us past it (no gap).
            cur_start = min(cur_start, a_start)
            cur_end = a_end
        else:
            cur_end = a_end
    if cur_end > cur_start:
        spans.append(_emit(text, cur_start, cur_end, cfg))

    # 3) Re-index, drop empty/dust trailing fragments (keep ≥ 1 chunk always).
    cleaned: list[ChunkSpan] = []
    for sp in spans:
        body = sp.text.strip() if cfg.strip_whitespace else sp.text
        if not body:
            continue
        if len(body) < cfg.min_chunk_chars and cleaned:
            continue
        cleaned.append(sp)
    if not cleaned and text.strip():
        cleaned = [ChunkSpan(index=0, text=text, char_start=0, char_end=len(text))]
    for i, sp in enumerate(cleaned):
        sp.index = i
    return cleaned


def _emit(text: str, start: int, end: int, cfg: ChunkingConfig) -> ChunkSpan:
    raw = text[start:end]
    body = raw.strip() if cfg.strip_whitespace else raw
    return ChunkSpan(index=0, text=body, char_start=start, char_end=end)


def _segment(text: str, cfg: ChunkingConfig) -> list[tuple[str, int, int]]:
    """Return ``[(atom_text, start, end), …]`` segmented on the best separator.

    Picks the highest-priority separator whose pieces are mostly under
    ``chunk_size``; if even the finest separator leaves an over-long piece it is
    hard-cut into ``chunk_size`` windows so no atom ever exceeds the budget.
    """
    chosen = cfg.separators[-1] if cfg.separators else ""
    for sep in cfg.separators:
        if sep == "":
            chosen = sep
            break
        pieces = text.split(sep)
        if all(len(p) <= cfg.chunk_size for p in pieces):
            chosen = sep
            break
        chosen = sep
        if any(len(p) <= cfg.chunk_size for p in pieces):
            break

    atoms: list[tuple[str, int, int]] = []
    if chosen == "":
        # Hard character windows.
        for i in range(0, len(text), cfg.chunk_size):
            atoms.append(
                (text[i : i + cfg.chunk_size], i, min(i + cfg.chunk_size, len(text)))
            )
        return atoms

    pos = 0
    seplen = len(chosen)
    for piece in text.split(chosen):
        start = pos
        end = pos + len(piece)
        # Hard-cut any oversized atom so packing can never exceed chunk_size.
        if len(piece) > cfg.chunk_size:
            for j in range(0, len(piece), cfg.chunk_size):
                sub_start = start + j
                sub_end = min(start + j + cfg.chunk_size, end)
                atoms.append((piece[j : j + cfg.chunk_size], sub_start, sub_end))
        elif piece:
            atoms.append((piece, start, end))
        pos = end + seplen
    return atoms


# ── Materialized ontology objects ────────────────────────────────────────────


class DocumentChunk(BaseModel):
    """A materialized ``Chunk`` ontology object linked to its source Document.

    CONCEPT:AU-KG.ingest.chunk-overlap-stage — the exploded-and-embedded Chunk object. Carries position +
    provenance back to the parent document and its own embedding for semantic
    search (the ontology ``embedding`` PropertyType, 768-dim default).
    """

    id: str
    document_id: str
    position: int
    text: str
    char_start: int
    char_end: int
    content_hash: str
    word_count: int
    embedding: list[float] | None = None
    embedding_dim: int = 0
    context: str = ""


class ProcessedDocument(BaseModel):
    """Result of :meth:`DocumentProcessor.process` — the materialized graph slice.

    CONCEPT:AU-KG.ingest.chunk-overlap-stage. ``document_node`` and ``chunk_nodes`` are the node payloads
    written through the live write path (or returned offline); ``edges`` are the
    HAS_CHUNK / CHUNK_OF link payloads. ``persisted`` records whether the live
    graph write path actually committed the slice.
    """

    document_node: dict[str, Any]
    chunk_nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    #: Placeholder nodes for markdown-link targets not otherwise materialized
    #: (CONCEPT:AU-KG.ingest.broken-link-tolerance) — a broken/forward link becomes
    #: a ``dangling`` node so its ``LINKS_TO`` edge is never dropped (OKF SPEC §5).
    link_nodes: list[dict[str, Any]] = Field(default_factory=list)
    document_id: str = ""
    chunk_count: int = 0
    persisted: bool = False
    # Section-tree slice (CONCEPT:AU-KG.retrieval.section-tree) — populated only
    # when ``process(section_tree=True)``; empty otherwise so the existing
    # chunk-only pipeline stays byte-identical. ``section_roots`` is the nested
    # in-memory tree; ``section_nodes``/``section_edges`` are the flat payloads
    # written through the same live write path as the chunks.
    section_roots: list[SectionNode] = Field(default_factory=list)
    section_nodes: list[dict[str, Any]] = Field(default_factory=list)
    section_edges: list[dict[str, Any]] = Field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the ``{document_node, chunk_nodes, edges}`` mapping."""
        out: dict[str, Any] = {
            "document_node": self.document_node,
            "chunk_nodes": self.chunk_nodes,
            "edges": self.edges,
        }
        if self.section_nodes:
            out["section_nodes"] = self.section_nodes
            out["section_edges"] = self.section_edges
        return out


class DocumentExtractionError(RuntimeError):
    """Raised when a document's text cannot be extracted by any available reader.

    CONCEPT:AU-KG.ingest.chunk-overlap-stage — the *clear, explicit* degradation path. PDFs without
    ``pypdf``/``pdfminer`` (and no pre-extracted text supplied) raise this with an
    actionable message rather than silently materializing an empty document.
    """


EmbedFn = Callable[[Sequence[str]], list[list[float]]]


class DocumentProcessor:
    """End-to-end document → Chunk-objects pipeline (CONCEPT:AU-KG.ingest.chunk-overlap-stage).

    Wraps text extraction (KB parser + enrichment readers), separator-priority
    chunking with overlap, embedding, and materialization of ``Document`` +
    ``Chunk`` ontology objects through the live graph write path.

    Args:
        graph: A :class:`KnowledgeGraph` facade (or any object exposing a
            ``store`` with ``add_node`` / ``add_edge``, or those methods
            directly). When ``None`` the processor runs offline: it still
            extracts, chunks, embeds and *returns* the materialized structure,
            but performs no graph writes.
        chunking: :class:`ChunkingConfig`; sensible defaults when omitted.
        embed_fn: Optional embedding callable ``(texts) -> [vectors]``. When
            omitted, a lazy embedder backed by :func:`create_embedding_model`
            (768-dim) is built on first use; if the model is unavailable the
            pipeline degrades to ``None`` embeddings (chunks are still
            materialized + linked) rather than failing.
        embedding_dim: Expected embedding dimensionality (default 768).
    """

    def __init__(
        self,
        graph: Any = None,
        *,
        chunking: ChunkingConfig | None = None,
        embed_fn: EmbedFn | None = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        contextual: bool = False,
        enricher: Any = None,
        extract_links: bool = False,
    ) -> None:
        self.graph = graph
        self.chunking = chunking or ChunkingConfig()
        self.embedding_dim = embedding_dim
        self._embed_fn = embed_fn
        # CONCEPT:AU-KG.ingest.broken-link-tolerance — off by default so the
        # existing KG-2.48 slice is byte-identical; the OKF/openwiki ingest path
        # turns it on to capture markdown cross-links as LINKS_TO edges.
        self.extract_links = bool(extract_links)
        # CONCEPT:AU-KG.enrichment.contextual-retrieval-enrichment — contextual-retrieval enrichment. Default OFF so the
        # existing KG-2.48 pipeline is byte-identical; the connector ingestion
        # path turns it on. The enricher is lazy so importing this module never
        # requires the enrichment stack.
        self.contextual = contextual
        self._enricher = enricher

    # ── public API ───────────────────────────────────────────────────────

    def process(
        self,
        document: str | bytes | Path,
        *,
        document_id: str | None = None,
        title: str | None = None,
        doc_type: str | None = None,
        text: str | None = None,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
        section_tree: bool | SectionTreeConfig = False,
    ) -> ProcessedDocument:
        """Run the full pipeline and materialize Document + Chunk objects.

        CONCEPT:AU-KG.ingest.chunk-overlap-stage — ``media → extract/OCR → chunk(overlap) → explode →
        embed → materialize``. Writes through the live graph write path when a
        backend is reachable and ``persist`` is True; always returns the
        ``{document_node, chunk_nodes, edges}`` structure (via
        :class:`ProcessedDocument`).

        Args:
            document: A file path (``str``/``Path``), raw ``bytes``, or text
                content. If ``text`` is supplied explicitly it is used verbatim
                (the ``document`` arg then only informs id/title/provenance).
            document_id: Stable id for the Document node; derived from the source
                + content hash when omitted (idempotent re-processing).
            title: Document title; inferred from the first non-empty line / file
                name when omitted.
            doc_type: Document type (``paper``/``email``/…); auto-detected when
                omitted.
            text: Pre-extracted text — the explicit, non-silent fallback for
                formats this process can't read directly (e.g. OCR output, or a
                PDF on a host without pypdf). When given, extraction is skipped.
            source: Human-readable source label (path/URL) for provenance.
            metadata: Extra metadata merged onto the Document node.
            persist: When True and a live backend exists, commit the slice.

        Raises:
            DocumentExtractionError: when no text could be extracted and none was
                supplied.
        """
        raw_text, src_label, detected_type, derived_title = self._resolve_text(
            document, text=text, source=source
        )
        if not raw_text or not raw_text.strip():
            raise DocumentExtractionError(
                f"No extractable text for source={src_label!r}. "
                "Supply pre-extracted text via the `text=` argument "
                "(e.g. OCR output) or install a reader for this format "
                "(pypdf/pdfminer for PDF, python-docx for DOCX)."
            )

        content_hash = _sha(raw_text)
        doc_id = document_id or self._document_id(src_label, content_hash)
        final_title = title or derived_title or (src_label or doc_id)
        final_type = doc_type or detected_type or "document"

        spans = chunk_text(raw_text, self.chunking)

        # CONCEPT:AU-KG.enrichment.contextual-retrieval-enrichment — contextual-retrieval enrichment. Situate each chunk
        # within the whole document and embed ``context + chunk`` (Anthropic
        # contextual retrieval) so retrieval recall improves; the context is also
        # stored on the Chunk node for display/lexical match. Computed BEFORE
        # embedding. Off by default → no behaviour change for existing callers.
        contexts = self._enrich_contexts(
            raw_text, [sp.text for sp in spans], final_title
        )
        embed_inputs = [
            f"{ctx}\n\n{sp.text}" if ctx else sp.text
            for sp, ctx in zip(spans, contexts, strict=False)
        ]
        embeddings = self._embed(embed_inputs)

        chunks: list[DocumentChunk] = []
        for sp, emb, ctx in zip(spans, embeddings, contexts, strict=False):
            cid = f"{doc_id}::chunk::{sp.index}:{_sha(sp.text)[:12]}"
            chunks.append(
                DocumentChunk(
                    id=cid,
                    document_id=doc_id,
                    position=sp.index,
                    text=sp.text,
                    char_start=sp.char_start,
                    char_end=sp.char_end,
                    content_hash=_sha(sp.text),
                    word_count=len(sp.text.split()),
                    embedding=emb,
                    embedding_dim=len(emb) if emb else 0,
                    context=ctx,
                )
            )

        document_node = self._build_document_node(
            doc_id,
            title=final_title,
            doc_type=final_type,
            source=src_label,
            content_hash=content_hash,
            chunk_count=len(chunks),
            char_count=len(raw_text),
            extra=metadata,
        )
        chunk_nodes = [self._build_chunk_node(c) for c in chunks]
        edges = self._build_edges(doc_id, chunks)

        link_nodes: list[dict[str, Any]] = []
        if self.extract_links:
            link_nodes, link_edges = self._build_link_edges(doc_id, raw_text)
            edges.extend(link_edges)

        result = ProcessedDocument(
            document_node=document_node,
            chunk_nodes=chunk_nodes,
            edges=edges,
            link_nodes=link_nodes,
            document_id=doc_id,
            chunk_count=len(chunks),
        )

        # CONCEPT:AU-KG.retrieval.section-tree — optionally build the per-document
        # reasoning tree beside the flat chunks. Off by default so the chunk-only
        # pipeline is unchanged; the ingestion path / MCP tool turn it on.
        if section_tree:
            cfg = (
                section_tree
                if isinstance(section_tree, SectionTreeConfig)
                else SectionTreeConfig()
            )
            roots = build_section_tree(raw_text, config=cfg)
            # CONCEPT:AU-KG.ingest.structure-verify — confirm titles are inside
            # their claimed ranges (and repair drift) before we materialize.
            report = verify_section_tree(raw_text, roots, fix=True)
            if report["mismatched"]:
                logger.warning(
                    "[section-tree] %d section title(s) not found in range for %s",
                    report["mismatched"],
                    doc_id,
                )
            sec_nodes, sec_edges = section_nodes_and_edges(doc_id, roots)
            result.section_roots = roots
            result.section_nodes = sec_nodes
            result.section_edges = sec_edges

        if persist:
            result.persisted = self._persist(
                document_node, chunk_nodes + link_nodes, edges
            )
            if result.section_nodes:
                # Best-effort: section slice failures never abort the chunk slice.
                self._persist_sections(result.section_nodes, result.section_edges)
        return result

    # ── text extraction ──────────────────────────────────────────────────

    def _resolve_text(
        self,
        document: str | bytes | Path,
        *,
        text: str | None,
        source: str,
    ) -> tuple[str, str, str, str]:
        """Return ``(text, source_label, detected_doc_type, derived_title)``.

        Reuses the KB / enrichment readers for files; accepts bytes/inline text
        directly. PDFs route through readers that prefer pypdf/pdfminer and
        surface a clear error (via the empty-text → DocumentExtractionError path
        in :meth:`process`) when no reader is importable.
        """
        from ..enrichment.extractors.document import (
            detect_doc_type,
        )

        # 1) Explicit pre-extracted text wins (OCR / external extraction).
        if text is not None:
            label = source or "<text>"
            return (
                text,
                label,
                detect_doc_type(label, text),
                self._first_line(text, label),
            )

        # 2) A real file on disk → KB/enrichment reader.
        if isinstance(document, str | Path) and self._looks_like_path(document):
            path = Path(document)
            if path.exists() and path.is_file():
                label = source or str(path)
                extracted = self._read_file(path)
                return (
                    extracted,
                    label,
                    detect_doc_type(str(path), extracted),
                    (self._first_line(extracted, path.name)),
                )

        # 3) Raw bytes — decode as UTF-8 text (caller pre-extracts binary formats).
        if isinstance(document, bytes):
            label = source or "<bytes>"
            decoded = document.decode("utf-8", errors="replace")
            return (
                decoded,
                label,
                detect_doc_type(label, decoded),
                self._first_line(decoded, label),
            )

        # 4) Plain string content treated as the document text itself.
        if isinstance(document, str):
            label = source or "<text>"
            return (
                document,
                label,
                detect_doc_type(label, document),
                self._first_line(document, label),
            )

        # 5) Path-like that didn't resolve → empty (process() raises clearly).
        label = source or str(document)
        return "", label, "document", label

    def _read_file(self, path: Path) -> str:
        """Extract text from a file, reusing the KB parser, then enrichment reader.

        ``KBDocumentParser`` handles md/txt/html/pdf/docx/epub (pypdf/pdfminer for
        PDF via its ``_read_pdf``); the enrichment ``read_document_text`` is a
        second, lighter reader. Either returning empty text leads to the explicit
        :class:`DocumentExtractionError` in :meth:`process`.
        """
        from ..enrichment.extractors.document import read_document_text
        from ..kb.parser import KBDocumentParser

        try:
            parsed = KBDocumentParser(chunk_size=10_000).parse_file(path)
            if parsed and parsed.chunks:
                joined = "\n".join(c.content for c in parsed.chunks).strip()
                if joined:
                    return joined
        except Exception as exc:  # noqa: BLE001 — fall through to lighter reader
            logger.debug("KBDocumentParser failed for %s: %s", path, exc)

        try:
            return read_document_text(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.debug("read_document_text failed for %s: %s", path, exc)
            return ""

    @staticmethod
    def _looks_like_path(document: str | Path) -> bool:
        if isinstance(document, Path):
            return True
        s = document.strip()
        if "\n" in s or len(s) > 1024:
            return False
        try:
            return Path(s).exists()
        except OSError:
            return False

    @staticmethod
    def _first_line(text: str, fallback: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line.strip()[:200]
        return fallback

    # ── contextual enrichment (KG-2.50) ──────────────────────────────────

    def _enrich_contexts(
        self, doc_text: str, chunk_texts: list[str], title: str
    ) -> list[str]:
        """Return a situating context per chunk (empty strings when disabled).

        CONCEPT:AU-KG.enrichment.contextual-retrieval-enrichment. When ``contextual`` is off, returns ``[""] * n`` so the
        embedding input and chunk nodes are unchanged. When on, lazily builds a
        :class:`ContextualEnricher` (LLM if configured, deterministic heuristic
        otherwise) and situates each chunk. Never raises — enrichment failure
        degrades to empty context so ingest still completes.
        """
        if not self.contextual or not chunk_texts:
            return [""] * len(chunk_texts)
        try:
            enricher = self._enricher
            if enricher is None:
                from .contextual_enrichment import ContextualEnricher

                enricher = ContextualEnricher(self._contextual_llm_fn())
                self._enricher = enricher
            return enricher.enrich(doc_text, chunk_texts, title=title)
        except Exception as exc:  # noqa: BLE001 — enrichment must never break ingest
            logger.warning("[KG-2.50] contextual enrichment failed: %s", exc)
            return [""] * len(chunk_texts)

    @staticmethod
    def _contextual_llm_fn() -> Any:
        """Lazy lite-LLM fn for context summaries, or ``None`` (heuristic path)."""
        try:
            from ..enrichment.cards import make_lite_llm_fn

            return make_lite_llm_fn()
        except Exception:  # noqa: BLE001 — no LLM → deterministic heuristic
            return None

    # ── embedding ────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float] | None]:
        """Embed chunk texts via the configured model; degrade to ``None`` on miss.

        Uses the injected ``embed_fn`` if present, else builds one lazily over
        :func:`create_embedding_model` (the single EmbeddingFactory) using
        ``get_text_embedding_batch`` — the same call the enrichment ``make_embed_fn``
        makes. If the model is unavailable, returns ``None`` per chunk so the
        pipeline still materializes + links chunks (graph stays usable offline).
        """
        if not texts:
            return []
        fn = self._embed_fn
        if fn is None:
            fn = self._build_embed_fn()
            self._embed_fn = fn
        if fn is None:
            return [None] * len(texts)
        # Bounded call: an embedder that hangs (e.g. the serving GPU power-cycles
        # mid-request — its /health proxy still returns 200) must NOT block ingestion.
        # On timeout the worker thread is abandoned (finishes/dies in the background)
        # and we materialize without vectors so the graph stays usable. Tunable via
        # KG_EMBED_TIMEOUT (seconds).
        timeout_s = setting("KG_EMBED_TIMEOUT", 30.0, cast=float)
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            vectors = ex.submit(fn, texts).result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "[KG-2.48] embedding did not respond within %.0fs (embedder slow/down)"
                " — materializing without vectors",
                timeout_s,
            )
            return [None] * len(texts)
        except Exception as exc:  # noqa: BLE001 — never let embedding break ingest
            logger.warning(
                "[KG-2.48] embedding failed, materializing without vectors: %s", exc
            )
            return [None] * len(texts)
        finally:
            ex.shutdown(wait=False)
        out: list[list[float] | None] = []
        for v in vectors:
            out.append([float(x) for x in v] if v else None)
        # Pad/truncate defensively so lengths line up with texts.
        while len(out) < len(texts):
            out.append(None)
        return out[: len(texts)]

    def _build_embed_fn(self) -> EmbedFn | None:
        try:
            from agent_utilities.core.embedding_utilities import create_embedding_model

            model = create_embedding_model()

            def _fn(texts: Sequence[str]) -> list[list[float]]:
                return list(model.get_text_embedding_batch(list(texts)))

            return _fn
        except Exception as exc:  # noqa: BLE001
            logger.warning("[KG-2.48] embedding model unavailable: %s", exc)
            return None

    # ── node / edge payload construction ─────────────────────────────────

    def _build_document_node(
        self,
        doc_id: str,
        *,
        title: str,
        doc_type: str,
        source: str,
        content_hash: str,
        chunk_count: int,
        char_count: int,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        node: dict[str, Any] = {
            "id": doc_id,
            "type": DOCUMENT_NODE_TYPE,
            "name": title,
            "doc_type": doc_type,
            "source": source,
            "file_path": source,
            "content_hash": content_hash,
            "ast_hash": content_hash,
            "chunk_count": chunk_count,
            "char_count": char_count,
            "ingested_at": _now(),
        }
        if extra:
            for k, v in extra.items():
                node.setdefault(k, v)
        return node

    def _build_chunk_node(self, chunk: DocumentChunk) -> dict[str, Any]:
        node: dict[str, Any] = {
            "id": chunk.id,
            "type": CHUNK_NODE_TYPE,
            "name": f"{chunk.document_id}#{chunk.position}",
            "document_id": chunk.document_id,
            "position": chunk.position,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "content": chunk.text,
            "content_hash": chunk.content_hash,
            "word_count": chunk.word_count,
            "embedding_dim": chunk.embedding_dim,
        }
        if chunk.embedding is not None:
            node["embedding"] = chunk.embedding
        # CONCEPT:AU-KG.enrichment.contextual-retrieval-enrichment — the situating context is stored on the Chunk node so
        # it is available for display and lexical match (the HybridRetriever reads
        # ``content``/``summary``; ``contextual_summary`` extends that surface).
        if chunk.context:
            node["context"] = chunk.context
            node["contextual_summary"] = chunk.context
        return node

    def _build_edges(
        self, doc_id: str, chunks: list[DocumentChunk]
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for c in chunks:
            edges.append(
                {
                    "source": doc_id,
                    "target": c.id,
                    "type": HAS_CHUNK_EDGE,
                    "position": c.position,
                }
            )
            edges.append(
                {
                    "source": c.id,
                    "target": doc_id,
                    "type": CHUNK_OF_EDGE,
                    "position": c.position,
                }
            )
        return edges

    def _build_link_edges(
        self, doc_id: str, raw_text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Markdown cross-links → ``(dangling placeholder nodes, LINKS_TO edges)``.

        CONCEPT:AU-KG.ingest.broken-link-tolerance — every ``[label](target)`` is
        an edge; a target with no already-materialized node gets a lightweight
        ``dangling`` placeholder so the edge is never dropped. Later id-interlinking
        can reconcile the placeholder onto a real Document when the target lands.
        """
        link_nodes: list[dict[str, Any]] = []
        link_edges: list[dict[str, Any]] = []
        for label, target in extract_markdown_links(raw_text):
            tid = _link_target_id(target)
            is_external = "://" in target
            link_nodes.append(
                {
                    "id": tid,
                    "type": DOCUMENT_NODE_TYPE,
                    "name": label or target,
                    "dangling": True,
                    "external": is_external,
                    "source": target,
                    "ingested_at": _now(),
                }
            )
            link_edges.append(
                {
                    "source": doc_id,
                    "target": tid,
                    "type": LINKS_TO_EDGE,
                    "label": label,
                    "href": target,
                }
            )
        return link_nodes, link_edges

    # ── live graph write path ────────────────────────────────────────────

    def _resolve_writer(self) -> Any:
        """Return an object exposing ``add_node`` / ``add_edge``, or ``None``.

        Accepts a :class:`KnowledgeGraph` facade (uses ``.store``), a backend
        directly, or any duck-typed writer. Mirrors how ``ingestion/engine.py``
        resolves ``self.backend``.
        """
        g = self.graph
        if g is None:
            return None
        if hasattr(g, "add_node") and hasattr(g, "add_edge"):
            return g
        store = getattr(g, "store", None)
        if (
            store is not None
            and hasattr(store, "add_node")
            and hasattr(store, "add_edge")
        ):
            return store
        return None

    def _persist(
        self,
        document_node: dict[str, Any],
        chunk_nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> bool:
        """Write the materialized slice through the live backend.

        Returns True if the write path was exercised. Best-effort per element:
        a single failed write is logged but does not abort the slice (the
        returned structure is still correct for the caller).
        """
        writer = self._resolve_writer()
        if writer is None:
            return False

        # When the backend has an engine bulk path, materialize the whole slice
        # (document + N chunks + M edges) in batched RPCs instead of one socket
        # round-trip per element. Reuse the enrichment batcher (nodes flushed
        # before edges so endpoints exist); fall back to the robust per-item path
        # — which carries the label-less retry — when there is no bulk path.
        from agent_utilities.knowledge_graph.enrichment.pipeline import _BatchedBackend

        batched = _BatchedBackend(writer)
        if batched.bulk_available:
            for node in (document_node, *chunk_nodes):
                props = {k: v for k, v in node.items() if k not in ("id", "type")}
                batched.add_node(
                    node["id"], label=node["type"], type=node["type"], **props
                )
            for e in edges:
                props = {
                    k: v for k, v in e.items() if k not in ("source", "target", "type")
                }
                batched.add_edge(e["source"], e["target"], rel_type=e["type"], **props)
            try:
                batched.flush()
                return True
            except Exception as exc:  # noqa: BLE001 — degrade to per-item below
                logger.debug("[KG-2.48] batched persist failed (%s); per-item", exc)

        ok = False
        ok |= self._write_node(writer, document_node)
        for cn in chunk_nodes:
            ok |= self._write_node(writer, cn)
        for e in edges:
            try:
                props = {
                    k: v for k, v in e.items() if k not in ("source", "target", "type")
                }
                writer.add_edge(e["source"], e["target"], rel_type=e["type"], **props)
                ok = True
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[KG-2.48] add_edge failed %s->%s: %s",
                    e["source"],
                    e["target"],
                    exc,
                )
        return ok

    def _persist_sections(
        self,
        section_nodes: list[dict[str, Any]],
        section_edges: list[dict[str, Any]],
    ) -> bool:
        """Write the section-tree slice through the live backend (best-effort).

        CONCEPT:AU-KG.retrieval.section-tree. Reuses the same per-item write path
        as the chunk slice; nodes are flushed before their HAS_SUBSECTION edges so
        endpoints exist. Returns True if the write path was exercised.
        """
        writer = self._resolve_writer()
        if writer is None:
            return False
        ok = False
        for sn in section_nodes:
            ok |= self._write_node(writer, sn)
        for e in section_edges:
            try:
                props = {
                    k: v for k, v in e.items() if k not in ("source", "target", "type")
                }
                writer.add_edge(e["source"], e["target"], rel_type=e["type"], **props)
                ok = True
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[section-tree] add_edge failed %s->%s: %s",
                    e["source"],
                    e["target"],
                    exc,
                )
        return ok

    @staticmethod
    def _write_node(writer: Any, node: dict[str, Any]) -> bool:
        try:
            props = {k: v for k, v in node.items() if k not in ("id", "type")}
            writer.add_node(node["id"], label=node["type"], type=node["type"], **props)
            return True
        except TypeError:
            # Backend without a ``label`` kwarg — retry with ``type`` only.
            try:
                props = {k: v for k, v in node.items() if k != "id"}
                writer.add_node(node["id"], **props)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[KG-2.48] add_node failed for %s: %s", node.get("id"), exc
                )
                return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("[KG-2.48] add_node failed for %s: %s", node.get("id"), exc)
            return False

    # ── id helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _document_id(source: str, content_hash: str) -> str:
        """Stable, idempotent Document id from source + content hash."""
        digest = hashlib.sha256(f"{source}\x1f{content_hash}".encode()).hexdigest()[:16]
        return f"doc:{digest}"


# ── Section-tree — PageIndex-style per-document reasoning tree ────────────────
#
# CONCEPT:AU-KG.retrieval.section-tree — a first-class per-document section tree
# built *beside* the flat Chunk objects. Where the Chunk pipeline is similarity
# -first (overlap chunks → embed → ANN/BM25 + rerank), the section tree is a
# navigable table-of-contents: one node per heading (markdown) or TOC entry
# (PDF), each carrying its title, a pre-order-DFS ``node_id``, the source
# ``char_start``/``char_end`` (and optional ``page_start``/``page_end``) range,
# an optional summary, and nested children. A retriever can *walk* this tree by
# node relevance and return cited ranges — no recall ceiling from an embedder,
# superior for long single documents. Ported from PageIndex
# (``pageindex/page_index_md.py``: ``extract_nodes_from_markdown`` +
# ``build_tree_from_nodes`` level-stack + ``tree_thinning_for_index``
# token-budget prune) — the deterministic markdown path is LLM-free; the PDF
# path (``build_section_tree_from_pages``) is LLM-assisted (CONCEPT:AU-KG.ingest.toc-detection).

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CODE_FENCE_RE = re.compile(r"^```")

# Default token budget below which a section is collapsed into its parent during
# thinning — mirrors PageIndex ``min_node_token`` (keeps the tree navigable
# instead of one node per trivial sub-heading).
DEFAULT_MIN_NODE_TOKENS = 200
DEFAULT_SUMMARY_TOKEN_THRESHOLD = 200


class SectionTreeConfig(BaseModel):
    """Configuration for :func:`build_section_tree` (CONCEPT:AU-KG.retrieval.section-tree).

    Attributes:
        thin: Collapse sub-``min_node_tokens`` sections into their parent so the
            tree stays a navigable map rather than one node per trivial heading
            (PageIndex ``tree_thinning_for_index``).
        min_node_tokens: Token budget below which a section is merged upward.
        summarize: Compute a short per-node summary so the *structure* view is
            text-free (the map the tree-navigation retriever reasons over).
        summary_token_threshold: Sections shorter than this keep their own text
            as the summary; longer ones get an LLM/heuristic summary.
        max_pages_for_toc: PDF path — how many leading pages to scan for a TOC.
    """

    thin: bool = True
    min_node_tokens: int = DEFAULT_MIN_NODE_TOKENS
    summarize: bool = False
    summary_token_threshold: int = DEFAULT_SUMMARY_TOKEN_THRESHOLD
    max_pages_for_toc: int = 20


class SectionNode(BaseModel):
    """A node in a document's section tree (CONCEPT:AU-KG.retrieval.section-tree).

    ``node_id`` is a stable pre-order-DFS index (zero-padded, e.g. ``"0004"``);
    ``char_start``/``char_end`` cite the node's own span in the source text
    (heading → next heading, matching PageIndex per-node text); ``page_start``/
    ``page_end`` are set on the PDF path. ``summary`` gives the text-free map;
    ``children`` are the nested sub-sections.
    """

    node_id: str
    title: str
    level: int
    char_start: int
    char_end: int
    line_start: int = 0
    page_start: int | None = None
    page_end: int | None = None
    summary: str = ""
    text: str = ""
    children: list[SectionNode] = Field(default_factory=list)


SectionNode.model_rebuild()
# ProcessedDocument's ``section_roots`` field forward-references SectionNode
# (defined here, after that class) — resolve it now that SectionNode exists.
ProcessedDocument.model_rebuild()

# A summarizer maps ``(title, text) -> summary`` (LLM or heuristic).
SectionSummarizer = Callable[[str, str], str]


def iter_sections(roots: Sequence[SectionNode]) -> list[SectionNode]:
    """Return every node in ``roots`` in pre-order DFS (parents before children)."""
    out: list[SectionNode] = []

    def _walk(nodes: Sequence[SectionNode]) -> None:
        for n in nodes:
            out.append(n)
            _walk(n.children)

    _walk(roots)
    return out


def _line_offsets(lines: list[str]) -> list[int]:
    """Char offset (into the original text) of each line's start."""
    offsets: list[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1  # +1 for the stripped '\n'
    return offsets


def _extract_heading_lines(text: str) -> list[dict[str, Any]]:
    """Markdown headings outside code fences, with char offsets.

    Port of PageIndex ``extract_nodes_from_markdown`` extended to carry each
    heading's ``char_start`` (offset of its line) so the tree can cite character
    ranges, not just line numbers.
    """
    lines = text.split("\n")
    offsets = _line_offsets(lines)
    headings: list[dict[str, Any]] = []
    in_code = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if _CODE_FENCE_RE.match(stripped):
            in_code = not in_code
            continue
        if in_code or not stripped:
            continue
        m = _MD_HEADING_RE.match(stripped)
        if m:
            headings.append(
                {
                    "title": m.group(2).strip(),
                    "level": len(m.group(1)),
                    "line_num": i + 1,
                    "char_start": offsets[i],
                }
            )
    return headings


def _sections_with_spans(
    headings: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    """Attach each heading's own text span (heading → next heading, any level)."""
    flat: list[dict[str, Any]] = []
    n = len(headings)
    for i, h in enumerate(headings):
        start = int(h["char_start"])
        end = int(headings[i + 1]["char_start"]) if i + 1 < n else len(text)
        flat.append(
            {
                "title": h["title"],
                "level": h["level"],
                "line_num": h["line_num"],
                "char_start": start,
                "char_end": end,
                "text": text[start:end].strip(),
            }
        )
    return flat


def _descendant_slice(flat: list[dict[str, Any]], i: int) -> int:
    """Index one-past the contiguous descendants of ``flat[i]`` (deeper level)."""
    lvl = int(flat[i]["level"])
    j = i + 1
    while j < len(flat) and int(flat[j]["level"]) > lvl:
        j += 1
    return j


def _annotate_token_counts(flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set ``token_count`` = tokens of a node's text + all its descendants' text.

    Port of PageIndex ``update_node_list_with_text_token_count`` using the shared
    :func:`estimate_tokens` heuristic (no tokenizer dependency).
    """
    from ..memory.agent_context import estimate_tokens

    for i in range(len(flat)):
        end = _descendant_slice(flat, i)
        joined = "\n".join(
            str(flat[k].get("text", "")) for k in range(i, end) if flat[k].get("text")
        )
        flat[i]["token_count"] = estimate_tokens(joined)
    return flat


def _thin_sections(flat: list[dict[str, Any]], min_tokens: int) -> list[dict[str, Any]]:
    """Collapse sub-``min_tokens`` sections into their parent (PageIndex thinning).

    A small node absorbs the text of its whole descendant subtree and those
    descendants are dropped; the node's ``char_end`` is extended to cover the
    merged range so the cited span stays accurate.
    """
    from ..memory.agent_context import estimate_tokens

    remove: set[int] = set()
    for i in range(len(flat) - 1, -1, -1):
        if i in remove:
            continue
        if int(flat[i].get("token_count", 0)) >= min_tokens:
            continue
        end = _descendant_slice(flat, i)
        child_texts: list[str] = []
        max_end = int(flat[i]["char_end"])
        for k in range(i + 1, end):
            if k in remove:
                continue
            child = flat[k]
            if str(child.get("text", "")).strip():
                child_texts.append(str(child["text"]))
            max_end = max(max_end, int(child["char_end"]))
            remove.add(k)
        if child_texts:
            merged = str(flat[i].get("text", ""))
            for ct in child_texts:
                if merged and not merged.endswith("\n"):
                    merged += "\n\n"
                merged += ct
            flat[i]["text"] = merged
            flat[i]["char_end"] = max_end
            flat[i]["token_count"] = estimate_tokens(merged)
    return [f for k, f in enumerate(flat) if k not in remove]


def _assign_node_ids(roots: Sequence[SectionNode]) -> None:
    """Assign zero-padded pre-order-DFS ids in place (PageIndex ``node_id``)."""
    counter = [0]

    def _walk(nodes: Sequence[SectionNode]) -> None:
        for n in nodes:
            counter[0] += 1
            n.node_id = str(counter[0]).zfill(4)
            _walk(n.children)

    _walk(roots)


def _build_tree_from_flat(
    flat: list[dict[str, Any]], *, summarizer: SectionSummarizer | None
) -> list[SectionNode]:
    """Level-stack tree build (PageIndex ``build_tree_from_nodes``) → SectionNodes."""
    roots: list[SectionNode] = []
    stack: list[tuple[SectionNode, int]] = []
    for f in flat:
        node = SectionNode(
            node_id="",
            title=str(f["title"]),
            level=int(f["level"]),
            char_start=int(f["char_start"]),
            char_end=int(f["char_end"]),
            line_start=int(f.get("line_num", 0)),
            page_start=f.get("page_start"),
            page_end=f.get("page_end"),
            text=str(f.get("text", "")),
        )
        while stack and stack[-1][1] >= node.level:
            stack.pop()
        if not stack:
            roots.append(node)
        else:
            stack[-1][0].children.append(node)
        stack.append((node, node.level))
    _assign_node_ids(roots)
    if summarizer is not None:
        for s in iter_sections(roots):
            try:
                s.summary = summarizer(s.title, s.text)
            except Exception as exc:  # noqa: BLE001 — summary must never break build
                logger.debug(
                    "[section-tree] summarizer failed for %r: %s", s.title, exc
                )
    return roots


def _default_section_summarizer(
    threshold: int = DEFAULT_SUMMARY_TOKEN_THRESHOLD,
) -> SectionSummarizer:
    """Build the default per-node summarizer (CONCEPT:AU-KG.retrieval.section-tree).

    Reuses the contextual enricher's :meth:`ContextualEnricher.summarize_section`
    (LLM when configured, deterministic heuristic otherwise). Short sections keep
    their own text as the summary (PageIndex ``summary_token_threshold``).
    """
    from ..memory.agent_context import estimate_tokens
    from .contextual_enrichment import ContextualEnricher

    enricher = ContextualEnricher(DocumentProcessor._contextual_llm_fn())

    def _summarize(title: str, text: str) -> str:
        if estimate_tokens(text) < threshold:
            return text.strip()
        return enricher.summarize_section(title, text)

    return _summarize


def build_section_tree(
    text: str,
    *,
    config: SectionTreeConfig | None = None,
    summarizer: SectionSummarizer | None = None,
) -> list[SectionNode]:
    """Build a document's section tree from markdown headings (LLM-free).

    CONCEPT:AU-KG.retrieval.section-tree. Deterministic port of PageIndex's
    markdown path: extract headings (skipping code fences) → attach per-node text
    spans → optionally token-budget-thin → level-stack into a tree with pre-order
    ids. When the document has no headings a single root spanning the whole text
    is returned so callers always get a usable tree.

    Args:
        text: The document text (markdown or plain).
        config: :class:`SectionTreeConfig`; sensible defaults when omitted.
        summarizer: Optional ``(title, text) -> summary``. When omitted and
            ``config.summarize`` is set, the default enricher-backed summarizer is
            used; otherwise summaries are left empty.
    """
    cfg = config or SectionTreeConfig()
    eff_summarizer = summarizer
    if eff_summarizer is None and cfg.summarize:
        eff_summarizer = _default_section_summarizer(cfg.summary_token_threshold)

    headings = _extract_heading_lines(text)
    if not headings:
        title = DocumentProcessor._first_line(text, "Document")
        root = SectionNode(
            node_id="0001",
            title=title,
            level=1,
            char_start=0,
            char_end=len(text),
            text=text.strip(),
        )
        if eff_summarizer is not None:
            try:
                root.summary = eff_summarizer(root.title, root.text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[section-tree] root summary failed: %s", exc)
        return [root]

    flat = _sections_with_spans(headings, text)
    if cfg.thin:
        flat = _annotate_token_counts(flat)
        flat = _thin_sections(flat, cfg.min_node_tokens)
    return _build_tree_from_flat(flat, summarizer=eff_summarizer)


def build_section_tree_from_pages(
    pages: Sequence[str],
    *,
    llm_fn: Any = None,
    config: SectionTreeConfig | None = None,
    summarizer: SectionSummarizer | None = None,
) -> list[SectionNode]:
    """Build a section tree for a paginated (PDF) document.

    CONCEPT:AU-KG.ingest.toc-detection — LLM-assisted table-of-contents detection
    (PageIndex ``check_toc``/``toc_transformer``). The leading ``max_pages_for_toc``
    pages are scanned with ``llm_fn`` (the same remote vLLM lite-LLM the contextual
    enricher uses — no new dependency) for a table of contents; a detected TOC is
    transformed into ``{title, level, page}`` entries, and each entry's char span
    is resolved against the concatenated page text with ``page_start``/``page_end``
    populated. When no ``llm_fn`` is available, or no TOC is found, this degrades
    deterministically: markdown headings across the concatenated text if present,
    else one section per page — so the PDF path always yields a usable tree.

    Args:
        pages: Per-page extracted text, in physical page order (1-indexed).
        llm_fn: Optional ``(prompt) -> completion``. When omitted the enricher's
            lite-LLM is used if configured; absent that, the deterministic path.
        config: :class:`SectionTreeConfig`.
        summarizer: Optional per-node summarizer (see :func:`build_section_tree`).
    """
    cfg = config or SectionTreeConfig()
    page_list = [p or "" for p in pages]
    full_text, page_bounds = _concat_pages(page_list)

    eff_summarizer = summarizer
    if eff_summarizer is None and cfg.summarize:
        eff_summarizer = _default_section_summarizer(cfg.summary_token_threshold)

    fn = llm_fn if llm_fn is not None else DocumentProcessor._contextual_llm_fn()
    toc: list[dict[str, Any]] = []
    if fn is not None:
        toc = _detect_toc(page_list[: cfg.max_pages_for_toc], fn)

    if toc:
        flat = _flat_from_toc(toc, full_text, page_bounds)
        if flat:
            return _build_tree_from_flat(flat, summarizer=eff_summarizer)

    # Deterministic fallbacks: markdown headings, else one node per page.
    if _extract_heading_lines(full_text):
        return build_section_tree(full_text, config=cfg, summarizer=eff_summarizer)
    flat = [
        {
            "title": f"Page {idx + 1}",
            "level": 1,
            "line_num": 0,
            "char_start": start,
            "char_end": end,
            "text": page_list[idx].strip(),
            "page_start": idx + 1,
            "page_end": idx + 1,
        }
        for idx, (start, end) in enumerate(page_bounds)
    ]
    return _build_tree_from_flat(flat, summarizer=eff_summarizer)


def _concat_pages(pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate pages with ``\\f`` separators; return text + per-page char bounds."""
    parts: list[str] = []
    bounds: list[tuple[int, int]] = []
    pos = 0
    for i, p in enumerate(pages):
        start = pos
        parts.append(p)
        pos += len(p)
        bounds.append((start, pos))
        if i < len(pages) - 1:
            parts.append("\f")
            pos += 1
    return "".join(parts), bounds


_TOC_PROMPT = (
    "You are given the opening pages of a document. If they contain a table of "
    "contents, return it as a JSON list of objects with keys 'title' (string), "
    "'level' (1-based heading depth as an integer) and 'page' (the 1-based page "
    "number the section starts on, an integer). Return ONLY the JSON list, or the "
    "empty list [] if there is no table of contents.\n\n<pages>\n{pages}\n</pages>"
)


def _detect_toc(pages: list[str], llm_fn: Any) -> list[dict[str, Any]]:
    """Ask ``llm_fn`` for a normalized ``[{title, level, page}]`` TOC (or [])."""
    import json as _json

    joined = "\n\n".join(
        f"[page {i + 1}]\n{p[:4000]}" for i, p in enumerate(pages) if p
    )
    if not joined.strip():
        return []
    try:
        out = llm_fn(_TOC_PROMPT.format(pages=joined[:_MAX_TOC_CHARS]))
    except Exception as exc:  # noqa: BLE001 — no TOC on LLM failure
        logger.debug("[KG toc-detection] llm_fn failed: %s", exc)
        return []
    if not out:
        return []
    raw = out[out.find("[") : out.rfind("]") + 1] if "[" in out else ""
    try:
        data = _json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001 — malformed → treat as no TOC
        return []
    entries: list[dict[str, Any]] = []
    for row in data if isinstance(data, list) else []:
        if not isinstance(row, dict) or not row.get("title"):
            continue
        try:
            entries.append(
                {
                    "title": str(row["title"]).strip(),
                    "level": max(1, int(row.get("level", 1))),
                    "page": max(1, int(row.get("page", 1))),
                }
            )
        except (TypeError, ValueError):
            continue
    return entries


_MAX_TOC_CHARS = 16_000


def _flat_from_toc(
    toc: list[dict[str, Any]],
    full_text: str,
    page_bounds: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Turn a ``[{title, level, page}]`` TOC into flat span dicts.

    Each entry's char span runs from its page start to the next entry's page
    start (PageIndex per-node text); ``page_start``/``page_end`` are recorded.
    """
    n_pages = len(page_bounds)
    flat: list[dict[str, Any]] = []
    for i, entry in enumerate(toc):
        page = min(max(1, int(entry["page"])), n_pages)
        start = page_bounds[page - 1][0]
        if i + 1 < len(toc):
            next_page = min(max(1, int(toc[i + 1]["page"])), n_pages)
            end = page_bounds[next_page - 1][0]
            end_page = max(page, next_page - 1)
        else:
            end = len(full_text)
            end_page = n_pages
        if end < start:
            end = page_bounds[page - 1][1]
            end_page = page
        flat.append(
            {
                "title": entry["title"],
                "level": int(entry["level"]),
                "line_num": 0,
                "char_start": start,
                "char_end": end,
                "text": full_text[start:end].strip(),
                "page_start": page,
                "page_end": end_page,
            }
        )
    return flat


# ── Section-tree self-verification ────────────────────────────────────────────


def _normalize_title(s: str) -> str:
    """Lowercase + collapse whitespace/markdown markers for title matching."""
    return " ".join(re.sub(r"[#*_`>]", " ", s or "").lower().split())


def verify_section_tree(
    full_text: str, roots: Sequence[SectionNode], *, fix: bool = True
) -> dict[str, Any]:
    """Confirm each section's title appears within its claimed char range.

    CONCEPT:AU-KG.ingest.structure-verify — port of PageIndex ``verify_toc`` /
    ``fix_incorrect_toc``. A tree built from markdown headings is self-consistent
    by construction, but the LLM-assisted TOC path (and aggressive thinning) can
    produce a node whose title is *not* inside its ``[char_start, char_end)`` span.
    This pass checks every node; when ``fix`` is set and the title is found
    elsewhere in the document, the node's ``char_start`` is re-anchored to the
    title's true position (its ``char_end`` extended if needed) so the cited range
    is trustworthy before the tree is committed.

    Returns ``{"checked", "verified", "mismatched", "repaired", "mismatches":[…]}``.
    """
    haystack = full_text.lower()
    checked = verified = repaired = 0
    mismatches: list[dict[str, Any]] = []
    for node in iter_sections(roots):
        checked += 1
        title_norm = _normalize_title(node.title)
        span_norm = _normalize_title(full_text[node.char_start : node.char_end])
        if not title_norm or title_norm in span_norm:
            verified += 1
            continue
        # Title not in its claimed span — try to locate it in the whole document.
        found = haystack.find(node.title.lower())
        entry: dict[str, Any] = {"node_id": node.node_id, "title": node.title}
        if fix and found >= 0:
            node.char_start = found
            if node.char_end <= found:
                node.char_end = min(len(full_text), found + max(1, len(node.title)))
            repaired += 1
            entry["repaired_to"] = found
        else:
            mismatches.append(entry)
    return {
        "checked": checked,
        "verified": verified,
        "mismatched": len(mismatches),
        "repaired": repaired,
        "mismatches": mismatches,
    }


# ── Section-tree materialization (nodes/edges + reconstruction) ───────────────


def section_nodes_and_edges(
    document_id: str, roots: Sequence[SectionNode]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build ``Section`` node payloads + HAS_SECTION/SECTION_OF/HAS_SUBSECTION edges.

    CONCEPT:AU-KG.retrieval.section-tree. Each node stores its ``parent_id`` so the
    tree can be reconstructed from the graph by :func:`rebuild_section_tree`.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def _emit(section: SectionNode, parent_full_id: str | None) -> None:
        full_id = f"{document_id}::section::{section.node_id}"
        node: dict[str, Any] = {
            "id": full_id,
            "type": SECTION_NODE_TYPE,
            "name": section.title,
            "document_id": document_id,
            # ``tree_node_id`` (not ``node_id``) so the property never collides
            # with the keyword-writer's ``node_id`` positional parameter.
            "tree_node_id": section.node_id,
            "parent_id": parent_full_id or "",
            "title": section.title,
            "level": section.level,
            "char_start": section.char_start,
            "char_end": section.char_end,
            "line_start": section.line_start,
            "summary": section.summary,
            # The node's own text is kept for range-fetch (get_page_content) and
            # stripped only in the structure view (get_document_structure), exactly
            # as PageIndex does — the *map* is text-free, the store is not.
            "content": section.text,
        }
        if section.page_start is not None:
            node["page_start"] = section.page_start
        if section.page_end is not None:
            node["page_end"] = section.page_end
        nodes.append(node)
        edges.append(
            {
                "source": document_id,
                "target": full_id,
                "type": HAS_SECTION_EDGE,
                "tree_node_id": section.node_id,
            }
        )
        edges.append(
            {"source": full_id, "target": document_id, "type": SECTION_OF_EDGE}
        )
        if parent_full_id:
            edges.append(
                {
                    "source": parent_full_id,
                    "target": full_id,
                    "type": HAS_SUBSECTION_EDGE,
                }
            )
        for child in section.children:
            _emit(child, full_id)

    for root in roots:
        _emit(root, None)
    return nodes, edges


def rebuild_section_tree(section_nodes: Sequence[dict[str, Any]]) -> list[SectionNode]:
    """Reconstruct the nested tree from flat stored ``Section`` node dicts.

    CONCEPT:AU-KG.retrieval.section-tree — the inverse of
    :func:`section_nodes_and_edges`, used by the hierarchical retriever after it
    loads a document's Section nodes from the graph. Ordering is by ``node_id``
    (the pre-order index) so children attach under their recorded ``parent_id``.
    """

    def _tid(raw: dict[str, Any]) -> str:
        return str(raw.get("tree_node_id") or raw.get("node_id") or "")

    by_id: dict[str, SectionNode] = {}
    parent_of: dict[str, str] = {}
    order = sorted(section_nodes, key=_tid)
    for raw in order:
        full_id = str(raw.get("id") or _tid(raw))
        node = SectionNode(
            node_id=_tid(raw),
            title=str(raw.get("title") or raw.get("name") or ""),
            level=int(raw.get("level", 1) or 1),
            char_start=int(raw.get("char_start", 0) or 0),
            char_end=int(raw.get("char_end", 0) or 0),
            line_start=int(raw.get("line_start", 0) or 0),
            page_start=raw.get("page_start"),
            page_end=raw.get("page_end"),
            summary=str(raw.get("summary", "") or ""),
            text=str(raw.get("content", "") or raw.get("text", "") or ""),
        )
        by_id[full_id] = node
        parent_of[full_id] = str(raw.get("parent_id", "") or "")

    roots: list[SectionNode] = []
    for full_id, node in by_id.items():
        parent_id = parent_of.get(full_id, "")
        parent = by_id.get(parent_id) if parent_id else None
        if parent is not None:
            parent.children.append(node)
        else:
            roots.append(node)
    return roots


def process_document(
    document: str | bytes | Path,
    graph: Any = None,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience entry: process ``document`` and return the materialized mapping.

    CONCEPT:AU-KG.ingest.chunk-overlap-stage — the one-call form the ingestion engine / MCP wrapper invokes:
    ``{document_node, chunk_nodes, edges}``. Writes through ``graph`` when given.
    """
    proc = DocumentProcessor(
        graph, chunking=ChunkingConfig(chunk_size=chunk_size, overlap=overlap)
    )
    return proc.process(document, **kwargs).as_dict()


__all__ = [
    "ChunkingConfig",
    "ChunkSpan",
    "chunk_text",
    "DocumentChunk",
    "ProcessedDocument",
    "DocumentExtractionError",
    "DocumentProcessor",
    "process_document",
    "DEFAULT_EMBEDDING_DIM",
    "DOCUMENT_NODE_TYPE",
    "CHUNK_NODE_TYPE",
    "HAS_CHUNK_EDGE",
    "CHUNK_OF_EDGE",
    # Section-tree (CONCEPT:AU-KG.retrieval.section-tree)
    "SectionNode",
    "SectionTreeConfig",
    "SectionSummarizer",
    "build_section_tree",
    "build_section_tree_from_pages",
    "iter_sections",
    "section_nodes_and_edges",
    "rebuild_section_tree",
    "verify_section_tree",
    "SECTION_NODE_TYPE",
    "HAS_SECTION_EDGE",
    "SECTION_OF_EDGE",
    "HAS_SUBSECTION_EDGE",
]
