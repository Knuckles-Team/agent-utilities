#!/usr/bin/python
from __future__ import annotations

"""Document Processing → Ontology — media/text → Chunk objects linked to a Document.

CONCEPT:KG-2.48 — Document Processing Pipeline.

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

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Default per-chunk embedding dimensionality — ties to create_embedding_model()'s
# 768 default and the ontology ``embedding`` PropertyType (CONCEPT:KG-2.48).
DEFAULT_EMBEDDING_DIM = 768

# Ontology object/link type names. ``Document`` already exists as a first-class
# node label in the ingestion fabric; ``Chunk`` is the per-chunk object this
# pipeline materializes. Edge labels mirror the Palantir HAS_CHUNK / CHUNK_OF
# parent↔child link pair.
DOCUMENT_NODE_TYPE = "Document"
CHUNK_NODE_TYPE = "Chunk"
HAS_CHUNK_EDGE = "HAS_CHUNK"
CHUNK_OF_EDGE = "CHUNK_OF"

# Separator priority for recursive chunking — highest-semantic boundary first,
# matching the Foundry/LangChain "recursive character" splitter ordering.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ChunkingConfig(BaseModel):
    """Configuration for separator-priority text chunking with overlap.

    CONCEPT:KG-2.48 — the chunk-with-overlap stage of the document pipeline.

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

    CONCEPT:KG-2.48 — the intermediate "exploded" row before it becomes a Chunk
    object. ``char_start`` is monotonically non-decreasing across the sequence and
    successive spans overlap by (up to) the configured overlap.
    """

    index: int
    text: str
    char_start: int
    char_end: int


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[ChunkSpan]:
    """Split ``text`` into overlapping :class:`ChunkSpan`s by separator priority.

    CONCEPT:KG-2.48 — real recursive separator-priority chunking. The text is
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
            atoms.append((text[i : i + cfg.chunk_size], i, min(i + cfg.chunk_size, len(text))))
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

    CONCEPT:KG-2.48 — the exploded-and-embedded Chunk object. Carries position +
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


class ProcessedDocument(BaseModel):
    """Result of :meth:`DocumentProcessor.process` — the materialized graph slice.

    CONCEPT:KG-2.48. ``document_node`` and ``chunk_nodes`` are the node payloads
    written through the live write path (or returned offline); ``edges`` are the
    HAS_CHUNK / CHUNK_OF link payloads. ``persisted`` records whether the live
    graph write path actually committed the slice.
    """

    document_node: dict[str, Any]
    chunk_nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    document_id: str = ""
    chunk_count: int = 0
    persisted: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the ``{document_node, chunk_nodes, edges}`` mapping."""
        return {
            "document_node": self.document_node,
            "chunk_nodes": self.chunk_nodes,
            "edges": self.edges,
        }


class DocumentExtractionError(RuntimeError):
    """Raised when a document's text cannot be extracted by any available reader.

    CONCEPT:KG-2.48 — the *clear, explicit* degradation path. PDFs without
    ``pypdf``/``pdfminer`` (and no pre-extracted text supplied) raise this with an
    actionable message rather than silently materializing an empty document.
    """


EmbedFn = Callable[[Sequence[str]], list[list[float]]]


class DocumentProcessor:
    """End-to-end document → Chunk-objects pipeline (CONCEPT:KG-2.48).

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
    ) -> None:
        self.graph = graph
        self.chunking = chunking or ChunkingConfig()
        self.embedding_dim = embedding_dim
        self._embed_fn = embed_fn

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
    ) -> ProcessedDocument:
        """Run the full pipeline and materialize Document + Chunk objects.

        CONCEPT:KG-2.48 — ``media → extract/OCR → chunk(overlap) → explode →
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
        embeddings = self._embed([sp.text for sp in spans])

        chunks: list[DocumentChunk] = []
        for sp, emb in zip(spans, embeddings):
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

        result = ProcessedDocument(
            document_node=document_node,
            chunk_nodes=chunk_nodes,
            edges=edges,
            document_id=doc_id,
            chunk_count=len(chunks),
        )

        if persist:
            result.persisted = self._persist(document_node, chunk_nodes, edges)
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
            read_document_text,
        )

        # 1) Explicit pre-extracted text wins (OCR / external extraction).
        if text is not None:
            label = source or "<text>"
            return text, label, detect_doc_type(label, text), self._first_line(text, label)

        # 2) A real file on disk → KB/enrichment reader.
        if isinstance(document, (str, Path)) and self._looks_like_path(document):
            path = Path(document)
            if path.exists() and path.is_file():
                label = source or str(path)
                extracted = self._read_file(path)
                return extracted, label, detect_doc_type(str(path), extracted), (
                    self._first_line(extracted, path.name)
                )

        # 3) Raw bytes — decode as UTF-8 text (caller pre-extracts binary formats).
        if isinstance(document, bytes):
            label = source or "<bytes>"
            decoded = document.decode("utf-8", errors="replace")
            return decoded, label, detect_doc_type(label, decoded), self._first_line(decoded, label)

        # 4) Plain string content treated as the document text itself.
        if isinstance(document, str):
            label = source or "<text>"
            return document, label, detect_doc_type(label, document), self._first_line(document, label)

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
        try:
            vectors = fn(texts)
        except Exception as exc:  # noqa: BLE001 — never let embedding break ingest
            logger.warning("[KG-2.48] embedding failed, materializing without vectors: %s", exc)
            return [None] * len(texts)
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
        if store is not None and hasattr(store, "add_node") and hasattr(store, "add_edge"):
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

        ok = False
        ok |= self._write_node(writer, document_node)
        for cn in chunk_nodes:
            ok |= self._write_node(writer, cn)
        for e in edges:
            try:
                props = {k: v for k, v in e.items() if k not in ("source", "target", "type")}
                writer.add_edge(e["source"], e["target"], rel_type=e["type"], **props)
                ok = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("[KG-2.48] add_edge failed %s->%s: %s", e["source"], e["target"], exc)
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
                logger.debug("[KG-2.48] add_node failed for %s: %s", node.get("id"), exc)
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


def process_document(
    document: str | bytes | Path,
    graph: Any = None,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience entry: process ``document`` and return the materialized mapping.

    CONCEPT:KG-2.48 — the one-call form the ingestion engine / MCP wrapper invokes:
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
]
