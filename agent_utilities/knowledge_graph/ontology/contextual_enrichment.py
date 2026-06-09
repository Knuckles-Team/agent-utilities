from __future__ import annotations

"""Contextual-retrieval enrichment — per-chunk situating context.

CONCEPT:KG-2.50 — Contextual-Retrieval Enrichment

Provenance: Anthropic "Contextual Retrieval" (situating each chunk within its
document before embedding markedly improves retrieval recall) and the contextual
enrichment stage of Onyx's indexing pipeline. This module ports that idea onto the
KG-2.48 ``DocumentProcessor``: before a chunk is embedded, a short *context*
string is computed that situates the chunk inside the whole document, and the
chunk is embedded as ``"{context}\n\n{chunk text}"`` (and the context is stored on
the ``Chunk`` node as ``context`` / ``contextual_summary`` for display + lexical
match).

Two paths, the second always available offline:

  * **LLM path** — when an ``llm_fn`` (e.g. ``enrichment.cards.make_lite_llm_fn``)
    is supplied, a single per-document summary is computed once and reused as a
    header, then each chunk gets a 1–2 sentence situating context. The per-doc
    summary keeps token cost bounded (one extra call per document, not per chunk).
  * **Heuristic fallback** — when no ``llm_fn`` is given (or it raises), a
    *deterministic* context is built from the document title, the nearest
    preceding markdown heading, the chunk's ``part i/n`` position, and the
    document's first sentence. Deterministic → offline tests are stable and the
    pipeline never depends on a live model.

Default OFF in ``DocumentProcessor`` so existing KG-2.48 behaviour is byte
identical; the connector ingestion path turns it ON.
"""

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ["ContextualEnricher"]

LLMFn = Callable[[str], str]

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Anthropic contextual-retrieval prompt: situate the chunk, return context only.
_CHUNK_PROMPT = (
    "<document>\n{doc_summary}\n</document>\n"
    "Here is a chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Give a short (1-2 sentence) context that situates this chunk within the "
    "overall document, for improving search retrieval of the chunk. Answer with "
    "ONLY the succinct context and nothing else."
)
_DOC_SUMMARY_PROMPT = (
    "Summarize the following document in 2-3 sentences capturing its subject and "
    "purpose, for use as retrieval context. Answer with only the summary.\n\n"
    "<document>\n{doc}\n</document>"
)

# Bound the document text fed to the LLM so a huge doc cannot blow the context.
_MAX_DOC_CHARS = 12_000


class ContextualEnricher:
    """Compute situating context for chunks before embedding (CONCEPT:KG-2.50).

    Args:
        llm_fn: Optional ``(prompt) -> completion`` callable. When ``None`` (or it
            raises), the deterministic heuristic is used so the pipeline works
            offline.
        max_context_chars: Cap on a single context string (keeps the embedded
            ``context + chunk`` from ballooning).
    """

    def __init__(
        self, llm_fn: LLMFn | None = None, *, max_context_chars: int = 400
    ) -> None:
        self.llm_fn = llm_fn
        self.max_context_chars = max_context_chars

    # -- public API --------------------------------------------------------

    def document_summary(self, doc_text: str, title: str = "") -> str:
        """Return a short document summary (LLM if available, else heuristic).

        Computed once per document and reused as the context header for every
        chunk so the LLM path costs one extra call per document.
        """
        if self.llm_fn is not None:
            try:
                out = self.llm_fn(
                    _DOC_SUMMARY_PROMPT.format(doc=doc_text[:_MAX_DOC_CHARS])
                )
                if out and out.strip():
                    return out.strip()
            except Exception as exc:  # noqa: BLE001 — degrade to heuristic
                logger.debug("[KG-2.50] doc summary LLM failed: %s", exc)
        return self._heuristic_summary(doc_text, title)

    def enrich(
        self, doc_text: str, chunk_texts: list[str], *, title: str = ""
    ) -> list[str]:
        """Return one context string per chunk, aligned to ``chunk_texts``.

        CONCEPT:KG-2.50. The LLM path computes the doc summary once then situates
        each chunk; the heuristic path is fully deterministic.
        """
        if not chunk_texts:
            return []
        summary = self.document_summary(doc_text, title)
        first_sentence = self._first_sentence(doc_text)
        n = len(chunk_texts)
        out: list[str] = []
        for i, chunk in enumerate(chunk_texts):
            ctx = self._context_for_chunk(
                chunk,
                summary=summary,
                doc_text=doc_text,
                title=title,
                first_sentence=first_sentence,
                index=i,
                total=n,
            )
            out.append(ctx[: self.max_context_chars].strip())
        return out

    # -- LLM / heuristic per-chunk context ---------------------------------

    def _context_for_chunk(
        self,
        chunk: str,
        *,
        summary: str,
        doc_text: str,
        title: str,
        first_sentence: str,
        index: int,
        total: int,
    ) -> str:
        if self.llm_fn is not None:
            try:
                out = self.llm_fn(
                    _CHUNK_PROMPT.format(doc_summary=summary, chunk=chunk[:4000])
                )
                if out and out.strip():
                    return out.strip()
            except Exception as exc:  # noqa: BLE001 — degrade to heuristic
                logger.debug("[KG-2.50] chunk context LLM failed: %s", exc)
        return self._heuristic_context(
            chunk,
            doc_text=doc_text,
            title=title,
            first_sentence=first_sentence,
            index=index,
            total=total,
        )

    def _heuristic_context(
        self,
        chunk: str,
        *,
        doc_text: str,
        title: str,
        first_sentence: str,
        index: int,
        total: int,
    ) -> str:
        """Deterministic situating context from document structure.

        CONCEPT:KG-2.50 — title + nearest preceding heading + ``part i/n`` +
        opening sentence. No randomness → identical output for identical input, so
        offline tests are stable.
        """
        parts: list[str] = []
        doc_label = title.strip() or "the document"
        parts.append(f"From {doc_label}")
        heading = self._nearest_heading(doc_text, chunk)
        if heading:
            parts.append(f"under section '{heading}'")
        if total > 1:
            parts.append(f"(part {index + 1} of {total})")
        prefix = " ".join(parts) + "."
        if first_sentence and first_sentence.lower() not in chunk.lower():
            prefix += f" The document opens: {first_sentence}"
        return prefix

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _heuristic_summary(doc_text: str, title: str) -> str:
        sentences = _SENTENCE_RE.split(doc_text.strip())
        lead = " ".join(s.strip() for s in sentences[:2] if s.strip())
        label = title.strip()
        if label and lead:
            return f"{label}. {lead}"[:600]
        return (label or lead or "Document")[:600]

    @staticmethod
    def _first_sentence(doc_text: str) -> str:
        for line in doc_text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return _SENTENCE_RE.split(stripped)[0][:200]
        return ""

    @staticmethod
    def _nearest_heading(doc_text: str, chunk: str) -> str:
        """The last markdown heading appearing before the chunk's position."""
        pos = doc_text.find(chunk[:80]) if chunk else -1
        if pos < 0:
            return ""
        best = ""
        for m in _HEADING_RE.finditer(doc_text):
            if m.start() <= pos:
                best = m.group(1).strip()
            else:
                break
        return best[:120]
