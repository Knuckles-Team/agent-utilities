"""Document metadata + concept extraction (CONCEPT:KG-2.8 Phase 3).

Extracts a typed ``Document`` (paper/email/BRD/SOW/book/...) with type-aware
metadata, plus the ``Concept`` nodes it mentions (LLM, injectable). Concepts are
canonicalised by name so the same idea mentioned by many documents — and realized
by code — converges on one node. This is what makes cross-ingestion discovery and
research→code distillation possible.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from typing import Any

from ..models import Concept, Document, EnrichmentEdge

LLMFn = Callable[[str], str]

_EMAIL_EXT = {".eml", ".msg"}
_DOC_EXT = {".md", ".txt", ".rst", ".pdf", ".docx", ".doc"}


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:80] or "concept"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def detect_doc_type(file_path: str, text: str) -> str:
    """Classify the document by extension + filename + content cues."""
    base = os.path.basename(file_path).lower()
    ext = os.path.splitext(base)[1]
    head = text[:2000].lower()
    if ext in _EMAIL_EXT or re.search(r"^\s*from:.*\n.*^\s*to:", head, re.M):
        return "email"
    if "statement of work" in head or base.startswith(("sow", "sow_")) or "sow" in base:
        return "sow"
    if "business requirement" in head or "brd" in base or "requirements" in base:
        return "brd"
    if (
        ext == ".pdf"
        and re.search(r"\babstract\b", head)
        and re.search(r"\breferences\b", text.lower())
    ):
        return "paper"
    if "isbn" in head or "chapter 1" in head or "table of contents" in head:
        return "book"
    if base.startswith("paper") or "/papers/" in file_path.replace("\\", "/"):
        return "paper"
    return "document"


def read_document_text(file_path: str, max_chars: int = 200_000) -> str:
    """Best-effort text read. Plain text/markdown directly; PDF via pypdf if present."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in (".md", ".txt", ".rst", ".json", ".eml"):
            return open(file_path, encoding="utf-8", errors="ignore").read()[:max_chars]
        if ext == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(file_path)
                return "\n".join(p.extract_text() or "" for p in reader.pages)[
                    :max_chars
                ]
            except Exception:
                return ""
    except OSError:
        return ""
    return ""


def extract_metadata(file_path: str, text: str, doc_type: str) -> dict[str, Any]:
    """Type-aware lightweight metadata (LLM can enrich later)."""
    md: dict[str, Any] = {"doc_type": doc_type, "chars": len(text)}
    head = text[:4000]
    if doc_type == "email":
        for field in ("From", "To", "Subject", "Date", "Cc"):
            m = re.search(rf"^\s*{field}:\s*(.+)$", head, re.M | re.I)
            if m:
                md[field.lower()] = m.group(1).strip()[:300]
    elif doc_type == "paper":
        m = re.search(
            r"\babstract\b[:\s]*(.+?)(?:\n\n|\bintroduction\b)", head, re.I | re.S
        )
        if m:
            md["abstract"] = re.sub(r"\s+", " ", m.group(1)).strip()[:1000]
    # Title: first non-empty line, else filename.
    for line in text.splitlines():
        if line.strip():
            md.setdefault("title", line.strip()[:200])
            break
    md.setdefault("title", os.path.basename(file_path))
    return md


_CONCEPT_PROMPT = """From this {doc_type} titled "{title}", extract the key
concepts, techniques, methods, or claims that could inform software design or
implementation. Focus on reusable, actionable ideas.

TEXT (excerpt):
{excerpt}

Output ONLY a JSON array of objects, each with keys "name" (short noun phrase),
"kind" (one of: concept, technique, method, claim, requirement), and "summary"
(one sentence). Max {limit} items. No other text."""


def extract_concepts(
    text: str,
    source_id: str,
    llm_fn: LLMFn,
    *,
    source_type: str = "document",
    title: str = "",
    limit: int = 12,
) -> list[Concept]:
    """LLM-extract canonical Concept nodes from any text.

    ``source_id`` becomes each concept's provenance (``source_ids``). Works for
    documents, chat threads, prompts, etc. — the generic ``extract_text_concepts``
    wrapper (``extractors/text.py``) builds the MENTIONS edges.
    """
    import json

    prompt = _CONCEPT_PROMPT.format(
        doc_type=source_type, title=title or source_id, excerpt=text[:8000], limit=limit
    )
    try:
        raw = llm_fn(prompt)
        start, end = raw.index("["), raw.rindex("]") + 1
        items = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError, Exception):
        return []
    concepts: list[Concept] = []
    seen: set[str] = set()
    for it in items[:limit]:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        concepts.append(
            Concept(
                id=f"concept:{slug(name)}",
                name=name,
                kind=str(it.get("kind", "concept")).strip() or "concept",
                summary=str(it.get("summary", "")).strip(),
                source_ids=[source_id],
            )
        )
    return concepts


def extract_document(
    file_path: str, text: str, llm_fn: LLMFn, max_concepts: int = 12
) -> tuple[Document, list[Concept], list[EnrichmentEdge]]:
    """Extract a Document + its Concepts + MENTIONS edges."""
    content_hash = _sha(text)
    doc_type = detect_doc_type(file_path, text)
    metadata = extract_metadata(file_path, text, doc_type)
    doc = Document(
        id=f"doc:{_sha(file_path)[:16]}",
        title=metadata.get("title", os.path.basename(file_path)),
        doc_type=doc_type,
        file_path=file_path,
        content_hash=content_hash,
        metadata=metadata,
    )
    concepts = extract_concepts(
        text,
        doc.id,
        llm_fn,
        source_type=doc.doc_type,
        title=doc.title,
        limit=max_concepts,
    )
    doc.concept_ids = [c.id for c in concepts]
    edges = [
        EnrichmentEdge(source=doc.id, target=c.id, rel_type="MENTIONS")
        for c in concepts
    ]
    return doc, concepts, edges
