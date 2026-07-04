"""Document metadata + concept extraction (CONCEPT:EG-KG.storage.nonblocking-checkpoint Phase 3).

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

from ..models import (
    Concept,
    Document,
    EnrichmentEdge,
    Fact,
    Framework,
    Insight,
    Playbook,
)

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


def read_document_text(file_path: str, max_chars: int = 8_000_000) -> str:
    """Best-effort text read. Plain text/markdown directly; PDF via PyMuPDF.

    The cap is generous (whole books are the target) — size is bounded downstream
    by chunking, not by truncating the verbatim ``Document`` content.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in (".md", ".txt", ".rst", ".json", ".eml"):
            return open(file_path, encoding="utf-8", errors="ignore").read()[:max_chars]
        if ext == ".pdf":
            return _read_pdf_text(file_path)[:max_chars]
    except OSError:
        return ""
    return ""


def _read_pdf_text(file_path: str) -> str:
    """Extract a PDF's text. PyMuPDF (fitz) first — a GIL-releasing C parser
    ~100x faster than pypdf, which can stall for minutes on large PDFs and wedge
    the host (the same fast path ``kb/parser._read_pdf`` uses). Falls back to
    pypdf only if PyMuPDF is unavailable."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(file_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass
    except Exception:
        return ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
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


def _parse_json_objects(raw: str) -> list[dict[str, Any]]:
    """Extract JSON objects from an LLM array response, tolerating truncation.

    LLM completions are routinely cut off by ``max_tokens`` mid-array, so the
    closing ``]`` is often missing and a strict ``json.loads`` of ``[...]`` fails
    (dropping ALL items). This salvages every COMPLETE ``{...}`` object via a
    string-aware brace scan, ignoring a trailing incomplete one — turning a
    truncated array into the concepts it did manage to emit.
    """
    import json

    objs: list[dict[str, Any]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(raw[start : i + 1])
                    if isinstance(parsed, dict):
                        objs.append(parsed)
                except (ValueError, json.JSONDecodeError):
                    pass
                start = -1
    return objs


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
    prompt = _CONCEPT_PROMPT.format(
        doc_type=source_type, title=title or source_id, excerpt=text[:8000], limit=limit
    )
    try:
        raw = llm_fn(prompt)
    except Exception:  # noqa: BLE001
        return []
    items = _parse_json_objects(raw)
    if not items:
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


_INTEL_PROMPT = """From this {doc_type} titled "{title}", extract reusable
operating intelligence — the kind of takeaways a team would turn into training,
playbooks, or risk flags (e.g. from a sales call: objections, positioning
signals, next-step procedures).

TEXT (excerpt):
{excerpt}

Output ONLY a JSON object with these keys, each a JSON array (omit or empty if
none), max {limit} items each:
- "insights": objects with "title" and "reasoning" (one sentence each).
- "facts": objects with "statement" (a discrete checkable assertion).
- "frameworks": objects with "name", "summary", and "steps" (array of strings).
- "playbooks": objects with "name", "steps" (array), "preconditions" (array),
  and "expected_outcome".
No other text."""


def _safe_json_obj(raw: str) -> dict:
    import json

    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        obj = json.loads(raw[start:end])
        return obj if isinstance(obj, dict) else {}
    except (ValueError, json.JSONDecodeError, Exception):
        return {}


def extract_intelligence(
    text: str,
    source_id: str,
    llm_fn: LLMFn,
    *,
    source_type: str = "document",
    title: str = "",
    limit: int = 8,
) -> tuple[list[Any], list[EnrichmentEdge]]:
    """LLM-extract Insight/Fact/Framework/Playbook nodes + DERIVED_FROM edges.

    Turns a call/doc into reusable operating intelligence (CONCEPT:EG-KG.storage.nonblocking-checkpoint). Each
    returned node carries ``source_ids=[source_id]``; edges link ``source_id``
    (the document/conversation) to each derived node via ``DERIVED_FROM``.
    """
    prompt = _INTEL_PROMPT.format(
        doc_type=source_type, title=title or source_id, excerpt=text[:8000], limit=limit
    )
    obj = _safe_json_obj(llm_fn(prompt)) if text.strip() else {}
    nodes: list[Any] = []
    edges: list[EnrichmentEdge] = []

    def _str(v: Any) -> str:
        return str(v).strip()

    def _list(v: Any) -> list[str]:
        return [_str(s) for s in v if _str(s)] if isinstance(v, list) else []

    for it in (obj.get("insights") or [])[:limit]:
        if isinstance(it, dict) and _str(it.get("title")):
            t = _str(it["title"])
            nodes.append(
                Insight(
                    id=f"insight:{slug(t)}",
                    title=t,
                    reasoning=_str(it.get("reasoning")),
                    source_ids=[source_id],
                )
            )
    for it in (obj.get("facts") or [])[:limit]:
        if isinstance(it, dict) and _str(it.get("statement")):
            s = _str(it["statement"])
            nodes.append(
                Fact(id=f"fact:{slug(s)}", statement=s, source_ids=[source_id])
            )
    for it in (obj.get("frameworks") or [])[:limit]:
        if isinstance(it, dict) and _str(it.get("name")):
            n = _str(it["name"])
            nodes.append(
                Framework(
                    id=f"framework:{slug(n)}",
                    name=n,
                    summary=_str(it.get("summary")),
                    steps=_list(it.get("steps")),
                    source_ids=[source_id],
                )
            )
    for it in (obj.get("playbooks") or [])[:limit]:
        if isinstance(it, dict) and _str(it.get("name")):
            n = _str(it["name"])
            nodes.append(
                Playbook(
                    id=f"playbook:{slug(n)}",
                    name=n,
                    steps=_list(it.get("steps")),
                    preconditions=_list(it.get("preconditions")),
                    expected_outcome=_str(it.get("expected_outcome")),
                    source_ids=[source_id],
                )
            )

    edges = [
        EnrichmentEdge(source=source_id, target=n.id, rel_type="DERIVED_FROM")
        for n in nodes
    ]
    return nodes, edges


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
        content=text,
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
