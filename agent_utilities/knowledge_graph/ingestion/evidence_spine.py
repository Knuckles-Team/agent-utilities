"""``Artifact`` → ``Fragment`` — the canonical, addressable evidence spine.

CONCEPT:AU-KG.ingest.evidence-spine-artifact — one retrieved source object.
CONCEPT:AU-KG.ingest.stable-fragment-address — the citable unit inside it.

**What already existed, and what it lacked.**
:class:`~..ingestion.change_envelope.ChangeEnvelope` already carries the whole
connector contract (source identity, revision, ACL/classification/retention,
payload, bitemporal change events, provenance) and
:func:`~..ingestion.envelope_ingest.ingest_graph_slice` already commits a
multi-node slice atomically.  The engine's wire protocol even declares an
``Artifact`` projection
(:class:`agent_utilities.protocols.epistemic_operations.Artifact`: ``digest``,
``content_ref``, ``segment_ids``, ``loci``) — but **nothing in Python ever
constructed one**, and ``segment_ids`` had no Python type behind it at all.  And
``ontology/document_processing.py`` already chunks a document into ``Chunk``
nodes, with ids of the form ``{doc}::chunk::{index}:{sha(text)[:12]}`` — an id
that is *both* positional *and* content-hashed, so it breaks on an insert above
**and** on a typo fix.  That is the gap this module closes: a **stable,
hashed, orderable, nestable citation address** that survives both.

**The identity scheme, and its trade-off (stated plainly).**

Two different questions get two different fields, because one value cannot
answer both:

* :attr:`Fragment.fragment_id` — the **address**.  Derived from the artifact id
  plus a *scoped structural path* (``h2:getting-started/p:2``), never from the
  fragment's own body text.  This is what a citation stores.
* :attr:`Fragment.content_hash` — the **revision**.  ``sha256`` over the
  fragment's normalized text.  This is what tells a reader whether the thing
  they cited still says what it said.

The rejected alternatives, and why:

* A **purely positional** id (``chunk 7``) is stable across body edits but every
  insert above it renumbers every later fragment — one added paragraph
  invalidates the entire tail of the document's citations.
* A **purely content-hashed** id is stable across inserts but changes the moment
  a typo is fixed, silently breaking the citation to a passage that still says
  the same thing; identical paragraphs also collide onto one id.

The chosen scheme is neither.  Each path segment is ``<kind>:<anchor>``, where
``anchor`` is a **slug of the node's own label** when it has one (a heading, a
captioned table) and a **sibling ordinal** when it does not (a paragraph, a list
item, a table row).  So:

* fixing a typo in a paragraph → id unchanged, ``content_hash`` changes;
* inserting a paragraph in a *different* section → id unchanged;
* inserting a heading anywhere → ids unchanged (headings are addressed by slug,
  not by ordinal);
* re-ingesting the document unchanged → every id byte-identical;
* **renaming a heading** → ids beneath it change.  This is the accepted cost;
  a renamed section is arguably a different section, and the alternative (a
  content-independent synthetic id) requires durable state we would then have to
  keep correct across every re-ingest.
* **inserting a sibling before an anonymous fragment** → that fragment's ordinal
  shifts, so its id changes.  This is the residual weakness, and it is why
  :func:`epistemic_graph.ingestion.citation.resolve_fragment` exists: resolution matches on ``fragment_id`` first
  and falls back to ``(artifact_id, content_hash)``, so a paragraph that merely
  *moved* is still found by the citation that pointed at it.

Fragments are **orderable** (:attr:`Fragment.ordinal` among siblings,
:attr:`Fragment.sequence` in whole-artifact document order) and **nestable**
(:attr:`Fragment.parent_fragment_id` / :attr:`Fragment.depth`), so a row inside a
table inside a section reconstructs exactly.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from epistemic_graph.ingestion.evidence_address import slugify
from epistemic_graph.ingestion.evidence_model import (
    ARTIFACT_NODE_TYPE,
    ARTIFACT_OF_EDGE,
    FRAGMENT_KINDS,
    FRAGMENT_NODE_TYPE,
    FRAGMENT_OF_EDGE,
    HAS_ARTIFACT_EDGE,
    HAS_CHILD_FRAGMENT_EDGE,
    HAS_FRAGMENT_EDGE,
    NEXT_FRAGMENT_EDGE,
    PARENT_FRAGMENT_EDGE,
    Artifact,
    Fragment,
    FragmentKind,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_NODE_TYPE",
    "FRAGMENT_NODE_TYPE",
    "HAS_FRAGMENT_EDGE",
    "FRAGMENT_OF_EDGE",
    "HAS_CHILD_FRAGMENT_EDGE",
    "PARENT_FRAGMENT_EDGE",
    "NEXT_FRAGMENT_EDGE",
    "HAS_ARTIFACT_EDGE",
    "ARTIFACT_OF_EDGE",
    "FRAGMENT_KINDS",
    "fragment_markdown",
    "fragment_pdf",
    "fragment_record",
    "fragment_rowset",
    "ingest_artifact",
    "load_fragments",
    "FragmentKind",
    "Artifact",
    "Fragment",
]

# ── Markdown fragmenter ──────────────────────────────────────────────────────
# The reference producer: a real markdown document -> its fragment tree.
# Dependency-free by design (Dependency discipline: core is the serving plane) —
# ATX headings, fenced code, pipe tables, blockquotes, lists, paragraphs.

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*(\S*)")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


class _Scope:
    """One open heading scope while fragmenting — the parent of what follows."""

    __slots__ = ("level", "path", "fragment_id", "counters", "slugs")

    def __init__(
        self, level: int, path: tuple[str, ...], fragment_id: str | None
    ) -> None:
        self.level = level
        self.path = path
        self.fragment_id = fragment_id
        # Ordinals are counted PER (parent, kind), not per parent.  Inserting a
        # code block therefore never renumbers the paragraphs around it — one
        # more edit class the addresses survive.
        self.counters: dict[str, int] = {}
        # Duplicate heading slugs under the same parent get a deterministic
        # suffix, so two "## Notes" sections never collide onto one address.
        self.slugs: dict[str, int] = {}

    def next_ordinal(self, kind: str) -> int:
        value = self.counters.get(kind, 0)
        self.counters[kind] = value + 1
        return value

    def unique_label(self, label: str) -> str:
        slug = slugify(label)
        seen = self.slugs.get(slug, 0)
        self.slugs[slug] = seen + 1
        return label if seen == 0 else f"{label} {seen + 1}"


@dataclass
class _MarkdownFragmenter:
    """Stateful implementation for :func:`fragment_markdown`.

    Keeping block dispatch and each block's span bookkeeping in small methods
    leaves the public producer easy to audit while preserving one shared
    emitter for sequence, parent, and provenance metadata.
    """

    artifact_id: str
    lines: list[str]
    offsets: list[int]
    stack: list[_Scope]
    fragments: list[Fragment]

    @classmethod
    def from_text(cls, text: str, artifact_id: str) -> _MarkdownFragmenter:
        """Build parser state while retaining original character offsets."""
        lines = (text or "").splitlines(keepends=True)
        offsets: list[int] = []
        cursor = 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line)
        return cls(
            artifact_id=artifact_id,
            lines=lines,
            offsets=offsets,
            stack=[_Scope(0, (), None)],
            fragments=[],
        )

    def _end_of(self, index: int) -> int:
        """Return the exclusive original-text offset for one line."""
        return self.offsets[index] + len(self.lines[index])

    def _emit(
        self,
        kind: FragmentKind,
        *,
        body: str,
        start: int,
        end: int,
        label: str = "",
        scope: _Scope | None = None,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Fragment:
        owner = scope if scope is not None else self.stack[-1]
        ordinal = owner.next_ordinal(kind)
        fragment = Fragment.at(
            artifact_id=self.artifact_id,
            kind=kind,
            parent_path=owner.path,
            text=body,
            label=label,
            ordinal=ordinal,
            sequence=len(self.fragments),
            parent_fragment_id=(
                parent_id if parent_id is not None else owner.fragment_id
            ),
            char_start=start,
            char_end=end,
            attributes=attributes or {},
        )
        self.fragments.append(fragment)
        return fragment

    def run(self) -> tuple[Fragment, ...]:
        """Consume all lines and return fragments in source order."""
        index = 0
        while index < len(self.lines):
            index = self._consume(index)
        return tuple(self.fragments)

    def _consume(self, index: int) -> int:
        """Consume the block beginning at *index*."""
        if not self.lines[index].strip():
            return index + 1
        for consumer in (
            self._consume_fence,
            self._consume_heading,
            self._consume_table,
            self._consume_quote,
            self._consume_list,
        ):
            next_index = consumer(index)
            if next_index is not None:
                return next_index
        return self._consume_paragraph(index)

    def _consume_fence(self, index: int) -> int | None:
        """Consume one fenced code block, if present."""
        fence = _FENCE_RE.match(self.lines[index])
        if not fence:
            return None
        marker, language = fence.group(1), fence.group(2)
        close = index + 1
        while close < len(self.lines) and not self.lines[close].strip().startswith(
            marker[:3]
        ):
            close += 1
        last = min(close, len(self.lines) - 1)
        self._emit(
            "code_block",
            body="".join(self.lines[index + 1 : close]),
            start=self.offsets[index],
            end=self._end_of(last),
            attributes={"language": language} if language else {},
        )
        return close + 1

    def _consume_heading(self, index: int) -> int | None:
        """Consume one ATX heading and open its child scope, if present."""
        heading = _ATX_RE.match(self.lines[index])
        if not heading:
            return None
        level = len(heading.group(1))
        title = heading.group(2).strip()
        while len(self.stack) > 1 and self.stack[-1].level >= level:
            self.stack.pop()
        parent = self.stack[-1]
        fragment = self._emit(
            "heading",
            body=title,
            start=self.offsets[index],
            end=self._end_of(index),
            label=parent.unique_label(title),
            scope=parent,
            attributes={"level": level},
        )
        self.stack.append(_Scope(level, fragment.path, fragment.fragment_id))
        return index + 1

    def _consume_table(self, index: int) -> int | None:
        """Consume a pipe table and its row children, if present."""
        if not self.lines[index].strip().startswith("|"):
            return None
        if index + 1 >= len(self.lines) or not _TABLE_DIVIDER_RE.match(
            self.lines[index + 1]
        ):
            return None
        header = self.lines[index].strip()
        close = index + 2
        while close < len(self.lines) and self.lines[close].strip().startswith("|"):
            close += 1
        table = self._emit(
            "table",
            body=header,
            start=self.offsets[index],
            end=self._end_of(close - 1),
            label=header,
            attributes={"row_count": close - index - 2},
        )
        table_scope = _Scope(0, table.path, table.fragment_id)
        for row_index in range(index + 2, close):
            cells = _split_row(self.lines[row_index])
            self._emit(
                "table_row",
                body=self.lines[row_index].strip(),
                start=self.offsets[row_index],
                end=self._end_of(row_index),
                label=cells[0] if cells else "",
                scope=table_scope,
                parent_id=table.fragment_id,
                attributes={"cells": len(cells)},
            )
        return close

    def _consume_quote(self, index: int) -> int | None:
        """Consume one contiguous blockquote, if present."""
        if not self.lines[index].strip().startswith(">"):
            return None
        close = index
        while close < len(self.lines) and self.lines[close].strip().startswith(">"):
            close += 1
        body = "\n".join(
            self.lines[line_index].strip().lstrip(">").strip()
            for line_index in range(index, close)
        )
        self._emit(
            "quote",
            body=body,
            start=self.offsets[index],
            end=self._end_of(close - 1),
        )
        return close

    def _consume_list(self, index: int) -> int | None:
        """Consume list items and their continuation lines, if present."""
        if not _LIST_RE.match(self.lines[index]):
            return None
        close = index
        while close < len(self.lines) and (
            _LIST_RE.match(self.lines[close]) or self.lines[close].strip()
        ):
            close += 1
        for item_index in range(index, close):
            match = _LIST_RE.match(self.lines[item_index])
            if match:
                self._emit(
                    "list_item",
                    body=match.group(1).strip(),
                    start=self.offsets[item_index],
                    end=self._end_of(item_index),
                )
        return close

    def _consume_paragraph(self, index: int) -> int:
        """Consume prose through the next blank line or block boundary."""
        close = index
        while (
            close < len(self.lines)
            and self.lines[close].strip()
            and not _is_block_start(self.lines[close], close, self.lines)
        ):
            close += 1
        if close == index:
            close += 1
        self._emit(
            "paragraph",
            body="".join(self.lines[index:close]).strip(),
            start=self.offsets[index],
            end=self._end_of(close - 1),
        )
        return close


def fragment_markdown(text: str, *, artifact_id: str) -> tuple[Fragment, ...]:
    """Fragment a markdown document into its addressable citation units.

    CONCEPT:AU-KG.ingest.stable-fragment-address.  Returns fragments in document
    order (``sequence`` 0..N-1), each nested under the innermost enclosing
    heading, with table rows nested under their table.  Character spans are
    tracked against the ORIGINAL string so a fragment can be re-read verbatim.

    Re-running this on unchanged text yields byte-identical ``fragment_id``s;
    that is the property :func:`fragment_markdown`'s callers depend on and the
    wiring test asserts against a real file.
    """
    return _MarkdownFragmenter.from_text(text, artifact_id).run()


def _is_block_start(line: str, index: int, lines: list[str]) -> bool:
    """True when *line* begins a non-paragraph block (so a paragraph ends here)."""
    if index == 0:
        return False
    stripped = line.strip()
    if _ATX_RE.match(line) or _FENCE_RE.match(line) or _LIST_RE.match(line):
        return True
    if stripped.startswith(">"):
        return True
    return (
        stripped.startswith("|")
        and index + 1 < len(lines)
        and bool(_TABLE_DIVIDER_RE.match(lines[index + 1]))
    )


def _split_row(line: str) -> list[str]:
    """Split one markdown table row into its cell texts."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


# ── PDF fragmenter ────────────────────────────────────────────────────────────
# D-ES-3: fragment_markdown() was the only producer even though Fragment/
# Artifact already carry the ``page``/``record``/``field``/``table_cell`` kinds
# and the engine's ``ArtifactLocus`` vocabulary (``page_box``/``row_version``/
# ``table_cell_range``) for the rest — the gap was producers, not contract.
# ``extraction/pdf.py``'s ``read_pdf_pages`` already does the sandboxed
# extraction (subprocess, byte/page/time limits); this fragments text it is
# HANDED rather than a path, so the resource-sandboxing concern (a genuinely
# different, security-adjacent surface) stays exactly where it already lived.
_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")


def fragment_pdf(pages: Sequence[str], *, artifact_id: str) -> tuple[Fragment, ...]:
    """Fragment already-extracted PDF page text into ``page`` + ``paragraph``.

    ``pages`` is per-page text (see
    :func:`~..extraction.pdf.read_pdf_pages`, which preserves page
    boundaries — ``read_pdf_text``'s single joined string cannot, since a
    ``"\\n"`` inside one page's own extracted text is indistinguishable from
    a page break once joined). Each page becomes a ``page`` fragment
    (``locus_kind="page_box"``, matching the engine's ``ArtifactLocus``);
    each blank-line-separated block of text within it becomes a child
    ``paragraph`` fragment — PDF text extraction carries no markdown syntax,
    so there is no heading/table structure to recover, only page + block.
    """
    fragments: list[Fragment] = []
    for page_index, page_text in enumerate(pages):
        page_number = page_index + 1
        page = Fragment.at(
            artifact_id=artifact_id,
            kind="page",
            parent_path=(),
            text="",
            label=f"page {page_number}",
            ordinal=page_index,
            sequence=len(fragments),
            parent_fragment_id=None,
            attributes={"page_number": page_number},
            locus_kind="page_box",
        )
        fragments.append(page)
        block_ordinal = 0
        for block in _BLOCK_SPLIT_RE.split(page_text or ""):
            body = block.strip()
            if not body:
                continue
            fragments.append(
                Fragment.at(
                    artifact_id=artifact_id,
                    kind="paragraph",
                    parent_path=page.path,
                    text=body,
                    ordinal=block_ordinal,
                    sequence=len(fragments),
                    parent_fragment_id=page.fragment_id,
                )
            )
            block_ordinal += 1
    return tuple(fragments)


# ── JSON / row fragmenters ───────────────────────────────────────────────────
# D-ES-3: a JSON object -> record/field fragments keyed by JSON-pointer-shaped
# path segments (a naturally stable address — renaming a sibling key never
# moves this one), and DB rows -> the same, keyed by primary key instead of
# row position (a naturally stable address across a re-sort or an insert
# elsewhere in the result set).


def fragment_record(value: Any, *, artifact_id: str) -> tuple[Fragment, ...]:
    """Fragment one JSON-like value into ``record``/``list``/``field`` fragments.

    The JSON analogue of :func:`fragment_markdown`: a ``dict`` becomes a
    ``record`` fragment and each of its scalar values a child ``field``
    fragment addressed by its key (slugified, mirroring how a heading anchors
    a markdown section); a ``list``/``tuple`` becomes a ``list`` fragment and
    each scalar item a child ``list_item``. A nested ``dict``/``list`` value
    recurses into its own nested ``record``/``list`` fragment rather than a
    leaf, so the tree — and every fragment's ``parent_fragment_id`` chain —
    exactly mirrors the JSON structure.
    """
    fragments: list[Fragment] = []
    _fragment_value(
        value,
        artifact_id=artifact_id,
        kind="record",
        parent_path=(),
        parent_id=None,
        label="",
        ordinal=0,
        fragments=fragments,
    )
    return tuple(fragments)


def fragment_rowset(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_id: str,
    key_field: str | Sequence[str] = "id",
) -> tuple[Fragment, ...]:
    """Fragment a set of DB rows, each keyed by its primary key.

    Each row becomes a top-level ``record`` fragment (see
    :func:`fragment_record`) whose path is anchored to the row's
    ``key_field`` value(s) instead of its position in ``rows`` — a re-sort or
    an insert elsewhere in the result set moves no other row's address.
    ``key_field`` may name a composite key (``("tenant_id", "id")``); the
    fallback to the row's ordinal (only used when every key field is missing)
    still guarantees distinct addresses.
    """
    fragments: list[Fragment] = []
    keys = (key_field,) if isinstance(key_field, str) else tuple(key_field)
    for row_index, row in enumerate(rows):
        pk = "/".join(str(row.get(k, "")) for k in keys).strip("/")
        _fragment_value(
            row,
            artifact_id=artifact_id,
            kind="record",
            parent_path=(),
            parent_id=None,
            label=pk,
            ordinal=row_index,
            fragments=fragments,
        )
    return tuple(fragments)


def _fragment_value(
    value: Any,
    *,
    artifact_id: str,
    kind: FragmentKind,
    parent_path: tuple[str, ...],
    parent_id: str | None,
    label: str,
    ordinal: int,
    fragments: list[Fragment],
) -> Fragment:
    """Recursively fragment one JSON value; appends to *fragments* in order."""
    if isinstance(value, Mapping):
        node = Fragment.at(
            artifact_id=artifact_id,
            kind=kind,
            parent_path=parent_path,
            text="",
            label=label,
            ordinal=ordinal,
            sequence=len(fragments),
            parent_fragment_id=parent_id,
            attributes={"keys": len(value)},
        )
        fragments.append(node)
        for key, item in value.items():
            child_kind = _child_kind(item, is_named=True)
            _fragment_value(
                item,
                artifact_id=artifact_id,
                kind=child_kind,
                parent_path=node.path,
                parent_id=node.fragment_id,
                label=str(key),
                ordinal=0,
                fragments=fragments,
            )
        return node

    if isinstance(value, list | tuple):
        node = Fragment.at(
            artifact_id=artifact_id,
            kind=kind,
            parent_path=parent_path,
            text="",
            label=label,
            ordinal=ordinal,
            sequence=len(fragments),
            parent_fragment_id=parent_id,
            attributes={"item_count": len(value)},
        )
        fragments.append(node)
        for item_index, item in enumerate(value):
            child_kind = _child_kind(item, is_named=False)
            _fragment_value(
                item,
                artifact_id=artifact_id,
                kind=child_kind,
                parent_path=node.path,
                parent_id=node.fragment_id,
                label="",
                ordinal=item_index,
                fragments=fragments,
            )
        return node

    # Leaf scalar (str/int/float/bool/None).
    node = Fragment.at(
        artifact_id=artifact_id,
        kind=kind,
        parent_path=parent_path,
        text="" if value is None else str(value),
        label=label,
        ordinal=ordinal,
        sequence=len(fragments),
        parent_fragment_id=parent_id,
    )
    fragments.append(node)
    return node


def _child_kind(value: Any, *, is_named: bool) -> FragmentKind:
    """The fragment kind for one child value — a dict/list child recurses."""
    if isinstance(value, Mapping):
        return "record"
    if isinstance(value, list | tuple):
        return "list"
    return "field" if is_named else "list_item"


# ── Ingest wiring ────────────────────────────────────────────────────────────


def ingest_artifact(
    engine: Any,
    artifact: Artifact,
    *,
    document_id: str = "",
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Commit an artifact and its whole fragment tree atomically.

    Rides the EXISTING ingestion path — :func:`..ingestion.envelope_ingest.ingest_graph_slice`,
    which wraps the slice in one ``ChangeEnvelope`` and one native
    ``ApplyChangeEnvelope`` transaction — rather than opening a second write
    route.  A half-written spine (an artifact whose fragments did not land, or
    fragments orphaned from their artifact) is not a state any reader should
    ever have to handle, so it is not a state this can produce.

    ``version_field="content_hash"`` makes re-ingesting an unchanged artifact
    dedupe on the envelope's idempotency ledger instead of rewriting the slice.
    """
    from .envelope_ingest import ingest_graph_slice

    entities, relationships = artifact.to_graph_slice(document_id=document_id)
    return ingest_graph_slice(
        engine,
        artifact.connector,
        entities,
        relationships,
        source_instance=artifact.source_instance,
        checkpoint=checkpoint,
        version_field="content_hash",
    )


def _fragment_read_query(
    *, artifact_id: str, document_id: str
) -> tuple[str, dict[str, str]]:
    """Build the one projection shared by both fragment lookup keys."""
    projection = (
        "RETURN f.id AS id, f.artifact_id AS artifact_id, "
        "f.fragment_kind AS fragment_kind, f.address AS address, "
        "f.text AS text, f.content_hash AS content_hash, "
        "f.ordinal AS ordinal, f.sequence AS sequence, f.depth AS depth, "
        "f.parent_fragment_id AS parent_fragment_id, "
        "f.char_start AS char_start, f.char_end AS char_end, "
        "f.label AS label, f.locus_kind AS locus_kind"
    )
    if artifact_id:
        return (
            f"MATCH (f:{FRAGMENT_NODE_TYPE}) WHERE f.artifact_id = $key {projection}",
            {"key": artifact_id},
        )
    return (
        f"MATCH (d:Document)-[:{HAS_ARTIFACT_EDGE}]->(a:{ARTIFACT_NODE_TYPE})"
        f"-[:{HAS_FRAGMENT_EDGE}]->(f:{FRAGMENT_NODE_TYPE}) "
        f"WHERE d.id = $key {projection}",
        {"key": document_id},
    )


def _read_fragment_rows(
    engine: Any, cypher: str, params: dict[str, str], *, target: str
) -> list[Any]:
    """Read fragment rows through the engine, with the backend fallback."""
    try:
        run = getattr(engine, "query_cypher", None)
        if callable(run):
            return list(run(cypher, params) or [])
        backend = getattr(engine, "backend", None)
        execute = getattr(backend, "execute", None)
        return list(execute(cypher, params) or []) if callable(execute) else []
    except Exception as exc:  # noqa: BLE001 — advisory reads report no stored spine
        logger.debug("fragment spine read failed for %r: %s", target, exc)
        return []


def _fragment_row_value(row: dict[str, Any], key: str, default: Any) -> Any:
    """Apply the stored-row defaults used by the materialized spine reader."""
    return row.get(key) or default


def _fragment_from_row(row: Any) -> Fragment | None:
    """Rehydrate one stored row, logging and dropping corrupt evidence."""
    if not isinstance(row, dict) or not row.get("address"):
        return None
    try:
        return Fragment(
            fragment_id=str(_fragment_row_value(row, "id", "")),
            artifact_id=str(_fragment_row_value(row, "artifact_id", "")),
            kind=str(_fragment_row_value(row, "fragment_kind", "span")),  # type: ignore[arg-type]
            path=tuple(str(row["address"]).split("/")),
            text=str(_fragment_row_value(row, "text", "")),
            content_hash=str(_fragment_row_value(row, "content_hash", "")),
            ordinal=int(_fragment_row_value(row, "ordinal", 0)),
            sequence=int(_fragment_row_value(row, "sequence", 0)),
            depth=int(_fragment_row_value(row, "depth", 0)),
            parent_fragment_id=_fragment_row_value(row, "parent_fragment_id", None),
            char_start=int(_fragment_row_value(row, "char_start", -1)),
            char_end=int(_fragment_row_value(row, "char_end", -1)),
            label=str(_fragment_row_value(row, "label", "")),
            locus_kind=str(_fragment_row_value(row, "locus_kind", "document_span")),
        )
    except (ValueError, TypeError) as exc:
        # A stored row that fails Fragment's own invariants (a mismatched
        # address/id pair, a missing digest) is CORRUPT evidence.  Dropping
        # it is right — returning it would let a citation resolve against a
        # fragment whose address no longer proves anything — but it is never
        # dropped silently.
        logger.warning("discarding corrupt Fragment row %r: %s", row.get("id"), exc)
        return None


def load_fragments(
    engine: Any, *, artifact_id: str = "", document_id: str = ""
) -> tuple[Fragment, ...]:
    """Read a materialized fragment spine back out of the graph, in document order.

    Either key works: ``artifact_id`` addresses the source object directly,
    ``document_id`` resolves through the ``HAS_ARTIFACT`` join the
    ``DocumentProcessor`` path writes.  Returns an empty tuple (never raises)
    when nothing is stored — an absent spine is a legitimate answer, and a
    citation check must not fail closed into "cannot tell".
    """
    if engine is None or not (artifact_id or document_id):
        return ()
    cypher, params = _fragment_read_query(
        artifact_id=artifact_id, document_id=document_id
    )
    rows = _read_fragment_rows(
        engine, cypher, params, target=artifact_id or document_id
    )
    fragments: list[Fragment] = []
    for row in rows:
        fragment = _fragment_from_row(row)
        if fragment is not None:
            fragments.append(fragment)
    fragments.sort(key=lambda f: f.sequence)
    return tuple(fragments)
