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
  :func:`resolve_fragment` exists: resolution matches on ``fragment_id`` first
  and falls back to ``(artifact_id, content_hash)``, so a paragraph that merely
  *moved* is still found by the citation that pointed at it.

Fragments are **orderable** (:attr:`Fragment.ordinal` among siblings,
:attr:`Fragment.sequence` in whole-artifact document order) and **nestable**
(:attr:`Fragment.parent_fragment_id` / :attr:`Fragment.depth`), so a row inside a
table inside a section reconstructs exactly.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from .change_envelope import ChangeEnvelope

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
    "ingest_artifact",
    "load_fragments",
    "citation_status",
    "FragmentKind",
    "Artifact",
    "Fragment",
    "content_digest",
    "artifact_id_for",
    "fragment_id_for",
    "path_anchor",
    "resolve_fragment",
    "slugify",
]

# ── Graph vocabulary ─────────────────────────────────────────────────────────
# Node/edge labels the spine materializes through ``ingest_graph_slice``.  They
# mirror the existing Document/Chunk pair's HAS_CHUNK / CHUNK_OF convention
# (``ontology/document_processing.py``) rather than inventing a new one.
ARTIFACT_NODE_TYPE = "Artifact"
FRAGMENT_NODE_TYPE = "Fragment"
HAS_FRAGMENT_EDGE = "HAS_FRAGMENT"
FRAGMENT_OF_EDGE = "FRAGMENT_OF"
HAS_CHILD_FRAGMENT_EDGE = "HAS_CHILD_FRAGMENT"
PARENT_FRAGMENT_EDGE = "PARENT_FRAGMENT"
NEXT_FRAGMENT_EDGE = "NEXT_FRAGMENT"
HAS_ARTIFACT_EDGE = "HAS_ARTIFACT"
ARTIFACT_OF_EDGE = "ARTIFACT_OF"

#: The structural kinds a fragment may take.  Deliberately aligned with the
#: engine's ``ArtifactLocus.kind`` vocabulary (``document_span`` /
#: ``table_cell_range`` / ``page_box`` / ``row_version``) so a fragment renders
#: straight into an engine evidence locus without a second taxonomy.
FRAGMENT_KINDS: frozenset[str] = frozenset(
    {
        "document",
        "section",
        "heading",
        "paragraph",
        "list",
        "list_item",
        "table",
        "table_row",
        "table_cell",
        "code_block",
        "quote",
        "page",
        "span",
        "record",
        "field",
    }
)
FragmentKind = Literal[
    "document",
    "section",
    "heading",
    "paragraph",
    "list",
    "list_item",
    "table",
    "table_row",
    "table_cell",
    "code_block",
    "quote",
    "page",
    "span",
    "record",
    "field",
]

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def slugify(text: str, *, max_length: int = 48) -> str:
    """Return a stable, ASCII, lowercase slug for a fragment label.

    Used to anchor a *named* path segment (a heading, a captioned table).  NFKD
    normalization first, so ``Café`` and ``Café`` produce the same slug and
    a re-ingest of a re-encoded file does not move every id beneath it.
    """
    folded = unicodedata.normalize("NFKD", str(text or ""))
    ascii_only = folded.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_only).strip("-")
    return slug[:max_length].rstrip("-")


def content_digest(content: str | bytes) -> str:
    """Return ``sha256:<hex>`` over *content*.

    Text is normalized (NFC, whitespace collapsed, trimmed) before hashing so a
    re-export that only changed line wrapping or trailing spaces does NOT read as
    a content change.  Bytes are hashed verbatim — a binary artifact has no
    meaningful normalization.  The ``sha256:`` prefix matches the engine
    protocol's ``Artifact.digest`` pattern (``^sha256:[0-9a-f]{64}$``).
    """
    if isinstance(content, bytes):
        raw = content
    else:
        normalized = _WS.sub(" ", unicodedata.normalize("NFC", content)).strip()
        raw = normalized.encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def artifact_id_for(connector: str, source_instance: str, source_object_id: str) -> str:
    """Deterministic artifact id for one upstream *object*.

    Keyed to source identity, NOT to content: the artifact is the object, and its
    revisions are distinguished by :attr:`Artifact.content_hash`.  So a second
    ingest of an edited file updates the same artifact rather than forking a new
    one — which is what makes ``HAS_FRAGMENT`` edges survive an edit.
    """
    digest = hashlib.sha256(
        "\x1f".join((connector, source_instance, source_object_id)).encode("utf-8")
    ).hexdigest()
    return f"artifact:{digest[:40]}"


def path_anchor(kind: str, *, label: str = "", ordinal: int = 0) -> str:
    """Return one ``<kind>:<anchor>`` path segment.

    A *label* (a heading's text, a table's caption) anchors the segment by slug —
    content-derived but only from the node's own **name**, so editing its body
    never moves it.  Without a label the segment falls back to its sibling
    *ordinal*, which is the residual instability :func:`resolve_fragment` covers.
    """
    slug = slugify(label) if label else ""
    return f"{kind}:{slug or ordinal}"


def fragment_id_for(artifact_id: str, path: tuple[str, ...] | list[str]) -> str:
    """Deterministic fragment address from its artifact + structural path.

    Pure function of ``(artifact_id, path)`` and nothing else — in particular NOT
    of the fragment's own text, which is what lets a citation survive a typo fix.
    """
    joined = "/".join(str(segment) for segment in path)
    digest = hashlib.sha256(f"{artifact_id}\x1f{joined}".encode()).hexdigest()
    return f"fragment:{digest[:40]}"


@dataclass(frozen=True)
class Fragment:
    """One addressable citation unit inside an :class:`Artifact`.

    CONCEPT:AU-KG.ingest.stable-fragment-address.

    Attributes:
        fragment_id: The stable **address** — derived from ``artifact_id`` +
            ``path`` only.  Survives a body edit; a citation stores THIS.
        artifact_id: The owning artifact.
        kind: One of :data:`FRAGMENT_KINDS`.
        path: The scoped structural path, one ``<kind>:<anchor>`` segment per
            ancestor level ending with this fragment's own segment.  Human
            readable via :attr:`address`.
        text: The fragment's own text (a table's ``text`` is its caption/header
            line, not its rows — rows are child fragments).
        content_hash: ``sha256:<hex>`` over the normalized text.  Changes when
            and only when the content changes.
        ordinal: Position among *siblings* under the same parent (0-based).
        sequence: Position in whole-artifact document order (0-based) — a total
            order over every fragment, so a flat "give me fragments 12..18" read
            works without walking the tree.
        depth: Nesting depth (0 = top level under the artifact).
        parent_fragment_id: The enclosing fragment, or ``None`` at top level.
        char_start / char_end: Character span in the artifact's extracted text.
            ``-1`` when the artifact has no linear text (e.g. an API record).
        label: The fragment's own name when it has one (heading text, table
            caption) — what anchored its path segment.
        locus_kind: The engine ``ArtifactLocus.kind`` this fragment renders to.
        attributes: Kind-specific extras (a table row's column values, a page
            number, a code block's language).  Never governance — governance
            lives on the artifact/envelope.
    """

    fragment_id: str
    artifact_id: str
    kind: FragmentKind
    path: tuple[str, ...]
    text: str
    content_hash: str
    ordinal: int = 0
    sequence: int = 0
    depth: int = 0
    parent_fragment_id: str | None = None
    char_start: int = -1
    char_end: int = -1
    label: str = ""
    locus_kind: str = "document_span"
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in FRAGMENT_KINDS:
            raise ValueError(
                f"Fragment.kind must be one of {sorted(FRAGMENT_KINDS)}, "
                f"got {self.kind!r}"
            )
        if not self.content_hash.startswith("sha256:"):
            raise ValueError(
                "Fragment.content_hash must be a 'sha256:<hex>' digest "
                f"(see content_digest()), got {self.content_hash!r}"
            )
        if not self.path:
            raise ValueError(
                "Fragment.path must have at least one '<kind>:<anchor>' segment — "
                "an empty path has no stable address."
            )
        expected = fragment_id_for(self.artifact_id, self.path)
        if self.fragment_id != expected:
            raise ValueError(
                "Fragment.fragment_id must be fragment_id_for(artifact_id, path) — "
                f"got {self.fragment_id!r}, expected {expected!r}.  Build fragments "
                "with Fragment.at() so the address can never drift from the path."
            )

    @classmethod
    def at(
        cls,
        *,
        artifact_id: str,
        kind: FragmentKind,
        parent_path: tuple[str, ...] = (),
        text: str = "",
        label: str = "",
        ordinal: int = 0,
        **kwargs: Any,
    ) -> Fragment:
        """Build a fragment at ``parent_path + <this segment>``.

        The single sanctioned constructor: it derives the path segment, the
        address, and the content hash together, so the three can never disagree.
        """
        path = (*parent_path, path_anchor(kind, label=label, ordinal=ordinal))
        return cls(
            fragment_id=fragment_id_for(artifact_id, path),
            artifact_id=artifact_id,
            kind=kind,
            path=path,
            text=text,
            content_hash=content_digest(text),
            ordinal=ordinal,
            label=label,
            depth=len(parent_path),
            **kwargs,
        )

    @property
    def address(self) -> str:
        """The human-readable structural path, e.g. ``h2:getting-started/p:2``."""
        return "/".join(self.path)

    @property
    def version_id(self) -> str:
        """``<fragment_id>@<short content hash>`` — a content-pinned citation.

        Use this when a citation must be immutable (an audit record, a published
        claim's evidence).  Use :attr:`fragment_id` when it must *follow* the
        fragment through edits.  Both are needed; neither substitutes.
        """
        return f"{self.fragment_id}@{self.content_hash[7:23]}"

    def to_locus(self) -> dict[str, Any]:
        """Render an engine ``ArtifactLocus``-shaped selector for this fragment.

        Matches ``agent_utilities.protocols.epistemic_operations.ArtifactLocus``
        (``kind`` / ``start`` / ``end`` / ``selector``) so a candidate claim can
        cite this fragment as engine-native evidence without a second mapping.
        """
        return {
            "kind": self.locus_kind,
            "start": self.char_start if self.char_start >= 0 else None,
            "end": self.char_end if self.char_end >= 0 else None,
            "selector": {
                "fragment_id": self.fragment_id,
                "artifact_id": self.artifact_id,
                "address": self.address,
                "content_hash": self.content_hash,
            },
        }

    def to_node(self) -> dict[str, Any]:
        """Render the graph-slice entity row for this fragment.

        ``node_type``-keyed (never ``type``) so it is directly admissible to
        :func:`~..ingestion.envelope_ingest.ingest_graph_slice`.
        """
        row: dict[str, Any] = {
            "id": self.fragment_id,
            "node_type": FRAGMENT_NODE_TYPE,
            "artifact_id": self.artifact_id,
            "fragment_kind": self.kind,
            "address": self.address,
            "text": self.text,
            "content_hash": self.content_hash,
            "version_id": self.version_id,
            "ordinal": self.ordinal,
            "sequence": self.sequence,
            "depth": self.depth,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "locus_kind": self.locus_kind,
        }
        if self.label:
            row["label"] = self.label
        if self.parent_fragment_id:
            row["parent_fragment_id"] = self.parent_fragment_id
        for key, value in self.attributes.items():
            row.setdefault(f"attr_{key}", value)
        return row


@dataclass(frozen=True)
class Artifact:
    """One retrieved source object, keyed to the :class:`ChangeEnvelope`

    that delivered it (CONCEPT:AU-KG.ingest.evidence-spine-artifact).

    An artifact is the *object* — a markdown file, a PDF, an API record, a row
    set — not one delivery of it.  :attr:`artifact_id` is therefore keyed to
    source identity and stays put across revisions, while
    :attr:`content_hash` identifies the revision.  Governance is NOT re-declared
    here: it is carried verbatim off the envelope, which is the trust boundary
    that decided it.

    Attributes:
        artifact_id: Deterministic id from connector + instance + source object.
        connector / source_instance / source_object_id: Source identity, copied
            from the envelope.
        media_type: IANA media type of the retrieved bytes
            (``text/markdown``, ``application/pdf``, ``application/json``).
        content_hash: ``sha256:<hex>`` over the artifact's content.
        byte_length: Size of the retrieved content in bytes.
        content_ref: Where the bytes live when they are not inline (a blob key /
            URI — the envelope's ``blob_ref``), else ``""``.
        envelope_id / idempotency_key / source_version / schema_version: The
            delivery this artifact was extracted from.
        classification / retention / legal_hold / external_access: Governance,
            copied from the envelope.
        fragments: The artifact's fragments in document order.
        provenance: Free-form lineage, copied from the envelope and extended
            with the fragmenter that produced :attr:`fragments`.
    """

    artifact_id: str
    connector: str
    media_type: str
    content_hash: str

    source_instance: str = ""
    source_object_id: str = ""
    byte_length: int = 0
    content_ref: str = ""

    envelope_id: str = ""
    idempotency_key: str = ""
    source_version: str = ""
    schema_version: str = "1"
    tenant: str = ""

    classification: str = "internal"
    retention: str | None = None
    legal_hold: bool = False
    external_access: dict[str, Any] | None = None

    title: str = ""
    fragments: tuple[Fragment, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash.startswith("sha256:"):
            raise ValueError(
                "Artifact.content_hash must be a 'sha256:<hex>' digest "
                f"(see content_digest()), got {self.content_hash!r}"
            )
        for fragment in self.fragments:
            if fragment.artifact_id != self.artifact_id:
                raise ValueError(
                    "Artifact.fragments must all belong to this artifact — "
                    f"{fragment.fragment_id!r} claims artifact "
                    f"{fragment.artifact_id!r}, not {self.artifact_id!r}."
                )

    @classmethod
    def from_envelope(
        cls,
        envelope: ChangeEnvelope,
        *,
        content: str | bytes,
        media_type: str = "",
        fragments: tuple[Fragment, ...] | list[Fragment] = (),
        title: str = "",
        fragmenter: str = "",
        source_object_id: str = "",
    ) -> Artifact:
        """Build an artifact for the object *envelope* delivered.

        Governance, source identity, and revision are read off the envelope
        rather than re-derived — the envelope is the gate that already decided
        them, and re-deriving is how a payload gets to spoof its own ACL.

        ``source_object_id`` overrides the envelope's own object id for the
        SINGLE case where the envelope's id is content-derived (the
        ``DocumentProcessor`` path hashes content into its ``doc_id``).  The
        artifact must key to the stable *object* — the path/URL — or every edit
        forks a new artifact and every citation to it is orphaned.
        """
        object_id = source_object_id or envelope.source_object_id
        artifact_id = artifact_id_for(
            envelope.connector, envelope.source_instance, object_id
        )
        raw = content.encode("utf-8") if isinstance(content, str) else content
        provenance = dict(envelope.provenance)
        if fragmenter:
            provenance["fragmenter"] = fragmenter
        return cls(
            artifact_id=artifact_id,
            connector=envelope.connector,
            source_instance=envelope.source_instance,
            source_object_id=object_id,
            media_type=media_type or _media_type_for(envelope.payload_type),
            content_hash=content_digest(content),
            byte_length=len(raw),
            content_ref=envelope.blob_ref or "",
            envelope_id=envelope.envelope_id,
            idempotency_key=envelope.idempotency_key,
            source_version=envelope.source_version,
            schema_version=envelope.schema_version,
            tenant=envelope.tenant,
            classification=envelope.classification.value,
            retention=envelope.retention,
            legal_hold=envelope.legal_hold,
            external_access=(
                envelope.source_acl.model_dump()
                if envelope.source_acl is not None
                else None
            ),
            title=title,
            fragments=tuple(fragments),
            provenance=provenance,
        )

    def to_node(self) -> dict[str, Any]:
        """Render the graph-slice entity row for this artifact."""
        row: dict[str, Any] = {
            "id": self.artifact_id,
            "node_type": ARTIFACT_NODE_TYPE,
            "connector": self.connector,
            "source_instance": self.source_instance,
            "source_object_id": self.source_object_id,
            "media_type": self.media_type,
            "content_hash": self.content_hash,
            "byte_length": self.byte_length,
            "fragment_count": len(self.fragments),
            "envelope_id": self.envelope_id,
            "idempotency_key": self.idempotency_key,
            "source_version": self.source_version,
            "schema_version": self.schema_version,
            "classification": self.classification,
            "legal_hold": self.legal_hold,
        }
        if self.title:
            row["title"] = self.title
        if self.content_ref:
            row["content_ref"] = self.content_ref
        if self.retention:
            row["retention"] = self.retention
        if self.external_access is not None:
            row["external_access"] = self.external_access
        return row

    def to_graph_slice(
        self, *, document_id: str = ""
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``(entities, relationships)`` for ``ingest_graph_slice``.

        The artifact is entity[0] — the slice's primary object — and every
        fragment follows, so the whole spine commits in ONE atomic envelope
        rather than as a fragment-per-write stream that could half-land.

        Args:
            document_id: When this artifact was retrieved alongside an extracted
                ``Document`` node (the ``DocumentProcessor`` path), link the two
                so a reader can pivot from the retrieval unit (``Chunk``) to the
                citation unit (``Fragment``) through their shared source object.
        """
        entities: list[dict[str, Any]] = [self.to_node()]
        relationships: list[dict[str, Any]] = []
        if document_id:
            relationships.append(
                {
                    "source": document_id,
                    "target": self.artifact_id,
                    "relationship": HAS_ARTIFACT_EDGE,
                }
            )
            relationships.append(
                {
                    "source": self.artifact_id,
                    "target": document_id,
                    "relationship": ARTIFACT_OF_EDGE,
                }
            )
        by_parent: dict[str | None, list[Fragment]] = {}
        for fragment in self.fragments:
            entities.append(fragment.to_node())
            relationships.append(
                {
                    "source": self.artifact_id,
                    "target": fragment.fragment_id,
                    "relationship": HAS_FRAGMENT_EDGE,
                    "sequence": fragment.sequence,
                }
            )
            relationships.append(
                {
                    "source": fragment.fragment_id,
                    "target": self.artifact_id,
                    "relationship": FRAGMENT_OF_EDGE,
                }
            )
            if fragment.parent_fragment_id:
                relationships.append(
                    {
                        "source": fragment.parent_fragment_id,
                        "target": fragment.fragment_id,
                        "relationship": HAS_CHILD_FRAGMENT_EDGE,
                        "ordinal": fragment.ordinal,
                    }
                )
                relationships.append(
                    {
                        "source": fragment.fragment_id,
                        "target": fragment.parent_fragment_id,
                        "relationship": PARENT_FRAGMENT_EDGE,
                    }
                )
            by_parent.setdefault(fragment.parent_fragment_id, []).append(fragment)
        # Sibling order is an explicit edge, not an implied property read — a
        # reader walking evidence needs "what came next" without a sort.
        for siblings in by_parent.values():
            ordered = sorted(siblings, key=lambda f: (f.ordinal, f.sequence))
            for left, right in zip(ordered, ordered[1:], strict=False):
                relationships.append(
                    {
                        "source": left.fragment_id,
                        "target": right.fragment_id,
                        "relationship": NEXT_FRAGMENT_EDGE,
                    }
                )
        return entities, relationships


def _media_type_for(payload_type: str) -> str:
    """Map a :attr:`ChangeEnvelope.payload_type` tag to an IANA media type."""
    return {
        "json": "application/json",
        "markdown": "text/markdown",
        "text": "text/plain",
        "html": "text/html",
        "blob": "application/octet-stream",
    }.get(payload_type, "application/octet-stream")


def resolve_fragment(
    fragments: tuple[Fragment, ...] | list[Fragment],
    *,
    fragment_id: str = "",
    content_hash: str = "",
) -> Fragment | None:
    """Resolve a citation against a freshly re-ingested artifact.

    Address first, content second.  This two-step is the whole point of carrying
    both identities: a fragment whose *body* changed is still found by its
    address, and a fragment that merely *moved* (a sibling was inserted before
    it, shifting its ordinal) is still found by its content hash.  Only a
    fragment that both moved AND changed is genuinely lost — and returning
    ``None`` for that is correct, because guessing would silently re-point a
    citation at text it never supported.
    """
    if fragment_id:
        for fragment in fragments:
            if fragment.fragment_id == fragment_id:
                return fragment
    if content_hash:
        matches = [f for f in fragments if f.content_hash == content_hash]
        # Exactly one match is a confident relocation.  Two identical paragraphs
        # are ambiguous, and ambiguity must be preserved, not resolved by luck.
        if len(matches) == 1:
            return matches[0]
    return None


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
    root = _Scope(0, (), None)
    stack: list[_Scope] = [root]
    fragments: list[Fragment] = []
    lines = (text or "").splitlines(keepends=True)

    # Precompute each line's start offset so every fragment carries a true span.
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    end_of = lambda i: offsets[i] + len(lines[i])  # noqa: E731 — local span helper

    def emit(
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
        owner = scope if scope is not None else stack[-1]
        ordinal = owner.next_ordinal(kind)
        fragment = Fragment.at(
            artifact_id=artifact_id,
            kind=kind,
            parent_path=owner.path,
            text=body,
            label=label,
            ordinal=ordinal,
            sequence=len(fragments),
            parent_fragment_id=(
                parent_id if parent_id is not None else owner.fragment_id
            ),
            char_start=start,
            char_end=end,
            attributes=attributes or {},
        )
        fragments.append(fragment)
        return fragment

    index = 0
    total = len(lines)
    while index < total:
        raw = lines[index]
        stripped = raw.strip()

        if not stripped:
            index += 1
            continue

        # ── fenced code block ────────────────────────────────────────────────
        fence = _FENCE_RE.match(raw)
        if fence:
            marker, language = fence.group(1), fence.group(2)
            start = offsets[index]
            close = index + 1
            while close < total and not lines[close].strip().startswith(marker[:3]):
                close += 1
            last = min(close, total - 1)
            body = "".join(lines[index + 1 : close])
            emit(
                "code_block",
                body=body,
                start=start,
                end=end_of(last),
                attributes={"language": language} if language else {},
            )
            index = close + 1
            continue

        # ── ATX heading — opens a new scope ──────────────────────────────────
        heading = _ATX_RE.match(raw)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1]
            label = parent.unique_label(title)
            fragment = emit(
                "heading",
                body=title,
                start=offsets[index],
                end=end_of(index),
                label=label,
                scope=parent,
                attributes={"level": level},
            )
            scope = _Scope(level, fragment.path, fragment.fragment_id)
            stack.append(scope)
            index += 1
            continue

        # ── pipe table — rows are child fragments ────────────────────────────
        if stripped.startswith("|") and index + 1 < total:
            if _TABLE_DIVIDER_RE.match(lines[index + 1]):
                header = stripped
                close = index + 2
                while close < total and lines[close].strip().startswith("|"):
                    close += 1
                last = close - 1
                table = emit(
                    "table",
                    body=header,
                    start=offsets[index],
                    end=end_of(last),
                    # A table rarely has a caption; its HEADER ROW is its real
                    # name and is stable under row insert/delete/sort, so it
                    # anchors the address.
                    label=header,
                    attributes={"row_count": close - index - 2},
                )
                table_scope = _Scope(0, table.path, table.fragment_id)
                for row_index in range(index + 2, close):
                    cells = _split_row(lines[row_index])
                    emit(
                        "table_row",
                        body=lines[row_index].strip(),
                        start=offsets[row_index],
                        end=end_of(row_index),
                        # The first cell is a row's key in every table that has
                        # one, so a re-sorted table keeps every row address.
                        label=cells[0] if cells else "",
                        scope=table_scope,
                        parent_id=table.fragment_id,
                        attributes={"cells": len(cells)},
                    )
                index = close
                continue

        # ── blockquote ───────────────────────────────────────────────────────
        if stripped.startswith(">"):
            close = index
            while close < total and lines[close].strip().startswith(">"):
                close += 1
            body = "\n".join(
                lines[i].strip().lstrip(">").strip() for i in range(index, close)
            )
            emit("quote", body=body, start=offsets[index], end=end_of(close - 1))
            index = close
            continue

        # ── list — each item is its own citable fragment ─────────────────────
        if _LIST_RE.match(raw):
            close = index
            while close < total and (
                _LIST_RE.match(lines[close]) or lines[close].strip()
            ):
                if lines[close].strip() and not _LIST_RE.match(lines[close]):
                    # A continuation line belongs to the item above it.
                    close += 1
                    continue
                close += 1
            for item_index in range(index, close):
                match = _LIST_RE.match(lines[item_index])
                if not match:
                    continue
                emit(
                    "list_item",
                    body=match.group(1).strip(),
                    start=offsets[item_index],
                    end=end_of(item_index),
                )
            index = close
            continue

        # ── paragraph — everything up to the next blank line ─────────────────
        close = index
        while (
            close < total
            and lines[close].strip()
            and not _is_block_start(lines[close], close, lines)
        ):
            close += 1
        if close == index:
            close = index + 1
        body = "".join(lines[index:close]).strip()
        emit("paragraph", body=body, start=offsets[index], end=end_of(close - 1))
        index = close

    return tuple(fragments)


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
    if artifact_id:
        cypher = (
            f"MATCH (f:{FRAGMENT_NODE_TYPE}) WHERE f.artifact_id = $key "
            "RETURN f.id AS id, f.artifact_id AS artifact_id, "
            "f.fragment_kind AS fragment_kind, f.address AS address, "
            "f.text AS text, f.content_hash AS content_hash, "
            "f.ordinal AS ordinal, f.sequence AS sequence, f.depth AS depth, "
            "f.parent_fragment_id AS parent_fragment_id, "
            "f.char_start AS char_start, f.char_end AS char_end, "
            "f.label AS label, f.locus_kind AS locus_kind"
        )
        params = {"key": artifact_id}
    else:
        cypher = (
            f"MATCH (d:Document)-[:{HAS_ARTIFACT_EDGE}]->(a:{ARTIFACT_NODE_TYPE})"
            f"-[:{HAS_FRAGMENT_EDGE}]->(f:{FRAGMENT_NODE_TYPE}) "
            "WHERE d.id = $key "
            "RETURN f.id AS id, f.artifact_id AS artifact_id, "
            "f.fragment_kind AS fragment_kind, f.address AS address, "
            "f.text AS text, f.content_hash AS content_hash, "
            "f.ordinal AS ordinal, f.sequence AS sequence, f.depth AS depth, "
            "f.parent_fragment_id AS parent_fragment_id, "
            "f.char_start AS char_start, f.char_end AS char_end, "
            "f.label AS label, f.locus_kind AS locus_kind"
        )
        params = {"key": document_id}

    try:
        run = getattr(engine, "query_cypher", None)
        if callable(run):
            rows = list(run(cypher, params) or [])
        else:
            backend = getattr(engine, "backend", None)
            execute = getattr(backend, "execute", None)
            rows = list(execute(cypher, params) or []) if callable(execute) else []
    except Exception as exc:  # noqa: BLE001 — a spine read is advisory: an unavailable/uningested graph is reported as "no fragments stored", never as a citation failure; the cause is preserved on the debug record below
        logger.debug(
            "fragment spine read failed for %r: %s", artifact_id or document_id, exc
        )
        return ()

    fragments: list[Fragment] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("address"):
            continue
        try:
            fragments.append(
                Fragment(
                    fragment_id=str(row.get("id") or ""),
                    artifact_id=str(row.get("artifact_id") or ""),
                    kind=str(row.get("fragment_kind") or "span"),  # type: ignore[arg-type]
                    path=tuple(str(row["address"]).split("/")),
                    text=str(row.get("text") or ""),
                    content_hash=str(row.get("content_hash") or ""),
                    ordinal=int(row.get("ordinal") or 0),
                    sequence=int(row.get("sequence") or 0),
                    depth=int(row.get("depth") or 0),
                    parent_fragment_id=row.get("parent_fragment_id") or None,
                    char_start=int(row.get("char_start", -1) or -1),
                    char_end=int(row.get("char_end", -1) or -1),
                    label=str(row.get("label") or ""),
                    locus_kind=str(row.get("locus_kind") or "document_span"),
                )
            )
        except (ValueError, TypeError) as exc:
            # A stored row that fails Fragment's own invariants (a mismatched
            # address/id pair, a missing digest) is CORRUPT evidence.  Dropping
            # it is right — returning it would let a citation resolve against a
            # fragment whose address no longer proves anything — but it is never
            # dropped silently.
            logger.warning("discarding corrupt Fragment row %r: %s", row.get("id"), exc)
    fragments.sort(key=lambda f: f.sequence)
    return tuple(fragments)


def citation_status(
    fragments: tuple[Fragment, ...] | list[Fragment],
    *,
    fragment_id: str = "",
    content_hash: str = "",
) -> dict[str, Any]:
    """Report what happened to a citation since it was made.

    Four honest outcomes, because collapsing them is how a citation silently
    starts pointing at text it never supported:

    * ``current`` — the address resolved AND the content is unchanged.
    * ``moved`` — exactly one fragment still carries the cited content, at a
      different address; the citation is re-pointable to ``fragment_id``.
    * ``stale`` — the address resolved but the content changed and the cited
      content exists nowhere else; the fragment is still there, the quote is not.
    * ``lost`` — neither resolves (or the content is ambiguous across several
      fragments).  Reported, never guessed.

    ``moved`` is checked BEFORE ``stale`` deliberately.  When a same-kind sibling
    is inserted above a fragment, the *address* is inherited by the newcomer
    while the cited text is still present one slot down — reporting that as
    "stale" would tell the reader their quote was rewritten when in fact it just
    shifted, and would leave the recoverable citation unrecovered.
    """
    at_address = next(
        (f for f in fragments if fragment_id and f.fragment_id == fragment_id), None
    )
    if at_address is not None and (
        not content_hash or at_address.content_hash == content_hash
    ):
        return {
            "status": "current",
            "fragment_id": at_address.fragment_id,
            "address": at_address.address,
            "content_hash": at_address.content_hash,
            "cited_content_hash": content_hash,
            "text": at_address.text,
        }
    relocated = resolve_fragment(fragments, content_hash=content_hash)
    if relocated is not None and relocated is not at_address:
        return {
            "status": "moved",
            "fragment_id": relocated.fragment_id,
            "address": relocated.address,
            "content_hash": relocated.content_hash,
            "cited_content_hash": content_hash,
            "text": relocated.text,
        }
    if at_address is not None:
        return {
            "status": "stale",
            "fragment_id": at_address.fragment_id,
            "address": at_address.address,
            "content_hash": at_address.content_hash,
            "cited_content_hash": content_hash,
            "text": at_address.text,
        }
    return {
        "status": "lost",
        "fragment_id": "",
        "address": "",
        "content_hash": "",
        "cited_content_hash": content_hash,
        "text": "",
    }
