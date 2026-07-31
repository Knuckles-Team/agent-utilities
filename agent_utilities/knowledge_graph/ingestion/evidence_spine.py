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
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

from .change_envelope import ChangeEnvelope

__all__ = [
    "ARTIFACT_NODE_TYPE",
    "FRAGMENT_NODE_TYPE",
    "HAS_FRAGMENT_EDGE",
    "FRAGMENT_OF_EDGE",
    "HAS_CHILD_FRAGMENT_EDGE",
    "PARENT_FRAGMENT_EDGE",
    "NEXT_FRAGMENT_EDGE",
    "FRAGMENT_KINDS",
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
    :attr:`content_hash` identifies the revision.  Governance is NOT redeclared
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
    ) -> Artifact:
        """Build an artifact for the object *envelope* delivered.

        Governance, source identity, and revision are read off the envelope
        rather than re-derived — the envelope is the gate that already decided
        them, and re-deriving is how a payload gets to spoof its own ACL.
        """
        artifact_id = artifact_id_for(
            envelope.connector, envelope.source_instance, envelope.source_object_id
        )
        raw = content.encode("utf-8") if isinstance(content, str) else content
        provenance = dict(envelope.provenance)
        if fragmenter:
            provenance["fragmenter"] = fragmenter
        return cls(
            artifact_id=artifact_id,
            connector=envelope.connector,
            source_instance=envelope.source_instance,
            source_object_id=envelope.source_object_id,
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

    def to_graph_slice(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``(entities, relationships)`` for ``ingest_graph_slice``.

        The artifact is entity[0] — the slice's primary object — and every
        fragment follows, so the whole spine commits in ONE atomic envelope
        rather than as a fragment-per-write stream that could half-land.
        """
        entities: list[dict[str, Any]] = [self.to_node()]
        relationships: list[dict[str, Any]] = []
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
