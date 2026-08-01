"""Provisional ``Artifact``/``Fragment`` contract (CONCEPT:AU-KG.ingest.mapping-dsl).

**Why this module exists.** The universal-ingestion program
(``reports/program/universal-ingestion.md``) splits the evidence spine
(``Artifact`` -> addressable ``Fragment``, track 1) and the domain-pack +
mapping-DSL layer (tracks 2-3, this package) across two concurrent lanes. As of
this writing the evidence-spine lane (``feat/evidence-spine``) has not yet
published its ``Artifact``/``Fragment`` dataclasses — a repo-wide grep for
``class Fragment``/``class Artifact`` finds neither. The charter's own
instruction for exactly this situation: *"design against the charter's
description and say what you assumed."*

**What this is, and is not.** This is a small, structural stand-in — NOT a
competing canonical type. The mapping DSL (:mod:`.dsl`) depends on a fragment
only through the handful of fields below, expressed as ``Protocol`` classes
(:class:`ArtifactLike`, :class:`FragmentLike`) so the concrete
:class:`Artifact`/:class:`Fragment` dataclasses here are swappable for the
evidence-spine lane's real ones with **zero mapping-engine changes** once they
land — anything with the same fields satisfies the protocol. Replace the
``fragment_contract`` import in :mod:`.dsl`/:mod:`.pack_loader` with the real
module at that point and delete this one.

**Assumptions made** (stated so they're easy to correct against the real
contract):

* ``artifact_id`` / ``fragment_id`` are stable, deterministic strings — the
  mapping engine uses them verbatim as the seed for a produced fact's node id
  template (``{artifact_id}``, ``{fragment_id}``).
* ``fragment_type`` is one of the concrete kinds this DSL maps
  (:data:`FRAGMENT_TYPES`) — a refinement of the charter's abstract units
  (paragraph/heading/table/row/page/span): the charter names the *class* of
  addressable unit, this module names the *concrete* structural kinds a
  frontmatter+table+heading+link+JSON corpus actually produces.
* ``value`` carries the already-decoded scalar/structured payload for that
  fragment (a frontmatter value, a table row's ``{column: cell}`` dict, a
  link's ``{"text":.., "href":..}``, a JSON field's raw value) — turning raw
  bytes/markdown into that shape is the *connector's* job
  (see :mod:`.markdown_fragmenter`), never the DSL's.
* ``content_hash`` hashes the fragment's own content — used for
  provenance/dedup on the produced fact, never as identity by itself (identity
  comes from the pack's own ``id_template``, see :mod:`.domain_pack`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "FRAGMENT_TYPES",
    "ArtifactLike",
    "FragmentLike",
    "Artifact",
    "Fragment",
    "content_hash",
]

# The concrete fragment kinds this DSL's mapping rules match against
# (CONCEPT:AU-KG.ingest.mapping-dsl) — a refinement of the evidence-spine
# charter's abstract unit list (paragraph/heading/table/row/page/span).
FRAGMENT_TYPES: frozenset[str] = frozenset(
    {
        "frontmatter_key",
        "heading_section",
        "table",
        "table_row",
        "paragraph",
        "link",
        "json_field",
    }
)


def content_hash(value: Any) -> str:
    """Deterministic sha256 over any JSON-ish fragment value."""
    import json

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class ArtifactLike(Protocol):
    """The structural shape the mapping DSL needs from an ``Artifact``."""

    artifact_id: str
    source_path: str
    content_hash: str


@runtime_checkable
class FragmentLike(Protocol):
    """The structural shape the mapping DSL needs from a ``Fragment``."""

    fragment_id: str
    artifact_id: str
    fragment_type: str
    locator: str
    content_hash: str
    value: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Artifact:
    """Provisional stand-in for the evidence-spine ``Artifact`` — see module docstring."""

    artifact_id: str
    source_path: str
    content_hash: str
    connector: str = ""
    media_type: str = "text/markdown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Fragment:
    """Provisional stand-in for the evidence-spine ``Fragment`` — see module docstring.

    ``locator`` is a stable, human-readable structural address within the
    artifact (e.g. ``"frontmatter.owner"``, ``"table[0].row[2]"``,
    ``"heading[Intro/Setup]"``, ``"link[3]"``) — the citation unit an operator
    can point at to see exactly what a mapping rule matched.
    """

    fragment_id: str
    artifact_id: str
    fragment_type: str
    locator: str
    content_hash: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fragment_type not in FRAGMENT_TYPES:
            raise ValueError(
                f"Fragment.fragment_type must be one of {sorted(FRAGMENT_TYPES)}, "
                f"got {self.fragment_type!r}"
            )
