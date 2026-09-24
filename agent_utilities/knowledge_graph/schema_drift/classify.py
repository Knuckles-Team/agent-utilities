"""Deterministic schema/contract drift classification (EH-402).

``classify(approved, observed)`` compares the approved record contract with
what a source just delivered and names every difference with one of six
classes. It is a pure function of the two shapes: the same inputs always give
the same, sorted changes, so a drift report is reproducible evidence.

Classes (from the reader's side -- AU consumes these records):

* ``additive_nullable`` -- a new field that may be absent or null. Compatible.
* ``additive_required`` -- a new field present and non-null in every record.
  Compatible for a reader (it may ignore the field).
* ``type_narrow`` -- a field now carries a strict subset of its declared types.
  Compatible for a reader; only reported for a DECLARED observed shape, since a
  sample that lacks a type proves nothing.
* ``rename_candidate`` -- a required field vanished while exactly one new field
  with the identical type signature appeared (and no other vanished field shares
  that signature). Breaking until a mapping is approved.
* ``type_widen`` -- a field now carries a type it did not (``null`` included,
  and a required field that is now sometimes absent). Breaking.
* ``removal`` -- a required field is absent from every record. Breaking.

A field the contract marks optional that is absent from a sample is not a
removal: absence of evidence is not evidence of absence. Declared shapes
(``sampled=False``) report it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .shape import FieldShape, RecordShape, covers


class DriftClass(StrEnum):
    """The six drift classes, by their stable wire names."""

    ADDITIVE_NULLABLE = "additive_nullable"
    ADDITIVE_REQUIRED = "additive_required"
    RENAME_CANDIDATE = "rename_candidate"
    TYPE_WIDEN = "type_widen"
    TYPE_NARROW = "type_narrow"
    REMOVAL = "removal"


#: The classes a reader can absorb without a mapping change. Only these may
#: ever be auto-continued by a declared evolution policy.
COMPATIBLE_CLASSES = frozenset(
    {
        DriftClass.ADDITIVE_NULLABLE,
        DriftClass.ADDITIVE_REQUIRED,
        DriftClass.TYPE_NARROW,
    }
)


@dataclass(frozen=True, slots=True)
class DriftChange:
    """One classified difference. ``renamed_from`` is set for a rename."""

    kind: DriftClass
    field: str
    detail: str = ""
    renamed_from: str = ""

    def to_json(self) -> dict[str, str]:
        out = {"kind": self.kind.value, "field": self.field}
        if self.detail:
            out["detail"] = self.detail
        if self.renamed_from:
            out["renamed_from"] = self.renamed_from
        return out


def _signature(shape: FieldShape) -> tuple[str, ...]:
    return tuple(sorted(shape.types))


def _unique_by_signature(
    names: Iterable[str], shapes: dict[str, FieldShape]
) -> dict[tuple[str, ...], str]:
    """Signature -> the one field carrying it; ambiguous signatures dropped."""
    buckets: dict[tuple[str, ...], list[str]] = {}
    for name in names:
        buckets.setdefault(_signature(shapes[name]), []).append(name)
    return {sig: found[0] for sig, found in buckets.items() if len(found) == 1}


def _renames(
    removed: list[str], added: list[str], approved: dict, observed: dict
) -> dict[str, str]:
    """Added field -> the removed field it uniquely replaces."""
    gone = _unique_by_signature(removed, approved)
    new = _unique_by_signature(added, observed)
    return {new[sig]: gone[sig] for sig in sorted(set(gone) & set(new))}


def _removed_fields(approved: dict, observed: dict, *, sampled: bool) -> list[str]:
    """Approved fields the observation proves are gone."""
    return sorted(
        name
        for name, shape in approved.items()
        if name not in observed and (shape.required or not sampled)
    )


def _addition(name: str, shape: FieldShape) -> DriftChange:
    nullable = "null" in shape.types or not shape.required
    kind = DriftClass.ADDITIVE_NULLABLE if nullable else DriftClass.ADDITIVE_REQUIRED
    return DriftChange(kind, name, "+".join(_signature(shape)))


def _type_change(
    name: str, approved: FieldShape, observed: FieldShape, *, sampled: bool
) -> DriftChange | None:
    widened = sorted(t for t in observed.types if not covers(approved.types, t))
    if approved.required and not observed.required:
        widened.append("absent")
    if widened:
        return DriftChange(DriftClass.TYPE_WIDEN, name, "+".join(widened))
    narrowed = sorted(approved.types - observed.types)
    if narrowed and not sampled:
        return DriftChange(DriftClass.TYPE_NARROW, name, "-".join(narrowed))
    return None


def _shared_changes(
    approved: dict, observed: dict, *, sampled: bool
) -> list[DriftChange]:
    changes = (
        _type_change(name, approved[name], observed[name], sampled=sampled)
        for name in sorted(set(approved) & set(observed))
    )
    return [change for change in changes if change is not None]


def classify(
    approved: RecordShape, observed: RecordShape, *, sampled: bool = True
) -> tuple[DriftChange, ...]:
    """Every difference between the approved contract and an observation."""
    before, after = approved.as_dict(), observed.as_dict()
    removed = _removed_fields(before, after, sampled=sampled)
    added = sorted(name for name in after if name not in before)
    renames = _renames(removed, added, before, after)
    changes = [
        DriftChange(DriftClass.RENAME_CANDIDATE, new, renamed_from=old)
        for new, old in renames.items()
    ]
    renamed_away = set(renames.values())
    changes += [
        DriftChange(DriftClass.REMOVAL, name)
        for name in removed
        if name not in renamed_away
    ]
    changes += [_addition(name, after[name]) for name in added if name not in renames]
    changes += _shared_changes(before, after, sampled=sampled)
    return tuple(sorted(changes, key=lambda change: (change.field, change.kind.value)))


def is_compatible(changes: Iterable[DriftChange]) -> bool:
    """True when every change is one a reader absorbs without a mapping."""
    return all(change.kind in COMPATIBLE_CLASSES for change in changes)


__all__ = [
    "COMPATIBLE_CLASSES",
    "DriftChange",
    "DriftClass",
    "classify",
    "is_compatible",
]
