#!/usr/bin/python
from __future__ import annotations

"""Object-index staleness tracking (CONCEPT:KG-2.44).

Palantir provenance: *object-indexing/overview* (the Object Data Funnel) and
*object-backend* — specifically the **metadata sync / staleness** half of the
funnel, where the search index must be kept consistent with the source-of-truth
object store. Foundry tracks, per indexed object, a *version* (a content hash /
watermark) and detects when the source has advanced past what the index last
saw, flagging the object for re-index.

This module provides the freshness ledger that makes that real, not a flag:

* :class:`ObjectVersion` — the per-object fingerprint actually committed to the
  index (a content hash + a monotonic source watermark + the indexing time).
* :class:`StalenessLedger` — the source-of-truth-vs-index comparison engine. It
  records what was indexed (:meth:`record_indexed`), is shown the current source
  state (:meth:`observe_source`), and from the divergence of the two computes
  real drift: :meth:`needs_reindex`, :meth:`drift`, :meth:`stale_ids`, plus the
  inverse bookkeeping (:meth:`record_deleted`, :meth:`clear`).

The hash is computed from the object's *index-relevant payload* (embedding +
capabilities + the props that affect retrieval), so a no-op touch that does not
change any indexed field does **not** register as drift — this is content-based
staleness, not last-modified-timestamp staleness.
"""

import hashlib
import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ObjectVersion",
    "StalenessReport",
    "StalenessLedger",
    "content_hash",
]


def _canonical(value: Any) -> Any:
    """Return a JSON-canonicalizable, order-stable view of ``value``.

    Sets/iterables of scalars become sorted lists; mappings keep sorted keys via
    ``json.dumps(sort_keys=True)`` downstream. Numbers are rounded to a stable
    precision so float noise below 1e-9 does not spuriously change the hash.
    """
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in value}
    if isinstance(value, set | frozenset):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, float):
        return round(value, 9)
    return value


def content_hash(payload: Mapping[str, Any]) -> str:
    """Compute a stable content hash over an object's index-relevant payload.

    The hash is independent of mapping/iteration order and of sub-1e-9 float
    noise, so two semantically identical payloads hash equal. Used as the index
    version watermark — drift is detected when the source payload hash differs
    from the hash recorded at index time.
    """
    canon = _canonical(dict(payload))
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObjectVersion:
    """The fingerprint of one object as it was last committed to the index.

    Attributes:
        object_id: The indexed entity id.
        content_hash: SHA-256 over the index-relevant payload at index time.
        source_watermark: Monotonic source revision (e.g. a graph commit seq or
            an updated-at epoch). Advances when the source changes; lets the
            ledger detect drift even if the hash collides (it never should).
        indexed_at: Epoch seconds when this version entered the index.
    """

    object_id: str
    content_hash: str
    source_watermark: float
    indexed_at: float


@dataclass
class StalenessReport:
    """Snapshot of index-vs-source consistency at a moment in time.

    Attributes:
        stale: ids present in both index and source whose content changed.
        missing: ids the source has but the index never indexed (new objects).
        orphaned: ids the index still holds but the source no longer has.
        fresh: ids whose index version matches the current source.
    """

    stale: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    orphaned: set[str] = field(default_factory=set)
    fresh: set[str] = field(default_factory=set)

    @property
    def needs_reindex(self) -> bool:
        """Whether any drift exists (stale, missing, or orphaned objects)."""
        return bool(self.stale or self.missing or self.orphaned)

    @property
    def drift_ids(self) -> set[str]:
        """All ids that require an index mutation to become consistent."""
        return self.stale | self.missing | self.orphaned

    def as_dict(self) -> dict[str, Any]:
        """Serialize for audit / MCP transport."""
        return {
            "stale": sorted(self.stale),
            "missing": sorted(self.missing),
            "orphaned": sorted(self.orphaned),
            "fresh_count": len(self.fresh),
            "needs_reindex": self.needs_reindex,
        }


class StalenessLedger:
    """Per-object index freshness ledger (the metadata-sync half of the funnel).

    Holds the authoritative record of *what version of each object is currently
    in the search index*. Callers:

    1. record what they indexed via :meth:`record_indexed` (or
       :meth:`record_payload`, which hashes for you);
    2. show the ledger the current source state via :meth:`observe_source`;
    3. ask :meth:`needs_reindex` / :meth:`compare` for the drift set;
    4. after re-indexing, call :meth:`record_indexed` again to clear the drift.

    The ledger is backend-agnostic: it knows nothing about HNSW or the funnel,
    only about versions, so it is the single staleness source of truth for both
    batch and streaming sync paths.
    """

    def __init__(self) -> None:
        # object_id -> the version currently committed to the index.
        self._indexed: dict[str, ObjectVersion] = {}

    def __len__(self) -> int:
        return len(self._indexed)

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._indexed

    # ── recording what the index holds ───────────────────────────────────────
    def record_indexed(
        self,
        object_id: str,
        content_hash_value: str,
        source_watermark: float = 0.0,
    ) -> ObjectVersion:
        """Commit that ``object_id`` is now indexed at the given version."""
        version = ObjectVersion(
            object_id=str(object_id),
            content_hash=content_hash_value,
            source_watermark=float(source_watermark),
            indexed_at=time.time(),
        )
        self._indexed[version.object_id] = version
        return version

    def record_payload(
        self,
        object_id: str,
        payload: Mapping[str, Any],
        source_watermark: float = 0.0,
    ) -> ObjectVersion:
        """Hash ``payload`` and record it as the indexed version of ``object_id``."""
        return self.record_indexed(object_id, content_hash(payload), source_watermark)

    def record_deleted(self, object_id: str) -> bool:
        """Forget the indexed version of a removed object. Returns whether present."""
        return self._indexed.pop(str(object_id), None) is not None

    def clear(self) -> None:
        """Drop the whole ledger (e.g. before a full batch rebuild records anew)."""
        self._indexed.clear()

    def version_of(self, object_id: str) -> ObjectVersion | None:
        """Return the indexed version of ``object_id`` (``None`` if not indexed)."""
        return self._indexed.get(str(object_id))

    def indexed_ids(self) -> set[str]:
        """All ids the index currently holds, per the ledger."""
        return set(self._indexed)

    # ── comparing the ledger to the live source ──────────────────────────────
    def compare(
        self, source: Mapping[str, str] | Mapping[str, Mapping[str, Any]]
    ) -> StalenessReport:
        """Compare the ledger to the current source and classify every id.

        Args:
            source: Mapping ``object_id -> hash`` *or* ``object_id -> payload``.
                When values are mappings they are hashed via :func:`content_hash`;
                when they are strings they are treated as precomputed hashes.

        Returns:
            A :class:`StalenessReport` classifying every id as stale / missing /
            orphaned / fresh.
        """
        source_hashes: dict[str, str] = {}
        for oid, val in source.items():
            if isinstance(val, Mapping):
                source_hashes[str(oid)] = content_hash(val)
            else:
                source_hashes[str(oid)] = str(val)

        report = StalenessReport()
        indexed_ids = set(self._indexed)
        source_ids = set(source_hashes)

        report.missing = source_ids - indexed_ids
        report.orphaned = indexed_ids - source_ids

        for oid in source_ids & indexed_ids:
            if self._indexed[oid].content_hash == source_hashes[oid]:
                report.fresh.add(oid)
            else:
                report.stale.add(oid)
        return report

    def observe_source(
        self, source: Mapping[str, str] | Mapping[str, Mapping[str, Any]]
    ) -> StalenessReport:
        """Alias of :meth:`compare` emphasising the funnel's source-sync verb."""
        return self.compare(source)

    def needs_reindex(
        self,
        source: Mapping[str, str] | Mapping[str, Mapping[str, Any]],
    ) -> bool:
        """Whether the index is inconsistent with the current source state."""
        return self.compare(source).needs_reindex

    def drift(
        self,
        source: Mapping[str, str] | Mapping[str, Mapping[str, Any]],
    ) -> set[str]:
        """The set of ids requiring an index mutation to reach consistency."""
        return self.compare(source).drift_ids

    def stale_ids(
        self,
        source: Mapping[str, str] | Mapping[str, Mapping[str, Any]],
    ) -> set[str]:
        """Just the changed-content ids (excluding new/orphaned)."""
        return self.compare(source).stale

    def is_stale(self, object_id: str, current_hash: str) -> bool:
        """Whether a single object's current source hash differs from the index."""
        version = self._indexed.get(str(object_id))
        if version is None:
            return True
        return version.content_hash != str(current_hash)

    def mark_reindexed(
        self,
        ids: Iterable[str],
        source: Mapping[str, str] | Mapping[str, Mapping[str, Any]],
    ) -> int:
        """Record that ``ids`` were just re-indexed to the current source version.

        Convenience that re-commits the source hash for each id (and drops
        orphans absent from ``source``), clearing their drift. Returns the count
        of ids whose version advanced.
        """
        advanced = 0
        for oid in ids:
            oid = str(oid)
            val = source.get(oid)
            if val is None:
                # Orphan that was just removed from the index.
                if self.record_deleted(oid):
                    advanced += 1
                continue
            h = content_hash(val) if isinstance(val, Mapping) else str(val)
            prev = self._indexed.get(oid)
            if prev is None or prev.content_hash != h:
                advanced += 1
            self.record_indexed(oid, h)
        return advanced
