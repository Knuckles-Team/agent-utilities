#!/usr/bin/env python3
"""Owner + review-by expiry for the liveness ratchet's DEFERRED findings.

GOC-68's named failure mode: "a bare ratchet lets deferral become permanent."
This is the THIRD baseline found rotting under that exact defect — after
``epistemic-graph/tests/protocol_unbound_baseline.txt`` (fixed 2026-08-13) and
``scripts/surface_parity_baseline.txt`` (still rotting, and independently
tracking the same ``domains/finance`` modules this ratchet also touches — see
``plans/graph-os-completion-program/designs/DEAD-CODE-INTENT-RECOVERY.md``
Section 6.4). Format and enforcement mirror the protocol baseline's proven
pattern verbatim rather than inventing a fourth scheme.

Format, one entry per line, TAB-separated:

    <category>\t<pattern>\t# owner=<@handle> review-by=<YYYY-MM-DD> [note=...]
    <category>\t<pattern>\t# owner=<@handle> PERMANENT reason=<text>

``category`` is ``orphan_modules`` / ``dead_definitions`` / ``excluded_generated``.
``pattern`` is a repo-relative path (a single file, or a ``dir/*`` glob-style
prefix covering every file under it). ``review_by`` is set for a time-boxed
deferral; ``permanent`` marks a genuinely-intentional, never-fixed disposition
(a generated file, a deliberately-parked design gap) exempt from the date
check. Exactly one of the two must be present — enforced by
``is_well_formed`` — so a new line with neither can't silently join the
ratchet with no expiry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / "liveness_deferred.tsv"

_OWNER_TOKEN_RE = re.compile(r"\bowner=(\S+)")
_REVIEW_BY_TOKEN_RE = re.compile(r"\breview-by=(\d{4}-\d{2}-\d{2})\b")
_PERMANENT_TOKEN_RE = re.compile(r"\bPERMANENT\b")
_REASON_TOKEN_RE = re.compile(r"\breason=(.+)$")


@dataclass(frozen=True)
class DeferredEntry:
    category: str
    pattern: str
    line_no: int
    owner: str | None
    review_by: date | None
    permanent: bool
    reason: str | None


def parse_entries(text: str) -> list[DeferredEntry]:
    """Parse baseline-file TEXT into entries — pure, no file I/O, so the
    review-date gate can be proven against fabricated in-memory text (both a
    stale and a fresh entry) without waiting on the wall clock or mutating the
    real baseline."""
    entries: list[DeferredEntry] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(
                f"scripts/liveness_deferred.tsv:{line_no}: expected TAB-separated "
                f"`category<TAB>pattern<TAB># meta`, got {raw!r}"
            )
        category = fields[0].strip()
        pattern = fields[1].strip()
        meta = ""
        if len(fields) > 2:
            trailing = "\t".join(fields[2:])
            hash_idx = trailing.find("#")
            meta = trailing[hash_idx + 1 :] if hash_idx != -1 else ""
        owner_m = _OWNER_TOKEN_RE.search(meta)
        review_by_m = _REVIEW_BY_TOKEN_RE.search(meta)
        reason_m = _REASON_TOKEN_RE.search(meta)
        entries.append(
            DeferredEntry(
                category=category,
                pattern=pattern,
                line_no=line_no,
                owner=owner_m.group(1) if owner_m else None,
                review_by=date.fromisoformat(review_by_m.group(1))
                if review_by_m
                else None,
                permanent=bool(_PERMANENT_TOKEN_RE.search(meta)),
                reason=reason_m.group(1).strip() if reason_m else None,
            )
        )
    return entries


def load_entries(path: Path = BASELINE) -> list[DeferredEntry]:
    if not path.exists():
        return []
    return parse_entries(path.read_text(encoding="utf-8"))


def is_well_formed(entry: DeferredEntry) -> bool:
    """An entry must have an owner AND (a review-by date OR a justified
    PERMANENT marker) — never neither."""
    if not entry.owner:
        return False
    if entry.review_by is not None:
        return True
    return entry.permanent and bool((entry.reason or "").strip())


def stale_entries(entries: list[DeferredEntry], as_of: date) -> list[DeferredEntry]:
    """Time-boxed entries whose ``review_by`` has passed as of ``as_of``. Pure
    function of its inputs (no wall-clock read) so it is exercised both by the
    real gate (called with ``date.today()``) and by a self-contained proof
    test fabricating entries on both sides of a fixed ``as_of``."""
    return [
        e
        for e in entries
        if not e.permanent and e.review_by is not None and e.review_by < as_of
    ]
