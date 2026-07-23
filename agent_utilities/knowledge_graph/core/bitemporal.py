"""CONCEPT:AU-KG.temporal.bi-temporal-memory-layers — Bi-Temporal Memory Core.

Pure, dependency-free helpers implementing Quarq Agent's *Temporal Truth Protocol*
(agent-oss/agent.py:2370-2477, 3114-3161) as **structural graph metadata** rather than
prompt-only date discipline. Four time concepts are kept distinct:

- ``storage_time`` — when the fact was saved (= ``created_at`` / wall-clock at write).
- ``event_time``   — when the event actually happened (the *narrative* date).
- ``valid_from`` / ``valid_to`` — the validity interval a fact is believed true over.

These functions are intentionally side-effect-free so they are trivially testable and can be
wired into the relationship-creation hot path (``engine.create_relationship``) and into
query-time as-of resolution without importing the engine. The contradiction-precedence rule
(later ``event_time`` wins) drives the existing ``SUPERSEDES`` / ``CONTRADICTS`` edges.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# Open-interval sentinel: a None ``valid_to`` means "still valid" (no upper bound).
OPEN_INTERVAL: None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse(ts: str | None) -> datetime | None:
    """Best-effort ISO-8601 parse; returns None on empty/unparseable input."""
    if not ts:
        return None
    try:
        # Tolerate a trailing 'Z'.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def stamp_bitemporal(
    props: dict[str, Any],
    *,
    event_time: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Inject bi-temporal metadata into an edge/node property dict (in place, returned).

    - ``storage_time`` is set to ``now`` (defaults to wall-clock) if absent.
    - ``event_time`` defaults to an explicit arg, else an existing value, else ``storage_time``.
    - ``valid_from`` defaults to ``event_time`` (when the fact became true).
    - ``valid_to`` defaults to the open interval (``None``) — "still valid".

    Idempotent: existing keys are preserved, so re-stamping never clobbers a known event date.
    """
    stamp = now or _now_iso()
    props.setdefault("storage_time", stamp)
    resolved_event = event_time or props.get("event_time") or props["storage_time"]
    props["event_time"] = resolved_event
    props.setdefault("valid_from", resolved_event)
    props.setdefault("valid_to", OPEN_INTERVAL)
    return props


def is_valid_as_of(props: dict[str, Any], as_of: str) -> bool:
    """True if the fact described by ``props`` is valid at instant ``as_of``.

    A fact is valid when ``valid_from <= as_of < valid_to`` (open upper bound counts as valid).
    Missing ``valid_from`` is treated as "always started" (-inf); missing ``valid_to`` as +inf.
    Unparseable ``as_of`` conservatively returns True (do not silently drop rows).
    """
    t = _parse(as_of)
    if t is None:
        return True
    vf = _parse(props.get("valid_from"))
    vt = _parse(props.get("valid_to"))
    if vf is not None and t < vf:
        return False
    if vt is not None and t >= vt:
        return False
    return True


def filter_as_of(rows: list[dict[str, Any]], as_of: str | None) -> list[dict[str, Any]]:
    """Filter a list of property dicts to those valid at ``as_of`` (no-op if ``as_of`` is None)."""
    if not as_of:
        return rows
    return [r for r in rows if is_valid_as_of(r, as_of)]


def resolve_precedence(
    fact_a: dict[str, Any], fact_b: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(winner, loser)`` for two contradicting facts by event-time precedence.

    Implements Quarq's "newer memory supersedes older" rule structurally: the fact with the
    later ``event_time`` (falling back to ``storage_time``) wins. Ties keep ``fact_a`` as winner
    (stable). The caller writes a ``SUPERSEDES`` edge winner→loser and sets the loser's
    ``valid_to`` to the winner's ``event_time`` (see :func:`supersede`).
    """

    def _key(f: dict[str, Any]) -> datetime:
        ts = _parse(f.get("event_time")) or _parse(f.get("storage_time"))
        return ts or datetime.min.replace(tzinfo=UTC)

    if _key(fact_b) > _key(fact_a):
        return fact_b, fact_a
    return fact_a, fact_b


def supersede(winner: dict[str, Any], loser: dict[str, Any]) -> dict[str, Any]:
    """Close the loser's validity interval at the winner's event_time (never delete).

    Returns the mutated ``loser`` props. The winner's ``event_time`` becomes the loser's
    ``valid_to``, so an as-of query before that instant still sees the (then-true) loser fact.
    """
    boundary = winner.get("event_time") or winner.get("storage_time") or _now_iso()
    loser["valid_to"] = boundary
    return loser


def stamp_valid_from_source(
    props: dict[str, Any], *, valid_from: str | None
) -> dict[str, Any]:
    """Stamp ``valid_from`` ONLY from a genuine source-reported timestamp (in place, returned).

    CONCEPT:AU-KG.temporal.ambient-connector-valid-time — the ambient-epistemics
    counterpart of :func:`stamp_bitemporal` for connector-ingested rows: unlike
    that function (which falls back to wall-clock "now" when no event time is
    known), this one NEVER fabricates a validity start. A falsy ``valid_from``
    (the source reported no usable modification/creation timestamp) leaves
    ``props`` untouched — :func:`is_valid_as_of` already treats a missing
    ``valid_from`` as "always valid" (-inf), which is the honest default for
    data with no source-asserted validity, rather than inventing today's date
    as if the source had said so. Idempotent: an already-stamped ``valid_from``
    is preserved (``setdefault``), never overwritten by a later, possibly less
    precise, re-ingest of the same row.
    """
    if valid_from:
        props.setdefault("valid_from", valid_from)
    return props


def stamp_valid_until_from_source(
    props: dict[str, Any], *, valid_until: str | None
) -> dict[str, Any]:
    """Close ``valid_to`` from a genuine supersession instant (in place, returned).

    CONCEPT:AU-KG.temporal.ambient-connector-valid-time — the delete/tombstone
    twin of :func:`stamp_valid_from_source`: called when a connector reports a
    row is gone (a genuine "this fact stopped being true" event), so the read
    path's ``valid_from <= as_of < valid_to`` window closes at the right
    instant instead of staying open forever. Same never-fabricate contract: a
    falsy ``valid_until`` leaves ``props`` untouched.
    """
    if valid_until:
        props.setdefault("valid_to", valid_until)
    return props
