"""Atomic OKF-CIS concept-ID reservation (CONCEPT:AU-OS.governance.concept-id-allocation).

Many sessions work the package ecosystem at once, each in its own git worktree,
all merging to a shared branch. Semantic concept IDs are a contended resource:
two sessions can independently author the same canonical ID and collide at merge.

The claim is made **atomic and self-correcting** by three mechanisms, each an
application of a lane-arbitration class (``agent_utilities.governance.lanes``):

* **APPEND-ONLY writes.** A lane appends immutable records to *its own* fragment,
  ``docs/concept_reservations.d/<lane>.yaml``. It never rewrites a file another
  lane writes, so two lanes reserving at once produce two files that git merges
  without a conflict and neither can clobber. Status changes (landed / expired /
  released) are *new appended records*, not edits — the ledger is event-sourced.
* **A single generated view.** ``docs/concept_reservations.yaml`` is the folded
  union of every fragment: one record per id, deterministically ordered. Readers
  consult that one file exactly as before; only writers know fragments exist.
* **Host-shared arbitration.** The uniqueness check is serialized by a lock in
  the repository's shared ``--git-common-dir``, and in-flight claims are
  published to an append-only claims log in that same directory. That directory
  is identical from every worktree, so a claim in one lane is visible to all the
  others **immediately**, long before any merge.

Two defects this replaced, both observed in production:

1. The ledger was a mutable shared file rewritten whole by every writer, so
   concurrent lanes clobbered each other and it needed repair by hand-verified
   line-union.
2. The ledger root was derived from ``__file__``. With an editable install
   pointing at the canonical checkout, a lane running from a worktree reserved
   into — and reconciled against — the **canonical** tree, which is why a lane
   could reserve an id on a feature branch and have ``concept reconcile`` refuse
   to flip it: the marker scan never saw the lane's own code. The root now
   resolves from the caller's working tree (:func:`default_repo_root`).

Top-level imports are stdlib-only so the canonical :data:`MARKER_RE` can be
imported cheaply by ``scripts/build_concepts_yaml.py`` / ``scripts/check_concepts.py``
without dragging in heavy deps; ``yaml``/``platformdirs`` load lazily.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# CONCEPT:AU-OS.governance.concept-id-allocation — Multi-session concept-ID allocation & coordination protocol.
# The grammar lives in concept_hierarchy; every scanner imports this exact regex.
from agent_utilities.governance.concept_hierarchy import (
    OKF_MARKER_RE,
    is_valid_domain,
    iter_okf_markers,
    parse_okf_id,
)
from agent_utilities.governance.lanes import (
    FragmentStore,
    current_tree,
    lane_name,
    render_view,
    shared_arbitration_dir,
    write_view,
)

MARKER_RE = OKF_MARKER_RE
LEDGER_FILENAME = "concept_reservations.yaml"
FRAGMENT_DIRNAME = "concept_reservations.d"
CLAIMS_LOG_FILENAME = "concept-claims.jsonl"
DEFAULT_TTL_SECONDS = 86_400  # 24h — a reservation older than this is reclaimable.
_LEDGER_REFERENCE_RE = re.compile(r"^pref_[a-z0-9_]+_[0-9a-f]{64}$")
_LEDGER_REQUIRED_KEYS = frozenset(
    {
        "id",
        "slug",
        "pillar",
        "domain",
        "session_ref",
        "reserved_at",
        "expires_at",
        "status",
    }
)
_LEDGER_OPTIONAL_KEYS = frozenset(
    {
        "namespace",
        "range_start",
        "range_end",
        "policy_version",
        "provenance_refs",
        "design_ref",
        "landed_at",
        "materialized_at",
        "expired_at",
        "released_at",
        "tombstoned_at",
        "reservation_ref",
        "fence",
        "visibility",
    }
)
# The central authority may project its complete lifecycle into this generated
# view.  The legacy CLI still uses reserved/landed/expired; the additional
# states are accepted only as append-only projections and never make the local
# file lock a cross-host authority.
_LEDGER_STATUSES = frozenset(
    {"reserved", "materialized", "landed", "released", "expired", "tombstoned"}
)
# A later event supersedes an earlier one; same-instant events break the tie by
# how far the record has advanced (a reconcile that lands or expires a claim
# writes the same reserved_at as the claim it supersedes).
_STATUS_RANK = {
    "reserved": 0,
    "materialized": 1,
    "released": 2,
    "expired": 2,
    "landed": 3,
    "tombstoned": 4,
}

_VIEW_HEADER = [
    "# Concept-ID reservation ledger — GENERATED, do not edit by hand.",
    "# Truth is docs/concept_reservations.d/<lane>.yaml — one append-only fragment",
    "# per writing lane. Regenerated on reserve/release/reconcile; rebuild with",
    "# `agent-utilities concept reconcile`. See docs/concept_coordination.md.",
]

# Where the *package* lives. Only a fallback: the ledger belongs to the caller's
# working tree, not to wherever agent_utilities happens to be installed from.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def default_repo_root() -> Path:
    """The ledger root for the **caller's** working tree.

    Falls back to the installed package root only when the process is not inside
    a git working tree that carries a ledger (e.g. an installed-wheel CLI run
    from an unrelated directory).
    """
    tree = current_tree()
    if tree is not None and (
        (tree / "docs" / LEDGER_FILENAME).exists()
        or (tree / "docs" / FRAGMENT_DIRNAME).is_dir()
    ):
        return tree
    return PACKAGE_ROOT


def _root(repo_root: Path | None) -> Path:
    return repo_root if repo_root is not None else default_repo_root()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Source-of-truth scanners
# ---------------------------------------------------------------------------
def scan_code_markers(roots: list[Path]) -> dict[str, list[str]]:
    """Map every ``CONCEPT:<id>`` marker found under *roots* to its files."""
    found: dict[str, list[str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".py", ".rs") or not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.as_posix()
            for marker in iter_okf_markers(content):
                found.setdefault(marker.id, []).append(rel)
    return found


def registry_ids(concepts_yaml: Path) -> set[str]:
    """Read the registered concept ids from a generated ``concepts.yaml``."""
    if not concepts_yaml.exists():
        return set()
    import yaml

    data = yaml.safe_load(concepts_yaml.read_text(encoding="utf-8")) or {}
    concepts = data.get("concepts", [])
    if not isinstance(concepts, list):
        raise ValueError("concept registry must contain a concepts list")
    ids: set[str] = set()
    for concept in concepts:
        if not isinstance(concept, dict) or not concept.get("id"):
            raise ValueError("concept registry contains an invalid entry")
        concept_id = str(concept["id"])
        parse_okf_id(concept_id)
        ids.add(concept_id)
    return ids


# ---------------------------------------------------------------------------
# APPEND-ONLY storage: per-lane fragments folded into one generated view
# ---------------------------------------------------------------------------
def ledger_path(repo_root: Path | None = None) -> Path:
    """The single generated view every reader consults."""
    return _root(repo_root) / "docs" / LEDGER_FILENAME


def fragment_dir(repo_root: Path | None = None) -> Path:
    """The directory of per-lane append-only fragments (what writers touch)."""
    return _root(repo_root) / "docs" / FRAGMENT_DIRNAME


def _store(repo_root: Path) -> FragmentStore:
    return FragmentStore(root=fragment_dir(repo_root), key="id")


def _event_time(record: dict[str, Any]) -> str:
    return str(record.get("landed_at") or record.get("reserved_at") or "")


def _resolve_record(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse every appended record for one id down to its latest state."""
    return max(
        group,
        key=lambda r: (_event_time(r), _STATUS_RANK.get(str(r.get("status")), -1)),
    )


def _validate(record: dict[str, Any]) -> dict[str, Any]:
    """Reject a record that does not match the current, privacy-safe schema."""
    keys = set(record)
    if not _LEDGER_REQUIRED_KEYS <= keys or keys - (
        _LEDGER_REQUIRED_KEYS | _LEDGER_OPTIONAL_KEYS
    ):
        raise ValueError("concept reservation ledger record has invalid fields")
    concept_id = str(record["id"])
    parsed = parse_okf_id(concept_id)
    if (
        str(record["slug"]) != parsed.slug
        or str(record["pillar"]) != parsed.pillar
        or str(record["domain"]) != parsed.domain
    ):
        raise ValueError("concept reservation ledger identity fields disagree")
    if str(record["status"]) not in _LEDGER_STATUSES:
        raise ValueError("concept reservation ledger status is invalid")
    for field in ("session_ref", "design_ref"):
        if field in record and not _LEDGER_REFERENCE_RE.fullmatch(str(record[field])):
            raise ValueError("concept reservation ledger contains a raw identity")
    if "namespace" in record and not concept_id.startswith(f"{record['namespace']}."):
        raise ValueError("concept reservation ledger namespace disagrees with identity")
    if "policy_version" in record and (
        not isinstance(record["policy_version"], str)
        or not record["policy_version"].strip()
    ):
        raise ValueError("concept reservation ledger policy version is invalid")
    for field in ("range_start", "range_end"):
        if field in record and (
            not isinstance(record[field], int) or record[field] < 0
        ):
            raise ValueError("concept reservation ledger range is invalid")
    if (
        "range_start" in record
        and "range_end" in record
        and record["range_end"] < record["range_start"]
    ):
        raise ValueError("concept reservation ledger range is invalid")
    if "provenance_refs" in record:
        refs = record["provenance_refs"]
        if not isinstance(refs, list) or any(
            not _LEDGER_REFERENCE_RE.fullmatch(str(ref)) for ref in refs
        ):
            raise ValueError("concept reservation ledger provenance is invalid")
    for field in (
        "reserved_at",
        "expires_at",
        "landed_at",
        "materialized_at",
        "expired_at",
        "released_at",
        "tombstoned_at",
    ):
        if field in record:
            try:
                datetime.fromisoformat(str(record[field]))
            except ValueError:
                raise ValueError(
                    "concept reservation ledger timestamp is invalid"
                ) from None
    return record


def read_ledger(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Return current-schema reservation records from the generated view."""
    path = ledger_path(repo_root)
    if not path.exists():
        return []
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError("concept reservation ledger must be a list")
    records: list[dict[str, Any]] = []
    for record in data:
        if not isinstance(record, dict):
            raise ValueError("concept reservation ledger contains a non-record")
        records.append(_validate(record))
    return records


def _fold(repo_root: Path) -> list[dict[str, Any]]:
    """Every fragment, folded to one validated record per concept id."""
    return [_validate(r) for r in _store(repo_root).fold(_resolve_record)]


def render_view_for(repo_root: Path | None = None) -> str:
    """The exact text the generated view should hold, without writing it.

    Lets a gate prove a staged view really is the fold of the fragments rather
    than a hand edit of the shared file.
    """
    return render_view(_fold(_root(repo_root)), header=_VIEW_HEADER)


def regenerate_view(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Fold the fragments and rewrite the generated view. Returns the records."""
    root = _root(repo_root)
    records = _fold(root)
    write_view(ledger_path(root), render_view(records, header=_VIEW_HEADER))
    return records


# ---------------------------------------------------------------------------
# Host-shared arbitration: one lock and one claims log per repository
# ---------------------------------------------------------------------------
def _lock_path(repo_root: Path) -> Path:
    """The mutex serializing concept claims across **all** worktrees of a repo.

    The previous lock was keyed by the *worktree* path, so two worktrees of the
    same repository took two different locks and never excluded each other — the
    serialization it promised did not exist across lanes. The shared git
    directory resolves identically from every worktree, so the lock is now real.
    """
    shared = shared_arbitration_dir(repo_root)
    if shared is not None:
        shared.mkdir(parents=True, exist_ok=True)
        return shared / "concept-ledger.lock"
    import hashlib

    import platformdirs

    lock_dir = Path(platformdirs.user_runtime_dir("agent-utilities"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()[:32]
    return lock_dir / f"concept_ledger.{digest}.lock"


def _claims_log(repo_root: Path) -> Path | None:
    shared = shared_arbitration_dir(repo_root)
    return None if shared is None else shared / CLAIMS_LOG_FILENAME


def _record_claim(repo_root: Path, event: dict[str, Any]) -> None:
    """Publish a claim event where every sibling worktree sees it immediately."""
    log = _claims_log(repo_root)
    if log is None:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(log), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        os.write(fd, (json.dumps(event, sort_keys=True) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _claimed_elsewhere(repo_root: Path, *, now: datetime) -> set[str]:
    """Ids claimed by any lane on this host and not yet released or expired."""
    log = _claims_log(repo_root)
    if log is None or not log.exists():
        return set()
    latest: dict[str, dict[str, Any]] = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        latest[str(event.get("id"))] = event
    return {
        cid
        for cid, event in latest.items()
        if event.get("event") == "reserved"
        and not _is_expired(event.get("expires_at"), now)
    }


# ---------------------------------------------------------------------------
# Reservation state
# ---------------------------------------------------------------------------
def _open_reservation_ids(records: list[dict[str, Any]], *, now: datetime) -> set[str]:
    """Ids of reservations that still hold a claim (reserved & not expired, or landed)."""
    out: set[str] = set()
    for rec in records:
        status = rec.get("status")
        if status == "landed":
            out.add(str(rec["id"]))
        elif status in {"reserved", "materialized"}:
            expires = rec.get("expires_at")
            if not _is_expired(expires, now):
                out.add(str(rec["id"]))
    return out


def _is_expired(expires_at: Any, now: datetime) -> bool:
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(str(expires_at)) < now
    except ValueError:
        return False


def _default_scan_roots(repo_root: Path) -> list[Path]:
    """Every source tree a ``CONCEPT:`` marker can land in, for reservation
    scans and ``reconcile()`` (D-25-8).

    ``agent_utilities/`` is the main package; ``mcp_v2_gateway/`` is the one
    other in-repo, deliberately-isolated Python sidecar package (see its own
    ``AGENTS.md`` / ``pyproject.toml`` — separate wheel, separate deploy
    surface, same repo). Before this fix a marker landing there (e.g.
    ``AU-ECO.mcp.v2-gateway-otel-tracing``) never left ``reserved`` status
    because ``scan_code_markers`` never walked it.  ``scan_code_markers``
    already skips a root that doesn't exist, so this is safe for a synthetic
    test ``repo_root`` fixture that only creates ``agent_utilities/``.
    """
    return [repo_root / "agent_utilities", repo_root / "mcp_v2_gateway"]


def _taken_union(
    repo_root: Path,
    records: list[dict[str, Any]],
    *,
    now: datetime,
    scan_roots: list[Path] | None = None,
) -> set[str]:
    """Every id that is spoken for: in code, in the registry, or claimed in flight."""
    roots = scan_roots if scan_roots is not None else _default_scan_roots(repo_root)
    code = set(scan_code_markers(roots))
    reg = registry_ids(repo_root / "docs" / "concepts.yaml")
    open_res = _open_reservation_ids(records, now=now)
    in_flight = _claimed_elsewhere(repo_root, now=now)
    return code | reg | open_res | in_flight


class _Arbiter:
    """Exclusive hold on a repo's concept ledger, shared by all of its worktrees."""

    def __init__(self, repo_root: Path) -> None:
        self._path = _lock_path(repo_root)
        self._fd = -1

    def __enter__(self) -> _Arbiter:
        self._fd = os.open(str(self._path), os.O_CREAT | os.O_WRONLY, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)


# ---------------------------------------------------------------------------
# Public API — reserve / release / reconcile / list
# ---------------------------------------------------------------------------
def reserve_concept_id(
    concept_id: str,
    *,
    session_id: str,
    design_doc: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    repo_root: Path | None = None,
    scan_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Atomically reserve an exact canonical ID by appending to this lane's fragment.

    Serialized by a lock in the repository's shared git directory, so concurrent
    callers in *different worktrees* — not merely different processes in one
    worktree — can never claim the same ID. The claim is published to the shared
    claims log before the lock is released, so a sibling lane sees it without
    waiting for a merge.
    """
    if not str(session_id or "").strip():
        raise ValueError("session_id is required")
    parsed = parse_okf_id(concept_id)
    if not is_valid_domain(parsed.pillar, parsed.domain):
        raise ValueError(
            f"domain {parsed.domain!r} is not registered for pillar {parsed.pillar!r}"
        )
    root = _root(repo_root)

    from agent_utilities.security.persistence_privacy import persistence_reference

    with _Arbiter(root):
        now = _utcnow()
        records = _fold(root)
        taken = _taken_union(root, records, now=now, scan_roots=scan_roots)
        if concept_id in taken:
            raise ValueError(
                f"concept id is already registered or reserved: {concept_id}"
            )
        expires_at = _iso(now + timedelta(seconds=ttl_seconds))
        record: dict[str, Any] = {
            "id": concept_id,
            "slug": parsed.slug,
            "pillar": parsed.pillar,
            "domain": parsed.domain,
            "session_ref": persistence_reference("concept_session", session_id),
            "reserved_at": _iso(now),
            "expires_at": expires_at,
            "status": "reserved",
        }
        if design_doc:
            record["design_ref"] = persistence_reference("design_doc", design_doc)
        lane = lane_name(root)
        _store(root).append(_validate(record), lane=lane)
        _record_claim(
            root,
            {
                "id": concept_id,
                "event": "reserved",
                "lane": lane,
                "at": record["reserved_at"],
                "expires_at": expires_at,
            },
        )
        regenerate_view(root)
        return record


def materialize_authoritative_record(
    record: dict[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Append a graph-authority claim to the caller's local fragment.

    This is a compatibility projection for Repository Manager and merge
    tooling.  It never allocates, checks global uniqueness, or substitutes for
    the native graph transaction.  The ``reservation_ref`` and ``fence`` fields
    make retries after a crash between the authority transition and fragment
    write idempotent: an equal-or-newer projection is a no-op, while a local
    claim with the same ID is rejected rather than overwritten.
    """

    required = {
        "reservation_id",
        "concept_id",
        "namespace",
        "tenant_ref",
        "owner_ref",
        "created_at",
        "expires_at",
        "state",
        "fence",
        "visibility",
    }
    if not required <= set(record):
        raise ValueError("authoritative concept reservation record is incomplete")
    concept_id = str(record["concept_id"])
    parsed = parse_okf_id(concept_id)
    state = str(record["state"])
    projection_status = {
        "reserved": "reserved",
        "materialized": "materialized",
        "landed": "landed",
        "released": "released",
        "expired": "expired",
        "tombstoned": "tombstoned",
    }.get(state)
    if projection_status is None:
        raise ValueError("authoritative concept reservation state is invalid")
    from agent_utilities.security.persistence_privacy import persistence_reference

    reservation_ref = persistence_reference(
        "concept_reservation", str(record["reservation_id"])
    )
    projection: dict[str, Any] = {
        "id": concept_id,
        "slug": parsed.slug,
        "pillar": parsed.pillar,
        "domain": parsed.domain,
        "namespace": str(record["namespace"]),
        "policy_version": str(record.get("policy_version") or "native-unknown"),
        "session_ref": str(record["owner_ref"]),
        "reserved_at": str(record["created_at"]),
        "expires_at": str(record["expires_at"]),
        "status": projection_status,
        "reservation_ref": reservation_ref,
        "fence": int(record["fence"]),
        "visibility": str(record["visibility"]),
    }
    if record.get("provenance_refs") is not None:
        projection["provenance_refs"] = list(record["provenance_refs"])
    for field in ("range_start", "range_end"):
        if record.get(field) is not None:
            projection[field] = int(record[field])
    for source, target in (
        ("design_ref", "design_ref"),
        ("landed_at", "landed_at"),
        ("materialized_at", "materialized_at"),
        ("expired_at", "expired_at"),
        ("released_at", "released_at"),
        ("tombstoned_at", "tombstoned_at"),
    ):
        if record.get(source) is not None:
            projection[target] = str(record[source])
    _validate(projection)
    root = _root(repo_root)
    with _Arbiter(root):
        current = {str(item["id"]): item for item in _fold(root)}
        previous = current.get(concept_id)
        if previous is not None:
            previous_ref = previous.get("reservation_ref")
            previous_fence = int(previous.get("fence", 0) or 0)
            if previous_ref != reservation_ref:
                raise ValueError(
                    f"concept id {concept_id} is already projected by another reservation"
                )
            if previous_fence >= projection["fence"]:
                return previous
        lane = lane_name(root)
        _store(root).append(projection, lane=lane)
        regenerate_view(root)
        return projection


def release_concept_id(concept_id: str, *, repo_root: Path | None = None) -> bool:
    """Release a reservation this lane owns (e.g. the work was abandoned).

    A lane may only release a claim recorded in **its own** fragment; releasing
    another lane's claim would be exactly the cross-lane clobber this design
    exists to prevent, so it raises rather than silently succeeding.
    """
    parse_okf_id(concept_id)
    root = _root(repo_root)
    with _Arbiter(root):
        store = _store(root)
        lane = lane_name(root)
        mine = store.read_fragment(lane)
        kept = [r for r in mine if str(r.get("id")) != concept_id]
        if len(kept) == len(mine):
            if any(str(r.get("id")) == concept_id for r in _fold(root)):
                raise ValueError(
                    f"{concept_id} was reserved by another lane — ask that lane to "
                    "release it, or let its TTL expire"
                )
            return False
        store.rewrite_fragment(lane, kept)
        _record_claim(
            root,
            {
                "id": concept_id,
                "event": "released",
                "lane": lane,
                "at": _iso(_utcnow()),
            },
        )
        regenerate_view(root)
        return True


def reconcile(
    *, repo_root: Path | None = None, scan_roots: list[Path] | None = None
) -> dict[str, list[str]]:
    """Close out reservations against reality, in **this lane's** working tree.

    * A reservation whose marker now appears in code → ``landed``.
    * A still-``reserved`` reservation past its TTL → ``expired`` (its id is freed).

    Each transition is a *new appended record*, never an edit to an existing one,
    so a lane can reconcile a claim another lane originally wrote without touching
    that lane's fragment. Because the scan roots come from the caller's working
    tree, a marker that landed on a feature branch is now seen — the previous
    behaviour scanned only the canonical checkout and silently refused to flip it.

    Returns ``{"landed": [...], "expired": [...]}``. Safe to call from
    ``build_concepts_yaml.main`` so the ledger self-cleans on every regeneration.
    """
    root = _root(repo_root)
    with _Arbiter(root):
        now = _utcnow()
        records = _fold(root)
        roots = scan_roots if scan_roots is not None else _default_scan_roots(root)
        code = set(scan_code_markers(roots))
        landed: list[str] = []
        expired: list[str] = []
        transitions: list[dict[str, Any]] = []
        for rec in records:
            cid = str(rec.get("id"))
            if rec.get("status") not in {"reserved", "materialized"}:
                continue
            if cid in code:
                transitions.append(
                    {
                        **rec,
                        "status": "landed",
                        "landed_at": _iso(now),
                        "visibility": "repository",
                    }
                )
                landed.append(cid)
            elif _is_expired(rec.get("expires_at"), now):
                transitions.append(
                    {**rec, "status": "expired", "expired_at": _iso(now)}
                )
                expired.append(cid)
        if transitions:
            store = _store(root)
            lane = lane_name(root)
            for transition in transitions:
                store.append(_validate(transition), lane=lane)
            for cid in expired:
                _record_claim(
                    root,
                    {"id": cid, "event": "released", "lane": lane, "at": _iso(now)},
                )
            regenerate_view(root)
        return {"landed": landed, "expired": expired}


def list_reservations(
    *, repo_root: Path | None = None, status: str | None = None
) -> list[dict[str, Any]]:
    """Return ledger reservations, optionally filtered by status."""
    records = read_ledger(repo_root)
    if status:
        records = [r for r in records if r.get("status") == status]
    return records
