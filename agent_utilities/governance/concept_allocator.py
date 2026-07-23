"""Atomic OKF-CIS concept-ID reservation (CONCEPT:AU-OS.governance.concept-id-allocation).

Many sessions work the package ecosystem at once, each in its own configured git
worktree, all merging to a shared branch.
Semantic concept IDs are a contended resource: two sessions can independently
author the same canonical ID and collide at merge.

This module makes a claim **atomic and self-correcting**:

* Callers propose a complete canonical OKF-CIS ID; the allocator never invents
  semantic names.
* The set of taken IDs is the union of three sources — markers already in
  code, ids already in the registry (``docs/concepts.yaml``), and *open
  reservations* — so an in-flight claim in another worktree is counted before
  its marker ever lands.
* Reservations live in a committed, **line-oriented** ledger
  (``docs/concept_reservations.yaml``) so concurrent worktrees reconcile via a
  ``merge=union`` git driver instead of overwriting each other.
* Within a host, the read-modify-write is serialized by an ``fcntl.flock``.

The ledger is authoritative for *claiming* (it works offline and across
worktrees). The MCP/REST surface additionally projects reservations into the
Knowledge Graph when the gateway is healthy, for queryability — see
``agent_utilities/mcp/tools/ontology_tools.py`` (``concept_registry``).

Top-level imports are stdlib-only so the canonical :data:`MARKER_RE` can be
imported cheaply by ``scripts/build_concepts_yaml.py`` / ``scripts/check_concepts.py``
without dragging in heavy deps; ``yaml``/``platformdirs`` load lazily.
"""

from __future__ import annotations

import fcntl
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
    parse_okf_id,
)

MARKER_RE = OKF_MARKER_RE
LEDGER_FILENAME = "concept_reservations.yaml"
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
_LEDGER_OPTIONAL_KEYS = frozenset({"design_ref", "landed_at"})
_LEDGER_STATUSES = frozenset({"reserved", "landed", "expired"})

# Repo root = three parents up from this file (.../agent_utilities/governance/x.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


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
            for m in MARKER_RE.finditer(content):
                found.setdefault(m.group("id"), []).append(rel)
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
# Ledger I/O (line-oriented YAML, merge=union friendly)
# ---------------------------------------------------------------------------
def ledger_path(repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "docs" / LEDGER_FILENAME


def _lock_path(repo_root: Path = REPO_ROOT) -> Path:
    """Per-repo advisory lock.

    Each repo has its own ledger (agent-utilities, and every per-package repo),
    so each gets its own lock keyed by a stable hash of the resolved repo path —
    distinct ledgers never serialize against each other, and concurrent reservers
    of the *same* ledger always do.
    """
    import hashlib

    import platformdirs

    lock_dir = Path(platformdirs.user_runtime_dir("agent-utilities"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    # Collision resistance matters: two roots must never share a ledger lock.
    digest = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()[:32]
    return lock_dir / f"concept_ledger.{digest}.lock"


def read_ledger(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Return current-schema reservation records, rejecting stale/raw fields."""
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
            if field in record and not _LEDGER_REFERENCE_RE.fullmatch(
                str(record[field])
            ):
                raise ValueError("concept reservation ledger contains a raw identity")
        for field in ("reserved_at", "expires_at", "landed_at"):
            if field in record:
                try:
                    datetime.fromisoformat(str(record[field]))
                except ValueError:
                    raise ValueError(
                        "concept reservation ledger timestamp is invalid"
                    ) from None
        records.append(record)
    return records


def _dump_ledger(records: list[dict[str, Any]], repo_root: Path = REPO_ROOT) -> None:
    """Write the ledger as one ``- {…}`` flow-mapping per line.

    One reservation per physical line is what makes ``merge=union`` safe: two
    worktrees that each append a distinct reservation merge without a textual
    conflict. Written via a temp file + ``os.replace`` for atomicity.
    """
    import yaml

    path = ledger_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: str(r.get("id", "")))
    lines = [
        "# Concept-ID reservation ledger — one reservation per line (merge=union safe).",
        "# Managed by agent_utilities.governance.concept_allocator; see",
        "# docs/concept_coordination.md. Reserve via `agent-utilities concept reserve`.",
    ]
    for rec in records:
        flow = yaml.safe_dump(
            rec, default_flow_style=True, sort_keys=False, width=10_000
        ).strip()
        lines.append(f"- {flow}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _open_reservation_ids(records: list[dict[str, Any]], *, now: datetime) -> set[str]:
    """Ids of reservations that still hold a claim (reserved & not expired, or landed)."""
    out: set[str] = set()
    for rec in records:
        status = rec.get("status")
        if status == "landed":
            out.add(str(rec["id"]))
        elif status == "reserved":
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


# ---------------------------------------------------------------------------
# Public API — reserve / release / reconcile / list
# ---------------------------------------------------------------------------
def _default_scan_roots(repo_root: Path) -> list[Path]:
    return [repo_root / "agent_utilities"]


def _taken_union(
    repo_root: Path,
    records: list[dict[str, Any]],
    *,
    now: datetime,
    scan_roots: list[Path] | None = None,
) -> set[str]:
    roots = scan_roots if scan_roots is not None else _default_scan_roots(repo_root)
    code = set(scan_code_markers(roots))
    reg = registry_ids(repo_root / "docs" / "concepts.yaml")
    open_res = _open_reservation_ids(records, now=now)
    return code | reg | open_res


def reserve_concept_id(
    concept_id: str,
    *,
    session_id: str,
    design_doc: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    repo_root: Path = REPO_ROOT,
    scan_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Atomically reserve an exact canonical ID and append it to the ledger.

    Serialized by an ``fcntl.flock`` so concurrent callers on the same host can
    never claim the same ID; the committed line-oriented ledger plus the
    union-of-everything taken set extend that guarantee across worktrees.
    """
    if not str(session_id or "").strip():
        raise ValueError("session_id is required")
    parsed = parse_okf_id(concept_id)
    if not is_valid_domain(parsed.pillar, parsed.domain):
        raise ValueError(
            f"domain {parsed.domain!r} is not registered for pillar {parsed.pillar!r}"
        )

    from agent_utilities.security.persistence_privacy import persistence_reference

    lock_fd = open(_lock_path(repo_root), "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        now = _utcnow()
        records = read_ledger(repo_root)
        taken = _taken_union(repo_root, records, now=now, scan_roots=scan_roots)
        if concept_id in taken:
            raise ValueError(
                f"concept id is already registered or reserved: {concept_id}"
            )
        record = {
            "id": concept_id,
            "slug": parsed.slug,
            "pillar": parsed.pillar,
            "domain": parsed.domain,
            "session_ref": persistence_reference("concept_session", session_id),
            "reserved_at": _iso(now),
            "expires_at": _iso(now + timedelta(seconds=ttl_seconds)),
            "status": "reserved",
        }
        if design_doc:
            record["design_ref"] = persistence_reference("design_doc", design_doc)
        records.append(record)
        _dump_ledger(records, repo_root)
        return record
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            lock_fd.close()


def release_concept_id(concept_id: str, *, repo_root: Path = REPO_ROOT) -> bool:
    """Release a reservation (e.g. the work was abandoned). Returns True if found."""
    parse_okf_id(concept_id)
    lock_fd = open(_lock_path(repo_root), "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        records = read_ledger(repo_root)
        kept = [r for r in records if str(r.get("id")) != concept_id]
        found = len(kept) != len(records)
        if found:
            _dump_ledger(kept, repo_root)
        return found
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            lock_fd.close()


def reconcile(
    *, repo_root: Path = REPO_ROOT, scan_roots: list[Path] | None = None
) -> dict[str, list[str]]:
    """Close out reservations against reality.

    * A reservation whose marker now appears in code → ``landed``.
    * A still-``reserved`` reservation past its TTL → ``expired`` (its id is freed).

    Returns ``{"landed": [...], "expired": [...]}``. Safe to call from
    ``build_concepts_yaml.main`` so the ledger self-cleans on every regeneration.
    """
    lock_fd = open(_lock_path(repo_root), "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        now = _utcnow()
        records = read_ledger(repo_root)
        roots = scan_roots if scan_roots is not None else _default_scan_roots(repo_root)
        code = set(scan_code_markers(roots))
        landed: list[str] = []
        expired: list[str] = []
        changed = False
        for rec in records:
            cid = str(rec.get("id"))
            if rec.get("status") == "reserved" and cid in code:
                rec["status"] = "landed"
                rec["landed_at"] = _iso(now)
                landed.append(cid)
                changed = True
            elif rec.get("status") == "reserved" and _is_expired(
                rec.get("expires_at"), now
            ):
                rec["status"] = "expired"
                expired.append(cid)
                changed = True
        if changed:
            _dump_ledger(records, repo_root)
        return {"landed": landed, "expired": expired}
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            lock_fd.close()


def list_reservations(
    *, repo_root: Path = REPO_ROOT, status: str | None = None
) -> list[dict[str, Any]]:
    """Return ledger reservations, optionally filtered by status."""
    records = read_ledger(repo_root)
    if status:
        records = [r for r in records if r.get("status") == status]
    return records
