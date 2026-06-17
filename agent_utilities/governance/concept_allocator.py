"""Atomic concept-ID allocation for parallel Claude sessions (CONCEPT:OS-5.42).

Many sessions work the agent-packages ecosystem at once, each in its own git
worktree under ``/home/apps/worktrees/``, all merging to a shared ``main``.
Concept ids (``KG-2.101``, ``AHE-3.49``, the per-package ``KEY-001`` …) are the
contended resource: two sessions that both read "the current max is .100" both
pick ``.101`` and collide at merge, forcing a renumber.

This module makes a claim **atomic and self-correcting**:

* The set of "taken" ids is the union of three sources — markers already in
  code, ids already in the registry (``docs/concepts.yaml``), and *open
  reservations* — so an in-flight claim in another worktree is counted before
  its marker ever lands.
* Reservations live in a committed, **line-oriented** ledger
  (``docs/concept_reservations.yaml``) so concurrent worktrees reconcile via a
  ``merge=union`` git driver instead of overwriting each other.
* Within a host, the read-modify-write is serialized by an ``fcntl.flock`` —
  the same primitive :class:`agent_utilities.mcp.kg_coordinator.KGCoordinator`
  uses for KG-server spawn election.

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

# CONCEPT:OS-5.42 — Multi-session concept-ID allocation & coordination protocol.

# ---------------------------------------------------------------------------
# Canonical marker grammar — the ONE definition. ``scripts/build_concepts_yaml.py``
# and ``scripts/check_concepts.py`` import this so the three scanners can never
# drift. A concept id is ``<PILLAR>-<n>`` optionally followed by ``.<sub>`` where
# the sub-index is numeric or a placeholder letter (``KG-2.20g``).
# ---------------------------------------------------------------------------
MARKER_RE = re.compile(r"CONCEPT:(?P<id>[A-Z]+-\d+(?:\.[0-9A-Za-z]+)?)")

# A pillar namespace carries the major number (``KG-2``, ``OS-5``) and mints
# dotted sub-indices. A package namespace is letters only (``KEY``, ``GL``) and
# mints zero-padded 3-digit indices.
_PILLAR_NS_RE = re.compile(r"^[A-Z]+-\d+$")
_PACKAGE_NS_RE = re.compile(r"^[A-Z]+$")

LEDGER_FILENAME = "concept_reservations.yaml"
DEFAULT_TTL_SECONDS = 86_400  # 24h — a reservation older than this is reclaimable.

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
    return {
        c["id"] for c in data.get("concepts", []) if isinstance(c, dict) and c.get("id")
    }


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
    # Not a security hash — just a short, stable filename token for the per-repo
    # lock path. usedforsecurity=False keeps it FIPS-safe and silences B324.
    digest = hashlib.sha1(
        str(repo_root.resolve()).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:12]
    return lock_dir / f"concept_ledger.{digest}.lock"


def read_ledger(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Return every reservation record in the committed ledger."""
    path = ledger_path(repo_root)
    if not path.exists():
        return []
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict) and r.get("id")]


def _dump_ledger(records: list[dict[str, Any]], repo_root: Path = REPO_ROOT) -> None:
    """Write the ledger as one ``- {…}`` flow-mapping per line.

    One reservation per physical line is what makes ``merge=union`` safe: two
    worktrees that each append a distinct reservation merge without a textual
    conflict. Written via a temp file + ``os.replace`` for atomicity.
    """
    import yaml

    path = ledger_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable order: by namespace then numeric position then id.
    records = sorted(records, key=lambda r: _sort_key(str(r.get("id", ""))))
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


# ---------------------------------------------------------------------------
# Next-id computation
# ---------------------------------------------------------------------------
def _sort_key(cid: str):
    nums = [int(n) for n in re.findall(r"\d+", cid)]
    prefix = re.match(r"^[A-Z]+", cid)
    return (prefix.group(0) if prefix else cid, nums, cid)


def _max_subindex(namespace: str, taken: set[str]) -> int:
    """Largest numeric sub-index already used in *namespace* across *taken*."""
    if _PILLAR_NS_RE.match(namespace):
        pat = re.compile(rf"^{re.escape(namespace)}\.(\d+)")
    elif _PACKAGE_NS_RE.match(namespace):
        pat = re.compile(rf"^{re.escape(namespace)}-(\d+)$")
    else:
        raise ValueError(
            f"unrecognized namespace {namespace!r}: expected a pillar like 'KG-2' "
            "or a package prefix like 'KEY'"
        )
    best = 0
    for cid in taken:
        m = pat.match(cid)
        if m:
            best = max(best, int(m.group(1)))
    return best


def format_id(namespace: str, index: int) -> str:
    if _PILLAR_NS_RE.match(namespace):
        return f"{namespace}.{index}"
    if _PACKAGE_NS_RE.match(namespace):
        return f"{namespace}-{index:03d}"
    raise ValueError(f"unrecognized namespace {namespace!r}")


def next_id(namespace: str, taken: set[str]) -> str:
    """Compute the next free id in *namespace* given the union of taken ids."""
    return format_id(namespace, _max_subindex(namespace, taken) + 1)


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
    namespace: str,
    *,
    session_id: str,
    design_doc: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    repo_root: Path = REPO_ROOT,
    scan_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Atomically reserve the next free id in *namespace* and append it to the ledger.

    Serialized by an ``fcntl.flock`` so concurrent callers on the same host can
    never mint the same id; the committed line-oriented ledger plus the
    union-of-everything ``taken`` set extend that guarantee across worktrees.
    """
    lock_fd = open(_lock_path(repo_root), "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        now = _utcnow()
        records = read_ledger(repo_root)
        taken = _taken_union(repo_root, records, now=now, scan_roots=scan_roots)
        cid = next_id(namespace, taken)
        record = {
            "id": cid,
            "namespace": namespace,
            "session": session_id,
            "reserved_at": _iso(now),
            "expires_at": _iso(now + timedelta(seconds=ttl_seconds)),
            "status": "reserved",
        }
        if design_doc:
            record["design_doc"] = design_doc
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
