#!/usr/bin/python
"""Auto-reserve CONCEPT: markers on write — invisible coordination (CONCEPT:OS-5.51).

Removes the reserve-via-CLI papercut: the act of *writing* a ``CONCEPT:<ID>`` marker
reserves it. Run as a pre-commit hook (or manually with file args), it scans the
given/staged files for concept markers, and for any id not already in the ledger it
appends a ``landed`` reservation under a file lock (the ledger is append-only,
merge=union, so concurrent sessions never collide). The author never has to remember
to reserve.

Pre-commit wiring (optional):
    - id: reserve-concepts
      name: auto-reserve CONCEPT markers
      entry: python scripts/reserve_concepts_hook.py
      language: system
      files: \\.(py|md)$
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "docs" / "concept_reservations.yaml"
_MARKER = re.compile(r"CONCEPT:([A-Z]+-[0-9]+(?:\.[0-9a-z]+)*)")


def _staged_files() -> list[str]:
    import subprocess

    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        return [ln for ln in res.stdout.splitlines() if ln.endswith((".py", ".md"))]
    except Exception:
        return []


def _markers_in(files: list[str]) -> set[str]:
    ids: set[str] = set()
    for f in files:
        p = (REPO / f) if not Path(f).is_absolute() else Path(f)
        if not p.is_file():
            continue
        try:
            ids.update(_MARKER.findall(p.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            continue
    return ids


def _existing_ids() -> set[str]:
    if not LEDGER.is_file():
        return set()
    return set(
        _MARKER.findall(
            "CONCEPT:" + LEDGER.read_text(encoding="utf-8").replace("id: ", "CONCEPT:")
        )
    )


def main() -> int:
    files = [a for a in sys.argv[1:]] or _staged_files()
    if not files:
        return 0
    found = _markers_in(files)
    if not found:
        return 0
    existing = _existing_ids()
    new = sorted(found - existing)
    if not new:
        return 0

    # Append under a lock (union-safe). Each new marker lands as "landed" (it's in code).
    import fcntl

    now = datetime.now(UTC).isoformat()
    lines = []
    for cid in new:
        ns = cid.rsplit("-", 1)[0] if "-" in cid else cid
        lines.append(
            f"- {{id: {cid}, namespace: {ns}, session: 'reserve-hook', "
            f"reserved_at: '{now}', expires_at: '{now}', status: landed, "
            f"landed_at: '{now}'}}\n"
        )
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.writelines(lines)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    print(f"auto-reserved {len(new)} concept marker(s): {', '.join(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
