#!/usr/bin/env python3
r"""Auto-reserve CONCEPT: markers on write (CONCEPT:AU-OS.governance.reserve-concepts-hook).

Removes the reserve-via-CLI papercut: the act of *writing* a ``CONCEPT:<ID>``
marker reserves it. It scans the given (or staged) files for concept markers and,
for any id not already claimed, appends a ``landed`` reservation to **this lane's
own fragment** through the allocator — which serializes on the repository's shared
git directory, publishes the claim where sibling worktrees see it immediately, and
regenerates the single view readers consult.

It previously appended hand-rolled lines with a retired schema (``namespace`` /
``session``) straight into the shared ledger file. Those records are rejected by
``read_ledger``, so every run of this hook corrupted the ledger it was meant to
maintain — the exact "many sessions write one mutable shared file" failure the
append-only design removes.

Pre-commit wiring (optional):
    - id: reserve-concepts
      name: auto-reserve CONCEPT markers
      entry: python3 scripts/reserve_concepts_hook.py
      language: system
      files: \.(py|md|rs)$
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent_utilities.governance import concept_allocator as ca  # noqa: E402
from agent_utilities.governance import lanes  # noqa: E402


def _staged_files(tree: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=str(tree),
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in proc.stdout.splitlines() if f.endswith((".py", ".md", ".rs"))]


def _markers_in(tree: Path, files: list[str]) -> set[str]:
    found: set[str] = set()
    for name in files:
        path = Path(name) if Path(name).is_absolute() else tree / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found.update(m.group("id") for m in ca.MARKER_RE.finditer(text))
    return found


def main() -> int:
    tree = lanes.current_tree(REPO) or REPO
    files = sys.argv[1:] or _staged_files(tree)
    if not files:
        return 0
    found = _markers_in(tree, files)
    if not found:
        return 0
    claimed = {str(r["id"]) for r in ca.read_ledger(tree)}
    reserved: list[str] = []
    for concept_id in sorted(found - claimed):
        # A marker in code is already landed by definition; a zero TTL records
        # that without ever holding an open claim.
        ca.reserve_concept_id(
            concept_id, session_id="reserve-hook", ttl_seconds=0, repo_root=tree
        )
        reserved.append(concept_id)
    if reserved:
        ca.reconcile(repo_root=tree)
        print(f"auto-reserved {len(reserved)} concept marker(s): {', '.join(reserved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
