#!/usr/bin/env python3
"""Pre-commit gate that makes lane-concurrency rules unreachable, not just documented.

Every rule this repo has about concurrent development was already written down
when it was broken. Documentation does not stop a commit; a gate does. This hook
enforces:

1. **The canonical checkout is not a workspace.** A non-merge commit authored in
   the main worktree is refused, because uncommitted work there sits in the blast
   radius of every background sync — and the window in which that work is
   unrecoverable is precisely the window (mid-pre-commit) in which the lane
   cannot yet commit it. A merge/rebase/cherry-pick in progress is the sanctioned
   canonical mutation and is detected from git's own state, and a pure version
   bump is allowed because every file it touches is declared in
   ``.bumpversion.cfg``. Neither carve-out is a flag an agent can set. **Generic
   over any repo** — see ``current_tree()`` below.

2. **A generated view stays generated.** ``docs/concept_reservations.yaml`` is the
   fold of the per-lane append-only fragments (agent-utilities only — a no-op
   check everywhere else, since no other repo stages that path). Staging a
   hand-edited view is how the shared ledger got clobbered in the first place, so
   a staged view that does not match the fold is refused with the command that
   regenerates it.

3. **A ``CARGO_TARGET_DIR`` env override defeats PARTITION.** cargo's own
   precedence lets an exported env var beat a repo's ``.cargo/config.toml``, so a
   stray global export (the exact hazard PARTITION exists to prevent — see
   CONCEPT:AU-OS.governance.lane-partitioned-resources) would silently re-share
   the target dir. This cannot be *prevented* from here (a lease only binds
   actors that take it — same residual gap ``lanes.hold_lease`` documents), so it
   is *detected* loudly instead: refuse the commit rather than let it pass quietly.

Exit code 1 = refused. Run it directly to check the repo containing the cwd
(the same contract pre-commit itself uses — it always runs hooks with cwd at the
repo root of the commit being made, so THIS SAME SCRIPT is reused, unmodified,
by every other repo's ``.pre-commit-config.yaml`` — see D-CP-3,
``reports/deferred/lane-concurrency-protocol.md``):

    python3 scripts/check_lane_guard.py
"""

from __future__ import annotations

import configparser
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent_utilities.governance import lanes  # noqa: E402

LEDGER_VIEW = "docs/concept_reservations.yaml"


def _staged_files(tree: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=str(tree),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


#: `[bumpversion:file:PATH]` and the KEYED form `[bumpversion:file(NAME):PATH]`.
#: bump2version allows the same file to appear under several distinct keys
#: (agent-utilities declares `compatibility-matrix.yml` twice -- once for its own
#: version line, once for the `agent-utilities:` dependency entry), and the key
#: is the only thing making those section names unique.
_BUMPVERSION_SECTION = re.compile(r"^bumpversion:file(?:\([^)]*\))?:(?P<path>.+)$")


def _bumpversion_files(tree: Path) -> set[str]:
    """Files a version bump is allowed to rewrite, per ``.bumpversion.cfg``.

    Two things this MUST include beyond the obvious, both learned by the
    carve-out silently failing to fire during a fleet release:

    1. ``.bumpversion.cfg`` itself. bump2version rewrites its own
       ``current_version`` and stages it, but the file is never declared as a
       ``[bumpversion:file:...]`` section -- so a set-containment check against
       the declared sections alone can never match a real bump.
    2. The KEYED section form. Matching only the literal ``bumpversion:file:``
       prefix missed every ``[bumpversion:file(NAME):PATH]`` stanza; in
       agent-utilities that was 8 of the 12 files a bump touches.

    Together these made the "a pure version bump is allowed" carve-out
    unreachable in BOTH repos that have one, so `bump2version` -- which commits
    in the canonical checkout by design, and does not retry -- wrote and staged
    every file and then had its commit refused, leaving a half-applied bump in
    the index with no commit and no tag. The gate was right to exist and simply
    never matched the thing it was written to permit.
    """
    cfg_path = tree / ".bumpversion.cfg"
    if not cfg_path.is_file():
        return set()
    parser = configparser.ConfigParser()
    parser.read(cfg_path, encoding="utf-8")
    declared = {
        match.group("path")
        for section in parser.sections()
        if (match := _BUMPVERSION_SECTION.match(section))
    }
    # bump2version rewrites its own config as part of every bump.
    return declared | {".bumpversion.cfg"}


def _check_canonical(scope: lanes.LaneScope, staged: list[str]) -> str | None:
    if not scope.is_canonical or scope.merge_in_progress:
        return None
    # Nothing staged means no commit is being authored here at all (e.g. this
    # hook ran for a PUSH, which stages nothing) -- there is no uncommitted
    # work to lose, so there is nothing to refuse. Only non-empty staged
    # content that isn't purely the bumpversion carve-out is a real risk.
    if not staged or set(staged) <= _bumpversion_files(scope.tree):
        return None
    listing = "\n      ".join(staged[:10]) or "(nothing staged)"
    return (
        f"REFUSED: this commit is being authored in the CANONICAL checkout\n"
        f"  {scope.tree}\n"
        "  Uncommitted work here can be reset by any background actor, and the\n"
        "  window where it is unrecoverable is exactly the window you cannot\n"
        "  commit from. Move it to a worktree:\n\n"
        f"      git -C {scope.tree} worktree add ../<lane> -b <branch> main\n"
        f"      git -C {scope.tree} stash create   # then apply in the worktree;\n"
        "                                          # never `git stash` (shared ref)\n\n"
        f"  Staged here:\n      {listing}"
    )


def _check_generated_view(tree: Path, staged: list[str]) -> str | None:
    """agent-utilities-only: a no-op everywhere else (no other repo stages this path)."""
    if LEDGER_VIEW not in staged:
        return None
    from agent_utilities.governance import concept_allocator as ca

    view = (tree / LEDGER_VIEW).read_text(encoding="utf-8")
    expected = ca.render_view_for(tree)
    if view == expected:
        return None
    return (
        f"REFUSED: {LEDGER_VIEW} is GENERATED and was hand-edited.\n"
        "  Reservations are append-only: write to your own fragment under\n"
        "  docs/concept_reservations.d/<lane>.yaml (the CLI does this for you),\n"
        "  then regenerate the view:\n\n"
        "      agent-utilities concept reserve --id <ID>\n"
        "      agent-utilities concept reconcile"
    )


def _check_cargo_target_override(scope: lanes.LaneScope) -> str | None:
    """Refuse a commit made with a stray ``CARGO_TARGET_DIR`` env override.

    Only fires when this repo actually builds with cargo (a ``Cargo.toml`` at the
    tree root) — every other repo skips this check entirely. See module docstring
    point 3: this is DETECTION, not prevention (a lease only binds actors that
    take it; an exported env var is never "taken").
    """
    override = os.environ.get("CARGO_TARGET_DIR", "")
    if not override or not (scope.tree / "Cargo.toml").is_file():
        return None
    expected = str(lanes.partitioned_paths(scope.tree).cargo_target_dir)
    if Path(override).expanduser().resolve() == Path(expected).resolve():
        return None
    return (
        "REFUSED: CARGO_TARGET_DIR is exported to a path that is NOT this lane's\n"
        f"  own partitioned target dir:\n"
        f"      exported: {override}\n"
        f"      expected: {expected}\n"
        "  A shared/global CARGO_TARGET_DIR both serializes and CORRUPTS concurrent\n"
        "  cargo builds across worktrees (CONCEPT:AU-OS.governance.lane-partitioned-resources).\n"
        "  cargo's env var always wins over this repo's .cargo/config.toml, so this\n"
        "  export would silently defeat the per-worktree binding. Unset it —\n"
        '  `.cargo/config.toml` (target-dir = "target-isolated") already gives this\n'
        "  worktree its own target dir with no export needed."
    )


def main() -> int:
    tree = lanes.current_tree()
    if tree is None:
        print("lane-guard: not a git working tree; nothing to check")
        return 0
    scope = lanes.lane_scope(tree)
    staged = _staged_files(tree)
    problems = [
        problem
        for problem in (
            _check_canonical(scope, staged),
            _check_generated_view(tree, staged),
            _check_cargo_target_override(scope),
        )
        if problem
    ]
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
            print(file=sys.stderr)
        return 1
    print(f"lane-guard: ok (lane {scope.lane!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
