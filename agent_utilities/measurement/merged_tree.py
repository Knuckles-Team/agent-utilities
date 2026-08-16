"""Merged-tree helper (measurement harness, capability E).

CONCEPT:AU-OS.measurement.merged-tree

Direct response to the incident where ``git diff main..HEAD`` was used to
ask "does this branch delete X?" A two-dot diff against a moving ``main``
shows every commit main has gained since the branch point as a "deletion"
from the branch's point of view — it answers "what is different", not "what
would land". This nearly triggered a session-wide false alarm that a branch
was reverting other lanes' work, when the branch had simply not yet merged
main's later commits (and merging it back would keep them).

The correct instrument for "what will main contain after merging this
branch" is ``git merge-tree --write-tree`` (git >= 2.38): it computes the
tree that WOULD result from merging ``branch`` into ``base``, without
touching the working tree or creating a commit. :func:`merged_tree` wraps
it; :func:`files_deleted_by_merge` diffs base-tree vs merged-tree to report
the files a real merge would actually remove — the number that matters,
as opposed to :func:`naive_two_dot_diff_deletions`, kept here only as the
incident's own (wrong) instrument for contrast in tests/docs.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path


class MergeTreeError(Exception):
    """Raised when ``git merge-tree`` cannot compute a tree (e.g. a real conflict)."""


@dataclasses.dataclass(frozen=True)
class MergedTreeResult:
    tree_oid: str
    conflicted_paths: tuple[str, ...]
    had_conflicts: bool


def _git(
    repo: Path, *args: str, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        input=input_bytes.decode() if input_bytes is not None else None,
        check=False,
    )


def merged_tree(repo: Path, base: str, branch: str) -> MergedTreeResult:
    """Compute the tree that would result from merging ``branch`` into ``base``.

    Wraps ``git merge-tree --write-tree <base> <branch>``. This never
    touches the working tree, the index, or HEAD, and creates no commit —
    safe to call against a live checkout mid-work. On a clean merge, exit
    code 0 and stdout is the resulting tree OID. On conflicts, git's
    porcelain merge-tree (2.38+) still exits >0 but prints the tree OID as
    the first line followed by conflict info; this function surfaces both.
    """
    proc = _git(repo, "merge-tree", "--write-tree", base, branch)
    lines = proc.stdout.splitlines()
    if not lines:
        raise MergeTreeError(
            f"git merge-tree produced no output (rc={proc.returncode}): {proc.stderr}"
        )
    tree_oid = lines[0].strip()
    if proc.returncode == 0:
        return MergedTreeResult(
            tree_oid=tree_oid, conflicted_paths=(), had_conflicts=False
        )

    # Non-zero: either a real conflict (tree_oid line still present in
    # --write-tree porcelain output, followed by conflicted file info) or a
    # hard failure (bad refs, not a repo, etc — no tree_oid to trust).
    if not tree_oid or len(tree_oid) < 40:
        raise MergeTreeError(
            f"git merge-tree failed (rc={proc.returncode}): {proc.stderr or proc.stdout}"
        )
    conflicted = tuple(
        sorted({ln.split("\t", 1)[-1].strip() for ln in lines[1:] if ln.strip()})
    )
    return MergedTreeResult(
        tree_oid=tree_oid, conflicted_paths=conflicted, had_conflicts=True
    )


def _tree_file_set(repo: Path, treeish: str) -> set[str]:
    proc = _git(repo, "ls-tree", "-r", "--name-only", treeish)
    if proc.returncode != 0:
        raise MergeTreeError(f"git ls-tree {treeish} failed: {proc.stderr}")
    return {line for line in proc.stdout.splitlines() if line}


def files_deleted_by_merge(repo: Path, base: str, branch: str) -> set[str]:
    """Return the files a real merge of ``branch`` into ``base`` would actually remove.

    This is base-tree-vs-merged-tree, not base-vs-branch — the number that
    answers "does this branch delete X" correctly. Files ``base`` has
    gained since ``branch`` diverged (main's later commits) are present in
    both the base tree and the merged tree, so they never appear here, even
    though a two-dot ``git diff base..branch`` shows them as deletions.
    """
    result = merged_tree(repo, base, branch)
    base_files = _tree_file_set(repo, base)
    merged_files = _tree_file_set(repo, result.tree_oid)
    return base_files - merged_files


def naive_two_dot_diff_deletions(repo: Path, base: str, branch: str) -> set[str]:
    """The incident's own (WRONG) instrument for "what does this branch delete".

    Kept here only so tests/docs can show it disagreeing with
    :func:`files_deleted_by_merge` on the incident-2 shape. Do not use this
    to answer "will merging this branch delete X" — that is exactly the
    mistake this module exists to correct.
    """
    proc = _git(repo, "diff", "--diff-filter=D", "--name-only", f"{base}..{branch}")
    if proc.returncode != 0:
        raise MergeTreeError(f"git diff {base}..{branch} failed: {proc.stderr}")
    return {line for line in proc.stdout.splitlines() if line}
