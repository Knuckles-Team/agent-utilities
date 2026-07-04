"""Git change-coupling → FILE_CHANGES_WITH edges (CONCEPT:AU-KG.ingest.mine-git-history-files).

Two files that keep changing in the same commits are *coupled* even when nothing
in the AST connects them — a hidden dependency the call graph can't see. We mine
that from git history: files co-changed in ≥ ``min_support`` commits get a
symmetric ``FILE_CHANGES_WITH`` edge weighted by how often. It surfaces the real
blast radius of a change (the files that historically move together).
"""

from __future__ import annotations

import subprocess
from itertools import combinations

from .models import EnrichmentEdge

# A pair co-changing in fewer than this many commits is noise, not coupling.
DEFAULT_MIN_SUPPORT = 3
# Skip commits touching more than this many files (bulk reformats / vendoring)
# — they'd couple everything to everything.
_MAX_FILES_PER_COMMIT = 50


def parse_change_coupling(
    commits: list[list[str]], min_support: int = DEFAULT_MIN_SUPPORT
) -> list[EnrichmentEdge]:
    """Co-change coupling from a list of per-commit changed-file lists.

    Emits one symmetric ``FILE_CHANGES_WITH`` edge per file pair co-changed in
    ≥ ``min_support`` commits, with a ``support`` (count) property. Endpoints are
    ``file:<path>`` ids, matching the engine's file nodes (CONCEPT:AU-KG.ingest.mine-git-history-files)."""
    pair_support: dict[tuple[str, str], int] = {}
    for files in commits:
        uniq = sorted(set(f for f in files if f))
        if len(uniq) < 2 or len(uniq) > _MAX_FILES_PER_COMMIT:
            continue
        for a, b in combinations(uniq, 2):
            pair_support[(a, b)] = pair_support.get((a, b), 0) + 1

    return [
        EnrichmentEdge(
            source=f"file:{a}",
            target=f"file:{b}",
            rel_type="FILE_CHANGES_WITH",
            props={"support": str(support)},
        )
        for (a, b), support in pair_support.items()
        if support >= min_support
    ]


def git_log_file_changes(repo_path: str, max_commits: int = 500) -> list[list[str]]:
    """Per-commit changed-file lists from ``git log`` (newest first). Returns an
    empty list when ``repo_path`` is not a git work-tree or git is unavailable."""
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                repo_path,
                "log",
                f"-n{max_commits}",
                "--name-only",
                "--pretty=format:%x01",  # SOH record separator between commits
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    commits: list[list[str]] = []
    for block in out.stdout.split("\x01"):
        files = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if files:
            commits.append(files)
    return commits


def change_coupling_for_repo(
    repo_path: str, min_support: int = DEFAULT_MIN_SUPPORT, max_commits: int = 500
) -> list[EnrichmentEdge]:
    """Mine ``FILE_CHANGES_WITH`` edges from a repo's git history end-to-end."""
    return parse_change_coupling(
        git_log_file_changes(repo_path, max_commits), min_support
    )
