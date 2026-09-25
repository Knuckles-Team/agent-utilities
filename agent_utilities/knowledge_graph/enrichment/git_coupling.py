"""Adapt EG git change derivations to source graph edges.

Two files that keep changing in the same commits are *coupled* even when nothing
in the AST connects them — a hidden dependency the call graph can't see. We mine
that from git history: files co-changed in ≥ ``min_support`` commits get a
symmetric ``FILE_CHANGES_WITH`` edge weighted by how often. It surfaces the real
blast radius of a change (the files that historically move together).
"""

from __future__ import annotations

import subprocess

from epistemic_graph.git_derivation import DEFAULT_MIN_SUPPORT, derive_change_coupling

from .models import EdgeRung, EnrichmentEdge


def parse_change_coupling(
    commits: list[list[str]], min_support: int = DEFAULT_MIN_SUPPORT
) -> list[EnrichmentEdge]:
    """Co-change coupling from a list of per-commit changed-file lists.

    Emits one symmetric ``FILE_CHANGES_WITH`` edge per file pair co-changed in
    ≥ ``min_support`` commits, with a ``support`` (count) property. Endpoints are
    ``file:<path>`` ids, matching the engine's file nodes (CONCEPT:AU-KG.ingest.mine-git-history-files)."""
    return [
        EnrichmentEdge(
            source=f"file:{a}",
            target=f"file:{b}",
            rel_type="FILE_CHANGES_WITH",
            # Co-change frequency mined from git history -- a statistical
            # measure, DERIVED (rung 2), the exact "community/statistical"
            # category the EH-270 ladder names for this tier.
            rung=EdgeRung.DERIVED,
            props={"support": str(support)},
        )
        for a, b, support in derive_change_coupling(commits, min_support)
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
