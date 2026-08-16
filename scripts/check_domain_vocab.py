#!/usr/bin/env python3
"""Gate: every OKF-CIS marker's domain is in the closed vocabulary (CONCEPT:AU-OS.governance.concept-2).

The anti-sprawl guarantee: breadth is governed. A marker whose ``<domain>`` is not
listed in ``governance/domain_vocab.yaml`` for its pillar fails the build, preventing
ungoverned domain sprawl. The gate also verifies
each id parses under the OKF-CIS grammar and its SLUG is registered.

Usage: python scripts/check_domain_vocab.py [ROOT ...]  (default: cwd)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_utilities.governance import concept_hierarchy as ch

_EXT = {".py", ".rs", ".md"}
_SKIP = {"__pycache__", ".git", ".venv", "node_modules", "target", "build", "dist"}
_GIT_ENV_LEAK_KEYS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")


def _git_subprocess_env() -> dict[str, str]:
    """Environment for a ``git -C <subdir>`` call, scrubbed of inherited GIT_* vars.

    ``git commit`` exports ``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE`` (pinned to
    this worktree) for every hook it runs. A child ``git -C <subdir> ls-files``
    subprocess that inherits those vars stops computing the ``-C``-relative path
    prefix and instead emits repo-root-relative paths -- silently breaking every
    caller that joins them back onto ``root`` (``root / line`` no longer resolves,
    so callers that additionally check ``.is_file()`` filter the tracked set down
    to nearly nothing, which made this gate scan **zero** files -- and thus
    vacuously pass -- on every real ``git commit``). Reproduced directly: with
    ``GIT_DIR`` set, ``git -C docs ls-files -- "*.md"`` returns paths like
    ``.specify/design/README.md`` instead of ``CONTEXT.md``. Stripping these
    keys forces fresh, ``-C``-relative discovery regardless of the parent
    process's hook context.
    """
    return {k: v for k, v in os.environ.items() if k not in _GIT_ENV_LEAK_KEYS}


def _candidate_files(root: Path) -> list[Path]:
    """Files under ``root``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale ``CONCEPT:`` marker no longer in real source. Falls
    back to a filtered filesystem walk only when ``root`` is not inside a
    git working tree.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
            env=_git_subprocess_env(),
        ).stdout
        tracked = [root / line for line in out.splitlines() if line]
        if tracked:
            return tracked
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return [p for p in root.rglob("*") if not any(s in p.parts for s in _SKIP)]


def scan(root: Path) -> list[str]:
    errs: list[str] = []
    known_slugs = set(ch.load_slug_registry().values())
    for p in _candidate_files(root):
        if p.suffix not in _EXT or any(s in p.parts for s in _SKIP) or not p.is_file():
            continue
        if p.name in {
            "check_domain_vocab.py",
            "domain_vocab.yaml",
            "slug_registry.yaml",
        }:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for marker in ch.iter_okf_markers(text):
            cid = marker.id
            try:
                parsed = ch.parse_okf_id(cid)
            except ValueError as ex:
                errs.append(f"{p.relative_to(root)}: {cid} — {ex}")
                continue
            if parsed.slug not in known_slugs:
                errs.append(
                    f"{p.relative_to(root)}: {cid} — SLUG {parsed.slug!r} not registered"
                )
            if not ch.is_valid_domain(parsed.pillar, parsed.domain):
                errs.append(
                    f"{p.relative_to(root)}: {cid} — domain {parsed.domain!r} not in "
                    f"closed vocab for pillar {parsed.pillar}"
                )
    return errs


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path.cwd()]
    errs: list[str] = []
    for r in roots:
        errs.extend(scan(r))
    if errs:
        print(
            f"FAIL: {len(errs)} OKF-CIS marker(s) violate the closed vocab / grammar:"
        )
        for e in errs[:60]:
            print("  " + e)
        if len(errs) > 60:
            print(f"  … and {len(errs) - 60} more")
        return 1
    print("OK: all OKF-CIS markers use registered slugs + closed-vocab domains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
