#!/usr/bin/env python3
"""Reach sweep: add the `lane-guard` local hook to other repos' pre-commit config.

D-CP-3 (reports/deferred/lane-concurrency-protocol.md): the lane-arbitration
library is installed only in agent-utilities. This script is the mechanical
reach — it does NOT vendor a copy of the guard anywhere; every touched repo's
`.pre-commit-config.yaml` gains one `lane-guard` local hook that shells out to
agent-utilities' own, unmodified `scripts/check_lane_guard.py` via the SAME
`AGENT_UTILITIES_ROOT`-sibling-checkout idiom the fleet already uses for
`check-stubs`/`check-sprawl`/`check_no_legacy_markers.py` — no new dependency,
no relock of any repo's lockfile.

Safety, matching the protocol's own rules (docs/architecture/lane-concurrency.md):

* **Never touches a dirty tree.** A repo with any uncommitted change (staged or
  not) is SKIPPED, not stashed/reset/checked-out — the ~68 `agents/*` provider
  repos are deliberately dirty with signed re-certification bytes, and this
  script must never disturb that.
* **Never edits the canonical checkout directly.** Every change is made in a
  fresh worktree on its own branch, committed there, then fast-merged back into
  that repo's own canonical `main` — never in the shared working tree.
* **Idempotent.** A repo whose `.pre-commit-config.yaml` already declares
  `id: lane-guard` is reported unchanged, not re-edited.
* **Text-surgical, not YAML-round-tripped.** These configs are hand-formatted
  with comments; round-tripping through a YAML dumper would reformat the whole
  file. The hook block is inserted as text, immediately above the repo's own
  `- id: check-stubs` line (present in every repo built from the same scaffold
  the lane-guard hook was just added to), matching that repo's own indentation
  so the diff is exactly the one new block.

Usage:
    python3 scripts/rollout_lane_guard_hook.py --repo /path/to/repo [--repo ...]
    python3 scripts/rollout_lane_guard_hook.py --discover   # agents/* + the 3 frontends
    python3 scripts/rollout_lane_guard_hook.py --discover --apply   # actually write+commit+merge
Without --apply, every repo is only classified (dry run) — nothing is written.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def _workspace_root() -> Path:
    """The ``agent-packages`` directory — works whether this script is running
    from agent-utilities' canonical checkout or one of its worktrees, by asking
    git for THIS repo's own common dir rather than assuming a fixed number of
    ``parents[...]`` hops (a worktree adds a level a canonical checkout does not
    have)."""
    here = Path(__file__).resolve().parent
    common_dir = Path(
        subprocess.run(
            [
                "git",
                "-C",
                str(here),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    ).resolve()
    au_canonical = common_dir.parent  # .git's parent is the canonical checkout
    return au_canonical.parent  # agent-packages


WORKSPACE_ROOT = _workspace_root()
ANCHOR_RE = re.compile(r"^(?P<indent>[ \t]*)- id: check-stubs\s*$", re.MULTILINE)
HOOK_ID_MARKER = "id: lane-guard"


@dataclass
class Outcome:
    repo: str
    status: str  # "changed" | "skipped-dirty" | "skipped-already-present" | "skipped-no-anchor" | "skipped-no-config"
    detail: str = ""


def _run(args: list[str], cwd: Path, *, check: bool = True) -> str:
    proc = subprocess.run(
        args, cwd=str(cwd), capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} in {cwd} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _is_dirty(repo: Path) -> bool:
    status = _run(["git", "status", "--porcelain"], repo, check=False)
    return bool(status.strip())


def discover_repos() -> list[Path]:
    repos = []
    agents_dir = WORKSPACE_ROOT / "agents"
    if agents_dir.is_dir():
        for child in sorted(agents_dir.iterdir()):
            if (child / ".git").exists() and (child / "pyproject.toml").is_file():
                repos.append(child)
    for name in ("agent-webui", "agent-terminal-ui", "geniusbot"):
        p = WORKSPACE_ROOT / name
        if (p / ".git").exists():
            repos.append(p)
    return repos


def _hook_block(indent: str) -> str:
    inner = indent + "  "
    return (
        f"{indent}- id: lane-guard\n"
        f"{inner}name: Lane guard — canonical checkout read-only + stray CARGO_TARGET_DIR (D-CP-3/D-CP-4 reach)\n"
        f'{inner}entry: bash -c \'repo=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)"); '
        f'root=${{AGENT_UTILITIES_ROOT:-"$(dirname "$repo")/agent-utilities"}}; '
        f'python3 "$root/scripts/check_lane_guard.py"\'\n'
        f"{inner}language: system\n"
        f"{inner}pass_filenames: false\n"
        f"{inner}always_run: true\n"
    )


def classify_and_patch(repo: Path, *, apply: bool) -> Outcome:
    name = repo.name
    config = repo / ".pre-commit-config.yaml"
    if not config.is_file():
        return Outcome(name, "skipped-no-config")
    text = config.read_text(encoding="utf-8")
    if HOOK_ID_MARKER in text:
        return Outcome(name, "skipped-already-present")
    if _is_dirty(repo):
        return Outcome(name, "skipped-dirty", "protocol: never touch a dirty tree")
    match = ANCHOR_RE.search(text)
    if not match:
        return Outcome(name, "skipped-no-anchor", "no `- id: check-stubs` line found")
    if not apply:
        return Outcome(name, "would-change", f"anchor indent={match['indent']!r}")

    branch = "feat/lane-protocol-reach"
    worktree_root = Path.home() / ".local" / "state" / "repository-worktrees" / name
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree = worktree_root / branch.replace("/", "-")
    if not worktree.exists():
        default_branch = (
            _run(["git", "symbolic-ref", "--short", "HEAD"], repo, check=False)
            or "main"
        )
        _run(
            ["git", "worktree", "add", str(worktree), "-b", branch, default_branch],
            repo,
        )
    wt_config = worktree / ".pre-commit-config.yaml"
    wt_text = wt_config.read_text(encoding="utf-8")
    wt_match = ANCHOR_RE.search(wt_text)
    if not wt_match:
        return Outcome(name, "skipped-no-anchor", "anchor vanished in worktree")
    block = _hook_block(wt_match["indent"])
    new_text = wt_text[: wt_match.start()] + block + wt_text[wt_match.start() :]
    wt_config.write_text(new_text, encoding="utf-8")
    _run(["git", "add", ".pre-commit-config.yaml"], worktree)
    _run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            "lane-guard: reach the lane-concurrency canonical-checkout guard "
            "(D-CP-3)\n\n"
            "Adds the same lane-guard local hook agent-utilities/epistemic-graph/"
            "universal-skills already carry -- shells out, unmodified, to "
            "agent-utilities' check_lane_guard.py via the AGENT_UTILITIES_ROOT "
            "sibling-checkout idiom this repo already uses for "
            "check-stubs/check-sprawl. No new dependency.\n\n"
            "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>",
        ],
        worktree,
    )
    default_branch = (
        _run(["git", "symbolic-ref", "--short", "HEAD"], repo, check=False) or "main"
    )
    _run(["git", "merge", "--ff-only", branch], repo, check=False)
    merged = _run(["git", "log", "--oneline", "-1"], repo, check=False)
    if HOOK_ID_MARKER not in config.read_text(encoding="utf-8"):
        return Outcome(
            name,
            "committed-not-merged",
            f"branch {branch}; merge manually into {default_branch}",
        )
    _run(["git", "worktree", "remove", str(worktree)], repo, check=False)
    _run(["git", "branch", "-d", branch], repo, check=False)
    return Outcome(name, "changed", merged)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", action="append", default=[], help="explicit repo path (repeatable)"
    )
    parser.add_argument(
        "--discover", action="store_true", help="scan agents/* + the 3 frontends"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write/commit/merge (default: dry run)",
    )
    args = parser.parse_args()

    repos = [Path(r).resolve() for r in args.repo]
    if args.discover:
        repos.extend(discover_repos())
    if not repos:
        print("nothing to do -- pass --repo or --discover", file=sys.stderr)
        return 2

    outcomes = [classify_and_patch(r, apply=args.apply) for r in repos]
    by_status: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_status.setdefault(o.status, []).append(o)
    for status, items in sorted(by_status.items()):
        print(f"\n{status} ({len(items)}):")
        for o in items:
            print(f"  {o.repo}" + (f" — {o.detail}" if o.detail else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
