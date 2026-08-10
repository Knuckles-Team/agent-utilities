#!/usr/bin/env python3
"""Reach sweep: add the `lane-guard` local hook to other repos' pre-commit config.

D-CP-3 (reports/deferred/lane-concurrency-protocol.md): the lane-arbitration
library is installed only in agent-utilities. This script is the mechanical
reach — it does NOT vendor a copy of the guard anywhere; every touched repo's
`.pre-commit-config.yaml` gains one `lane-guard` local hook that shells out to
agent-utilities' own, unmodified `scripts/check_lane_guard.py` via an upward
ancestor search for the `agent-utilities` checkout (see `_hook_block` below),
honouring `AGENT_UTILITIES_ROOT` when set — no new dependency, no relock of
any repo's lockfile.

Safety, matching the protocol's own rules (docs/architecture/lane-concurrency.md):

* **Never touches a dirty tree.** A repo with any uncommitted change (staged or
  not) is SKIPPED, not stashed/reset/checked-out — the ~68 `agents/*` provider
  repos are deliberately dirty with signed re-certification bytes, and this
  script must never disturb that.
* **Never edits the canonical checkout directly.** Every change is made in a
  fresh worktree on its own branch, committed there, then fast-merged back into
  that repo's own canonical `main` — never in the shared working tree.
* **Idempotent by default.** A repo whose `.pre-commit-config.yaml` already
  declares `id: lane-guard` is reported unchanged, not re-edited — even if its
  block is stale (RMDD-D1: fixing the generator alone repairs zero already-
  rolled-out repos). Pass ``--force``/``--rewrite`` to replace an existing
  block with the current one; default behaviour is unchanged.
* **Text-surgical, not YAML-round-tripped.** These configs are hand-formatted
  with comments; round-tripping through a YAML dumper would reformat the whole
  file. A new hook block is inserted as text, immediately above the repo's own
  `- id: check-stubs` line (present in every repo built from the same scaffold
  the lane-guard hook was just added to), matching that repo's own indentation
  so the diff is exactly the one new block. A re-edit (``--force``) replaces
  only the matched `id: lane-guard` block span, leaving everything else intact.

Usage:
    python3 scripts/rollout_lane_guard_hook.py --repo /path/to/repo [--repo ...]
    python3 scripts/rollout_lane_guard_hook.py --discover   # agents/* + the 3 frontends
    python3 scripts/rollout_lane_guard_hook.py --discover --apply   # write+commit+merge
    python3 scripts/rollout_lane_guard_hook.py --discover --apply --force  # + rewrite existing blocks
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

# The defect this module fixes (RMDD-D1 / INFRA-1): the hook's entry used to
# derive the agent-utilities root positionally with exactly one `dirname`
# hop -- `"$(dirname "$repo")/agent-utilities"` -- which only resolves for a
# repo sitting directly under `agent-packages/` (e.g. the 3 frontends,
# epistemic-graph). Every repo one level deeper (everything under
# `agent-packages/agents/`) resolved to a nonexistent path and exited 2. This
# fragment identifies that old, depth-fragile formula in an existing block.
_OLD_DERIVATION_FRAGMENT = 'dirname "$repo")/agent-utilities'

# Matches an existing `- id: lane-guard` block: the id line plus every
# following line that is indented *more* than the id line (its continuation
# lines -- name/entry/language/pass_filenames/always_run, plus any leading
# comment lines a hand-authored block may carry, e.g. epistemic-graph's).
# Stops at the first line that is not more-indented than the id line, i.e.
# the next hook's `- id:` or a dedent -- so it never swallows a neighbour.
_EXISTING_BLOCK_RE = re.compile(
    r"^(?P<indent>[ \t]*)- id: lane-guard[ \t]*\n"
    r"(?P<continuation>(?:(?P=indent)[ \t]+.*\n?)*)",
    re.MULTILINE,
)


@dataclass
class Outcome:
    repo: str
    status: str
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
    """Render the `lane-guard` hook block at *indent*.

    The entry does an upward ancestor search from the invoking repo's
    canonical checkout (via `--git-common-dir`, so it resolves correctly even
    when pre-commit runs inside a linked worktree) for the first directory
    containing `agent-utilities/scripts/check_lane_guard.py` -- generic over
    any repo depth under the workspace, unlike the old single `dirname` hop
    (RMDD-D1/INFRA-1). `AGENT_UTILITIES_ROOT`, when set, is honoured and is
    itself validated -- pointed somewhere without `scripts/check_lane_guard.py`
    refuses loudly rather than silently no-op'ing. An unreachable guard is a
    degraded read, not an honest absence (H-12): if no root is found either
    way, the hook REFUSES (exit 1) naming exactly what it could not find,
    rather than passing.
    """
    inner = indent + "  "
    entry = (
        "bash -c '"
        'repo=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)"); '
        'if [ -n "$AGENT_UTILITIES_ROOT" ]; then '
        'root="$AGENT_UTILITIES_ROOT"; '
        'if [ ! -f "$root/scripts/check_lane_guard.py" ]; then '
        'echo "lane-guard REFUSED - AGENT_UTILITIES_ROOT=$root has no scripts/check_lane_guard.py" >&2; '
        "exit 1; "
        "fi; "
        "else "
        'dir="$repo"; root=""; '
        'while [ "$dir" != "/" ]; do '
        'if [ -f "$dir/agent-utilities/scripts/check_lane_guard.py" ]; then '
        'root="$dir/agent-utilities"; break; '
        "fi; "
        'dir=$(dirname "$dir"); '
        "done; "
        'if [ -z "$root" ]; then '
        'echo "lane-guard REFUSED - could not find agent-utilities/scripts/check_lane_guard.py in any ancestor of $repo; set AGENT_UTILITIES_ROOT to override" >&2; '
        "exit 1; "
        "fi; "
        "fi; "
        'python3 "$root/scripts/check_lane_guard.py"\''
    )
    return (
        f"{indent}- id: lane-guard\n"
        f"{inner}name: Lane guard — canonical checkout read-only + stray CARGO_TARGET_DIR (D-CP-3/D-CP-4 reach)\n"
        f"{inner}entry: {entry}\n"
        f"{inner}language: system\n"
        f"{inner}pass_filenames: false\n"
        f"{inner}always_run: true\n"
    )


def _find_existing_block(text: str) -> tuple[str, str] | None:
    """Return ``(full_matched_text, indent)`` for an existing lane-guard
    block, or ``None`` if the repo carries no such hook."""
    match = _EXISTING_BLOCK_RE.search(text)
    if not match:
        return None
    return match.group(0), match["indent"]


def _classify(repo: Path, text: str) -> str:
    """Classify a repo's current lane-guard state without mutating anything.

    - ``missing``               -- no `id: lane-guard` hook at all.
    - ``present-correct``       -- hook present and already textually
      identical to what `_hook_block` renders today.
    - ``present-old-broken``    -- hook present, uses the old one-`dirname`-
      hop formula, and this repo sits deep enough (not a direct child of
      `agent-packages/`) that the formula resolves to a nonexistent path
      (the RMDD-D1/INFRA-1 defect itself).
    - ``present-old-functional`` -- hook present, uses the old formula, but
      this repo sits directly under `agent-packages/` (e.g. a frontend) so
      the one-hop derivation happens to still resolve correctly today. Still
      worth adopting the robust version for depth-independence, but it is
      not the bug.
    - ``present-differs-other`` -- hook present, but neither the old nor the
      current formula -- some other hand edit or a partial rollout.
    """
    found = _find_existing_block(text)
    if found is None:
        return "missing"
    block, indent = found
    if block == _hook_block(indent):
        return "present-correct"
    if _OLD_DERIVATION_FRAGMENT in block:
        if repo.resolve().parent == WORKSPACE_ROOT:
            return "present-old-functional"
        return "present-old-broken"
    return "present-differs-other"


def classify_and_patch(repo: Path, *, apply: bool, force: bool) -> Outcome:
    name = repo.name
    config = repo / ".pre-commit-config.yaml"
    if not config.is_file():
        return Outcome(name, "skipped-no-config")
    text = config.read_text(encoding="utf-8")
    state = _classify(repo, text)

    if state == "present-correct":
        return Outcome(name, "skipped-already-correct")

    if state != "missing" and not force:
        # Idempotent-by-marker default preserved: a present block, however
        # stale, is left alone unless the caller explicitly opts into a
        # rewrite. This is the re-edit path's whole reason for existing --
        # see the module docstring and RMDD-D1.
        return Outcome(name, f"skipped-{state}", "pass --force/--rewrite to replace")

    if _is_dirty(repo):
        return Outcome(name, "skipped-dirty", "protocol: never touch a dirty tree")

    is_insert = state == "missing"
    if is_insert:
        match = ANCHOR_RE.search(text)
        if not match:
            return Outcome(name, "skipped-no-anchor", "no `- id: check-stubs` line found")
        if not apply:
            return Outcome(name, "would-insert", f"anchor indent={match['indent']!r}")
    else:
        if not apply:
            return Outcome(name, f"would-rewrite ({state})")

    branch = "feat/lane-protocol-reach" if is_insert else "fix/lane-guard-hook-resolution"
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
    if is_insert:
        wt_match = ANCHOR_RE.search(wt_text)
        if not wt_match:
            return Outcome(name, "skipped-no-anchor", "anchor vanished in worktree")
        block = _hook_block(wt_match["indent"])
        new_text = wt_text[: wt_match.start()] + block + wt_text[wt_match.start() :]
        commit_message = (
            "lane-guard: reach the lane-concurrency canonical-checkout guard "
            "(D-CP-3)\n\n"
            "Adds the same lane-guard local hook agent-utilities/epistemic-graph/"
            "universal-skills already carry -- shells out, unmodified, to "
            "agent-utilities' check_lane_guard.py via an upward ancestor search "
            "for the agent-utilities checkout, honouring AGENT_UTILITIES_ROOT "
            "when set. No new dependency.\n\n"
            "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
        )
    else:
        wt_existing = _find_existing_block(wt_text)
        if wt_existing is None:
            return Outcome(name, "skipped-block-vanished", "lane-guard block vanished in worktree")
        old_block, indent = wt_existing
        new_block = _hook_block(indent)
        start = wt_text.index(old_block)
        new_text = wt_text[:start] + new_block + wt_text[start + len(old_block) :]
        commit_message = (
            "lane-guard: repair the AGENT_UTILITIES_ROOT resolution "
            "(INFRA-1, RMDD-D1)\n\n"
            'Replaces the one-`dirname`-hop `"$(dirname "$repo")/agent-utilities"` '
            "guess with an upward ancestor search for "
            "agent-utilities/scripts/check_lane_guard.py, generic over repo depth "
            "-- the old formula resolved to a nonexistent path for every repo "
            "under agent-packages/agents/, so the guard (canonical-checkout and "
            "stray CARGO_TARGET_DIR refusal) never executed there. The hook now "
            "refuses loudly, naming what it could not find, instead of silently "
            "erroring or passing, when no root is found or AGENT_UTILITIES_ROOT "
            "is invalid.\n\n"
            "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
        )
    wt_config.write_text(new_text, encoding="utf-8")
    _run(["git", "add", ".pre-commit-config.yaml"], worktree)
    _run(["git", "commit", "-q", "-m", commit_message], worktree)
    default_branch = (
        _run(["git", "symbolic-ref", "--short", "HEAD"], repo, check=False) or "main"
    )
    _run(["git", "merge", "--ff-only", branch], repo, check=False)
    merged_text = config.read_text(encoding="utf-8")
    ok = (
        HOOK_ID_MARKER in merged_text
        and (is_insert or _hook_block(_find_existing_block(merged_text)[1]) in merged_text)
    )
    merged = _run(["git", "log", "--oneline", "-1"], repo, check=False)
    if not ok:
        return Outcome(
            name,
            "committed-not-merged",
            f"branch {branch}; merge manually into {default_branch}",
        )
    _run(["git", "worktree", "remove", str(worktree)], repo, check=False)
    _run(["git", "branch", "-d", branch], repo, check=False)
    return Outcome(name, "changed" if is_insert else "rewritten", merged)


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
    parser.add_argument(
        "--force",
        "--rewrite",
        dest="force",
        action="store_true",
        help=(
            "replace an existing `id: lane-guard` block with the current one "
            "instead of leaving it unchanged (default: idempotent-by-marker, "
            "an existing block is never re-edited)"
        ),
    )
    args = parser.parse_args()

    repos = [Path(r).resolve() for r in args.repo]
    if args.discover:
        repos.extend(discover_repos())
    if not repos:
        print("nothing to do -- pass --repo or --discover", file=sys.stderr)
        return 2

    outcomes = [
        classify_and_patch(r, apply=args.apply, force=args.force) for r in repos
    ]
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
