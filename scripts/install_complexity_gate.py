#!/usr/bin/env python3
"""Install the complexity gate into every agent-packages repo, idempotently.

Run with --dry-run first and READ THE PLAN. A previous fleet sweep in this
workspace shipped a hook that could not pass anywhere -- it resolved its tool
from PyPI, where the required version did not exist -- and produced 69 push
failures across 226 repos. The rules that came out of that are baked in here:

  * the hook resolves its script from the LOCAL agent-utilities checkout, never
    from a package index (same walk-up + AGENT_UTILITIES_ROOT pattern the fleet's
    existing okf-no-legacy-concepts / lane-guard hooks already use);
  * it uses `language: system` with NO third-party dependency for Python repos,
    because installing lizard into a pre-commit venv cost 6m04s per repo on first
    run -- a tax that gets hooks disabled;
  * every repo gets its OWN baseline, generated from its current tree, so
    installing the gate can never fail an unrelated commit on day one;
  * --verify actually RUNS the hook in each repo after installing it, because a
    gate nobody proved is a gate nobody can trust.

Usage:
    install_complexity_gate.py --root /home/apps/workspace/agent-packages --dry-run
    install_complexity_gate.py --root ... --apply [--cap N] [--exclude repo ...]
    install_complexity_gate.py --root ... --verify
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HOOK_ID = "complexity-gate"

# Mirrors the fleet's existing local-hook resolver so there is one pattern, not two.
_RESOLVER = (
    'repo=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)"); '
    'if [ -n "$AGENT_UTILITIES_ROOT" ]; then root="$AGENT_UTILITIES_ROOT"; '
    'else d="$repo"; root=""; while [ "$d" != "/" ]; do '
    'if [ -d "$d/agent-utilities/scripts" ]; then root="$d/agent-utilities"; break; fi; '
    'd=$(dirname "$d"); done; fi; '
    'if [ ! -d "$root/scripts" ]; then '
    'echo "REFUSED - cannot locate agent-utilities/scripts from $repo '
    '(set AGENT_UTILITIES_ROOT)" >&2; exit 1; fi; '
    'python3 "$root/scripts/check_complexity.py" '
    '--baseline .complexity-baseline.json{cap}'
)


def _hook_yaml(cap: int | None) -> str:
    entry = _RESOLVER.format(cap=f" --cap {cap}" if cap else "")
    return (
        f"  - id: {HOOK_ID}\n"
        f"    name: Cyclomatic complexity — no new complexity, ceiling enforced\n"
        f"    entry: bash -c '{entry}'\n"
        f"    language: system\n"
        f"    pass_filenames: false\n"
        f"    always_run: true\n"
    )


def _repos(root: Path, exclude: set[str]) -> list[Path]:
    out = []
    for git in sorted(root.glob("*/.git")) + sorted(root.glob("*/*/.git")):
        repo = git.parent
        rel = repo.relative_to(root).as_posix()
        if rel in exclude or repo.name in exclude:
            continue
        out.append(repo)
    return out


def _script(repo: Path) -> Path:
    return Path(__file__).resolve().parent / "check_complexity.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--exclude", nargs="*", default=[])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    repos = _repos(args.root, set(args.exclude))
    script = _script(args.root)
    if not script.is_file():
        print(f"REFUSED: gate script not found at {script}", file=sys.stderr)
        return 2

    installed = skipped = failed = 0
    for repo in repos:
        cfg = repo / ".config" / "pre-commit.yaml"
        if not cfg.is_file():
            cfg = repo / ".pre-commit-config.yaml"
        base = repo / ".complexity-baseline.json"

        if args.verify:
            r = subprocess.run(
                [sys.executable, str(script), "--baseline", str(base)]
                + (["--cap", str(args.cap)] if args.cap else []),
                cwd=repo, capture_output=True, text=True, check=False,
            )
            tag = "OK  " if r.returncode == 0 else f"FAIL({r.returncode})"
            print(f"  {tag} {repo.relative_to(args.root)}: "
                  f"{(r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr).strip() else ''}")
            failed += r.returncode != 0
            continue

        if not cfg.is_file():
            print(f"  SKIP {repo.relative_to(args.root)}: no tracked pre-commit config")
            skipped += 1
            continue
        text = cfg.read_text(encoding="utf-8")
        if f"id: {HOOK_ID}" in text:
            print(f"  HAVE {repo.relative_to(args.root)}: hook already present")
            skipped += 1
            continue
        if "- repo: local" not in text:
            print(f"  SKIP {repo.relative_to(args.root)}: no `- repo: local` block to extend")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  WOULD {repo.relative_to(args.root)}: add hook + baseline")
            installed += 1
            continue

        # Baseline FIRST: a hook installed without one fails closed with exit 2,
        # which would block every commit in the repo.
        r = subprocess.run(
            [sys.executable, str(script), "--baseline", str(base), "--write"],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        if r.returncode != 0:
            print(f"  FAIL {repo.relative_to(args.root)}: baseline: "
                  f"{(r.stderr or r.stdout).strip()[:120]}")
            failed += 1
            continue

        # Append to the LAST `- repo: local` block's hooks list.
        idx = text.rindex("- repo: local")
        end = text.find("\n- repo:", idx + 1)
        end = len(text) if end == -1 else end
        text = text[:end].rstrip("\n") + "\n" + _hook_yaml(args.cap) + text[end:]
        cfg.write_text(text, encoding="utf-8")
        print(f"  ADD  {repo.relative_to(args.root)}: {r.stdout.strip()}")
        installed += 1

    verb = "verified" if args.verify else ("would install" if args.dry_run else "installed")
    print(f"\n{verb}: {installed}, skipped: {skipped}, failed: {failed}, "
          f"repos considered: {len(repos)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
