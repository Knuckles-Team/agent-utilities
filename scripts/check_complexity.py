#!/usr/bin/env python3
"""Cyclomatic-complexity gate: hold the line today, lower the ceiling over time.

TWO MECHANISMS, because a single threshold cannot do this job:

  1. RATCHET (--baseline)  No function may get worse, and no NEW function may
     land above --track. This is what makes the gate adoptable on day one: the
     existing tail is frozen, not flagged, so the gate never blocks unrelated
     work and therefore never gets switched off.

  2. CEILING (--cap)  A hard ceiling nothing may exceed, regardless of the
     baseline. This is the mechanism that actually REDUCES complexity: start it
     above the current worst function, then step it down as remediation lands.
     Each step is enforced forever after, so the number cannot drift back up.

A ratchet alone only holds the line. A cap alone blocks every commit on day one.
Together they let the fleet converge on the target without a big-bang refactor.

Measurement is `lizard`, which scores Python, Rust, JS/TS and more with the SAME
definition, so every repo in the fleet is directly comparable. It does NOT agree
with ruff's C901: ruff attributes a nested closure's branches to the ENCLOSING
function (scoring `register_analysis_tools` at 320), while lizard attributes them
to the nested function that actually holds the branching (`_run_analysis_action`
at 290). Never mix the two tools' numbers in one report.

Exit codes: 0 pass, 1 violation, 2 the gate could not run (an ENVIRONMENT fact,
never reported as a clean pass -- a gate that could not run has not found nothing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Functions below this are not tracked: the baseline would churn on every
# refactor and the signal would drown in noise.
DEFAULT_TRACK = 15

#: Files the analyzer could not score in the last run (see measure()).
_UNMEASURED = 0

# Directories that are never our own source.
_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist",
    "target", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    ".tox", ".eggs", "htmlcov", "vendor", "third_party", ".idea", ".vscode",
}
_EXTS = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".rb"}


def _fail_env(msg: str) -> "None":
    print(f"complexity gate: CANNOT RUN: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _iter_files(roots: list[Path]):
    for root in roots:
        if root.is_file():
            if root.suffix in _EXTS:
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix in _EXTS:
                    yield p


def _autodetect(repo: Path) -> list[Path]:
    """Source roots for a repo, when the hook does not name them.

    Keeps the hook entry identical across every repo in the fleet, which is the
    whole point of a fleet-wide gate -- 79 bespoke entries would rot.
    """
    roots: list[Path] = []
    for cand in ("src", "crates", "lib", "app"):
        p = repo / cand
        if p.is_dir():
            roots.append(p)
    # Python packages: any top-level dir with an __init__.py that is not tests.
    for p in sorted(repo.iterdir()):
        if not p.is_dir() or p.name in _SKIP_DIRS or p.name.startswith("."):
            continue
        if p.name in {"tests", "test", "docs", "scripts", "examples"}:
            continue
        if (p / "__init__.py").is_file() and p not in roots:
            roots.append(p)
    return roots or [repo]


def _analyze_one(path_str: str):
    """Worker: returns (path, [(fn_name, ccn), ...]) or (path, None) on failure.

    Python goes through the stdlib counter so 78 of the fleet's 79 repos need NO
    third-party package and can run `language: system`. Installing lizard into a
    pre-commit venv cost 6m04s on first run in a pilot repo -- a tax that gets a
    hook disabled, and a disabled gate measures nothing.
    """
    if path_str.endswith(".py"):
        from _ccn_ast import analyze_python  # noqa: PLC0415

        return path_str, analyze_python(path_str)

    try:
        import lizard  # noqa: PLC0415 - only needed for non-Python (i.e. eg's Rust)
    except ImportError:
        return path_str, None
    try:
        res = lizard.analyze_file(path_str)
    except Exception:  # noqa: BLE001 - one unparseable file must not blind the gate
        return path_str, None
    return path_str, [(f.name, int(f.cyclomatic_complexity)) for f in res.function_list]


def measure(roots: list[Path], track: int, repo: Path) -> dict[str, int]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    files = [str(p) for p in _iter_files(roots)]

    # Mixed-language repos (agent-webui's TS/TSX, servicenow-api, universal-skills)
    # need lizard for the non-Python half. If it is absent we still gate Python and
    # say LOUDLY what went unmeasured, recording the count in the baseline so the
    # gap is durable rather than a one-off console line. Refusing outright would
    # leave those repos with NO gate, which is strictly worse; silently measuring
    # only Python would be a gate reporting more coverage than it has, which is
    # worse still.
    non_py = [f for f in files if not f.endswith(".py")]
    global _UNMEASURED
    _UNMEASURED = 0
    if non_py:
        try:
            import lizard  # noqa: F401,PLC0415
        except ImportError:
            _UNMEASURED = len(non_py)
            files = [f for f in files if f.endswith(".py")]
            print(f"complexity gate: PARTIAL COVERAGE: {_UNMEASURED} non-Python file(s) "
                  f"NOT measured (`lizard` unavailable). Python is still gated. "
                  f"Install lizard to close this gap.", file=sys.stderr)
    if not files:
        return {}

    # Parallel because agent-utilities and epistemic-graph are ~400k NLOC each and
    # a serial pass costs ~32s -- slow enough that a hook gets disabled. Small
    # fleet packages finish in well under a second either way.
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    workers = min(8, (os.cpu_count() or 2))
    found: dict[str, int] = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_analyze_one, files, chunksize=16))
    except Exception:  # noqa: BLE001 - fall back rather than fail the commit
        results = [_analyze_one(f) for f in files]

    for path_str, fns in results:
        if fns is None:
            print(f"complexity gate: WARNING: could not analyze {path_str}", file=sys.stderr)
            continue
        try:
            rel = Path(path_str).relative_to(repo).as_posix()
        except ValueError:
            rel = Path(path_str).as_posix()
        for name, ccn in fns:
            if ccn < track:
                continue
            # Key excludes the line number so the entry survives edits above it.
            key = f"{name}@{rel}"
            found[key] = max(found.get(key, 0), ccn)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="source roots (auto-detected if omitted)")
    ap.add_argument("--baseline", type=Path, default=Path(".complexity-baseline.json"))
    ap.add_argument("--track", type=int, default=DEFAULT_TRACK,
                    help=f"only track functions at or above this CCN (default {DEFAULT_TRACK})")
    ap.add_argument("--cap", type=int, default=None,
                    help="hard ceiling nothing may exceed; step down to reduce complexity")
    ap.add_argument("--write", action="store_true", help="(re)generate the baseline")
    ap.add_argument("--report", action="store_true", help="print the distribution and exit 0")
    args = ap.parse_args()

    repo = Path.cwd()
    roots = [Path(p) for p in args.paths] if args.paths else _autodetect(repo)
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("complexity gate: OK: no source roots to analyze")
        return 0

    current = measure(roots, args.track, repo)

    if args.report:
        allv = sorted(current.values(), reverse=True)
        print(f"complexity report: {len(allv)} functions >= {args.track}")
        for t in (15, 20, 30, 50, 100):
            print(f"  >{t:<4} {sum(1 for v in allv if v > t)}")
        for k, v in sorted(current.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {v:>4}  {k}")
        return 0

    if args.write:
        doc = {"track": args.track, "functions": dict(sorted(current.items()))}
        if _UNMEASURED:
            doc["unmeasured_non_python_files"] = _UNMEASURED
        args.baseline.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"complexity gate: wrote baseline: {len(current)} functions >= {args.track}")
        return 0

    if not args.baseline.is_file():
        _fail_env(f"no baseline at {args.baseline}; generate it with --write")

    base: dict[str, int] = json.loads(args.baseline.read_text(encoding="utf-8")).get("functions") or {}

    over_cap = ([(k, v) for k, v in current.items() if v > args.cap] if args.cap else [])
    added = [(k, v) for k, v in current.items() if k not in base]
    worse = [(k, base[k], v) for k, v in current.items() if k in base and v > base[k]]
    better = [k for k, v in current.items() if k in base and v < base[k]]
    gone = [k for k in base if k not in current]

    if over_cap or added or worse:
        print(f"complexity gate: FAIL: {len(added)} new, {len(worse)} worsened"
              + (f", {len(over_cap)} over the cap of {args.cap}" if args.cap else ""))
        for k, v in sorted(over_cap, key=lambda x: -x[1])[:20]:
            print(f"  OVER CAP  {v:>4}  {k}")
        for k, v in sorted(added, key=lambda x: -x[1])[:20]:
            print(f"  NEW       {v:>4}  {k}")
        for k, was, now in sorted(worse, key=lambda x: x[2] - x[1], reverse=True)[:20]:
            print(f"  WORSE  {was:>4}->{now:<4} {k}")
        print("\nSplit the function into named parts, or -- if the complexity is genuinely\n"
              "irreducible -- re-baseline deliberately with --write and justify it in the\n"
              "commit message. Do not raise the cap to make this pass.")
        return 1

    print(f"complexity gate: OK: {len(current)} tracked >= {args.track}"
          + (f" [PARTIAL: {_UNMEASURED} non-Python files unmeasured]" if _UNMEASURED else "")
          + (f", cap {args.cap}" if args.cap else "")
          + f", {len(better) + len(gone)} improved/removed since baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
