#!/usr/bin/env python3
"""Cyclomatic-complexity gate: an ABSOLUTE ceiling, driven down deliberately.

ONE MECHANISM. --cap N is a hard ceiling nothing may exceed. Start it above the
current worst function, step it down as remediation lands; each step is enforced
forever after, so the number cannot drift back up.

THERE IS NO BASELINE MODE, BY POLICY (CX program, MR-11). A baseline converts a
finding into invisible permanent debt. The evidence in this workspace is
decisive: every baseline here only ever grew, and each became a merge-conflict
surface because every concurrent lane rewrote it. So this gate REPORTS THE REAL
DISTRIBUTION ON EVERY RUN, unconditionally, and enforces an absolute number.

If the true number is too large to enforce at once, that is a program with
waves -- not a reason to hide it behind a file.

--baseline and --write are RETIRED and fail loudly rather than silently
reintroducing the mechanism.

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

# Functions below this are not tracked: the signal would drown in noise.
# The CX program targets CCN <= 10, so the gate must be able to SEE the
# 10-15 band -- with the old default of 15 it was structurally invisible.
DEFAULT_TRACK = 10

#: Distribution buckets printed on EVERY run. Extended past the old
#: (15,20,30,50,100) so the workspace's worst function (CCN 353) and the
#: sub-15 band are both visible.
_BUCKETS = (10, 12, 15, 20, 30, 50, 100, 200, 300)

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
            continue
        # One level down. agent-webui's package lives at agent/agent_webui/ and
        # `agent/` has no __init__.py, so a top-level-only scan measured NOTHING
        # there -- the gate reported OK on a repo holding a CCN-144 function.
        # A gate that silently scores nothing is worse than no gate at all.
        try:
            children = sorted(p.iterdir())
        except OSError:
            continue
        for q in children:
            if q.is_dir() and q.name not in _SKIP_DIRS \
                    and (q / "__init__.py").is_file() and q not in roots:
                roots.append(q)
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


def _print_distribution(current: dict[str, int], track: int) -> None:
    """Print the REAL distribution. Called on EVERY run, in every mode.

    This is the no-ratchet policy in code: the true number is on screen
    unconditionally, so debt cannot become invisible.
    """
    allv = sorted(current.values(), reverse=True)
    print(f"complexity: {len(allv)} functions >= {track}")
    for t in _BUCKETS:
        if t < track:
            continue
        print(f"  >{t:<4} {sum(1 for v in allv if v > t)}")
    for k, v in sorted(current.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {v:>4}  {k}")
    if _UNMEASURED:
        print(f"  [{_UNMEASURED} non-Python file(s) UNMEASURED -- see --require-lizard]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="source roots (auto-detected if omitted)")
    ap.add_argument("--track", type=int, default=DEFAULT_TRACK,
                    help=f"only track functions at or above this CCN (default {DEFAULT_TRACK})")
    ap.add_argument("--cap", type=int, default=None,
                    help="ABSOLUTE ceiling nothing may exceed; step it down over time")
    ap.add_argument("--cap-ext", action="append", default=[], metavar="EXT=N",
                    help="per-extension cap override, e.g. --cap-ext .rs=15 (repeatable)")
    ap.add_argument("--require-lizard", action="store_true",
                    help="exit 2 if any non-Python file could not be scored")
    ap.add_argument("--report", action="store_true",
                    help="print the distribution and exit 0 without enforcing")
    # RETIRED. Kept so a re-introduction fails loudly rather than being rejected
    # with a generic argparse error that reads like a typo.
    ap.add_argument("--baseline", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--write", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.baseline is not None or args.write:
        _fail_env(
            "--baseline/--write are RETIRED. This gate enforces an ABSOLUTE --cap "
            "only. A baseline converts a finding into invisible permanent debt "
            "(CX program, MR-11). Drop the flag from the hook entry and pass "
            "--cap N instead."
        )

    caps: dict[str, int] = {}
    for spec in args.cap_ext:
        ext, sep, n = spec.partition("=")
        if not sep or not n.strip().lstrip("-").isdigit():
            _fail_env(f"--cap-ext expects EXT=N, got {spec!r}")
        caps[ext if ext.startswith(".") else f".{ext}"] = int(n)

    if not args.report and args.cap is None and not caps:
        _fail_env("--cap (or --cap-ext) is required: there is no baseline mode, "
                  "and a gate with no threshold has not found nothing")

    # ★ A cap BELOW the tracking floor is SILENTLY UNENFORCEABLE: measure() drops
    # everything under --track before the cap comparison ever runs, so the gate
    # would report "nothing exceeds cap 10" while functions at 11-14 sat
    # unmeasured. A lane hit exactly this and only caught it by re-scanning by
    # hand. Clamp, and say so out loud, rather than let a caller believe a cap
    # that is not being applied.
    all_caps = ([args.cap] if args.cap is not None else []) + list(caps.values())
    effective = min(all_caps) if all_caps else None
    if effective is not None and args.track > effective:
        print(f"complexity gate: lowering --track {args.track} -> {effective} so the "
              f"cap is actually enforceable (measure() filters below --track before "
              f"the cap is applied)", file=sys.stderr)
        args.track = effective

    repo = Path.cwd()
    roots = [Path(p) for p in args.paths] if args.paths else _autodetect(repo)
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("complexity gate: OK: no source roots to analyze")
        return 0

    current = measure(roots, args.track, repo)

    # Unconditional, in every mode, before any verdict.
    _print_distribution(current, args.track)

    if _UNMEASURED and args.require_lizard:
        _fail_env(
            f"{_UNMEASURED} non-Python file(s) unmeasured: `lizard` is not "
            f"importable by {sys.executable}. A gate that could not score the "
            f"Rust/TypeScript half has not found nothing. Install lizard into "
            f"this repo's environment (dependency-group `guardrails`)."
        )

    if args.report:
        return 0

    def _cap_for(key: str) -> "int | None":
        # measure() keys findings as f"{name}@{rel}", so the extension is
        # recoverable from the key with no change to the measurement path.
        ext = Path(key.rsplit("@", 1)[-1]).suffix
        return caps.get(ext, args.cap)

    over = []
    for k, v in current.items():
        c = _cap_for(k)
        if c is not None and v > c:
            over.append((k, v, c))

    if over:
        print(f"\ncomplexity gate: FAIL: {len(over)} function(s) over the cap")
        for k, v, c in sorted(over, key=lambda x: -x[1])[:30]:
            print(f"  OVER CAP  {v:>4} (cap {c})  {k}")
        if len(over) > 30:
            print(f"  ... and {len(over) - 30} more")
        print("\nSplit the function into named parts. Do NOT raise the cap to make\n"
              "this pass, and do NOT add a suppression comment -- an in-line\n"
              "suppression is a one-line baseline and is a program-terminating\n"
              "finding. If the complexity is genuinely irreducible, record a\n"
              "time-boxed entry with an owner in scripts/gate_deferrals.tsv.")
        return 1

    shown = args.cap if args.cap is not None else "per-extension"
    print(f"\ncomplexity gate: OK: nothing exceeds cap {shown}"
          + (f" [PARTIAL: {_UNMEASURED} non-Python files unmeasured]" if _UNMEASURED else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
