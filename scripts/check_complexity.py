#!/usr/bin/env python3
"""Complexity gate: ABSOLUTE ceilings on cyclomatic AND cognitive complexity.

MEASURED BY `cccc` (https://github.com/moznion/cccc), which replaced `lizard`
2026-08-27. Three reasons the swap was worth breaking continuity for:

  * lizard cannot measure COGNITIVE complexity at all. This program drove
    cyclomatic down for a full day with no instrument watching cognitive, so it
    could have fragmented logic and nobody would have seen it. (Measured after
    the swap: cognitive on the changed files went DOWN 3.7%, and the worst
    function went 456 -> 189. The risk was real; the outcome was fine.)
  * cccc covers 18 languages against lizard's narrower set, so one instrument
    now scores Python, Rust and TypeScript identically.
  * cccc emits JSON directly and has --max-* gating built in, so this gate no
    longer parses text or reimplements thresholds.

THERE IS NO BASELINE MODE, BY POLICY (CX MR-11). A baseline converts a finding
into invisible permanent debt. This gate reports the REAL distribution on every
run and enforces absolute numbers. --baseline/--write are RETIRED and fail loud.

THE GOLD STANDARD is cccc's own documented configuration:
    max-cyclomatic = 10      (McCabe's limit; NIST SP 500-235 adopts it)
    max-cognitive  = 15      (SonarSource / G. Ann Campbell)
cccc itself ships NO compiled-in default -- both are Option<u32>, and unset means
no gating whatsoever. Leaving them unset is therefore a silently vacuous gate.

★ NESTED FUNCTIONS. cccc reports a nested function under its parent's `children`,
NOT as a flat entry. A flat read of `functions` sees an enclosing `def` at
cognitive 0 and MISSES the nested body entirely. That is not hypothetical here:
the three worst Python functions in this workspace (_run_analysis_action 289,
graph_configure 244, graph_ingest 215) are all nested inside register_*_tools,
so a flat read scores those files as clean. This gate RECURSES. Do not "simplify"
_walk away.

Exit codes: 0 pass, 1 violation, 2 the gate could not run (an ENVIRONMENT fact,
never reported as a clean pass -- a gate that could not run has not found nothing).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The gold standard, per the owner. Not derived from what any repo currently
# passes -- these are cccc's own documented values.
DEFAULT_MAX_CYCLOMATIC = 10
DEFAULT_MAX_COGNITIVE = 15

#: Distribution buckets printed on EVERY run, for both metrics.
_BUCKETS = (10, 15, 20, 30, 50, 100, 200, 300)

_SKIP_PARTS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist",
    "target", "target-isolated", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "site-packages", ".tox", ".eggs", "htmlcov", "vendor", "third_party",
    ".hypothesis", ".pytest_tmp", "dist-primary", "dist-reproduction",
    "build-artifacts",
}


def _fail_env(msg: str) -> "None":
    print(f"complexity gate: CANNOT RUN: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _resolve_cccc() -> str:
    """Find cccc WITHOUT consulting any package index.

    A hook that resolves a tool from an index at hook time is how a previous
    fleet sweep shipped a gate that could not pass anywhere, producing 69 push
    failures across 226 repos. Local paths only; absent means exit 2.
    """
    env = os.environ.get("CCCC_BIN")
    if env and Path(env).is_file():
        return env
    for cand in (Path.home() / ".local/bin/cccc", Path("/usr/local/bin/cccc")):
        if cand.is_file():
            return str(cand)
    found = shutil.which("cccc")
    if found:
        return found
    _fail_env(
        "`cccc` not found. Looked at $CCCC_BIN, ~/.local/bin/cccc, "
        "/usr/local/bin/cccc and $PATH. Build it with "
        "`cargo build --release` in open-source-libraries/cccc and copy the "
        "binary to ~/.local/bin/. This gate never installs anything itself."
    )


def _walk(fn: dict, rel: str, prefix: str, out: list) -> None:
    """Collect a function AND its nested children. See the docstring."""
    name = f"{prefix}{fn['name']}"
    out.append({"fn": name, "file": rel,
                "ccn": fn["cyclomatic"], "cog": fn["cognitive"]})
    for kid in fn.get("children", ()):
        _walk(kid, rel, f"{name}.", out)


def measure(paths: list[str]) -> list[dict]:
    exe = _resolve_cccc()
    real = [p for p in paths if Path(p).exists()]
    if not real:
        return []
    try:
        r = subprocess.run([exe, *real, "--min", "1", "--jobs", "8"],
                           capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        _fail_env(f"cccc timed out over {len(real)} path(s)")
    except OSError as exc:
        _fail_env(f"could not execute {exe}: {exc}")
    # exit 1 only means a --max-* threshold was hit; we pass none, so >1 is real.
    if r.returncode > 1:
        _fail_env(f"cccc exited {r.returncode}: {(r.stderr or '').strip()[:400]}")
    if not r.stdout.strip():
        _fail_env("cccc produced no output; refusing to report that as clean")
    try:
        doc = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        _fail_env(f"cccc output was not JSON: {exc}")

    out: list[dict] = []
    for f in doc.get("files", []):
        rel = f["path"]
        if any(part in _SKIP_PARTS for part in Path(rel).parts):
            continue
        for fn in f.get("functions", []):
            _walk(fn, rel, "", out)
    return out


def _print_distribution(fns: list[dict]) -> None:
    """Print the REAL distribution, both metrics, on EVERY run.

    This is the no-ratchet policy in code: the true numbers are on screen
    unconditionally, so debt cannot become invisible.
    """
    print(f"complexity: {len(fns)} function(s) measured by cccc")
    print(f"  {'':>6}{'cyclomatic':>12}{'cognitive':>11}")
    for t in _BUCKETS:
        c = sum(1 for f in fns if f["ccn"] > t)
        g = sum(1 for f in fns if f["cog"] > t)
        if c or g:
            print(f"  >{t:<5}{c:>12}{g:>11}")
    worst_cyc = sorted(fns, key=lambda f: -f["ccn"])[:5]
    worst_cog = sorted(fns, key=lambda f: -f["cog"])[:5]
    if worst_cyc:
        print("  worst cyclomatic:")
        for f in worst_cyc:
            print(f"    {f['ccn']:>4} (cog {f['cog']:>4})  {f['fn']}@{f['file']}")
    if worst_cog:
        print("  worst COGNITIVE:")
        for f in worst_cog:
            print(f"    {f['cog']:>4} (cyc {f['ccn']:>4})  {f['fn']}@{f['file']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["."])
    ap.add_argument("--max-cyclomatic", type=int, default=DEFAULT_MAX_CYCLOMATIC)
    ap.add_argument("--max-cognitive", type=int, default=DEFAULT_MAX_COGNITIVE)
    ap.add_argument("--report", action="store_true",
                    help="print the distribution and exit 0 without enforcing")
    # RETIRED, kept so a re-introduction fails loudly rather than looking like a typo.
    ap.add_argument("--baseline", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--write", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cap", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--track", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cap-ext", action="append", default=[], help=argparse.SUPPRESS)
    ap.add_argument("--require-lizard", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.baseline is not None or args.write:
        _fail_env(
            "--baseline/--write are RETIRED. This gate enforces ABSOLUTE ceilings "
            "only. A baseline converts a finding into invisible permanent debt "
            "(CX MR-11). Drop the flag and pass --max-cyclomatic/--max-cognitive."
        )
    if args.cap is not None or args.track is not None or args.cap_ext or args.require_lizard:
        _fail_env(
            "--cap/--track/--cap-ext/--require-lizard are RETIRED with the lizard "
            "backend. This gate measures BOTH cyclomatic and cognitive complexity "
            "via cccc. Use --max-cyclomatic N and --max-cognitive N."
        )

    paths = args.paths or ["."]
    fns = measure(paths)
    if not fns:
        print("complexity gate: OK: no measurable source in " + " ".join(paths))
        return 0

    _print_distribution(fns)
    if args.report:
        return 0

    over_cyc = [f for f in fns if f["ccn"] > args.max_cyclomatic]
    over_cog = [f for f in fns if f["cog"] > args.max_cognitive]
    if not over_cyc and not over_cog:
        print(f"\ncomplexity gate: OK: nothing exceeds cyclomatic "
              f"{args.max_cyclomatic} or cognitive {args.max_cognitive}")
        return 0

    print(f"\ncomplexity gate: FAIL: {len(over_cyc)} over cyclomatic "
          f"{args.max_cyclomatic}, {len(over_cog)} over cognitive {args.max_cognitive}")
    seen: set[tuple] = set()
    for f in sorted(over_cyc + over_cog, key=lambda f: -max(f["ccn"], f["cog"]))[:30]:
        key = (f["fn"], f["file"])
        if key in seen:
            continue
        seen.add(key)
        flag = []
        if f["ccn"] > args.max_cyclomatic:
            flag.append(f"cyc {f['ccn']}")
        if f["cog"] > args.max_cognitive:
            flag.append(f"COG {f['cog']}")
        print(f"  OVER  {', '.join(flag):<22} {f['fn']}@{f['file']}")
    print("\nSplit the function into named parts. Do NOT raise a threshold to make\n"
          "this pass, and do NOT add a suppression comment -- an in-line suppression\n"
          "is a one-line baseline and is a program-terminating finding. If the\n"
          "complexity is genuinely irreducible, record a time-boxed entry with an\n"
          "owner in scripts/gate_deferrals.tsv.\n"
          "Note cognitive complexity rewards FLAT structure: extracting a deeply\n"
          "nested block into a named function removes its nesting bonus entirely,\n"
          "whereas merely renaming things does not move the number.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
