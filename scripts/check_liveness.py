#!/usr/bin/env python3
"""Liveness / dead-pathway ratchet gate (CONCEPT:AU-KG.maintenance.periodic-code-health consumer).

Wire-First's static `check_wiring.py` catches modules with no import path. This
gate adds the two layers it can't: the **typed-seam / contract-drift** scan (public
functions returning an untyped ``dict``/``list[dict]`` — the seam where a producer
writes ``_score`` and a consumer reads ``score`` and silently gets ``0.00``) and,
when a coverage report is provided, the **never-executed** layer.

It does NOT duplicate the detector — that lives once in the code-enhancer skill
(`universal_skills/.../code-enhancer/scripts/analyze_liveness.py`). This gate
locates it, runs it over ``agent_utilities/``, and ratchets against a committed
baseline (`.liveness_baseline.json`): the build fails only when a category
REGRESSES (more findings than the baseline), so dead pathways can only shrink.

If the code-enhancer skill is not installed, the gate skips cleanly (exit 0) rather
than blocking CI — install `universal-skills` to enable it.

Run ``--update-baseline`` to re-baseline after an intentional, reviewed change.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "agent_utilities"
BASELINE = REPO / ".liveness_baseline.json"


def _head_commit() -> str:
    """The commit a baseline is being taken against (best effort)."""
    try:
        res = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return res.stdout.strip() or "unknown"


def _find_analyzer() -> Path | None:
    spec = importlib.util.find_spec("universal_skills")
    locs = list(getattr(spec, "submodule_search_locations", []) or []) if spec else []
    for loc in locs:
        cand = Path(loc) / "core" / "code-enhancer" / "scripts" / "analyze_liveness.py"
        if cand.exists():
            return cand
    return None


def main() -> int:
    analyzer = _find_analyzer()
    if analyzer is None:
        # Say NOT ENFORCED, loudly. This branch is not hypothetical: the
        # `guardrails` dependency group `.github/workflows/guardrails.yml` syncs
        # does NOT contain `universal-skills` (it is only in the `agent-runtime`
        # / `agent-headless` extras), so the CI job named "Liveness ratchet gate"
        # takes this path and exits 0 — a gate that reads green in the log while
        # ratcheting nothing (D-PCG-9). Until that group gains the dependency,
        # the wording must make a skip impossible to mistake for a pass.
        print(
            "liveness gate NOT ENFORCED (exit 0, nothing was checked): the "
            "code-enhancer detector (universal_skills) is not importable by "
            f"{sys.executable}. Add `universal-skills` to the environment this "
            "gate runs in — for CI that is the `guardrails` dependency group in "
            "pyproject.toml."
        )
        return 0

    update = "--update-baseline" in sys.argv
    cmd = [sys.executable, str(analyzer), str(TARGET)]
    cov = REPO / "coverage.json"
    if cov.exists():
        cmd += ["--coverage", str(cov)]
    if BASELINE.exists() and not update:
        cmd += ["--baseline", str(BASELINE)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode not in (0, 1):  # 0=ok/pass, 1=regression; other=crash
        sys.stderr.write(res.stderr)
        # A crashing analyzer is NOT a pass. This previously returned 0, which
        # made a broken detector indistinguishable from a clean tree — the same
        # silently-non-functional-gate failure mode `check_cpd.py` and
        # `check_surface_parity.py` were both in (see scripts/_gate_interpreter.py).
        print(
            f"liveness gate FAILED: the detector crashed (exit {res.returncode}); "
            "it enforced nothing. Fix the analyzer — do not re-baseline around it."
        )
        return 1

    report = json.loads(res.stdout)
    if update:
        # Record WHAT the baseline was taken against, not just the numbers. A
        # bare counts blob cannot be told apart from a fresh one, which is how
        # this file silently drifted ~4 days behind `main` (D-KCI-11) while the
        # gate kept reporting "regressed" against the wrong reference. Extra
        # keys are inert to the analyzer, which reads only ``counts``.
        BASELINE.write_text(
            json.dumps(
                {
                    "counts": report["counts"],
                    "taken_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "taken_against": _head_commit(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"liveness baseline updated: {report['counts']}")
        return 0

    print(
        f"Liveness {report['grade']} score={report['score']} counts={report['counts']}"
    )
    if report.get("gate") == "fail":
        print("\n❌ Liveness REGRESSED vs baseline (new dead pathways):")
        for cat, (now, base) in report["regressed"].items():
            print(f"  - {cat}: {base} → {now}")
        print(
            "\nWire the new code into a live path, type the seam, or — if intentional "
            "and reviewed — `python scripts/check_liveness.py --update-baseline`."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
