#!/usr/bin/env python3
"""Liveness / dead-pathway ratchet gate (CONCEPT:AU-KG.maintenance.periodic-code-health consumer).

Wire-First's static `check_wiring.py` catches modules with no import path. This
gate adds the two layers it can't: the **typed-seam / contract-drift** scan (public
functions returning an untyped ``dict``/``list[dict]`` — the seam where a producer
writes ``_score`` and a consumer reads ``score`` and silently gets ``0.00``) and,
when a coverage report is provided, the **never-executed** layer.

It does NOT duplicate the raw detector — that lives once in the code-enhancer
skill (`universal_skills/.../code-enhancer/scripts/analyze_liveness.py`). This
gate locates it, runs it over ``agent_utilities/``, and RECONCILES its raw
``orphan_modules``/``dead_definitions`` findings (``scripts/liveness_reconciler.py``)
before ratcheting — the vendored analyzer has a confirmed resolution bug (it
compares a module path computed relative to ``agent_utilities/`` against import
statements captured with the full ``agent_utilities.`` prefix, so this repo's
dominant absolute-import style almost never matches) plus a blind spot for
string-dispatched registries (entry-points, ``getattr`` registries,
``__getattr__`` lazy-import maps). See
``plans/graph-os-completion-program/designs/DEAD-CODE-INTENT-RECOVERY.md`` for
the full audit. The reconciler reuses ``check_wiring.py``'s already-correct
import-graph resolver rather than reimplementing one (`one capability, one
entrypoint`).

Other categories (``never_executed``, ``untyped_seams``, ``orphan_read_keys``,
``facade_handlers``, ``placeholder_markers``) pass through from the raw
analyzer unchanged — this reconciliation is scoped to the two categories with
a confirmed, located defect.

Ratchets against a committed baseline (`.liveness_baseline.json`, now storing
CORRECTED counts): the build fails only when a category REGRESSES (more
findings than the baseline), so dead pathways can only shrink. A second,
independent check enforces ``scripts/liveness_deferred.tsv``'s owner +
review-by expiry (GOC-68: "a bare ratchet lets deferral become permanent") —
a past-due deferred entry also fails the gate.

If the code-enhancer skill is not installed, the gate skips cleanly (exit 0) rather
than blocking CI — install `universal-skills` to enable it.

Run ``--update-baseline`` to re-baseline after an intentional, reviewed change.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "agent_utilities"
BASELINE = REPO / ".liveness_baseline.json"


def _load_sibling(name: str):
    """Load a ``scripts/<name>.py`` sibling module by path — ``scripts`` ships
    an ``__init__.py`` (making it a package) but this gate must work
    regardless of how it is invoked (bare ``python3 scripts/check_liveness.py``
    has no package context), same idiom already proven by
    ``tests/gates/test_wire_first_gate.py``."""
    spec = importlib.util.spec_from_file_location(
        f"_{name}_for_check_liveness", REPO / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec — `liveness_deferred.py`'s
    # `@dataclass` decorator introspects `sys.modules[cls.__module__]` at
    # class-definition time; skipping this step makes that lookup return
    # None and crash (a real failure this task's own dogfooding caught).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


liveness_reconciler = _load_sibling("liveness_reconciler")
liveness_deferred = _load_sibling("liveness_deferred")


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
    # No `--baseline` passed to the analyzer subprocess: its own gate/regressed
    # computation runs against the RAW (buggy) orphan_modules/dead_definitions
    # counts. This gate reconciles those two categories itself (below) and
    # computes the pass/fail decision against the CORRECTED counts instead —
    # the analyzer is used purely as a finding SOURCE now, not as the judge.

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
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
    recon = liveness_reconciler.reconcile(report.get("details", {}))
    om, dd = recon["orphan_modules"], recon["dead_definitions"]

    # Corrected counts: orphan_modules/dead_definitions replaced by the
    # reconciled "still genuinely unexplained" counts; every other category
    # passes through from the raw analyzer unchanged (out of this gate's
    # reconciliation scope).
    corrected_counts = dict(report["counts"])
    corrected_counts["orphan_modules"] = len(om["still"])
    corrected_counts["dead_definitions"] = len(dd["still"])

    raw_orphans = len(report["details"].get("orphan_modules", []))
    raw_dead = len(report["details"].get("dead_definitions", []))
    print(
        "Liveness reconciliation (raw analyzer -> corrected, "
        "scripts/liveness_reconciler.py):"
    )
    print(
        f"  orphan_modules:    raw={raw_orphans:4d}  corrected={corrected_counts['orphan_modules']:4d}"
        f"  (rescued={len(om['rescued'])}, excluded_generated={len(om['excluded_generated'])})"
    )
    print(
        f"  dead_definitions:  raw={raw_dead:4d}  corrected={corrected_counts['dead_definitions']:4d}"
        f"  (rescued={len(dd['rescued'])}, excluded_generated={len(dd['excluded_generated'])})"
    )
    if recon["dynamic_unresolvable_sites"]:
        print(
            f"  {len(recon['dynamic_unresolvable_sites'])} genuinely dynamic "
            "import call site(s) found (non-literal importlib.import_module()/"
            "__import__() target) — any remaining finding in that package "
            "directory was treated fail-open (not dead), not guessed at."
        )

    if update:
        # Record WHAT the baseline was taken against, not just the numbers. A
        # bare counts blob cannot be told apart from a fresh one, which is how
        # this file silently drifted ~4 days behind `main` (D-KCI-11) while the
        # gate kept reporting "regressed" against the wrong reference. Extra
        # keys are inert to callers that read only ``counts``.
        BASELINE.write_text(
            json.dumps(
                {
                    "counts": corrected_counts,
                    "taken_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "taken_against": _head_commit(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"liveness baseline updated (corrected counts): {corrected_counts}")
        return 0

    base_counts: dict[str, int] = {}
    if BASELINE.exists():
        base_counts = json.loads(BASELINE.read_text()).get("counts", {})
    regressed = {
        cat: (now, base_counts.get(cat, 0))
        for cat, now in corrected_counts.items()
        if now > base_counts.get(cat, 0)
    }

    print(f"Liveness counts={corrected_counts}")

    gate_failed = False
    if regressed:
        gate_failed = True
        print("\n❌ Liveness REGRESSED vs baseline (new dead pathways):")
        for cat, (now, base) in regressed.items():
            print(f"  - {cat}: {base} → {now}")
        print(
            "\nWire the new code into a live path, type the seam, or — if intentional "
            "and reviewed — `python scripts/check_liveness.py --update-baseline`."
        )

    # Ratchet expiry (GOC-68: "a bare ratchet lets deferral become permanent").
    # Independent of the count-regression check above: a deferred finding that
    # is well-formed and still within its review-by window never fails this;
    # a malformed entry, or one past its review-by date, always does.
    try:
        deferred_entries = liveness_deferred.load_entries()
    except ValueError as exc:
        gate_failed = True
        print(f"\n❌ liveness_deferred.tsv is unparsable: {exc}")
        deferred_entries = []

    malformed = [e for e in deferred_entries if not liveness_deferred.is_well_formed(e)]
    if malformed:
        gate_failed = True
        print(
            "\n❌ scripts/liveness_deferred.tsv entries missing owner + "
            "(review-by OR PERMANENT reason):"
        )
        for e in malformed:
            print(f"  - line {e.line_no}: {e.category}\t{e.pattern}")

    stale = liveness_deferred.stale_entries(deferred_entries, date.today())
    if stale:
        gate_failed = True
        print(
            "\n❌ scripts/liveness_deferred.tsv entries past their review-by date "
            "(bind, extend with a new reason, or mark PERMANENT with a justification):"
        )
        for e in stale:
            print(
                f"  - {e.category}\t{e.pattern} (owner={e.owner}, review-by={e.review_by})"
            )

    return 1 if gate_failed else 0


if __name__ == "__main__":
    sys.exit(main())
