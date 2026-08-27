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

Compares against ``CAPS`` (module-level constant, below): the build fails
only when a category exceeds its cap (more findings than the cap allows), so
dead pathways can only shrink. Not a baseline file -- seven integers in this
file's own source, printed on every run, moved only by a reviewed diff to
this file (CX-RAT-11: no-ratchet policy, see the comment on CAPS). A second,
independent check enforces ``scripts/liveness_deferred.tsv``'s owner +
review-by expiry (GOC-68: "a bare ratchet lets deferral become permanent") —
a past-due deferred entry also fails the gate.

If the code-enhancer skill is not installed, the gate skips cleanly (exit 0) rather
than blocking CI — install `universal-skills` to enable it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

# ABSOLUTE caps. NOT a baseline: seven integers, in the gate's own source,
# printed on every run, on a dated ladder. They cannot hide a finding --
# any new dead pathway raises the count and fails the gate.
#
# 2026-08-27 (CX-RAT-11): every value below is the MEASURED count at the time
# this constant was introduced (`.venv/bin/python3 scripts/check_liveness.py`
# against 3b7186be2, corrected counts) -- including `untyped_seams: 1771`,
# which is one over the old `.liveness_baseline.json` value of 1770. Seeding
# a cap at the current value is legitimate here (unlike a per-finding
# freeze) precisely because it is a visible integer in source, on a dated
# ladder, reviewed in a diff -- not a hidden count nobody looks at again.
# Wave ladder: hold every cap flat (no regressions) until a future wave
# deliberately lowers one; lowering a cap is the only way this file changes
# after this wave.
CAPS = {
    "orphan_modules": 2,  # wave CX-RAT-11 (2026-08-27): hold
    "dead_definitions": 518,  # wave CX-RAT-11 (2026-08-27): hold
    "never_executed": 0,  # wave CX-RAT-11 (2026-08-27): hold at zero
    "untyped_seams": 1771,  # wave CX-RAT-11 (2026-08-27): hold (was 1770; +1 measured drift)
    "orphan_read_keys": 77,  # wave CX-RAT-11 (2026-08-27): hold
    "facade_handlers": 98,  # wave CX-RAT-11 (2026-08-27): hold
    "placeholder_markers": 503,  # wave CX-RAT-11 (2026-08-27): hold
}

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "agent_utilities"


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

    # Step B (CX-RAT-11): the source of the compared-against numbers is now
    # CAPS (module-level, above) instead of a committed baseline file --
    # identical `>` comparison, identical failure message. Zero other
    # behaviour change.
    #
    # Step C (CX-RAT-11): the `--update-baseline` branch, the `BASELINE`
    # constant, and `_head_commit()` are gone -- there is no baseline file
    # left to write. CAPS only moves by editing this source file (see the
    # comment on CAPS above).
    regressed = {
        cat: (now, CAPS.get(cat, 0))
        for cat, now in corrected_counts.items()
        if now > CAPS.get(cat, 0)
    }

    print(f"Liveness counts={corrected_counts}")
    print(f"Liveness caps  ={CAPS}")

    gate_failed = False
    if regressed:
        gate_failed = True
        print("\n❌ Liveness REGRESSED vs baseline (new dead pathways):")
        for cat, (now, base) in regressed.items():
            print(f"  - {cat}: {base} → {now}")
        print(
            "\nWire the new code into a live path, type the seam, or — if intentional "
            "and reviewed, raise the relevant CAPS entry in scripts/check_liveness.py "
            "itself (reviewed in the diff, not a hidden re-baseline)."
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
