"""Meta-test: prove scripts/ci_gate_replica.py's anti-drift guard actually fires.

The whole point of parsing .github/workflows/release.yml at run time (see that
script's module docstring) instead of hand-copying its steps is that a NEW job
release.yml adds can never be silently skipped — --consistency-check must fail
loudly instead. This test proves that claim rather than merely asserting it in
a comment: it takes the real, live release.yml, injects one new top-level job
the script has never seen, and checks that consistency_check() rejects it (and
that the ORIGINAL, unmodified file still passes).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci_gate_replica.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ci_gate_replica", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_release_yaml_passes_consistency_check():
    m = _load_module()
    doc = m.load_workflow(m.DEFAULT_WORKFLOW_PATH)
    assert m.consistency_check(doc, verbose=False) is True


def test_injected_new_job_fails_consistency_check():
    m = _load_module()
    doc = m.load_workflow(m.DEFAULT_WORKFLOW_PATH)
    doc["jobs"]["totally-new-release-job"] = {
        "runs-on": "ubuntu-latest",
        "steps": [{"name": "do a new blocking thing", "run": "echo hi"}],
    }
    assert m.consistency_check(doc, verbose=False) is False


def test_removing_a_configured_job_fails_consistency_check():
    """The opposite drift direction: EXECUTABLE_JOBS/JOB_SKIP_REASONS naming a
    job that no longer exists in release.yml (renamed or deleted) must also
    fail, not silently pass with less coverage than the config claims."""
    m = _load_module()
    doc = m.load_workflow(m.DEFAULT_WORKFLOW_PATH)
    assert "gates" in m.EXECUTABLE_JOBS
    del doc["jobs"]["gates"]
    assert m.consistency_check(doc, verbose=False) is False


def test_gates_and_build_run_steps_cover_the_known_release_blockers():
    """Regression guard: the RUN plan for `gates`/`build` must include the
    exact release-blocking steps release.yml documents as release-critical
    (secret-history scan, OSV audit, the eg-PyPI-resolvable cross-repo gate,
    the test suite, and the byte-identical-wheel reproducibility check)."""
    m = _load_module()
    doc = m.load_workflow(m.DEFAULT_WORKFLOW_PATH)
    plan, _, _ = m.build_plan(doc)
    run_names = {p["name"] for p in plan if p["mode"] == "RUN"}
    for expected in (
        "Secret-history scan (D-CIP-13)",
        "Dependency vulnerability audit (OSV)",
        "Verify required epistemic-graph release is resolvable on PyPI",
        "Test suite",
        "Require byte-identical wheels and stage the sole candidate",
    ):
        assert expected in run_names, f"missing from parsed RUN plan: {expected!r}"


def test_numeric_runtime_gate_is_explicitly_skip_reasoned_not_silently_dropped():
    """Known case from the task brief: numeric-runtime-gate installs the
    wheel with its declared epistemic-graph[full] from PyPI (currently
    unpublishable) — it must show up as an explicit, reasoned skip, not be
    absent from the plan entirely."""
    m = _load_module()
    doc = m.load_workflow(m.DEFAULT_WORKFLOW_PATH)
    plan, _, _ = m.build_plan(doc)
    numeric_steps = [p for p in plan if p["job"] == "numeric-runtime-gate"]
    assert numeric_steps, "numeric-runtime-gate steps missing from the plan entirely"
    assert all(p["mode"] == "SKIP_LOUD" for p in numeric_steps)
    assert all("PyPI" in p["detail"] for p in numeric_steps)
