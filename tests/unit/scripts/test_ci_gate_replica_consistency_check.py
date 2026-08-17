"""Meta-tests: prove scripts/ci_gate_replica.py's anti-drift guards actually fire.

The whole point of parsing every workflow file under .github/workflows/ at run
time (see that script's module docstring) instead of hand-copying step lists
is that drift can never be silently invisible — --consistency-check must fail
loudly instead. These tests prove that claim rather than merely asserting it
in a comment, for all three classes of drift this script now guards against:

  1. a NEW JOB appearing in a known, registered workflow file
  2. a NEW WORKFLOW FILE appearing under .github/workflows/ with no registry entry
  3. a .cargo/config.toml change that adds an external-binary build dependency
     (rustc-wrapper/linker/runner) no workflow file ever installs — the exact
     shape of the incident that shipped epistemic-graph's commit 652f91c and
     broke every GitHub CI job (sccache not on the runner's PATH; cargo
     hard-errors rather than falling back). agent-utilities has no Rust build
     of its own today, so this class is proven against a synthetic
     cargo-invoking workflow text rather than a real one.

It also proves the previously-uncovered advisory.yml is now actually replayed
(GAP 1: its ~40-check `advisory` job was invisible to this script before), and
that the GAP 3 build-affecting file-set predicate used by `--skip-safe` is
correct for this repo's actual (Python packaging) build inputs.
"""

from __future__ import annotations

import importlib.util
import shutil
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


def test_real_workflows_pass_consistency_check():
    m = _load_module()
    assert m.consistency_check(verbose=False) is True


def test_injected_new_job_in_release_yml_fails_consistency_check():
    m = _load_module()
    release_doc = m.load_workflow(m.WORKFLOWS_DIR / "release.yml")
    release_doc["jobs"]["totally-new-release-job"] = {
        "runs-on": "ubuntu-latest",
        "steps": [{"name": "do a new blocking thing", "run": "echo hi"}],
    }
    ok = m.consistency_check(verbose=False, workflow_docs={"release.yml": release_doc})
    assert ok is False


def test_removing_a_configured_job_fails_consistency_check():
    """The opposite drift direction: a spec naming a job that no longer
    exists in the workflow (renamed or deleted) must also fail, not
    silently pass with less coverage than the config claims."""
    m = _load_module()
    release_doc = m.load_workflow(m.WORKFLOWS_DIR / "release.yml")
    assert "gates" in m.WORKFLOW_REGISTRY["release.yml"].executable_jobs
    del release_doc["jobs"]["gates"]
    ok = m.consistency_check(verbose=False, workflow_docs={"release.yml": release_doc})
    assert ok is False


def test_unregistered_workflow_file_fails_consistency_check(tmp_path):
    """GAP 1's core anti-drift claim: a brand new workflow file (advisory.yml
    was exactly this, once) must fail the check until it is registered, not
    be silently invisible the way advisory.yml was before this fix."""
    m = _load_module()
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    for fname in m.WORKFLOW_REGISTRY:
        shutil.copy(m.WORKFLOWS_DIR / fname, workflows_dir / fname)
    (workflows_dir / "newly-added.yml").write_text(
        "name: New\non:\n  push: {}\njobs:\n  x:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
    )
    ok = m.consistency_check(verbose=False, workflows_dir=workflows_dir)
    assert ok is False


def test_registered_workflow_missing_from_disk_fails_consistency_check(tmp_path):
    """Opposite direction: WORKFLOW_REGISTRY names a file no longer present."""
    m = _load_module()
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    shutil.copy(m.WORKFLOWS_DIR / "release.yml", workflows_dir / "release.yml")
    # advisory.yml deliberately not copied here.
    ok = m.consistency_check(verbose=False, workflows_dir=workflows_dir)
    assert ok is False


def test_gates_and_build_run_steps_cover_the_known_release_blockers():
    """Regression guard: the RUN plan for `gates`/`build` must include the
    exact release-blocking steps release.yml documents as release-critical
    (secret-history scan, OSV audit, the eg-PyPI-resolvable cross-repo gate,
    the test suite, and the byte-identical-wheel reproducibility check)."""
    m = _load_module()
    doc = m.load_workflow(m.WORKFLOWS_DIR / "release.yml")
    plan, _, _ = m.build_plan_for_workflow(m.WORKFLOW_REGISTRY["release.yml"], doc)
    run_names = {p["name"] for p in plan if p["mode"] == "RUN"}
    for expected in (
        "Secret-history scan (D-CIP-13)",
        "Dependency vulnerability audit (OSV)",
        "Verify required epistemic-graph release is resolvable on PyPI",
        "Test suite",
        "Require byte-identical wheels and stage the sole candidate",
    ):
        assert expected in run_names, f"missing from parsed RUN plan: {expected!r}"


def test_numeric_runtime_gate_is_matrix_expanded_and_explicitly_skip_reasoned():
    """Known case from the task brief: numeric-runtime-gate installs the
    wheel with its declared epistemic-graph[full] from PyPI (currently
    unpublishable) — it must show up as an explicit, reasoned skip for BOTH
    matrix legs (ubuntu-latest, windows-latest), not be absent from the plan
    entirely or collapsed into one un-expanded row."""
    m = _load_module()
    doc = m.load_workflow(m.WORKFLOWS_DIR / "release.yml")
    plan, _, _ = m.build_plan_for_workflow(m.WORKFLOW_REGISTRY["release.yml"], doc)
    numeric_steps = [p for p in plan if p["job"].startswith("numeric-runtime-gate")]
    assert numeric_steps, "numeric-runtime-gate steps missing from the plan entirely"
    legs = {p["job"] for p in numeric_steps}
    assert legs == {
        "numeric-runtime-gate#ubuntu-latest",
        "numeric-runtime-gate#windows-latest",
    }, legs
    assert all(p["mode"] == "SKIP_LOUD" for p in numeric_steps)
    assert all("PyPI" in p["detail"] for p in numeric_steps)


def test_advisory_job_is_now_covered():
    """GAP 1's headline example for this repo: advisory.yml's ~40-check
    `advisory` job used to be entirely unreplicated."""
    m = _load_module()
    doc = m.load_workflow(m.WORKFLOWS_DIR / "advisory.yml")
    plan, _, _ = m.build_plan_for_workflow(m.WORKFLOW_REGISTRY["advisory.yml"], doc)
    advisory_run_names = {
        p["name"] for p in plan if p["job"] == "advisory" and p["mode"] == "RUN"
    }
    for expected in (
        "No-stub gate",
        "CycloneDX SBOM and license policy",
        "Dependency vulnerability audit (OSV)",
    ):
        assert expected in advisory_run_names, (
            f"missing from parsed advisory RUN plan: {expected!r}"
        )
    # advisory.yml never blocks the pre-push gate.
    assert m.WORKFLOW_REGISTRY["advisory.yml"].blocking is False
    assert m.WORKFLOW_REGISTRY["release.yml"].blocking is True


def test_windows_and_pages_jobs_are_reported_not_silently_dropped():
    m = _load_module()
    doc = m.load_workflow(m.WORKFLOWS_DIR / "advisory.yml")
    plan, _, _ = m.build_plan_for_workflow(m.WORKFLOW_REGISTRY["advisory.yml"], doc)
    for job_id in ("windows", "pages"):
        rows = [p for p in plan if p["job"] == job_id]
        assert rows, f"{job_id} job produced zero plan rows -- silently dropped"
        assert all(p["mode"] == "SKIP_LOUD" for p in rows)


# ─────────────────────────────────────────────────────────────────────────
# GAP 2 — external build-tool dependency check. This repo has no
# .cargo/config.toml and no workflow step invokes cargo, so the proof here
# is against a SYNTHETIC cargo-invoking workflow text (proving the shared
# mechanism fires the same way it does for epistemic-graph's real
# .cargo/config.toml, not that this repo currently needs it).
# ─────────────────────────────────────────────────────────────────────────


def test_build_tool_dependency_check_is_a_noop_with_no_cargo_config():
    m = _load_module()
    assert m.find_required_build_binaries(m.CARGO_CONFIG_PATH) == []
    problems = m.check_build_tool_dependencies(
        m.CARGO_CONFIG_PATH,
        {"release.yml": (m.WORKFLOWS_DIR / "release.yml").read_text(encoding="utf-8")},
    )
    assert problems == []


def test_build_tool_dependency_check_fires_on_a_synthetic_missing_wrapper(tmp_path):
    """Known-bad proof: a rustc-wrapper the (synthetic, cargo-invoking)
    workflow never installs must fail and name the binary by name, without
    any hard-coded string match on 'sccache' in the checker itself."""
    m = _load_module()
    bad_config = tmp_path / "config.toml"
    bad_config.write_text('[build]\nrustc-wrapper = "sccache"\n')
    problems = m.check_build_tool_dependencies(
        bad_config,
        {
            "release.yml": "jobs:\n  gates:\n    steps:\n      - run: cargo build --release\n"
        },
    )
    assert any("sccache" in p for p in problems), (
        f"expected an sccache finding, got: {problems}"
    )


def test_build_tool_dependency_check_passes_when_binary_is_actually_installed():
    m = _load_module()
    bad_config_text = '[build]\nrustc-wrapper = "totally-fake-wrapper-xyz"\n'
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(bad_config_text)
        fake_path = Path(f.name)
    try:
        workflow_with_install = (
            "jobs:\n  gates:\n    steps:\n"
            "      - run: apt-get install -y totally-fake-wrapper-xyz\n"
            "      - run: cargo test --workspace\n"
        )
        problems = m.check_build_tool_dependencies(
            fake_path, {"release.yml": workflow_with_install}
        )
        assert problems == []
    finally:
        fake_path.unlink()


# ─────────────────────────────────────────────────────────────────────────
# GAP 3 — the authoritative build-affecting file-set predicate for THIS
# repo's actual (Python packaging) build inputs.
# ─────────────────────────────────────────────────────────────────────────


def test_build_affecting_predicate_matches_expected_patterns():
    from ci_gate_replica import is_build_affecting

    for path in (
        "pyproject.toml",
        "uv.lock",
        "requirements.txt",
        "requirements-dev.txt",
        "build_backend.py",
        "MANIFEST.in",
        ".github/workflows/release.yml",
        ".pre-commit-config.yaml",
    ):
        assert is_build_affecting(path) is True, (
            f"expected {path!r} to be build-affecting"
        )


def test_build_affecting_predicate_excludes_unrelated_paths():
    from ci_gate_replica import is_build_affecting

    for path in (
        "docs/architecture/overview.md",
        "README.md",
        "agent_utilities/some_module.py",
        "tests/unit/test_something.py",
    ):
        assert is_build_affecting(path) is False, (
            f"expected {path!r} to NOT be build-affecting"
        )
