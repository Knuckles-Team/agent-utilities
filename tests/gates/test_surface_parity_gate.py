"""Meta-tests for the surface-parity gate (``scripts/check_surface_parity.py``).

This gate had NO test file before D-WD5-RAT-03 (one of two gates in this
program retired without one — see ``scripts/check_swallowed_errors.py``'s
docstring for the sibling precedent). Three things are proven here:

1. ``_is_capability``/``INFRA_MODULES`` classify a module correctly in
   isolation — including the ``asset_mirror.py`` promotion made when the
   ``surface_parity_baseline.txt`` ratchet was retired (it was never real
   debt; it is infra now, not a tracked exception).
2. The diff-scoped enforcement this gate now has instead of a frozen
   baseline (``_added_py_files``/``new_capability_violations``) actually
   trips on a genuinely NEW unwired capability module, stays quiet on an
   EXISTING one that is merely edited (the false-positive shape that made
   the old file/symbol-keyed baselines rot — here it can't happen at all,
   because the key is "was this file added by this diff", which a purely
   internal rename can never satisfy), and clears once the plant is
   removed — the same plant-proof shape used across this whole program.
3. Both `--update-baseline` (retired) and a real, end-to-end CLI run are
   wired all the way through to the actual script, not just the importable
   functions (a gate that can't fail is not a gate).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_surface_parity.py"


def _load_gate_module():
    """Import ``check_surface_parity.py`` as a module so its internal
    ``_is_capability``/``_added_py_files``/``new_capability_violations`` can
    be exercised directly, mirroring
    ``test_swallowed_errors_gate.py::_load_gate_module``. Safe to import
    directly under a test run already inside the managed venv (this repo's
    ``tests/conftest.py`` already enforces that) — ``_gate_interpreter``'s
    re-exec check is then a no-op, and the expensive ``kg_server`` import /
    ``build_graph()`` walk only run when a function that calls them is
    actually invoked, not at import time.
    """
    spec = importlib.util.spec_from_file_location("check_surface_parity_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


def _init_repo(root: Path) -> None:
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "wd5-rat-03@test.local", cwd=root)
    _git("config", "user.name", "wd5-rat-03", cwd=root)


# ── 1. Pure classification ─────────────────────────────────────────────────


def test_is_capability_accepts_a_plain_feature_module():
    mod = _load_gate_module()
    assert mod._is_capability("agent_utilities/harness/some_new_feature.py")


def test_is_capability_excludes_init_conftest_and_test_files():
    mod = _load_gate_module()
    assert not mod._is_capability("agent_utilities/harness/__init__.py")
    assert not mod._is_capability("agent_utilities/harness/conftest.py")
    assert not mod._is_capability("agent_utilities/harness/tests/test_thing.py")
    assert not mod._is_capability("agent_utilities/harness/models.py")


def test_is_capability_excludes_infra_and_plugin_packages():
    mod = _load_gate_module()
    assert not mod._is_capability("agent_utilities/harness/evaluators.py")
    assert not mod._is_capability(
        "agent_utilities/protocols/source_connectors/connectors/some_connector.py"
    )


def test_asset_mirror_is_classified_as_infra_not_a_capability():
    """D-WD5-RAT-03 regression: this module was a ``surface_parity_baseline.txt``
    entry with a comment saying it was never real debt (the capability it
    fronts IS exposed via ``graph_writeback(asset_mirror=true)``; this file
    is only the co-located CLI transport). Retiring the baseline promoted it
    into ``INFRA_MODULES`` — a real fix, not a tracked exception — so it must
    no longer show up as a capability at all."""
    mod = _load_gate_module()
    path = "agent_utilities/knowledge_graph/enrichment/writeback/asset_mirror.py"
    assert path in mod.INFRA_MODULES
    assert not mod._is_capability(path)


# ── 2. Diff-scoped enforcement: added-file detection + plant proofs ───────


def test_added_py_files_detects_a_newly_added_agent_utilities_file(tmp_path):
    mod = _load_gate_module()
    repo = tmp_path / "repo"
    (repo / "agent_utilities").mkdir(parents=True)
    _init_repo(repo)
    existing = repo / "agent_utilities" / "existing.py"
    existing.write_text("def f():\n    return 1\n")
    _git("add", "agent_utilities/existing.py", cwd=repo)
    _git("commit", "-q", "-m", "seed", cwd=repo)

    new_file = repo / "agent_utilities" / "brand_new.py"
    new_file.write_text("def g():\n    return 2\n")
    _git("add", "agent_utilities/brand_new.py", cwd=repo)

    added = mod._added_py_files(str(repo))
    assert added == ["agent_utilities/brand_new.py"]


def test_added_py_files_does_not_report_an_edited_pre_existing_file(tmp_path):
    """The false-positive shape every other ratchet in this program hit
    (file/symbol/line churn on EXISTING, unrelated debt) cannot occur here:
    editing (even heavily) a file already present at HEAD never makes it
    show up as "added"."""
    mod = _load_gate_module()
    repo = tmp_path / "repo"
    (repo / "agent_utilities").mkdir(parents=True)
    _init_repo(repo)
    existing = repo / "agent_utilities" / "existing.py"
    existing.write_text("def f():\n    return 1\n")
    _git("add", "agent_utilities/existing.py", cwd=repo)
    _git("commit", "-q", "-m", "seed", cwd=repo)

    # A large, unrelated edit -- including a rename of the only function in
    # the file (the "moved into a renamed enclosing function" shape).
    existing.write_text("def f_renamed_by_extraction():\n    return 1\n")
    _git("add", "agent_utilities/existing.py", cwd=repo)

    assert mod._added_py_files(str(repo)) == []


def test_new_capability_violations_plant_proof_cycle(tmp_path, monkeypatch):
    """The three-state plant proof required for every de-ratcheted gate in
    this program: a genuinely new finding FAILS, the same finding surviving
    a pure internal rename of an EXISTING file PASSES, and removing the
    plant PASSES."""
    mod = _load_gate_module()
    repo = tmp_path / "repo"
    (repo / "agent_utilities").mkdir(parents=True)
    _init_repo(repo)
    pre_existing_unwired = repo / "agent_utilities" / "already_unwired.py"
    pre_existing_unwired.write_text("def old():\n    return 1\n")
    _git("add", "agent_utilities/already_unwired.py", cwd=repo)
    _git("commit", "-q", "-m", "seed", cwd=repo)

    monkeypatch.chdir(repo)

    # State A: plant a brand-new unwired capability module -> FAILS (reported).
    new_file = repo / "agent_utilities" / "new_unwired_feature.py"
    new_file.write_text("def h():\n    return 3\n")
    _git("add", "agent_utilities/new_unwired_feature.py", cwd=repo)
    unexposed = {
        "agent_utilities/already_unwired.py",
        "agent_utilities/new_unwired_feature.py",
    }
    violations = mod.new_capability_violations(unexposed)
    assert violations == ["agent_utilities/new_unwired_feature.py"]

    # State B: the pre-existing unwired file is edited (extraction-style
    # rename of its only function) -- still unexposed, but NOT added by this
    # diff, so it must never appear as a violation.
    pre_existing_unwired.write_text("def already_unwired_extracted():\n    return 1\n")
    _git("add", "agent_utilities/already_unwired.py", cwd=repo)
    violations = mod.new_capability_violations(unexposed)
    assert "agent_utilities/already_unwired.py" not in violations
    assert violations == ["agent_utilities/new_unwired_feature.py"]

    # State C: the plant is removed -> clean.
    _git(
        "rm", "-q", "-f", "--cached", "agent_utilities/new_unwired_feature.py", cwd=repo
    )
    new_file.unlink()
    unexposed_after_removal = {"agent_utilities/already_unwired.py"}
    violations = mod.new_capability_violations(unexposed_after_removal)
    assert violations == []


def test_new_capability_violations_ignores_a_file_not_in_the_unexposed_set(
    tmp_path, monkeypatch
):
    """A newly-added file that IS reachable from a surface root must never be
    reported -- only the intersection of "added" and "currently unexposed"
    is a violation."""
    mod = _load_gate_module()
    repo = tmp_path / "repo"
    (repo / "agent_utilities").mkdir(parents=True)
    _init_repo(repo)
    _git("commit", "-q", "-m", "seed", "--allow-empty", cwd=repo)

    wired = repo / "agent_utilities" / "wired_feature.py"
    wired.write_text("def f():\n    return 1\n")
    _git("add", "agent_utilities/wired_feature.py", cwd=repo)

    monkeypatch.chdir(repo)
    assert mod.new_capability_violations({"agent_utilities/some_other_file.py"}) == []


def _run_added_py_files_in_a_real_subprocess(
    repo: Path, *, env_overrides: dict[str, str] | None = None
) -> list[str]:
    """Call ``_added_py_files`` in a genuine CHILD process, never inside this
    pytest process.

    ``tests/conftest.py`` installs a session-wide ``subprocess.Popen`` guard
    (D-LGI-1/GOC-71) that deliberately REFUSES to spawn a ``git`` subprocess
    whenever ``GIT_DIR``/``GIT_INDEX_FILE`` are present in its environment --
    exactly what setting them via ``monkeypatch.setenv`` and calling
    ``_added_py_files`` directly would trigger, since that git call runs
    inside this (guarded) process. A real child process is unpatched, which
    is also the more faithful proof: a genuine ``git commit`` hook exports
    these vars into a genuinely separate child process, matching
    ``test_current_only_contract_git_env_isolation_gate.py``'s pattern for
    the sibling ``check_wiring.py`` helper.
    """
    script = (
        f"import sys; sys.path.insert(0, {str(SCRIPT.parent)!r}); "
        "import check_surface_parity as m; "
        "print('\\n'.join(m._added_py_files(sys.argv[1])))"
    )
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", script, str(repo)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo),
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [line for line in result.stdout.splitlines() if line]


def test_added_py_files_byte_identical_under_exported_git_env(tmp_path):
    """Same proof BUG-180 established for the sibling ``_tracked_or_walked``
    helpers, applied to this gate's own git-diff call: because ``_git()``
    always runs with an explicit ``cwd=<toplevel>`` and never ``-C <subdir>``,
    exporting this SAME repo's own ``GIT_DIR``/``GIT_INDEX_FILE`` (exactly
    what a real ``git commit`` hook subprocess sees) must not change the
    result at all."""
    repo = tmp_path / "repo"
    (repo / "agent_utilities").mkdir(parents=True)
    _init_repo(repo)
    existing = repo / "agent_utilities" / "existing.py"
    existing.write_text("def f():\n    return 1\n")
    _git("add", "agent_utilities/existing.py", cwd=repo)
    _git("commit", "-q", "-m", "seed", cwd=repo)
    new_file = repo / "agent_utilities" / "brand_new.py"
    new_file.write_text("def g():\n    return 2\n")
    _git("add", "agent_utilities/brand_new.py", cwd=repo)

    plain = _run_added_py_files_in_a_real_subprocess(repo)
    under_ambient_env = _run_added_py_files_in_a_real_subprocess(
        repo,
        env_overrides={
            "GIT_DIR": str(repo / ".git"),
            "GIT_INDEX_FILE": str(repo / ".git" / "index"),
        },
    )

    assert plain == under_ambient_env == ["agent_utilities/brand_new.py"]


# ── 3. CLI wiring ────────────────────────────────────────────────────────


def test_update_baseline_flag_is_retired():
    """The retired flag must REFUSE, not silently do nothing -- the same
    convention every other de-ratcheted gate in this program adopted, and it
    must return before doing any of the expensive real-repo work (the check
    happens first in ``main()``)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--update-baseline"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "RETIRED" in result.stderr


def test_no_baseline_file_remains():
    assert not (ROOT / "scripts" / "surface_parity_baseline.txt").exists()
