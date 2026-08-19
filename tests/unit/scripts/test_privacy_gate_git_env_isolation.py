"""NE-059: prove ``scripts/check_tracked_privacy.py`` is immune to inherited
``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE`` repository pointers -- the defect
``tests/unit/scripts/test_adopt_b_tracked_privacy_git_env_isolation.py``
originally documented and this track fixed.

``tests/conftest.py`` scrubs ``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE`` (and the rest of
``_DANGEROUS_GIT_ENV_VARS``) from ``os.environ`` at import time, and its
``_guarded_popen_init`` backstop additionally raises ``LeakedGitPointerEnvError``
for any ``subprocess.Popen`` call that still carries one of the *pointer*
vars and looks like a ``git`` invocation. Both protections are scoped to
calls made **inside this pytest process**. A real ``git commit``/``git push``
runs each ``language: system`` pre-commit hook (this gate included) as its
own, separate OS process, with those vars exported into ITS environment --
outside pytest, outside conftest, with nothing to intercept a leak.

So the tests below spawn ``scripts/check_tracked_privacy.py``'s scanning
logic in a genuine **child process** (``subprocess.run([sys.executable, ...],
env=...)``), with an explicit, hand-built ``env`` dict rather than
``monkeypatch.setenv`` on the live ``os.environ``. That subprocess.run call
itself is not intercepted by conftest's guard either: the guard only matches
invocations that look like ``git ...``, and this one spawns Python. This is
deliberate -- it reproduces the real "hook subprocess with leaked repository
pointers" shape with nothing standing between the poisoned env and the git
calls check_tracked_privacy.py's ``scan()`` makes internally, exactly the
scenario that is invisible when exercised in-process under pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE = _REPO_ROOT / "scripts" / "check_tracked_privacy.py"

# A machine-specific host identifier the gate's own
# `_MACHINE_HOST_ID_RE` (`(?:rw?|host)[0-9]{3,}`) reliably flags regardless
# of which account/host actually runs the test -- unlike the "local account
# or host identifier" category, which depends on the runtime environment's
# own username/hostname and would make this test flaky across machines.
_CANARY_LINE = "canary line: host12345 must not leak\n"
_CANARY_CATEGORY = "machine-specific host identifier"

_RUNNER_SOURCE = """
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
module_path = Path(sys.argv[2]).resolve()

spec = importlib.util.spec_from_file_location(
    "check_tracked_privacy_raw_subprocess", module_path
)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

violations = mod.scan(repo_root)
payload = {
    "tracked": sorted(
        p.relative_to(repo_root).as_posix() for p in mod._tracked_artifacts(repo_root)
    ),
    "runtime": sorted(
        p.relative_to(repo_root).as_posix()
        for p in mod._runtime_source_artifacts(repo_root)
    ),
    "violations": sorted(v.render() for v in violations),
}
print(json.dumps(payload, sort_keys=True))
"""


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo(root: Path, *, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "ne059@test.local", cwd=root)
    _git("config", "user.name", "ne059", cwd=root)
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "seed", cwd=root)
    return root


def _clean_env() -> dict[str, str]:
    """A base environment with no ambient repository-pointer vars at all --
    the positive control, independent of whatever is (or is not) set in the
    ambient environment this test session happens to run under."""
    env = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
    ):
        env.pop(name, None)
    return env


def _poisoned_env(decoy_git_dir: Path) -> dict[str, str]:
    env = _clean_env()
    env["GIT_DIR"] = str(decoy_git_dir)
    env["GIT_INDEX_FILE"] = str(decoy_git_dir / "index")
    env["GIT_WORK_TREE"] = str(decoy_git_dir.parent)
    return env


def _run_gate_subprocess(
    *, target_root: Path, runner: Path, env: dict[str, str]
) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(runner), str(target_root), str(_MODULE)],
        cwd=target_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"raw-subprocess gate run failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    return json.loads(result.stdout)


def test_poisoned_git_env_immunity_matches_clean_run(tmp_path: Path) -> None:
    """The immunity proof the track was assigned to build: run the gate's
    scan (via ``_git_file_names`` under ``scan()``/``_tracked_artifacts()``/
    ``_runtime_source_artifacts()``) as a raw subprocess twice against the
    SAME target repo -- once with a clean environment, once with
    all three repository pointers pointed at an unrelated decoy repository --
    and assert the tracked-file inventory and the resulting violations
    (the gate's verdict) are byte-for-byte identical either way.

    This must fail if the ``strip_inherited_git_repository_env()``/
    ``sanitized_git_env()`` fix in ``scripts/check_tracked_privacy.py`` is
    reverted -- confirmed by hand against the pre-fix module content (see the
    track's report), not re-verified automatically here.
    """
    runner = tmp_path / "runner.py"
    runner.write_text(_RUNNER_SOURCE, encoding="utf-8")

    target = _make_repo(tmp_path / "target-repo", files={"NOTES.md": _CANARY_LINE})
    decoy = _make_repo(tmp_path / "decoy-repo", files={"unrelated.md": "# nothing\n"})

    clean_result = _run_gate_subprocess(
        target_root=target, runner=runner, env=_clean_env()
    )
    poisoned_result = _run_gate_subprocess(
        target_root=target,
        runner=runner,
        env=_poisoned_env(decoy / ".git"),
    )

    assert clean_result == poisoned_result
    # And it must be the REAL inventory, not both sides trivially empty.
    assert clean_result["tracked"] == ["NOTES.md"]
    assert clean_result["violations"] == [f"NOTES.md:1: {_CANARY_CATEGORY}"]


def test_decoy_exclude_hidden_canary_is_not_missed_under_poisoned_git_env(
    tmp_path: Path,
) -> None:
    """The substantive failure mode this gate exists to prevent: a decoy
    repository -- reached only because the repository pointers leaked
    into the gate's subprocess -- whose own ``.git/info/exclude`` names the
    real tracked file in the repo under scan. Before the fix this caused
    ``_git_file_names`` to resolve against the decoy's index instead of the
    target's, so the real file (and the privacy-relevant canary it carries)
    silently vanished from the scan -- a confident wrong PASS, the worst
    outcome for a security gate. After the fix, the poisoned env never
    reaches the git subprocess at all, so the canary is found regardless.
    """
    runner = tmp_path / "runner.py"
    runner.write_text(_RUNNER_SOURCE, encoding="utf-8")

    target = _make_repo(tmp_path / "target-repo", files={"NOTES.md": _CANARY_LINE})
    decoy = _make_repo(tmp_path / "decoy-repo", files={"unrelated.md": "# nothing\n"})
    # The decoy's own exclude rules name the target's real tracked file --
    # the "hides a file" half of the known-bad input.
    (decoy / ".git" / "info" / "exclude").write_text("NOTES.md\n", encoding="utf-8")

    poisoned_result = _run_gate_subprocess(
        target_root=target,
        runner=runner,
        env=_poisoned_env(decoy / ".git"),
    )

    assert poisoned_result["tracked"] == ["NOTES.md"]
    assert poisoned_result["violations"] == [f"NOTES.md:1: {_CANARY_CATEGORY}"]


def test_gate_still_catches_a_planted_canary_with_no_poisoning_at_all(
    tmp_path: Path,
) -> None:
    """Guard against the fix silently weakening detection: a plain, unpoisoned
    run must still flag a genuine privacy-relevant line. A gate that has
    never rejected anything is not known to work."""
    runner = tmp_path / "runner.py"
    runner.write_text(_RUNNER_SOURCE, encoding="utf-8")

    target = _make_repo(tmp_path / "target-repo", files={"NOTES.md": _CANARY_LINE})

    result = _run_gate_subprocess(target_root=target, runner=runner, env=_clean_env())

    assert result["tracked"] == ["NOTES.md"]
    assert result["violations"] == [f"NOTES.md:1: {_CANARY_CATEGORY}"]
