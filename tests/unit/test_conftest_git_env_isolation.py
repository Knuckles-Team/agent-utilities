"""D-LGI-1: a real ``git commit`` exports ``GIT_DIR``/``GIT_INDEX_FILE`` into
the hooks it runs (including ``guardrail-gate-meta-tests``, which runs
``pytest tests/gates``), and ``git -C <other-dir>`` does **not** override
them -- the inherited env vars still win over path-based repository
discovery. Any test that shells out to ``git -C <tmp_path> ...`` without its
own explicit ``env=`` therefore silently mutates the REAL repository's index
instead of its own isolated fixture repo. Confirmed live: ``tests/gates/
test_docs_contract_gate.py::test_privacy_gate_scans_unchanged_runtime_source_not_only_the_diff``'s
``git -C tmp_path add -A`` replaced this repo's entire tracked-file index
with its own single fixture file (``docker/build-job.yaml``).

``tests/conftest.py``'s ``_strip_inherited_git_repository_env()`` is the
session-wide chokepoint fix: it clears the dangerous ``GIT_*`` vars from
``os.environ`` once, before any test collects, so every
``subprocess.run(["git", ...])`` call that does not pass its own ``env=``
inherits a clean environment for the rest of the session.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _conftest_module():
    conftest_path = str(Path(__file__).resolve().parents[1] / "conftest.py")
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) == conftest_path:
            return module
    pytest.skip("root conftest not loaded")


def test_dangerous_git_env_vars_are_absent_for_the_whole_session() -> None:
    """The chokepoint already ran at collection time -- this proves it held."""
    conftest = _conftest_module()
    for name in conftest._DANGEROUS_GIT_ENV_VARS:
        assert name not in os.environ, (
            f"{name} is set mid-session -- something re-introduced it after "
            "tests/conftest.py's own startup stripping"
        )


def test_strip_inherited_git_repository_env_clears_every_named_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct unit proof of the stripping function itself."""
    conftest = _conftest_module()
    for name in conftest._DANGEROUS_GIT_ENV_VARS:
        monkeypatch.setenv(name, "/somewhere/decoy")

    conftest._strip_inherited_git_repository_env()

    for name in conftest._DANGEROUS_GIT_ENV_VARS:
        assert name not in os.environ


def _git(
    args: list[str], cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inherited_git_index_file_redirects_a_dash_c_add_to_the_wrong_repo(
    tmp_path: Path,
) -> None:
    """Proves the underlying git behaviour this whole item is about, on its
    own, with no dependency on this repo's real index: ``-C`` changes the
    working directory but NOT which repository ``GIT_INDEX_FILE``/``GIT_DIR``
    name. This is what made a fixture's ``git -C tmp_path add -A`` write into
    the REAL repo's index during a real ``git commit`` (D-LGI-1) -- the exact
    mechanism, reproduced here in two disposable throwaway repos so it needs
    no assumption about (and takes no risk against) the real repository.
    """
    decoy = tmp_path / "decoy-real-repo"
    fixture = tmp_path / "fixture-repo"
    decoy.mkdir()
    fixture.mkdir()

    base_env = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
    ):
        base_env.pop(name, None)

    _git(["init", "-q"], decoy, base_env)
    _git(
        [
            "-c",
            "user.email=d@test",
            "-c",
            "user.name=d",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "base",
        ],
        decoy,
        base_env,
    )
    (decoy / "real-tracked-file.txt").write_text("real content\n", encoding="utf-8")
    _git(["add", "-A"], decoy, base_env)
    _git(
        [
            "-c",
            "user.email=d@test",
            "-c",
            "user.name=d",
            "commit",
            "-q",
            "-m",
            "add real file",
        ],
        decoy,
        base_env,
    )
    tracked_before = _git(["ls-files"], decoy, base_env).stdout.split()
    assert tracked_before == ["real-tracked-file.txt"]

    _git(["init", "-q"], fixture, base_env)
    (fixture / "fixture-file.txt").write_text("fixture content\n", encoding="utf-8")

    # The bug: an environment carrying GIT_DIR/GIT_INDEX_FILE for `decoy`,
    # with a `git -C fixture add -A` that does NOT pass its own `env=`.
    poisoned_env = dict(base_env)
    poisoned_env["GIT_DIR"] = str(decoy / ".git")
    poisoned_env["GIT_INDEX_FILE"] = str(decoy / ".git" / "index")

    subprocess.run(
        ["git", "-C", str(fixture), "add", "-A"],
        env=poisoned_env,
        check=True,
        capture_output=True,
        text=True,
    )

    # The corruption: the fixture's OWN file now shows up staged in the
    # `decoy` repo -- content that repo never had and never asked for,
    # because the `-C fixture add -A` call actually wrote into `decoy`'s
    # index (named by the inherited GIT_DIR/GIT_INDEX_FILE), not fixture's
    # own. (The exact residual state of decoy's pre-existing entries depends
    # on git's work-tree-boundary inference and is not asserted here --
    # what matters, and what is common to every real occurrence observed,
    # is that content from the WRONG repository lands in the index.)
    staged_in_decoy = _git(
        ["diff", "--cached", "--name-only"], decoy, base_env
    ).stdout.split()
    assert "fixture-file.txt" in staged_in_decoy, (
        "expected the poisoned env to redirect the `-C fixture` add into the "
        "decoy repo's index -- if this fails, the underlying git behaviour "
        "this item is about no longer reproduces the way it was observed"
    )


def test_strip_inherited_git_repository_env_prevents_the_redirect(
    tmp_path: Path,
) -> None:
    """The same reproduction as above, but with the fix applied first --
    proven against the restored bug by the previous test, which demonstrates
    the identical setup DOES corrupt the decoy repo without this call."""
    conftest = _conftest_module()

    decoy = tmp_path / "decoy-real-repo"
    fixture = tmp_path / "fixture-repo"
    decoy.mkdir()
    fixture.mkdir()

    base_env = dict(os.environ)
    for name in conftest._DANGEROUS_GIT_ENV_VARS:
        base_env.pop(name, None)

    _git(["init", "-q"], decoy, base_env)
    _git(
        [
            "-c",
            "user.email=d@test",
            "-c",
            "user.name=d",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "base",
        ],
        decoy,
        base_env,
    )
    (decoy / "real-tracked-file.txt").write_text("real content\n", encoding="utf-8")
    _git(["add", "-A"], decoy, base_env)
    _git(
        [
            "-c",
            "user.email=d@test",
            "-c",
            "user.name=d",
            "commit",
            "-q",
            "-m",
            "add real file",
        ],
        decoy,
        base_env,
    )

    _git(["init", "-q"], fixture, base_env)
    (fixture / "fixture-file.txt").write_text("fixture content\n", encoding="utf-8")

    # Apply the fix to THIS process's real os.environ the way a real
    # subprocess call (no explicit env=) would pick it up: mutate a process
    # environment that started out poisoned -- exactly like a hook process
    # inherits from a real `git commit` -- then run the fix's own stripping
    # function, then let a plain `git -C fixture add -A` (no explicit env=)
    # inherit whatever os.environ looks like afterward.
    saved = {
        name: os.environ.get(name)
        for name in ("GIT_DIR", "GIT_INDEX_FILE", *conftest._DANGEROUS_GIT_ENV_VARS)
    }
    try:
        os.environ["GIT_DIR"] = str(decoy / ".git")
        os.environ["GIT_INDEX_FILE"] = str(decoy / ".git" / "index")

        conftest._strip_inherited_git_repository_env()

        subprocess.run(
            ["git", "-C", str(fixture), "add", "-A"],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    staged_in_decoy = _git(
        ["diff", "--cached", "--name-only"], decoy, base_env
    ).stdout.split()
    assert staged_in_decoy == [], (
        "the fix must prevent the redirect into the decoy repo"
    )

    staged_in_fixture = _git(
        ["diff", "--cached", "--name-only"], fixture, base_env
    ).stdout.split()
    assert staged_in_fixture == ["fixture-file.txt"], (
        "the add must land in the fixture repo it was actually targeting"
    )
