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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the underlying git behaviour this whole item is about, on its
    own, with no dependency on this repo's real index: ``-C`` changes the
    working directory but NOT which repository ``GIT_INDEX_FILE``/``GIT_DIR``
    name. This is what made a fixture's ``git -C tmp_path add -A`` write into
    the REAL repo's index during a real ``git commit`` (D-LGI-1) -- the exact
    mechanism, reproduced here in two disposable throwaway repos so it needs
    no assumption about (and takes no risk against) the real repository.

    Deliberately bypasses the runtime guard added alongside it (see
    ``_guarded_popen_init``) by restoring the true, unpatched
    ``Popen.__init__`` for this test only -- this test's whole purpose is to
    reproduce the raw, unprotected vulnerability, and the guard would
    otherwise (correctly) refuse to let it. ``monkeypatch`` reverts the
    restore at teardown, same as the guard's own negative-control tests.
    """
    conftest = _conftest_module()
    monkeypatch.setattr(subprocess.Popen, "__init__", conftest._TRUE_POPEN_INIT)

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


# ---------------------------------------------------------------------------
# GOC-71 follow-up: GIT_AUTHOR_*/GIT_COMMITTER_*/GIT_CONFIG* -- the quieter,
# more damaging sibling of the GIT_DIR redirect above. This repo's own
# history carries 295 commits authored ``universal-ingestion-proof
# <proof@test.local>`` (the identity ``tests/integration/knowledge_graph/
# test_git_markdown_domain_packs_live_engine.py`` sets via `git -C tmp_path
# config user.name/email`) -- the SAME GIT_DIR redirect mechanism, but
# writing into the REAL repo's persistent local `.git/config` instead of a
# transient env var, so it kept mis-authoring commits for weeks after the
# triggering test run ended, until this audit found it. ``core.bare``
# corruption breaks loudly; this one broke silently.
# ---------------------------------------------------------------------------


def test_dangerous_git_env_prefixes_are_absent_for_the_whole_session() -> None:
    """The chokepoint strips GIT_AUTHOR_*/GIT_COMMITTER_*/GIT_CONFIG* too."""
    conftest = _conftest_module()
    for key in os.environ:
        assert not key.startswith(conftest._DANGEROUS_GIT_ENV_PREFIXES), key


def test_a_leaked_git_dir_redirects_a_config_call_into_the_real_repos_local_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces the *config-corruption* shape of the leak, on disposable
    stand-ins only: ``git -C fixture config user.name X`` under a leaked
    GIT_DIR does not just fail to touch ``fixture`` -- it silently writes
    ``user.name``/``user.email`` into the *real* (here: decoy stand-in)
    repo's persistent local config, exactly as it did to this repo's actual
    ``.git/config`` history. Unlike the env-var leak, this survives past the
    poisoned subprocess -- every later commit in the real repo made without
    explicit ``GIT_AUTHOR_*`` inherits the wrong identity until someone
    notices and fixes the local config by hand.

    Deliberately bypasses the runtime guard (see the module docstring on the
    test above) -- this test also reproduces the raw, unprotected behaviour.
    """
    conftest = _conftest_module()
    monkeypatch.setattr(subprocess.Popen, "__init__", conftest._TRUE_POPEN_INIT)

    decoy = tmp_path / "decoy-real-repo"
    fixture = tmp_path / "fixture-repo"
    decoy.mkdir()
    fixture.mkdir()

    base_env = dict(os.environ)
    for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        base_env.pop(name, None)

    _git(["init", "-q"], decoy, base_env)
    _git(["config", "user.email", "legit@example.invalid"], decoy, base_env)
    _git(["config", "user.name", "legit-owner"], decoy, base_env)
    _git(["init", "-q"], fixture, base_env)

    poisoned_env = dict(base_env)
    poisoned_env["GIT_DIR"] = str(decoy / ".git")

    # The bug: `git -C fixture config user.name/email ...` under a leaked
    # GIT_DIR writes into decoy's config, not fixture's.
    subprocess.run(
        ["git", "-C", str(fixture), "config", "user.name", "universal-ingestion-proof"],
        env=poisoned_env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(fixture), "config", "user.email", "proof@test.local"],
        env=poisoned_env,
        check=True,
        capture_output=True,
        text=True,
    )

    corrupted_name = _git(
        ["config", "--local", "--get", "user.name"], decoy, base_env
    ).stdout.strip()
    assert corrupted_name == "universal-ingestion-proof", (
        "expected the poisoned env to overwrite the decoy repo's identity -- "
        "if this fails, the underlying git behaviour this item is about no "
        "longer reproduces the way it was observed in this repo's own history"
    )


def test_the_runtime_guard_refuses_a_git_subprocess_with_a_leaked_pointer_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backstop proof: the session-wide strip already prevents the ambient
    leak (proved above), but a test can still *re-introduce* one of these
    vars mid-run (e.g. ``monkeypatch.setenv``, exactly how the real leak
    enters a hook process). ``_guarded_popen_init`` is independent of the
    strip and catches that case too -- refusing the corrupting subprocess
    outright rather than letting it silently redirect, entirely against
    disposable ``tmp_path`` stand-ins.
    """
    conftest = _conftest_module()

    real_repo = tmp_path / "real"
    throwaway_repo = tmp_path / "throwaway"
    real_repo.mkdir()
    throwaway_repo.mkdir()
    base_env = dict(os.environ)
    for name in conftest._DANGEROUS_GIT_ENV_VARS:
        base_env.pop(name, None)
    _git(["init", "-q"], real_repo, base_env)
    _git(["init", "-q"], throwaway_repo, base_env)

    monkeypatch.setenv("GIT_DIR", str(real_repo / ".git"))

    with pytest.raises(conftest.LeakedGitPointerEnvError):
        subprocess.run(
            ["git", "-C", str(throwaway_repo), "config", "core.bare", "true"],
            check=True,
        )

    monkeypatch.delenv("GIT_DIR", raising=False)
    corrupted = _git(
        ["config", "--get", "core.bare"], real_repo, base_env
    ).stdout.strip()
    assert corrupted in ("", "false"), (
        "GUARD FAILED: the leaked GIT_DIR corrupted the real stand-in repo"
    )


def test_without_the_guard_the_same_leak_genuinely_corrupts_the_real_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control for the guard test above: temporarily restores the
    true, unpatched ``Popen.__init__`` (captured at conftest import time) to
    prove the guard stops a real vulnerability, not a strawman.
    """
    conftest = _conftest_module()

    real_repo = tmp_path / "real"
    throwaway_repo = tmp_path / "throwaway"
    real_repo.mkdir()
    throwaway_repo.mkdir()
    base_env = dict(os.environ)
    for name in conftest._DANGEROUS_GIT_ENV_VARS:
        base_env.pop(name, None)
    _git(["init", "-q"], real_repo, base_env)
    _git(["init", "-q"], throwaway_repo, base_env)

    monkeypatch.setattr(subprocess.Popen, "__init__", conftest._TRUE_POPEN_INIT)
    monkeypatch.setenv("GIT_DIR", str(real_repo / ".git"))

    subprocess.run(
        ["git", "-C", str(throwaway_repo), "config", "core.bare", "true"],
        check=True,
    )

    monkeypatch.delenv("GIT_DIR", raising=False)
    corrupted = _git(
        ["config", "--get", "core.bare"], real_repo, base_env
    ).stdout.strip()
    assert corrupted == "true", (
        "expected the unguarded leak to hit the real stand-in repo -- if "
        "this fails, the reproduction is stale and the guard test above "
        "proves nothing"
    )
