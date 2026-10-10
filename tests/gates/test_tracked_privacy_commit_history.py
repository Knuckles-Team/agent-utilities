"""AU-QUAL-R006: the tracked-artifact privacy gate also scans commit history.

``check_tracked_privacy.py`` previously only classified tracked file
*content*; a credential, private path, or internal identifier typed straight
into a commit message was invisible to it. ``_commit_message_violations``
closes that gap by running the same ``classify_runtime_source_line``
classifier over every not-yet-public commit message.

Scope is deliberately bounded to commits reachable from ``HEAD`` but not
from ``origin/main`` (``_commit_message_range``): a commit already reachable
from the public remote was already disclosed, so re-flagging it on every
later commit would be a permanent, unfixable gate failure rather than a
leak-prevention signal. These tests build a disposable local repository and
move a fake ``refs/remotes/origin/main`` to prove that boundary directly,
rather than relying on this repository's own (large, shared) history.

The planted "leak" is built at runtime via string concatenation, never
written as one matchable literal -- the same convention
``tests/gates/test_tracked_privacy_gate.py`` uses -- so this test file does
not itself become a tracked-source finding under the very gate it tests.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _gate_module() -> ModuleType:
    source = Path(__file__).parents[2] / "scripts" / "check_tracked_privacy.py"
    spec = importlib.util.spec_from_file_location(
        "check_tracked_privacy_commit_history", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_ENV},
    )


def _commit(repo: Path, message: str, filename: str) -> None:
    (repo / filename).write_text("content\n", encoding="utf-8")
    _run(repo, "add", filename)
    _run(repo, "commit", "--no-gpg-sign", "-q", "-m", message)


def _head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _credential_uri_leak() -> str:
    """A credential-bearing URI, assembled so no literal match exists here."""
    return "postgres://" + "admin:pw@db" + "/x"


@pytest.mark.spec("AU-QUAL-R006")
def test_leaky_not_yet_public_commit_message_is_flagged(tmp_path: Path) -> None:
    gate = _gate_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    leak = _credential_uri_leak()
    _commit(repo, f"fix: rotate the leaked ref ({leak})", "file.txt")

    violations = gate._commit_message_violations(repo, {}, frozenset())

    assert any("credential-bearing URI" in v.category for v in violations)


def test_clean_commit_history_has_no_violations(tmp_path: Path) -> None:
    gate = _gate_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    _commit(repo, "fix: an ordinary, unremarkable change", "file.txt")

    assert gate._commit_message_violations(repo, {}, frozenset()) == []


@pytest.mark.spec("AU-QUAL-R006")
def test_already_public_commit_is_not_reflagged(tmp_path: Path) -> None:
    """A commit reachable from ``origin/main`` was already disclosed; scanning
    must stop re-flagging it the moment the public remote catches up, or the
    gate would be a permanent, unfixable failure instead of a leak gate."""
    gate = _gate_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    _commit(repo, "fix: base commit already on main", "base.txt")
    base_sha = _head(repo)
    _run(repo, "update-ref", "refs/remotes/origin/main", base_sha)

    leak = _credential_uri_leak()
    _commit(repo, f"fix: not yet public ({leak})", "feature.txt")
    before_publish = gate._commit_message_violations(repo, {}, frozenset())
    assert any("credential-bearing URI" in v.category for v in before_publish)

    # origin/main catches up to the leaky commit -- it is now public.
    _run(repo, "update-ref", "refs/remotes/origin/main", _head(repo))
    after_publish = gate._commit_message_violations(repo, {}, frozenset())
    assert after_publish == []
