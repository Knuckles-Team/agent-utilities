"""Regression tests for tracked-source candidate enumeration."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "security" / "check_cypher_write_subset.py"


def _gate():
    spec = importlib.util.spec_from_file_location("check_cypher_write_subset", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_candidate_files_use_tracked_working_tree_and_skip_generated_copies(
    tmp_path: Path,
) -> None:
    gate = _gate()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")

    tracked = repo / "src" / "tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "src/tracked.py")

    bad_query = 'backend.execute("MATCH (a) MATCH (b) MERGE (a)-[:REL]->(b)")\n'
    # The scanner must inspect the working-tree contents, not only the index.
    tracked.write_text(bad_query, encoding="utf-8")

    for relative in (
        Path(".venv") / "lib" / "generated.py",
        Path("build") / "lib" / "generated.py",
    ):
        generated = repo / relative
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(bad_query, encoding="utf-8")

    assert gate._candidate_files(repo) == [tracked]
    violations = gate.scan(repo)
    assert [(violation.path, violation.shape) for violation in violations] == [
        ("src/tracked.py", "multiple_match_clauses")
    ]


def test_candidate_enumeration_fails_closed_when_git_cannot_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()

    def _failed_git(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=128, stdout="", stderr="not a repository"
        )

    monkeypatch.setattr(gate.subprocess, "run", _failed_git)

    with pytest.raises(gate.CypherWriteSubsetGateError, match="git grep"):
        gate._candidate_files(tmp_path)
