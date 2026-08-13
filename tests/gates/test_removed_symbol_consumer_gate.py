"""Meta-tests for ``scripts/check_removed_symbol_consumers.py``: prove it trips on
a broken fixture (a genuinely-consumed public symbol removed/renamed), stays
quiet on a private symbol removal, and fails CLOSED — not open — when its
consumer index is missing or stale. A gate that can't fail is not a gate; per
this program's own finding, a gate that silently no-ops when its data is
absent is the exact anti-pattern to guard against here.

Fully self-contained: builds a throwaway git repo + package under ``tmp_path``
and a hand-written consumer index, so it needs neither the real
``agent_utilities`` package nor the real fleet checkout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
GATE = SCRIPTS / "check_removed_symbol_consumers.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        capture_output=True,
        text=True,
    )


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def _init_repo(tmp_path: Path, foo_body: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "test_pkg").mkdir(parents=True)
    (repo / "test_pkg" / "foo.py").write_text(foo_body)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "test-pkg"\nversion = "0"\ndependencies = []\n'
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "gate-meta-test@example.invalid")
    _git(repo, "config", "user.name", "gate-meta-test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _write_index(path: Path, consumers: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "consumers": consumers,
            }
        )
    )


def test_trips_on_removed_consumed_symbol(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    (repo / "test_pkg" / "foo.py").write_text("# bar deleted\n")
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {"test_pkg.foo.bar": [{"repo": "consumer-repo", "file": "m.py", "line": 3}]},
    )
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "HEAD", "--consumer-index", str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "test_pkg.foo.bar" in res.stdout
    assert "consumer-repo" in res.stdout
    assert "m.py:3" in res.stdout


def test_trips_on_renamed_symbol(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    (repo / "test_pkg" / "foo.py").write_text("def baz():\n    return 1\n")
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {"test_pkg.foo.bar": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "HEAD", "--consumer-index", str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "test_pkg.foo.bar" in res.stdout


def test_ignores_private_symbol_removal(tmp_path):
    repo = _init_repo(tmp_path, "def _bar():\n    return 1\n\n\ndef public_ok():\n    return 2\n")
    (repo / "test_pkg" / "foo.py").write_text("def public_ok():\n    return 2\n")
    idx = tmp_path / "index.json"
    # Even if (mistakenly) a consumer entry existed for the private name, the
    # gate must never even CONSIDER it removed — private symbols are never
    # part of the tracked public surface, base or head.
    _write_index(
        idx,
        {"test_pkg.foo._bar": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "HEAD", "--consumer-index", str(idx),
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "0 removed" in res.stdout


def test_passes_when_removed_symbol_has_no_consumers(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    (repo / "test_pkg" / "foo.py").write_text("# bar deleted\n")
    idx = tmp_path / "index.json"
    _write_index(idx, {})  # nobody imports it
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "HEAD", "--consumer-index", str(idx),
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_fails_closed_when_index_missing(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "HEAD", "--consumer-index", str(tmp_path / "does-not-exist.json"),
    )
    assert res.returncode == 1
    assert "not found" in (res.stdout + res.stderr)


def test_fails_closed_when_index_stale(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    idx = tmp_path / "index.json"
    idx.write_text(
        json.dumps({"generated_at": "2000-01-01T00:00:00+00:00", "consumers": {}})
    )
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "HEAD", "--consumer-index", str(idx),
        "--max-index-age-days", "30",
    )
    assert res.returncode == 1
    assert "stale" in (res.stdout + res.stderr)


def test_fails_closed_when_index_malformed_json(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    idx = tmp_path / "index.json"
    idx.write_text("{not valid json")
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "HEAD", "--consumer-index", str(idx),
    )
    assert res.returncode == 1


def test_fails_when_base_ref_unresolvable(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    idx = tmp_path / "index.json"
    _write_index(idx, {})
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "origin/does-not-exist", "--consumer-index", str(idx),
    )
    assert res.returncode == 1
    assert "does not resolve" in (res.stdout + res.stderr)


def test_reexported_symbol_traced_through_facade(tmp_path):
    """The ``setting``-style case: a facade module re-exports a name defined
    elsewhere; deleting the UPSTREAM definition must still be caught under the
    facade's dotted path, because that's what real consumers actually import.
    """
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    (repo / "test_pkg" / "_impl.py").write_text("def bar():\n    return 1\n")
    (repo / "test_pkg" / "facade.py").write_text("from test_pkg._impl import bar\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add facade + impl")
    # Now delete the UPSTREAM definition; the facade's import line is untouched.
    (repo / "test_pkg" / "_impl.py").write_text("# bar deleted upstream\n")
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {
            "test_pkg.facade.bar": [
                {"repo": "consumer-repo", "file": "m.py", "line": 1}
            ]
        },
    )
    res = _run(
        "--tree", str(repo), "--package", "test_pkg",
        "--base-ref", "HEAD", "--consumer-index", str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "test_pkg.facade.bar" in res.stdout
