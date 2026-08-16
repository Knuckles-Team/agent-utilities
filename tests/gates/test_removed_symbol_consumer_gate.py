"""Meta-tests for ``scripts/check_removed_symbol_consumers.py``: prove it trips on
a broken fixture (a genuinely-consumed public symbol removed/renamed, a whole
module renamed/removed, a function's signature gaining a required parameter,
a public class losing a public method), stays quiet on a private symbol
removal or a signature-preserving body refactor, and fails CLOSED — not open
— when its consumer index is missing or stale. A gate that can't fail is not
a gate; per this program's own finding, a gate that silently no-ops when its
data is absent is the exact anti-pattern to guard against here.

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
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
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
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "test_pkg.foo.bar" in res.stdout


def test_ignores_private_symbol_removal(tmp_path):
    repo = _init_repo(
        tmp_path, "def _bar():\n    return 1\n\n\ndef public_ok():\n    return 2\n"
    )
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
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "0 removed" in res.stdout


def test_passes_when_removed_symbol_has_no_consumers(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    (repo / "test_pkg" / "foo.py").write_text("# bar deleted\n")
    idx = tmp_path / "index.json"
    _write_index(idx, {})  # nobody imports it
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_fails_closed_when_index_missing(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(tmp_path / "does-not-exist.json"),
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
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
        "--max-index-age-days",
        "30",
    )
    assert res.returncode == 1
    assert "stale" in (res.stdout + res.stderr)


def test_fails_closed_when_index_malformed_json(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    idx = tmp_path / "index.json"
    idx.write_text("{not valid json")
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1


def test_fails_when_base_ref_unresolvable(tmp_path):
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    idx = tmp_path / "index.json"
    _write_index(idx, {})
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "origin/does-not-exist",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1
    assert "does not resolve" in (res.stdout + res.stderr)


def test_trips_on_module_rename_bare_import(tmp_path):
    """Check 4: a whole module renamed away, imported BARE (``import
    test_pkg.http``, no trailing symbol) — the shape Check 1 structurally
    cannot see (it only ever diffs ``module.symbol`` pairs), and the shape
    that actually broke ``kafka-mcp``/``portainer-agent`` in production when
    ``agent_utilities.http`` became ``agent_utilities.httpsupport``.
    """
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    (repo / "test_pkg" / "http.py").write_text("def get():\n    return 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add http module")
    (repo / "test_pkg" / "http.py").unlink()
    (repo / "test_pkg" / "httpsupport.py").write_text("def get():\n    return 1\n")
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {"test_pkg.http": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "MODULE REMOVED/RENAMED" in res.stdout
    assert "test_pkg.http" in res.stdout
    assert "consumer-repo" in res.stdout


def test_trips_on_new_required_parameter(tmp_path):
    """Check 3: a brand-new REQUIRED parameter on a still-present, still-
    consumed public function breaks every existing caller, purely via static
    signature diffing — no runtime behavior involved."""
    repo = _init_repo(tmp_path, "def bar(x=1):\n    return x\n")
    (repo / "test_pkg" / "foo.py").write_text("def bar(x=1, *, y):\n    return x\n")
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {"test_pkg.foo.bar": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "SIGNATURE/CONTRACT CHANGED" in res.stdout
    assert "test_pkg.foo.bar" in res.stdout
    assert "new required parameter 'y'" in res.stdout


def test_ignores_signature_unchanged_body_refactor(tmp_path):
    """True negative for Check 3: the function's PARAMETER LIST is untouched;
    only its body changed. No signature-change violation."""
    repo = _init_repo(tmp_path, "def bar(x=1):\n    return x + 1\n")
    (repo / "test_pkg" / "foo.py").write_text(
        "def bar(x=1):\n    y = x\n    return y + 1\n"
    )
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {"test_pkg.foo.bar": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "0 function signature change" in res.stdout


def test_trips_on_removed_public_class_method(tmp_path):
    """Check 5: a public class present at both refs loses a public method —
    flagged whenever the class itself has a recorded fleet consumer, since
    the index records imports, not per-method usage."""
    repo = _init_repo(
        tmp_path,
        "class Client:\n    def connect(self):\n        return 1\n"
        "    def close(self):\n        return 1\n",
    )
    (repo / "test_pkg" / "foo.py").write_text(
        "class Client:\n    def connect(self):\n        return 1\n"
    )
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {"test_pkg.foo.Client": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "LOST PUBLIC METHOD" in res.stdout
    assert "test_pkg.foo.Client" in res.stdout
    assert "method 'close'" in res.stdout


def test_reexported_signature_change_traced_through_facade(tmp_path):
    """Check 3 must resolve through a facade too, not just the defining
    module — the SAME transitive-re-export need Check 1 already has, and the
    dominant real-world shape (most fleet consumers import the package-root
    facade, never the true defining submodule)."""
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    (repo / "test_pkg" / "_impl.py").write_text("def bar(x=1):\n    return x\n")
    (repo / "test_pkg" / "facade.py").write_text("from test_pkg._impl import bar\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add facade + impl")
    (repo / "test_pkg" / "_impl.py").write_text("def bar(x=1, *, y):\n    return x\n")
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {"test_pkg.facade.bar": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "test_pkg.facade.bar" in res.stdout


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
        {"test_pkg.facade.bar": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "test_pkg.facade.bar" in res.stdout


def test_signature_change_traced_through_getattr_dispatch(tmp_path):
    """agent_utilities/__init__.py's OWN dominant re-export shape: a
    module-level ``def __getattr__(name):`` (PEP 562) that lazy-imports on
    ``elif name == "X":``/``elif name in [...]:``, not a plain top-level
    ``from x import y``. Most real fleet consumers import through exactly
    this kind of root-package facade (``agent_utilities.initialize_workspace``,
    never ``agent_utilities.core.workspace.initialize_workspace``), so a
    signature check that only sees top-level imports would have almost no
    real coverage on this codebase's own shape.
    """
    repo = _init_repo(tmp_path, "def bar():\n    return 1\n")
    (repo / "test_pkg" / "_impl.py").write_text("def bar(x=1):\n    return x\n")
    (repo / "test_pkg" / "__init__.py").write_text(
        "def __getattr__(name):\n"
        "    if name == 'bar':\n"
        "        from test_pkg._impl import bar\n"
        "        return bar\n"
        "    raise AttributeError(name)\n"
        "\n"
        "__all__ = ['bar']\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add __getattr__ facade + impl")
    (repo / "test_pkg" / "_impl.py").write_text("def bar(x=1, *, y):\n    return x\n")
    idx = tmp_path / "index.json"
    _write_index(
        idx,
        {"test_pkg.bar": [{"repo": "consumer-repo", "file": "m.py", "line": 1}]},
    )
    res = _run(
        "--tree",
        str(repo),
        "--package",
        "test_pkg",
        "--base-ref",
        "HEAD",
        "--consumer-index",
        str(idx),
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "test_pkg.bar" in res.stdout
    assert "new required parameter 'y'" in res.stdout
