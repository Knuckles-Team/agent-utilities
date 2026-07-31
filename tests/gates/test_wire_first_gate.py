"""Meta-test: the Wire-First gate (``scripts/check_wiring.py``'s D-OB-9/13/16
sweeps) trips on synthetic fixtures shaped like the real defects it exists to
catch, and stays clean on wired-up equivalents. "A gate that can't fail is
not a gate" (see ``test_swallowed_errors_gate.py``, the template for this
file).

Imported directly (not via subprocess, unlike most ``scripts/check_*.py``
meta-tests) because the checks under test take fixture ``Path`` overrides
(``src_dir``/``tests_dir``/``display_root``) rather than a CLI ``--root``
flag — ``scripts/`` is not a package, so ``importlib`` loads it by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_wiring.py"


def _load_check_wiring():
    spec = importlib.util.spec_from_file_location("_check_wiring_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_wiring = _load_check_wiring()


# ---------------------------------------------------------------------------
# D-OB-13a — uncollected test files
# ---------------------------------------------------------------------------


def test_orphan_gate_trips_on_a_loose_test_file_outside_testpaths(tmp_path):
    """A ``test_*.py`` sitting directly under a synthetic ``tests/`` root
    (not ``tests/unit``/``tests/integration``/``tests/retrieval``, and not
    named in any pre-commit/CI pytest invocation) must be flagged — this is
    exactly the ``tests/test_multiplexer_transports.py`` shape D-OB-13
    found.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_orphan.py").write_text("def test_x():\n    assert True\n")

    orphans = check_wiring.find_orphaned_test_files(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert "tests/test_orphan.py" in orphans


def test_orphan_gate_does_not_flag_a_file_under_tests_unit(tmp_path):
    """A test file under ``tests/unit`` (a real ``testpaths`` entry) is
    never flagged — the gate must not cry wolf on ordinary, collected
    tests."""
    tests_dir = tmp_path / "tests"
    (tests_dir / "unit").mkdir(parents=True)
    (tests_dir / "unit" / "test_ok.py").write_text("def test_x():\n    assert True\n")

    orphans = check_wiring.find_orphaned_test_files(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert "tests/unit/test_ok.py" not in orphans


# ---------------------------------------------------------------------------
# D-OB-13b — MagicMock(spec=[]) / patch(create=True) mock hygiene
# ---------------------------------------------------------------------------


def test_mock_hygiene_gate_trips_on_spec_empty_list(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_bad_mock.py").write_text(
        "from unittest.mock import MagicMock\n\n"
        "def test_x():\n"
        "    fake = MagicMock(spec=[])\n"
    )
    issues = check_wiring.find_mock_hygiene_issues(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert any(shape == "spec=[]" for _rel, _line, shape in issues)


def test_mock_hygiene_gate_ignores_a_docstring_mentioning_create_true(tmp_path):
    """A docstring merely mentioning ``create=True`` (explaining why the
    file does NOT use that shape, as the real ``test_graph_iter.py`` does)
    must never be flagged — this is AST ``ast.Call`` matching, not a raw
    line-regex, specifically to avoid that false positive."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_docstring_only.py").write_text(
        '"""Explains why we do NOT use patch(..., create=True) here."""\n\n'
        "def test_x():\n"
        "    assert True\n"
    )
    issues = check_wiring.find_mock_hygiene_issues(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert issues == []


def test_mock_hygiene_gate_trips_on_patch_create_true(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_bad_patch.py").write_text(
        "from unittest.mock import patch\n\n"
        "def test_x():\n"
        "    with patch('some.module.Thing', create=True):\n"
        "        pass\n"
    )
    issues = check_wiring.find_mock_hygiene_issues(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert any(shape == "create=True" for _rel, _line, shape in issues)


# ---------------------------------------------------------------------------
# D-OB-16 — silently-swallowed optional-extra import guards
# ---------------------------------------------------------------------------


def test_extras_gating_gate_trips_on_silent_import_error_pass(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_silent_skip.py").write_text(
        "def test_x():\n"
        "    try:\n"
        "        import some_optional_extra\n"
        "    except ImportError:\n"
        "        pass\n"
    )
    guards = check_wiring.find_silent_import_guards(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert guards != []


def test_extras_gating_gate_does_not_flag_a_visible_pytest_skip(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_visible_skip.py").write_text(
        "import pytest\n\n"
        "def test_x():\n"
        "    try:\n"
        "        import some_optional_extra\n"
        "    except ImportError:\n"
        "        pytest.skip('some_optional_extra not installed')\n"
    )
    guards = check_wiring.find_silent_import_guards(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert guards == []


# ---------------------------------------------------------------------------
# D-OB-9 — public symbol with no non-test caller
# ---------------------------------------------------------------------------


def test_symbol_gate_trips_on_a_class_referenced_only_from_tests(tmp_path):
    """The exact D-OB-9 shape: a class fully built, fully unit-tested, and
    never constructed anywhere else in ``agent_utilities/``."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "orphan_module.py").write_text(
        "class NeverCalledPolicy:\n"
        "    def decide_something_distinctive(self):\n"
        "        return True\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_orphan_module.py").write_text(
        "from agent_utilities.orphan_module import NeverCalledPolicy\n\n"
        "def test_x():\n"
        "    assert NeverCalledPolicy().decide_something_distinctive()\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "NeverCalledPolicy" in symbols


def test_symbol_gate_does_not_flag_a_class_with_a_live_caller(tmp_path):
    """The same class, but with a second module in agent_utilities/ that
    actually constructs it — the wired equivalent — must NOT be flagged."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "wired_module.py").write_text(
        "class LiveCalledPolicy:\n"
        "    def decide_something_distinctive(self):\n"
        "        return True\n"
    )
    (src_dir / "live_caller.py").write_text(
        "from agent_utilities.wired_module import LiveCalledPolicy\n\n"
        "def run():\n"
        "    return LiveCalledPolicy().decide_something_distinctive()\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wired_module.py").write_text(
        "from agent_utilities.wired_module import LiveCalledPolicy\n\n"
        "def test_x():\n"
        "    assert LiveCalledPolicy().decide_something_distinctive()\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "LiveCalledPolicy" not in symbols


def test_symbol_gate_ignores_a_comment_only_mention(tmp_path):
    """A symbol name that appears ONLY inside a comment in another file
    (never actually constructed/called) must still be flagged — a raw
    line-regex over source text would have missed this by counting the
    comment as a "reference"; the real ``AdmissionPolicy`` instance hid
    behind exactly this shape (mentioned in a comment in
    ``engine_tasks.py``, never constructed) until this gate's tokenize-based
    scan was written to see past comments."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "orphan_module2.py").write_text(
        "class StillNeverCalledPolicy:\n"
        "    def decide_something_else_distinctive(self):\n"
        "        return True\n"
    )
    (src_dir / "mentions_only.py").write_text(
        "# TODO: wire in StillNeverCalledPolicy() here eventually\n"
        "def run():\n"
        "    return None\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_orphan_module2.py").write_text(
        "from agent_utilities.orphan_module2 import StillNeverCalledPolicy\n\n"
        "def test_x():\n"
        "    assert StillNeverCalledPolicy().decide_something_else_distinctive()\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "StillNeverCalledPolicy" in symbols


# ---------------------------------------------------------------------------
# Regression lock — the real repo's ratchet must stay green
# ---------------------------------------------------------------------------


def test_gate_report_passes_against_the_frozen_repo_baseline():
    """The combined report must exit 0 against the real repo as long as
    nothing NEW beyond the frozen baseline was introduced — the regression
    lock proving the ratchet mechanics work end-to-end, not just on
    synthetic fixtures (mirrors ``test_swallowed_errors_gate.py``'s
    equivalent test)."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--wire-first-report"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
