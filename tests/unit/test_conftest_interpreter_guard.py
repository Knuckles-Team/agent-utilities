"""BUG-026 (GOC-59 B3): ``tests/conftest.py``'s ``_fail_fast_on_wrong_interpreter()``
used to reject **any** ``sys.executable`` outside ``<repo>/.venv`` -- including the
intentionally-separate CI-parity lean venv that ``scripts/run_guardrails_lean.py``
deliberately builds under a throwaway ``tempfile.mkdtemp(prefix="guardrails-lean-")``
directory to reproduce CI's lean install. That made the check a **blind gate**: four
sub-gates (``Prod-profile guard``, ``Gate meta-tests``, ``Exact local promoter
behavioral contract``, ``CI bootstrap and supply-chain policy behavioral contract``)
failed on the interpreter check alone, before the gate's own assertions ever ran, so
``guardrails-lean-parity`` reported failure without measuring what it names.

Fixed in commit ``a14486d6`` ("fix(guardrails): unblind guardrails-lean-parity's
interpreter check (B3)") by having ``run_guardrails_lean.py::_gate_env()`` set the
escape hatch ``_fail_fast_on_wrong_interpreter()`` itself already defines:
``AGENT_UTILITIES_ALLOW_ANY_INTERPRETER=1``.

This test proves the fix both ways, directly against the live function (not just
that the gate stopped erroring):

1. A **known-bad input** -- a wrong interpreter path with the escape hatch absent --
   is still rejected. The un-blinding must not have gone blind the OTHER direction
   (accepting everything).
2. The exact escape hatch ``run_guardrails_lean.py::_gate_env()`` sets is what the
   real fix depends on, so a regression that stops setting it would be caught by
   ``test_run_guardrails_lean.py``'s own ``_gate_env`` coverage combined with (1)
   here -- together they prove the un-blinded gate is real, not merely quiet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _conftest_module():
    conftest_path = str(Path(__file__).resolve().parents[1] / "conftest.py")
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) == conftest_path:
            return module
    pytest.skip("root conftest not loaded")


def test_wrong_interpreter_without_escape_hatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known-bad input: a `sys.executable` outside `<repo>/.venv`, escape hatch unset."""
    conftest = _conftest_module()
    monkeypatch.delenv("AGENT_UTILITIES_ALLOW_ANY_INTERPRETER", raising=False)
    monkeypatch.setattr(sys, "executable", "/home/example/.local/bin/python3")

    with pytest.raises(SystemExit, match="is not inside"):
        conftest._fail_fast_on_wrong_interpreter()


def test_wrong_interpreter_with_escape_hatch_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact env var `run_guardrails_lean.py::_gate_env()` sets must clear the
    same wrong-interpreter input the previous test proves is otherwise rejected --
    this is what makes the CI-parity lean venv runnable at all."""
    conftest = _conftest_module()
    monkeypatch.setenv("AGENT_UTILITIES_ALLOW_ANY_INTERPRETER", "1")
    monkeypatch.setattr(sys, "executable", "/home/example/.local/bin/python3")

    conftest._fail_fast_on_wrong_interpreter()  # must not raise


def test_run_guardrails_lean_sets_the_exact_escape_hatch_conftest_checks_for(
    tmp_path: Path,
) -> None:
    """Cross-checks the producer (`run_guardrails_lean.py::_gate_env`) against the
    consumer (`conftest.py::_fail_fast_on_wrong_interpreter`) so the two cannot
    silently drift apart -- e.g. one renaming the env var without the other."""
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "run_guardrails_lean.py"
    spec = importlib.util.spec_from_file_location(
        "run_guardrails_lean_crosscheck", script
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)

    conftest = _conftest_module()
    environment = runner._gate_env(tmp_path)

    # The literal name conftest reads via os.environ.get(...) at line ~166.
    assert environment.get("AGENT_UTILITIES_ALLOW_ANY_INTERPRETER") == "1"
    assert conftest._fail_fast_on_wrong_interpreter.__module__ == conftest.__name__
