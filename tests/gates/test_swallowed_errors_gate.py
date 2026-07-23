"""Meta-test: the swallowed-error gate trips on cause-dropping handlers and
passes clean on justified/cause-preserving ones. A gate that can't fail is
not a gate (see the ``check_cpd``/``check_prompt_schema`` tautology traps
this exact codebase already found and fixed).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_swallowed_errors.py"


def _run(target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), target],
        capture_output=True,
        text=True,
    )


def test_gate_trips_on_bare_except_pass(tmp_path):
    (tmp_path / "bad_bare.py").write_text(
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except:\n"
        "        pass\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "bad_bare.py" in result.stdout
    assert "bare_except" in result.stdout


def test_gate_trips_on_except_pass_with_no_justification(tmp_path):
    (tmp_path / "bad_pass.py").write_text(
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception:\n"
        "        pass\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "[pass]" in result.stdout


def test_gate_trips_on_bare_return_with_no_log(tmp_path):
    (tmp_path / "bad_return.py").write_text(
        "def f():\n"
        "    try:\n"
        "        return do_thing()\n"
        "    except Exception:\n"
        "        return\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "[return_none]" in result.stdout


def test_gate_trips_on_type_name_only_logging(tmp_path):
    """The exact antipattern this gate was built for: the class name is
    logged, the actual message is discarded."""
    (tmp_path / "bad_type_only.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:\n"
        '        logger.error("failed (%s)", type(exc).__name__)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "[log_type_name_only]" in result.stdout


def test_gate_passes_on_justified_noqa_with_reason(tmp_path):
    (tmp_path / "fine_noqa.py").write_text(
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception:  # noqa: BLE001 - telemetry must never block startup\n"
        "        pass\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_still_trips_on_bare_noqa_with_no_reason(tmp_path):
    """A bare ``# noqa: BLE001`` with nothing after it is NOT the justified
    convention — the convention is documenting *why*, not just silencing."""
    (tmp_path / "bad_bare_noqa.py").write_text(
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception:  # noqa: BLE001\n"
        "        pass\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout


def test_gate_passes_on_cause_preserving_log(tmp_path):
    (tmp_path / "fine_log.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:\n"
        '        logger.warning("failed: %s", exc)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_on_logger_exception_with_no_bound_name(tmp_path):
    """``logger.exception(...)`` always attaches the current traceback/message
    regardless of whether the exception is even bound to a name."""
    (tmp_path / "fine_logger_exception.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception:\n"
        '        logger.exception("failed")\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_when_the_handler_reraises(tmp_path):
    (tmp_path / "fine_reraise.py").write_text(
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:\n"
        '        raise RuntimeError("wrapped") from exc\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_ignores_typed_narrow_control_flow(tmp_path):
    """A narrow, typed fallback (not the broad-Exception silent-swallow
    antipattern) is ordinary Python control flow, not flagged."""
    (tmp_path / "fine_typed_fallback.py").write_text(
        "def f(raw):\n"
        "    try:\n"
        "        return int(raw)\n"
        "    except ValueError:\n"
        "        return 0\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_on_the_repo_baseline() -> None:
    """The gate's own baseline (frozen pre-existing debt) must stay green
    against the real repo — this is the regression lock proving the ratchet
    mechanics work end-to-end, not just on synthetic fixtures."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
