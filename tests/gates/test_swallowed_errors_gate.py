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


def test_type_name_only_does_not_fire_when_the_cause_is_ALSO_logged(tmp_path):
    """Regression (reconciliation gate 2): logging ``type(exc).__name__`` NEXT TO
    the exception itself is cause-preserving and must not be flagged.

    The classifier used to set its ``type_only`` flag on any log call that merely
    *mentioned* ``type(exc).__name__``, without checking whether the exception was
    also passed — contradicting its own docstring ("all of them only reference
    type(bound_name).__name__"). That false positive fired on the real
    ``HybridRetriever._neighbors_batch`` handler, which logs at WARNING with BOTH
    the class name and the message, and would have forced a pointless ``# noqa``
    onto correct code — training exactly the habit this gate exists to prevent.
    """
    (tmp_path / "cause_preserved.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as e:\n"
        '        logger.warning("failed (%s: %s)", type(e).__name__, e)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout
    assert "[log_type_name_only]" not in result.stdout


def test_type_name_only_still_fires_when_only_one_of_two_logs_carries_the_cause(
    tmp_path,
):
    """The flip side: the shape is reported only when EVERY log call drops the
    cause, so a handler with one cause-preserving line stays green."""
    (tmp_path / "mixed_logs.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:\n"
        '        logger.error("failed (%s)", type(exc).__name__)\n'
        '        logger.error("detail: %s", exc)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout


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


# ── D-SWG-2: DEBUG-only cause-preservation is still a violation ────────────


def test_gate_trips_on_debug_only_cause_preserving_swallow(tmp_path):
    """The exact D-DG-7 shape: the exception IS passed to the log call (so the
    OLD cause-preservation check would have called this fine), but the ONLY
    log call is at DEBUG — invisible at any production log level."""
    (tmp_path / "bad_debug_only.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:\n"
        '        logger.debug("failed: %s", exc)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "[debug_only_swallow]" in result.stdout


def test_gate_passes_on_debug_swallow_with_noqa_justification(tmp_path):
    """The prescribed escape hatch: a genuinely best-effort DEBUG swallow is
    fine once it says why."""
    (tmp_path / "fine_debug_justified.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:  # noqa: BLE001 - telemetry span close, non-load-bearing\n"
        '        logger.debug("failed: %s", exc)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_still_trips_on_bare_debug_noqa_with_no_reason(tmp_path):
    """A bare ``# noqa: BLE001`` (no reason) does not exempt a DEBUG-only
    swallow either — same bar as every other shape."""
    (tmp_path / "bad_debug_bare_noqa.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:  # noqa: BLE001\n"
        '        logger.debug("failed: %s", exc)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "[debug_only_swallow]" in result.stdout


def test_gate_passes_when_debug_swallow_also_logs_at_a_loud_level(tmp_path):
    """A handler that logs at DEBUG *and* at a level someone actually watches
    is fine — the loud call is what matters, regardless of call order."""
    (tmp_path / "fine_debug_plus_warning.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:\n"
        '        logger.debug("verbose context: %s", exc)\n'
        '        logger.warning("failed: %s", exc)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_when_debug_call_is_type_name_only_but_no_other_ref(tmp_path):
    """A DEBUG call that only logs the type name (no message) is caught by
    the pre-existing ``log_type_name_only`` shape, not the new DEBUG rule —
    both are violations, but the type-name-only message should name the
    right shape."""
    (tmp_path / "bad_debug_type_only.py").write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception as exc:\n"
        '        logger.debug("failed (%s)", type(exc).__name__)\n'
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "[log_type_name_only]" in result.stdout


# ── D-SWG-1: the baseline key is stable under unrelated line motion ────────


def _load_gate_module():
    """Import ``check_swallowed_errors.py`` as a module so its internal
    ``scan()``/``_load_baseline()``/``_write_baseline()`` can be exercised
    directly against an isolated ``tmp_path`` fixture — the CLI's
    explicit-ROOT mode deliberately skips baseline comparison entirely (see
    its own ``--help``), so proving the ratchet's line-motion stability
    requires calling the module functions, not shelling out."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_swallowed_errors_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_baseline_key_survives_unrelated_line_shift(tmp_path, monkeypatch):
    """D-SWG-1's core claim: an edit earlier in a baselined file that shifts
    every line number below it must NOT manufacture a phantom "new" entry."""
    mod = _load_gate_module()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "mod.py"
    source = (
        "def f():\n"
        "    try:\n"
        "        do_thing()\n"
        "    except Exception:\n"
        "        pass\n"
    )
    target.write_text(source)

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "PKG", pkg)
    monkeypatch.setattr(mod, "BASELINE", tmp_path / "baseline.txt")

    current = mod.scan(pkg, display_root=tmp_path)
    assert len(current) == 1
    mod._write_baseline(current)

    # Sanity: freshly frozen baseline is green.
    baseline = mod._load_baseline()
    assert set(current) == baseline

    # Unrelated edit: prepend 8 blank lines, shifting the handler from line 4
    # to line 12 with no change to the handler itself.
    target.write_text("\n" * 8 + source)
    shifted_current = mod.scan(pkg, display_root=tmp_path)
    shifted_baseline = mod._load_baseline()
    new_entries = set(shifted_current) - shifted_baseline
    assert not new_entries, f"line motion falsely trips the gate: {new_entries}"
    # And the key itself is byte-identical, not just "still not new".
    assert set(shifted_current) == set(current)


def test_baseline_key_disambiguates_two_identical_violations_in_one_symbol(
    tmp_path, monkeypatch
):
    """Two structurally-identical violations in the same function must get
    distinct keys (an ordinal collision would silently drop one from the
    baseline, hiding a real, distinct site)."""
    mod = _load_gate_module()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "mod.py"
    target.write_text(
        "def f():\n"
        "    try:\n"
        "        a()\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        b()\n"
        "    except Exception:\n"
        "        pass\n"
    )

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "PKG", pkg)
    monkeypatch.setattr(mod, "BASELINE", tmp_path / "baseline.txt")

    current = mod.scan(pkg, display_root=tmp_path)
    assert len(current) == 2, current
    ordinals = sorted(key[4] for key in current)
    assert ordinals == [0, 1]


def test_baseline_key_changes_when_the_handler_actually_moves_symbol(
    tmp_path, monkeypatch
):
    """Moving a violation into a DIFFERENT enclosing function (not just
    shifting lines) is genuinely new information — the key must change."""
    mod = _load_gate_module()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "mod.py"
    target.write_text(
        "def f():\n"
        "    try:\n"
        "        a()\n"
        "    except Exception:\n"
        "        pass\n"
    )

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "PKG", pkg)
    monkeypatch.setattr(mod, "BASELINE", tmp_path / "baseline.txt")

    current = mod.scan(pkg, display_root=tmp_path)
    mod._write_baseline(current)
    baseline_before = mod._load_baseline()

    target.write_text(
        "def g():\n"
        "    try:\n"
        "        a()\n"
        "    except Exception:\n"
        "        pass\n"
    )
    moved_current = mod.scan(pkg, display_root=tmp_path)
    new_entries = set(moved_current) - baseline_before
    assert new_entries, "renaming the enclosing function should be a new entry"
