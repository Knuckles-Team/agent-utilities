"""Meta-test: the event-loop-blocking gate trips on a known-bad async blocking
call and passes clean on hopped/typed-safe code. A gate that can't fail is not
a gate (see the ``check_cpd``/``check_prompt_schema`` tautology traps this exact
codebase already found and fixed).

WD4-RAT-01 (CX complexity-collapse program): this gate used to freeze its
findings into ``scripts/event_loop_blocking_baseline.txt`` keyed by
``(file, ENCLOSING FUNCTION, label)`` — not invariant under function
extraction/renaming, the same D-SWG-1 defect ``check_swallowed_errors.py`` was
rebuilt for earlier in this program. It is now content-keyed (the finding's
own label) and diff-scoped instead. There was no prior test file for this
gate at all, so nothing here needed inverting — but the content-key stability
tests below are the direct analog of
``tests/gates/test_swallowed_errors_gate.py``'s D-SWG-1 section.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_event_loop_blocking.py"


def _run(target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), target],
        capture_output=True,
        text=True,
    )


def test_gate_trips_on_a_bare_engine_write_in_an_async_def(tmp_path):
    (tmp_path / "bad_engine_write.py").write_text(
        "async def handle(engine, payload):\n    engine.add_node(payload)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "bad_engine_write.py" in result.stdout
    assert "engine write: add_node" in result.stdout


def test_gate_trips_on_time_sleep(tmp_path):
    """time.sleep is also the HARD_ZERO_SHAPES invariant, but the explicit-root
    mode reports every finding regardless — this proves the shape itself is
    still recognized."""
    (tmp_path / "bad_sleep.py").write_text(
        "import time\n\nasync def handle():\n    time.sleep(1)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "time.sleep (blocks the loop)" in result.stdout


def test_gate_trips_on_subprocess_run(tmp_path):
    (tmp_path / "bad_subprocess.py").write_text(
        "import subprocess\n\nasync def handle():\n    subprocess.run(['ls'])\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "subprocess.run" in result.stdout


def test_gate_trips_on_blocking_open(tmp_path):
    (tmp_path / "bad_open.py").write_text(
        "async def handle():\n    f = open('x.txt')\n    return f\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "open() (blocking file I/O)" in result.stdout


def test_gate_passes_when_hopped_via_run_blocking_ordered(tmp_path):
    """The sanctioned escape hatch: a nested sync closure passed by bare
    reference to ``run_blocking_ordered`` runs off-loop, so its blocking
    call inside must not be flagged."""
    (tmp_path / "fine_hopped.py").write_text(
        "from agent_utilities.core.event_loop import run_blocking_ordered\n\n"
        "async def handle(engine, payload):\n"
        "    def _do_write():\n"
        "        engine.add_node(payload)\n"
        "    await run_blocking_ordered(_do_write)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_on_plain_def_handlers(tmp_path):
    """Known scope limit, asserted rather than just documented: a plain
    (non-async) def is never walked by this scanner."""
    (tmp_path / "fine_sync.py").write_text(
        "def handle(engine, payload):\n    engine.add_node(payload)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_on_the_real_repo() -> None:
    """The gate must be green against the real repo with no change staged.

    The regression lock proving the census + diff-scoped mechanics work
    end-to-end, not just on synthetic fixtures. It is NOT a claim the repo has
    no blocking-call sites — it has a large, real, printed population every
    run. It is a claim that leaving them alone does not fail."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "census:" in result.stdout


def test_update_baseline_flag_is_retired() -> None:
    """The retired flag must REFUSE, not silently do nothing — the same
    convention check_swallowed_errors.py adopted when its baseline was
    removed."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--update-baseline"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "RETIRED" in result.stderr


# ── content-key stability (the D-SWG-1-class defect this gate was rebuilt for) ──


def _load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "check_event_loop_blocking_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ONE_BLOCKING_CALL = "async def f(engine):\n    engine.add_node(1)\n"


def test_content_key_survives_unrelated_line_shift() -> None:
    """An edit above a call site that shifts every line below it must NOT
    manufacture a phantom finding."""
    mod = _load_gate_module()
    before = mod._content_counts("mod.py", _ONE_BLOCKING_CALL)
    after = mod._content_counts("mod.py", "\n" * 8 + _ONE_BLOCKING_CALL)
    assert before == after, f"line motion changes the content key: {before} vs {after}"


def test_content_key_survives_extraction_into_a_new_function() -> None:
    """The regression class this gate was rebuilt for: moving the SAME
    blocking call from one enclosing async function into an extracted nested
    closure must not manufacture a phantom finding — only the call's own
    label (never the enclosing function name) participates in the key.

    The nested closure is a plain (sync) ``def``, matching the actual sweep
    pattern this codebase uses throughout (see module docstring: "define a
    nested sync closure... then ``await run_blocking_ordered(_do_thing)``").
    A nested ``async def`` closure is scanned twice by this gate's ``ast.walk``
    dispatch (once via its parent's recursive walk, once again because
    ``ast.walk`` also visits it directly as its own top-level
    ``AsyncFunctionDef`` target) — a pre-existing quirk of the original
    scanner, unrelated to and unchanged by this content-key rewrite, and out
    of scope for WD4-RAT-01. It never fires for the sync-``def`` shape the
    real sweep actually used, which is what this fixture exercises."""
    mod = _load_gate_module()
    inline = "async def create_agent(engine):\n    engine.add_node(1)\n"
    extracted = (
        "async def create_agent(engine):\n"
        "    def _setup_mcp_url_toolset():\n"
        "        engine.add_node(1)\n"
        "    _setup_mcp_url_toolset()\n"
    )
    assert mod._content_counts("mod.py", inline) == mod._content_counts(
        "mod.py", extracted
    ), "extraction/renaming must not manufacture a finding"


def test_content_key_does_report_a_genuinely_added_call() -> None:
    """...and the other half: a call that is actually ADDED must show up. Without
    this the test above would be satisfied by a key that never changes at all."""
    mod = _load_gate_module()
    one = "async def f(engine):\n    engine.add_node(1)\n"
    two = one + "async def g(engine):\n    engine.add_edge(1, 2)\n"
    before = mod._content_counts("mod.py", one)
    after = mod._content_counts("mod.py", two)
    added = [k for k, n in after.items() if n > before[k]]
    assert added, "adding a second, differently-shaped call must be visible"


def test_content_key_reports_a_second_identical_call_added_to_the_same_file() -> None:
    """A second call sharing the exact same label as an existing one must
    still register as +1 added — the Counter counts occurrences, not just
    presence/absence of the key."""
    mod = _load_gate_module()
    one = _ONE_BLOCKING_CALL
    two = "async def f(engine):\n    engine.add_node(1)\n    engine.add_node(2)\n"
    before = mod._content_counts("mod.py", one)
    after = mod._content_counts("mod.py", two)
    added = {k: n - before[k] for k, n in after.items() if n > before[k]}
    assert added, "a second occurrence of the same call shape must be visible"
    assert all(n == 1 for n in added.values())


# ── ambient GIT_DIR/GIT_INDEX_FILE isolation (BUG-043/BUG-174 class) ────────


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        env.pop(name, None)
    return env


def _ambient_env() -> dict[str, str]:
    """The real hook shape: GIT_DIR/GIT_INDEX_FILE exported and pointed at
    THIS repo's own .git (not a decoy) — exactly what every git hook
    subprocess inherits, and exactly the condition BUG-043/BUG-174 found
    silently re-rooting path resolution in other gates."""
    env = _clean_env()
    env["GIT_DIR"] = str(ROOT / ".git")
    env["GIT_INDEX_FILE"] = str(ROOT / ".git" / "index")
    return env


def test_gate_verdict_is_identical_plain_vs_ambient_git_env() -> None:
    plain = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
    )
    ambient = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=_ambient_env(),
        capture_output=True,
        text=True,
    )
    assert plain.stdout == ambient.stdout
    assert plain.stderr == ambient.stderr
    assert plain.returncode == ambient.returncode
