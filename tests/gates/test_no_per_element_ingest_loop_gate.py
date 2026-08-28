"""Meta-test: the per-element-ingest-loop gate trips on a known-bad ``for``
loop calling a per-element engine ingest method under ``agent_utilities/mcp/``,
and passes clean on a batched call or a loop outside its ingest-method set. A
gate that can't fail is not a gate.

WD4-RAT-01 (CX complexity-collapse program): this gate used to freeze its
findings into ``scripts/no_per_element_ingest_loop_baseline.txt`` keyed by
``(file, ENCLOSING FUNCTION, label)`` — the same structurally unstable key
``check_swallowed_errors.py``'s D-SWG-1 fix and this gate's sibling
(``check_event_loop_blocking.py``) both replaced with a content key earlier in
this program. There was no prior test file for this gate at all, so nothing
here needed inverting — but the content-key stability tests below are the
direct analog of that fix.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_no_per_element_ingest_loop.py"


def _run(target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), target],
        capture_output=True,
        text=True,
    )


def test_gate_trips_on_add_node_in_a_for_loop(tmp_path):
    (tmp_path / "bad_loop.py").write_text(
        "def sync_agents(engine, items):\n"
        "    for item in items:\n"
        "        engine.add_node(item)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "bad_loop.py" in result.stdout
    assert "per-element ingest call in a loop: engine.add_node" in result.stdout


def test_gate_trips_on_async_for_loop(tmp_path):
    (tmp_path / "bad_async_loop.py").write_text(
        "async def sync_agents(engine, items):\n"
        "    async for item in items:\n"
        "        engine.link_nodes(item.a, item.b)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 1, result.stdout
    assert "per-element ingest call in a loop: engine.link_nodes" in result.stdout


def test_gate_passes_on_a_batched_call_outside_a_loop(tmp_path):
    (tmp_path / "fine_batched.py").write_text(
        "def sync_agents(engine, items):\n    engine.batch_typed_mutations(items)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_on_a_loop_calling_an_out_of_scope_method(tmp_path):
    """delete_node/remove_* are deliberately a different operation family —
    see module docstring — and must not be flagged."""
    (tmp_path / "fine_delete_loop.py").write_text(
        "def prune(engine, items):\n"
        "    for item in items:\n"
        "        engine.delete_node(item)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_when_the_ingest_call_is_not_inside_a_loop(tmp_path):
    (tmp_path / "fine_single_call.py").write_text(
        "def add_one(engine, item):\n    engine.add_node(item)\n"
    )
    result = _run(str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_gate_passes_on_the_real_repo() -> None:
    """The gate must be green against the real repo with no change staged.

    Regression lock proving the census + diff-scoped mechanics work
    end-to-end. It is NOT a claim the repo has zero per-element ingest loops —
    it prints the real (small) population every run — only that leaving them
    alone does not fail."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "census:" in result.stdout


def test_update_baseline_flag_is_retired() -> None:
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
        "check_no_per_element_ingest_loop_mod", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ONE_LOOP = (
    "def f(engine, items):\n    for item in items:\n        engine.add_node(item)\n"
)


def test_content_key_survives_unrelated_line_shift() -> None:
    mod = _load_gate_module()
    before = mod._content_counts("mod.py", _ONE_LOOP)
    after = mod._content_counts("mod.py", "\n" * 8 + _ONE_LOOP)
    assert before == after, f"line motion changes the content key: {before} vs {after}"


def test_content_key_survives_extraction_into_a_new_function() -> None:
    """The regression class this gate was rebuilt for: moving the SAME
    per-element loop from one enclosing function into an extracted helper
    must not manufacture a phantom finding."""
    mod = _load_gate_module()
    inline = (
        "def sync_mcp_agents(engine, items):\n"
        "    for item in items:\n"
        "        engine.add_node(item)\n"
    )
    extracted = (
        "def sync_mcp_agents(engine, items):\n"
        "    def _sync_one_batch():\n"
        "        for item in items:\n"
        "            engine.add_node(item)\n"
        "    _sync_one_batch()\n"
    )
    assert mod._content_counts("mod.py", inline) == mod._content_counts(
        "mod.py", extracted
    ), "extraction/renaming must not manufacture a finding"


def test_content_key_does_report_a_genuinely_added_loop() -> None:
    mod = _load_gate_module()
    one = _ONE_LOOP
    two = one + (
        "def g(engine, items):\n"
        "    for item in items:\n"
        "        engine.add_edge(item.a, item.b)\n"
    )
    before = mod._content_counts("mod.py", one)
    after = mod._content_counts("mod.py", two)
    added = [k for k, n in after.items() if n > before[k]]
    assert added, "adding a second, differently-shaped loop must be visible"


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
