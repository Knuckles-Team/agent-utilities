"""Plan 10 meta-tests: prove each guardrail actually trips on a broken fixture
and passes on a clean one. A gate that can't fail is not a gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(script: str, arg: str) -> int:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), arg],
        capture_output=True,
        text=True,
    ).returncode


# ---- sprawl gate -------------------------------------------------------------


def test_sprawl_gate_trips_on_versioned_clone(tmp_path):
    (tmp_path / "router_v2.py").write_text("x = 1\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 1


def test_sprawl_gate_trips_on_merge_marker(tmp_path):
    (tmp_path / "m.py").write_text("# --- Merged from other.py\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 1


def test_sprawl_gate_trips_on_bare_merge_marker_in_markdown(tmp_path):
    """A real botched merge in a .md is still a violation."""
    (tmp_path / "d.md").write_text("# --- Merged from other.py\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 1


def test_sprawl_gate_ignores_quoted_merge_marker_in_markdown(tmp_path):
    """Documentation that quotes the marker is not a botched merge."""
    (tmp_path / "d.md").write_text(
        "The gate refuses the literal `# --- Merged from` marker.\n\n"
        "```\n# --- Merged from other.py\n```\n"
    )
    assert _run("check_sprawl.py", str(tmp_path)) == 0


def test_sprawl_gate_trips_on_reject_artifact(tmp_path):
    (tmp_path / "patch.orig").write_text("junk\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 1


def test_sprawl_gate_passes_clean(tmp_path):
    (tmp_path / "router.py").write_text("def f():\n    return 1\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 0


_BIG_BINARY = bytes(range(256)) * 8192  # 2 MiB of non-text: over the 1 MB cap


def _generated_artifact_repo(tmp_path, pinned):
    """A target repo with one large binary and, if `pinned`, a ledger entry for it."""
    import hashlib

    (tmp_path / "codec.wasm").write_bytes(_BIG_BINARY)
    if pinned == "correct":
        pinned = hashlib.sha256(_BIG_BINARY).hexdigest()
    if pinned is not None:
        (tmp_path / ".config").mkdir()
        (tmp_path / ".config" / "generated-artifacts.toml").write_text(
            '[[artifact]]\npath = "codec.wasm"\n'
            f'sha256 = "{pinned}"\n'
            'reproducer = "python3 scripts/build.py --check"\n'
            'proven_by = "language-clients"\nreason = "embedded codec"\n'
        )
    return tmp_path


def test_sprawl_gate_exempts_a_listed_binary_with_its_pinned_sha(tmp_path):
    repo = _generated_artifact_repo(tmp_path, "correct")
    assert _run("check_sprawl.py", str(repo)) == 0


def test_sprawl_gate_trips_on_a_listed_binary_whose_sha_changed(tmp_path):
    repo = _generated_artifact_repo(tmp_path, "0" * 64)
    assert _run("check_sprawl.py", str(repo)) == 1


def test_sprawl_gate_still_trips_on_an_unlisted_large_binary(tmp_path):
    repo = _generated_artifact_repo(tmp_path, None)
    assert _run("check_sprawl.py", str(repo)) == 1


def test_sprawl_gate_trips_on_a_listed_artifact_that_is_missing(tmp_path):
    repo = _generated_artifact_repo(tmp_path, "correct")
    (repo / "codec.wasm").unlink()
    assert _run("check_sprawl.py", str(repo)) == 1


def test_sprawl_gate_fails_closed_on_a_malformed_ledger(tmp_path):
    repo = _generated_artifact_repo(tmp_path, None)
    (repo / ".config").mkdir()
    (repo / ".config" / "generated-artifacts.toml").write_text('allow = ["*"]\n')
    assert _run("check_sprawl.py", str(repo)) == 1


# ---- no_stub gate (check_no_stub.py) ----------------------------------------


def test_no_stub_gate_trips_on_mock(tmp_path):
    (tmp_path / "m.py").write_text('def f():\n    return "[Mock] nope"\n')
    assert _run("check_no_stub.py", str(tmp_path)) == 1


def test_no_stub_gate_trips_on_notimplemented(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    raise NotImplementedError\n")
    assert _run("check_no_stub.py", str(tmp_path)) == 1


def test_no_stub_gate_allows_abstract_ok(tmp_path):
    (tmp_path / "m.py").write_text(
        "def f():\n    raise NotImplementedError  # ABSTRACT-OK\n"
    )
    assert _run("check_no_stub.py", str(tmp_path)) == 0


def test_no_stub_gate_passes_clean(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 42\n")
    assert _run("check_no_stub.py", str(tmp_path)) == 0
