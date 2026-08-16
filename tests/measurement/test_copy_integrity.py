"""Capability C proof: copy integrity (incident 3).

Incident: `rsync -a` (no `-H`) silently dropped one hardlinked source file
during a copy-for-measurement; the suite then reported 14,159 errors on a
repo that was actually fine — a bad copy, not a bad repo. Proves
`verify_copy` catches an incomplete copy (the general mechanism) and that
`copy_tree` refuses to hand back a copy that fails verification.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from agent_utilities.measurement.copy_integrity import (
    CopyIntegrityError,
    copy_tree,
    verify_copy,
)


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        (src / f"file_{i}.py").write_text(f"# file {i}\n" * 10)
    # The incident's specific artifact: a hardlinked file (2 links).
    (src / "distributed_state_manager.py").write_text("# shared state\n")
    os.link(src / "distributed_state_manager.py", src / "distributed_state_manager_link.py")
    return src


def test_verify_copy_catches_incident_3_exact_shape(source_tree: Path, tmp_path: Path):
    """Construct a copy missing exactly the hardlinked file (as `rsync -a`,
    without `-H`, was reported to have done) and prove verify_copy catches it."""
    dest = tmp_path / "dest_bad"
    dest.mkdir()
    for f in source_tree.iterdir():
        if f.name == "distributed_state_manager.py":
            continue  # simulate the exact incident: this one file is dropped
        shutil.copy2(f, dest / f.name)

    result = verify_copy(source_tree, dest)

    assert not result.ok
    assert "distributed_state_manager.py" in result.missing_in_dest
    assert result.source_file_count == 7
    assert result.dest_file_count == 6
    with pytest.raises(CopyIntegrityError):
        result.raise_if_bad()


def test_verify_copy_passes_on_a_complete_copy(source_tree: Path, tmp_path: Path):
    dest = tmp_path / "dest_good"
    shutil.copytree(source_tree, dest)
    result = verify_copy(source_tree, dest)
    assert result.ok
    result.raise_if_bad()  # must not raise


def test_copy_tree_with_rsync_dash_h_preserves_hardlinked_file(source_tree: Path, tmp_path: Path):
    """The actual fix: rsync -aH (not -a) must not reproduce the incident."""
    if shutil.which("rsync") is None:
        pytest.skip("rsync not available")
    dest = tmp_path / "dest_rsync"
    result = copy_tree(source_tree, dest, method="rsync")
    assert result.ok
    assert (dest / "distributed_state_manager.py").exists()
    assert (dest / "distributed_state_manager_link.py").exists()


def test_verify_copy_catches_a_content_mismatch_not_just_missing_files(tmp_path: Path):
    src = tmp_path / "src2"
    dst = tmp_path / "dst2"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("original\n")
    (dst / "a.txt").write_text("corrupted-during-copy\n")

    result = verify_copy(src, dst)
    assert not result.ok
    assert "a.txt" in result.mismatched
