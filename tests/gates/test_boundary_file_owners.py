"""Exercise the tracked-file inventory without requiring another checkout."""

import json
from pathlib import Path

import pytest

from scripts.check_boundary_file_owners import check


@pytest.fixture
def inventory(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "architecture/file-ownership.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "schema": "au-file-owners/v1",
                "files": [
                    {
                        "path": "agent_utilities/core/paths.py",
                        "disposition": "keep",
                        "owner": "meristem",
                    }
                ],
            }
        )
    )
    return tmp_path, path


def test_file_inventory_detects_added_and_removed_tracked_files(
    inventory: tuple[Path, Path],
) -> None:
    root, _ = inventory
    assert check(root, ["agent_utilities/core/paths.py"]) == []
    assert check(root, ["agent_utilities/core/new.py"]) == [
        "unowned file: agent_utilities/core/new.py",
        "stale file: agent_utilities/core/paths.py",
    ]


@pytest.mark.parametrize(
    "owner,disposition",
    [("unknown", "relocate"), ("", "keep"), ("meristem", "undecided")],
)
def test_file_inventory_refuses_unowned_or_undecided_entries(
    inventory: tuple[Path, Path], owner: str, disposition: str
) -> None:
    root, path = inventory
    manifest = json.loads(path.read_text())
    manifest["files"][0].update(owner=owner, disposition=disposition)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid file owner"):
        check(root, ["agent_utilities/core/paths.py"])


def test_file_inventory_refuses_duplicate_paths(inventory: tuple[Path, Path]) -> None:
    root, path = inventory
    manifest = json.loads(path.read_text())
    manifest["files"].append(manifest["files"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate file path"):
        check(root, ["agent_utilities/core/paths.py"])
