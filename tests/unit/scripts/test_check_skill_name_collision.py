from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _module():
    source = Path(__file__).parents[3] / "scripts" / "check_skill_name_collision.py"
    spec = importlib.util.spec_from_file_location("check_skill_name_collision", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _skill(root: Path, package: str, name: str) -> Path:
    path = root / "agents" / package / package.replace("-", "_") / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
    return path


def test_scan_excludes_detached_worktree_copies(tmp_path: Path) -> None:
    module = _module()
    canonical = _skill(tmp_path, "example-agent", "example-agent-operate")
    detached = (
        tmp_path
        / "agents"
        / ".worktrees"
        / "example-copy"
        / "example_agent"
        / "skills"
        / "example-agent-operate"
        / "SKILL.md"
    )
    detached.parent.mkdir(parents=True)
    detached.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")

    by_name, _convention, _prompts = module.scan(tmp_path)

    assert by_name == {"example-agent-operate": [canonical]}


def test_scan_rejects_symlink_inside_installable_skill_tree(tmp_path: Path) -> None:
    module = _module()
    skill_root = tmp_path / "agents" / "example-agent" / "example_agent" / "skills"
    skill_root.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (skill_root / "linked").symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="contains a symlink"):
        module.scan(tmp_path)
