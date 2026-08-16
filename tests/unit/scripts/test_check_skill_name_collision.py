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
    path = (
        root
        / "agents"
        / package
        / package.replace("-", "_")
        / "skills"
        / name
        / "SKILL.md"
    )
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

    by_name, _convention, _prompts, _warnings = module.scan(tmp_path)

    assert by_name == {"example-agent-operate": [canonical]}


def test_scan_rejects_symlink_escaping_the_tracked_tree(tmp_path: Path) -> None:
    """A symlink resolving OUTSIDE every tracked tree (BUG-233's real-finding
    case) must still fail closed — e.g. a scratch dir that happens to be a
    physical sibling of ``agents/``/``skills/``/``agent-utilities`` under the
    same fleet root is not itself part of any tracked installable tree."""
    module = _module()
    skill_root = tmp_path / "agents" / "example-agent" / "example_agent" / "skills"
    skill_root.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (skill_root / "linked").symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="escaping the tracked tree"):
        module.scan(tmp_path)


def test_scan_prunes_benign_symlink_inside_the_tracked_tree(tmp_path: Path) -> None:
    """BUG-233: a symlink whose target resolves INSIDE one of the tracked
    trees (the ``.uv-workspace-siblings/<pkg> -> agent-packages/<pkg>``
    editable-install shape) must not abort the whole scan — it is pruned
    (not re-descended into, so a SKILL.md reached through it is never
    double-counted) and surfaced only as a warning, while a clean tree
    containing it still scans successfully."""
    module = _module()
    canonical = _skill(tmp_path, "example-agent", "example-agent-operate")
    siblings_dir = (
        tmp_path
        / "agents"
        / "other-agent"
        / "other_agent"
        / ".uv-workspace-siblings-unlisted"
    )
    siblings_dir.mkdir(parents=True)
    # Resolves to a path INSIDE the tracked ``agents/`` tree (an ancestor of
    # a genuinely different package) — the benign case, even though this
    # exact directory name is not (yet) in the exclude list.
    (siblings_dir / "example-agent").symlink_to(
        tmp_path / "agents" / "example-agent", target_is_directory=True
    )

    by_name, _convention, _prompts, warnings = module.scan(tmp_path)

    assert by_name == {"example-agent-operate": [canonical]}
    assert any("pruned" in w for w in warnings)


def test_scan_prunes_uv_workspace_siblings_without_scanning(tmp_path: Path) -> None:
    """The concrete BUG-233 shape: ``.uv-workspace-siblings/agent-utilities``
    symlinked back to the hub repo must not raise, and — because
    ``.uv-workspace-siblings`` is now excluded outright — never even reaches
    the symlink classifier."""
    module = _module()
    canonical = _skill(tmp_path, "example-agent", "example-agent-operate")
    (tmp_path / "agent-utilities" / "agent_utilities" / "skills").mkdir(parents=True)
    siblings_dir = (
        tmp_path
        / "agents"
        / "example-agent"
        / "example_agent"
        / ".uv-workspace-siblings"
    )
    siblings_dir.mkdir(parents=True)
    (siblings_dir / "agent-utilities").symlink_to(
        tmp_path / "agent-utilities", target_is_directory=True
    )

    by_name, _convention, _prompts, warnings = module.scan(tmp_path)

    assert by_name == {"example-agent-operate": [canonical]}
    assert warnings == []
