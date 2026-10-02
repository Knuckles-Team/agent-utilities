"""Tests for the AU cross-owner module/console-script boundary gate."""

from __future__ import annotations

from pathlib import Path

from scripts.boundary_inventory import COVERAGE_PATH
from scripts.check_boundary_owners import SEEDS_PATH, check

_HEADER = (
    "| Directory | Approx. lines | Disposition | Covering requirement ID(s) | Notes |\n"
    "|---|---:|---|---|---|\n"
)


def test_repository_satisfies_boundary_owners_today() -> None:
    """Every real module/console-script that matches a graph-os or
    agent-connector-sdk seed today is recorded in coverage.md as relocating
    there -- nothing has silently come back after a cutover, and no new code
    has landed directly under a destination-owned seed path.
    """
    root = Path(__file__).resolve().parents[2]
    assert check(root) == []


def _row(directory: str, disposition: str, requirement: str = "AU-FIXTURE-R7") -> str:
    return f"| `{directory}/` | 10 | {disposition} | {requirement} | fixture row |\n"


def _write_coverage(root: Path, rows: str) -> None:
    path = root / COVERAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Fixture inventory\n\n## Deletion and relocation inventory\n\n"
        + _HEADER
        + rows,
        encoding="utf-8",
    )


def _write_seeds(root: Path, owned_module_paths: list[str]) -> None:
    path = root / SEEDS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    paths_yaml = "\n".join(f"      - {p}" for p in owned_module_paths)
    path.write_text(
        "schema: au-boundary-owner-seeds/v1\n"
        "seeds:\n"
        "  - owner_repository: graph-os\n"
        "    requirement_id: AU-FIXTURE-R7\n"
        "    owned_module_paths:\n" + paths_yaml + "\n",
        encoding="utf-8",
    )


def _write_pyproject(root: Path, scripts: dict[str, str]) -> None:
    lines = ["[project.scripts]"]
    lines.extend(f'{name} = "{target}"' for name, target in scripts.items())
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_module(root: Path, relative_py_path: str) -> None:
    file_path = root / relative_py_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("", encoding="utf-8")
    init_path = file_path.parent / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")


def test_wrong_owner_module_is_rejected(tmp_path: Path) -> None:
    _make_module(tmp_path, "agent_utilities/gateway/daemon.py")
    _write_seeds(tmp_path, ["agent_utilities/gateway"])
    _write_coverage(tmp_path, _row("agent_utilities", "keep in AU"))
    _write_pyproject(tmp_path, {})
    findings = check(tmp_path)
    assert any(
        "module at 'agent_utilities/gateway/daemon.py'" in f and "graph-os seed" in f
        for f in findings
    )


def test_wrong_owner_console_script_is_rejected(tmp_path: Path) -> None:
    _make_module(tmp_path, "agent_utilities/mcp/kg_server.py")
    _write_seeds(tmp_path, ["agent_utilities/mcp/kg_server.py"])
    _write_coverage(tmp_path, _row("agent_utilities", "keep in AU"))
    _write_pyproject(tmp_path, {"graph-os": "agent_utilities.mcp.kg_server:mcp_server"})
    findings = check(tmp_path)
    assert any("console script 'graph-os'" in f for f in findings)


def test_recorded_still_present_module_passes(tmp_path: Path) -> None:
    _make_module(tmp_path, "agent_utilities/gateway/daemon.py")
    _write_seeds(tmp_path, ["agent_utilities/gateway"])
    _write_coverage(tmp_path, _row("agent_utilities/gateway", "relocate to graph-os"))
    _write_pyproject(tmp_path, {})
    assert check(tmp_path) == []


def test_unrecorded_module_after_cutover_fails(tmp_path: Path) -> None:
    """Once a row stops saying "relocate to graph-os" (the module was cut
    over and the row updated), a reappearance of the module must be rejected
    even though a row for the directory still exists.
    """
    _make_module(tmp_path, "agent_utilities/gateway/daemon.py")
    _write_seeds(tmp_path, ["agent_utilities/gateway"])
    _write_coverage(tmp_path, _row("agent_utilities/gateway", "deleted; see graph-os"))
    _write_pyproject(tmp_path, {})
    findings = check(tmp_path)
    assert any("agent_utilities/gateway/daemon.py" in f for f in findings)


def test_module_with_no_seed_match_is_ignored(tmp_path: Path) -> None:
    _make_module(tmp_path, "agent_utilities/agent/factory.py")
    _write_seeds(tmp_path, ["agent_utilities/gateway"])
    _write_coverage(tmp_path, _row("agent_utilities", "keep in AU"))
    _write_pyproject(tmp_path, {})
    assert check(tmp_path) == []
