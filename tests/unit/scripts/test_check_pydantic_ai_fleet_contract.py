"""Focused tests for the bounded fleet Pydantic-AI parity checker."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_pydantic_ai_fleet_contract import (
    read_contract_version,
    scan_paths,
)


def _contract(tmp_path: Path) -> Path:
    source = tmp_path / "protocol_compat.py"
    source.write_text('_PYDANTIC_AI_CONTRACT_VERSION = "2.29.0"\n', encoding="utf-8")
    return source


def _lock(
    version: str, editable: str = ".uv-workspace-siblings/agent-utilities"
) -> str:
    # TOML basic strings treat backslash as an escape. Use a literal string for
    # the adversarial Windows-separator case so the parser reaches the gate.
    editable_literal = f"'{editable}'" if "\\" in editable else f'"{editable}"'
    return (
        "[[package]]\n"
        'name = "agent-utilities"\n'
        f"source = {{ editable = {editable_literal} }}\n\n"
        "[[package]]\n"
        'name = "pydantic-ai-slim"\n'
        f'version = "{version}"\n'
    )


def test_fixture_clean_contract_and_generated_editable_path(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(_lock("2.29.0"), encoding="utf-8")
    manifest = lock.parent / "pyproject.toml"
    manifest.write_text(
        '[project]\ndependencies = ["pydantic-ai-slim[mcp]>=2.29,<3"]\n',
        encoding="utf-8",
    )

    result = scan_paths([lock, manifest], contract_source=contract)

    assert result.ok
    assert result.expected_version == "2.29.0"
    assert result.files_scanned == 2


def test_fixture_reports_resolved_manifest_and_editable_drift(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        _lock("2.28.0", "/tmp/ne-old-worktree/agent-utilities"), encoding="utf-8"
    )
    manifest = lock.parent / "pyproject.toml"
    manifest.write_text(
        '[project]\ndependencies = ["pydantic-ai-slim[mcp]==2.25.0"]\n',
        encoding="utf-8",
    )

    result = scan_paths([lock, manifest], contract_source=contract)
    kinds = {finding.kind for finding in result.findings}

    assert not result.ok
    assert {
        "resolved-version-mismatch",
        "manifest-version-mismatch",
        "stale-editable-au",
    } <= kinds


def test_fixture_accepts_direct_transitive_sibling_editables(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        _lock("2.29.0")
        + "[[package]]\n"
        + 'name = "langfuse-agent"\n'
        + 'source = { editable = ".uv-workspace-siblings/langfuse-agent" }\n',
        encoding="utf-8",
    )

    result = scan_paths([lock], contract_source=contract)

    assert result.ok


def test_fixture_rejects_nested_transitive_sibling_editable_path(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        _lock("2.29.0")
        + "[[package]]\n"
        + 'name = "langfuse-agent"\n'
        + 'source = { editable = ".uv-workspace-siblings/agent-utilities/'
        '.uv-workspace-siblings/langfuse-agent" }\n',
        encoding="utf-8",
    )

    result = scan_paths([lock], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "nested-editable-source" for finding in result.findings)


def test_fixture_accepts_direct_manifest_sibling_source(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        "[tool.uv.sources]\n"
        'helper = { path = ".uv-workspace-siblings/helper", editable = true }\n',
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert result.ok


def test_fixture_accepts_pep503_normalized_sibling_identity(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        "[tool.uv.sources]\n"
        'my_package = { path = ".uv-workspace-siblings/My.Package", editable = true }\n',
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert result.ok


@pytest.mark.parametrize(
    "invalid_name",
    [
        "helper name",
        "helper/name",
        "éhelper",
        "_helper",
        "helper_",
        "-helper",
        "helper-",
        "helper..",
    ],
)
def test_fixture_rejects_non_ascii_or_malformed_pep503_identity(
    tmp_path: Path, invalid_name: str
) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        "[tool.uv.sources]\n"
        f'"{invalid_name}" = {{ path = ".uv-workspace-siblings/helper" }}\n',
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "nested-editable-source" for finding in result.findings)


@pytest.mark.parametrize("invalid_kind", ["project", "path"])
def test_fixture_rejects_malformed_project_or_path_identity(
    tmp_path: Path, invalid_kind: str
) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    if invalid_kind == "project":
        content = (
            "[project]\nname = 'consumer name'\n\n"
            "[tool.uv.sources]\n"
            "consumer = { path = '.', editable = true }\n"
        )
    else:
        content = (
            "[tool.uv.sources]\n"
            'helper = { path = ".uv-workspace-siblings/helper name", editable = true }\n'
        )
    manifest.write_text(content, encoding="utf-8")

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "nested-editable-source" for finding in result.findings)


def test_fixture_rejects_sibling_alias_identity(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        "[tool.uv.sources]\n"
        'agent-utilities = { path = ".uv-workspace-siblings/evil", editable = true }\n',
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "nested-editable-source" for finding in result.findings)


@pytest.mark.parametrize(
    ("project_name", "source_name", "expected_ok"),
    [
        ("my-project", "my_project", True),
        ("consumer", "dependency", False),
    ],
)
def test_fixture_self_dot_requires_project_identity(
    tmp_path: Path,
    project_name: str,
    source_name: str,
    expected_ok: bool,
) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        "[project]\n"
        f'name = "{project_name}"\n\n'
        "[tool.uv.sources]\n"
        f'{source_name} = {{ path = ".", editable = true }}\n',
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert result.ok is expected_ok


def test_fixture_accepts_duplicate_canonical_source_alternatives(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        '[project]\nname = "consumer"\n\n'
        "[tool.uv.sources]\n"
        "helper = [\n"
        '  { path = ".uv-workspace-siblings/helper", editable = true },\n'
        '  { path = ".uv-workspace-siblings/HELPER", editable = true, '
        "marker = \"sys_platform == 'linux'\" },\n"
        "]\n",
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert result.ok


def test_fixture_validates_every_source_alternative(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        '[project]\nname = "consumer"\n\n'
        "[tool.uv.sources]\n"
        "helper = [\n"
        '  { path = ".uv-workspace-siblings/helper" },\n'
        '  { path = ".uv-workspace-siblings/evil" },\n'
        "]\n",
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "nested-editable-source" for finding in result.findings)


def test_fixture_rejects_non_table_source_alternatives(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        '[tool.uv.sources]\nhelper = [[".uv-workspace-siblings/helper"]]\n',
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


def test_fixture_accepts_documented_remote_source_fields(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        "[tool.uv.sources]\n"
        'gitdep = { git = "https://example.invalid/gitdep", tag = "v1", '
        'subdirectory = "python", marker = "sys_platform == \'linux\'" }\n'
        'urldep = { url = "https://example.invalid/urldep.whl", '
        'subdirectory = "python", marker = "sys_platform == \'darwin\'" }\n'
        'indexed = { index = "private", extra = "gpu", marker = "python_version >= \'3.12\'" }\n'
        'git-lfs = { git = "https://example.invalid/git-lfs", lfs = true }\n'
        "workspace = { workspace = true }\n",
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert result.ok


def test_fixture_rejects_invalid_remote_source_identity(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        '[tool.uv.sources]\n"helper name" = { index = "private" }\n',
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


@pytest.mark.parametrize(
    "source_value",
    [
        "empty = {}\n",
        "empty-alternatives = []\n",
        'mixed = { git = "https://example.invalid/git", url = "https://example.invalid/pkg.whl" }\n',
        'selector-without-git = { branch = "main" }\n',
        'multiple-selectors = { git = "https://example.invalid/git", branch = "main", tag = "v1" }\n',
        'url-selector = { url = "https://example.invalid/pkg.whl", rev = "abc" }\n',
        "orphan-lfs = { lfs = true }\n",
        'empty-url = { url = "" }\n',
        "workspace-false = { workspace = false }\n",
        'path-workspace = { path = ".uv-workspace-siblings/path-workspace", workspace = true }\n',
    ],
)
def test_fixture_rejects_malformed_remote_source_shape(
    tmp_path: Path,
    source_value: str,
) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text("[tool.uv.sources]\n" + source_value, encoding="utf-8")

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


def test_fixture_rejects_workspace_path_string(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        '[tool.uv.sources]\nexternal = { workspace = "../other-workspace" }\n',
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


@pytest.mark.parametrize(
    "project_name, expected_ok", [("agent-utilities", True), ("consumer", False)]
)
def test_fixture_lock_self_dot_requires_adjacent_project_identity(
    tmp_path: Path,
    project_name: str,
    expected_ok: bool,
) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(_lock("2.29.0", "."), encoding="utf-8")
    (lock.parent / "pyproject.toml").write_text(
        f'[project]\nname = "{project_name}"\n', encoding="utf-8"
    )

    result = scan_paths([lock], contract_source=contract)

    assert result.ok is expected_ok


def test_fixture_accepts_uv_virtual_lock_source(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        "[[package]]\n"
        'name = "consumer"\n'
        'source = { virtual = "." }\n\n'
        "[[package]]\n"
        'name = "pydantic-ai-slim"\n'
        'version = "2.29.0"\n',
        encoding="utf-8",
    )
    (lock.parent / "pyproject.toml").write_text(
        '[project]\nname = "consumer"\n', encoding="utf-8"
    )

    result = scan_paths([lock], contract_source=contract)

    assert result.ok


def test_fixture_rejects_dependency_metadata_without_lock_resolution(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        "[[package]]\n"
        'name = "consumer"\n'
        'version = "1.0.0"\n'
        'dependencies = [{ name = "pydantic-ai-slim" }]\n',
        encoding="utf-8",
    )

    result = scan_paths([lock], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "missing-resolution" for finding in result.findings)


@pytest.mark.parametrize(
    "source_value",
    [
        'helper = { path = ".uv-workspace-siblings/helper", unknown = true }\n',
        'helper = { path = ".uv-workspace-siblings/helper", index = "private" }\n',
    ],
)
def test_fixture_rejects_unknown_or_mixed_manifest_source_fields(
    tmp_path: Path,
    source_value: str,
) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text("[tool.uv.sources]\n" + source_value, encoding="utf-8")

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


def test_fixture_rejects_nested_manifest_sibling_source(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text(
        "[tool.uv.sources]\n"
        'helper = { path = ".uv-workspace-siblings/au/.uv-workspace-siblings/helper", '
        "editable = true }\n",
        encoding="utf-8",
    )

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "nested-editable-source" for finding in result.findings)


def test_fixture_accepts_canonical_duplicate_sources_and_ignores_comments(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        '# source = { editable = "/tmp/ignored" }\n'
        + _lock("2.29.0")
        + "[[package]]\n"
        + 'name = "agent-utilities"\n'
        + 'source = { editable = ".uv-workspace-siblings/agent-utilities" }\n'
        + 'description = "source = { editable = \\"../ignored\\" }"\n',
        encoding="utf-8",
    )

    result = scan_paths([lock], contract_source=contract)

    assert result.ok


def test_fixture_accepts_pep503_normalized_lock_identity(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        "[[package]]\n"
        'name = "my_package"\n'
        'source = { editable = ".uv-workspace-siblings/My.Package" }\n'
        "[[package]]\n"
        'name = "pydantic-ai-slim"\n'
        'version = "2.29.0"\n',
        encoding="utf-8",
    )

    result = scan_paths([lock], contract_source=contract)

    assert result.ok


def test_fixture_rejects_lock_local_remote_conflict_and_unknown_field(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        "[[package]]\n"
        'name = "helper"\n'
        'source = { editable = ".uv-workspace-siblings/helper", '
        'registry = "https://pypi.org/simple", unknown = true }\n'
        "[[package]]\n"
        'name = "pydantic-ai-slim"\n'
        'version = "2.29.0"\n',
        encoding="utf-8",
    )

    result = scan_paths([lock], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


@pytest.mark.parametrize(
    "source_path",
    [
        "/tmp/agent-utilities",
        "../agent-utilities",
        r".uv-workspace-siblings\agent-utilities",
        ".uv-workspace-siblings/evil",
        ".uv-workspace-siblings/agent-utilities/.uv-workspace-siblings/langfuse-agent",
    ],
)
def test_fixture_rejects_noncanonical_lock_source_paths(
    tmp_path: Path,
    source_path: str,
) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(_lock("2.29.0", source_path), encoding="utf-8")

    result = scan_paths([lock], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "nested-editable-source" for finding in result.findings)


@pytest.mark.parametrize(
    "source_value",
    [
        "helper = { path = 42 }\n",
        'helper = { path = [".uv-workspace-siblings/helper"] }\n',
        "helper = { editable = true }\n",
    ],
)
def test_fixture_rejects_malformed_manifest_source_values(
    tmp_path: Path,
    source_value: str,
) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "consumer" / "pyproject.toml"
    manifest.parent.mkdir()
    manifest.write_text("[tool.uv.sources]\n" + source_value, encoding="utf-8")

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


def test_fixture_rejects_malformed_lock_source_type(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        '[[package]]\nname = "helper"\nsource = [".uv-workspace-siblings/helper"]\n',
        encoding="utf-8",
    )

    result = scan_paths([lock], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


@pytest.mark.parametrize(
    "source_value",
    [
        "editable = true",
        "virtual = true",
    ],
)
def test_fixture_rejects_malformed_lock_source_scalar_types(
    tmp_path: Path, source_value: str
) -> None:
    contract = _contract(tmp_path)
    lock = tmp_path / "consumer" / "uv.lock"
    lock.parent.mkdir()
    lock.write_text(
        "[[package]]\n"
        'name = "helper"\n'
        f"source = {{ {source_value} }}\n\n"
        "[[package]]\n"
        'name = "pydantic-ai-slim"\n'
        'version = "2.29.0"\n',
        encoding="utf-8",
    )

    result = scan_paths([lock], contract_source=contract)

    assert not result.ok
    assert any(finding.kind == "malformed-source" for finding in result.findings)


def test_fixture_bounds_files_and_output(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    locks: list[Path] = []
    for index in range(3):
        path = tmp_path / f"consumer-{index}" / "uv.lock"
        path.parent.mkdir()
        path.write_text(_lock("2.28.0"), encoding="utf-8")
        locks.append(path)

    result = scan_paths(locks, contract_source=contract, max_files=2, max_findings=1)

    assert result.truncated_files == 1
    assert result.truncated_findings >= 1
    assert len(result.findings) == 1


def test_fixture_rejects_malformed_pyproject(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        '[project\ndependencies = ["pydantic-ai-slim"]\n', encoding="utf-8"
    )

    result = scan_paths([manifest], contract_source=contract)

    assert not result.ok
    assert [finding.kind for finding in result.findings] == ["invalid-manifest"]


def test_real_au_webui_and_agent_manifest_paths_are_scannable() -> None:
    """Exercise representative checked-out paths without regenerating locks."""
    au_root = Path(__file__).resolve().parents[3]
    paths = [au_root / "pyproject.toml", au_root / "uv.lock"]
    try:
        worktrees = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=au_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        worktrees = []
    for line in worktrees:
        if not line.startswith("worktree "):
            continue
        candidate_root = Path(line.removeprefix("worktree "))
        if candidate_root.name != "agent-utilities":
            continue
        workspace = candidate_root.parent
        for candidate in (
            workspace / "agent-webui/pyproject.toml",
            workspace / "agents/ansible-tower-mcp/uv.lock",
        ):
            if candidate.is_file():
                paths.append(candidate)
        if len(paths) > 2:
            break

    result = scan_paths(paths)

    assert result.expected_version == read_contract_version()
    assert result.files_scanned == len(paths)
    assert all(
        finding.kind not in {"unreadable", "file-too-large"}
        for finding in result.findings
    )
