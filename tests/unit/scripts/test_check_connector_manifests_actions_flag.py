"""CA-32/DEC-CA-07 reachability proof: the ``--check-actions`` flag on the
real ``scripts/check_connector_manifests.py`` CLI entrypoint (the operator
surface for the connector-manifest gate; the `undeclared_mutating_tools`
mechanism has no other pre-existing REST/MCP surface to hang from — see the
CA-32 lane file Non-goals, which leave the approval/executor wiring to
CA-22/CA-28) actually reaches :func:`undeclared_mutating_tools`, end to end,
via a real subprocess invocation of the script — not merely an importable,
unit-tested-only function.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "check_connector_manifests.py"


def _write_pkg_with_mutating_tool(tmp_path: Path, pkg: str) -> Path:
    pkg_root = tmp_path / pkg
    pkg_root.mkdir(parents=True)
    (pkg_root / "mcp_server.py").write_text(
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('x')\n\n"
        "@mcp.tool(tags={'widgets', 'mutating'})\n"
        "async def delete_widget(widget_id: str) -> dict:\n"
        "    return {}\n",
        encoding="utf-8",
    )
    return pkg_root


def _write_minimal_manifest(pkg_root: Path, connector: str) -> Path:
    from agent_utilities.knowledge_graph.ontology.connector_manifest import (
        ConnectorManifest,
        IntegrityInfo,
        ProvenanceSpec,
    )

    manifest = ConnectorManifest(
        connector=connector,
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="0" * 64)),
    )
    path = pkg_root / "connector_manifest.yml"
    path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_check_actions_flag_fails_closed_on_undeclared_mutating_tool(
    tmp_path: Path,
):
    pkg_root = _write_pkg_with_mutating_tool(tmp_path, "widget-mcp")
    manifest_path = _write_minimal_manifest(pkg_root, "widget-mcp")

    result = _run_cli(
        "--manifest",
        str(manifest_path),
        "--agents-root",
        str(tmp_path),
        "--check-actions",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "delete_widget" in result.stdout
    assert "[actions]" in result.stdout


def test_cli_without_check_actions_flag_ignores_the_undeclared_tool(tmp_path: Path):
    """Same fixture, flag omitted: the run may still fail on unrelated,
    pre-existing checks this synthetic manifest doesn't satisfy (integrity
    hash, anti-sprawl `owl:imports`) — those are real, orthogonal gate rules
    this test isn't exercising. The only thing this proves is that omitting
    `--check-actions` never emits an `[actions]` violation for the same
    undeclared-mutating-tool fixture the flag-on test above DOES catch."""
    pkg_root = _write_pkg_with_mutating_tool(tmp_path, "widget-mcp")
    manifest_path = _write_minimal_manifest(pkg_root, "widget-mcp")

    result = _run_cli(
        "--manifest",
        str(manifest_path),
        "--agents-root",
        str(tmp_path),
    )
    assert "[actions]" not in result.stdout
