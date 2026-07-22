"""CLI: end-to-end over the fixture tree, static-only (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.skills.fleet_harness.cli import main

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_cli_runs_static_layer_and_writes_both_outputs(tmp_path: Path):
    json_out = tmp_path / "report.json"
    table_out = tmp_path / "report.md"
    rc = main(
        [
            "--root",
            str(_FIXTURES),
            "--json-out",
            str(json_out),
            "--table-out",
            str(table_out),
        ]
    )
    assert (
        rc == 0
    )  # --fail-on-static not requested, so a FAIL-containing run still exits 0
    payload = json.loads(json_out.read_text())
    assert payload["summary"]["total_skills"] > 0
    assert payload["summary"]["static_fail"] > 0
    assert table_out.exists()


def test_cli_fail_on_static_exits_nonzero_when_fixtures_are_broken(tmp_path: Path):
    rc = main(
        [
            "--root",
            str(_FIXTURES),
            "--json-out",
            str(tmp_path / "report.json"),
            "--table-out",
            str(tmp_path / "report.md"),
            "--fail-on-static",
        ]
    )
    assert rc == 1


def test_cli_errors_cleanly_on_missing_root(tmp_path: Path, capsys):
    rc = main(
        [
            "--root",
            str(tmp_path / "nowhere"),
            "--json-out",
            str(tmp_path / "report.json"),
            "--table-out",
            str(tmp_path / "report.md"),
        ]
    )
    assert rc == 2
    assert "no SKILL.md discovered" in capsys.readouterr().err
