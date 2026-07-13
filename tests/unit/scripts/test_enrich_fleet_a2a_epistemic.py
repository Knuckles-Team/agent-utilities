"""Tests for the fleet-wide a2a.json epistemic enrichment script (WS-4 item 6).

Builds a throwaway ``agents/<pkg>/a2a.json`` layout on disk and proves the
script: appends the epistemic capability additively, is idempotent (a second
pass changes nothing), never touches files that already carry the
capability, defaults to dry-run (no write without ``--apply``), and leaves
every other field of the manifest untouched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "enrich_fleet_a2a_epistemic",
    Path(__file__).resolve().parents[3] / "scripts" / "enrich_fleet_a2a_epistemic.py",
)
enrich = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(enrich)

_BOILERPLATE = {
    "name": "example-agent",
    "type": "agent",
    "version": "0.1.0",
    "description": "Agent package for example",
    "url": "https://github.com/user/example/tree/main",
    "license": "MIT",
    "capabilities": [
        {
            "id": "run_graph_flow",
            "name": "Graph Flow Execution",
            "description": "Execute a workflow through the agent's graph orchestration engine",
        }
    ],
    "tools": [
        {
            "id": "graph-flow",
            "type": "flow",
            "description": "Run complex multi-step workflows via Pydantic-Graph",
        }
    ],
}


def _write(tmp_path: Path, name: str, manifest: dict) -> Path:
    agent_dir = tmp_path / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "a2a.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# pure enrich_manifest()
# --------------------------------------------------------------------------- #


def test_enrich_manifest_appends_epistemic_capability():
    updated, changed = enrich.enrich_manifest(_BOILERPLATE)
    assert changed is True
    ids = [c["id"] for c in updated["capabilities"]]
    assert ids == ["run_graph_flow", "epistemic-answer"]  # additive, order preserved
    # Original capability untouched.
    assert updated["capabilities"][0] == _BOILERPLATE["capabilities"][0]
    # Every non-capabilities field untouched.
    for key in ("name", "type", "version", "description", "url", "license", "tools"):
        assert updated[key] == _BOILERPLATE[key]


def test_enrich_manifest_idempotent():
    once, changed1 = enrich.enrich_manifest(_BOILERPLATE)
    twice, changed2 = enrich.enrich_manifest(once)
    assert changed1 is True
    assert changed2 is False
    assert once == twice


def test_enrich_manifest_creates_capabilities_list_if_absent():
    manifest = {"name": "no-caps-agent"}
    updated, changed = enrich.enrich_manifest(manifest)
    assert changed is True
    assert [c["id"] for c in updated["capabilities"]] == ["epistemic-answer"]


def test_already_enriched_detects_existing_capability():
    already = {
        "capabilities": [{"id": "epistemic-answer", "name": "x", "description": "y"}]
    }
    assert enrich.already_enriched(already) is True
    assert enrich.already_enriched({"capabilities": []}) is False
    assert enrich.already_enriched({}) is False


# --------------------------------------------------------------------------- #
# process_file() / find_a2a_manifests() — filesystem behavior
# --------------------------------------------------------------------------- #


def test_dry_run_default_never_writes(tmp_path):
    path = _write(tmp_path, "scholarx", _BOILERPLATE)
    before = path.read_text(encoding="utf-8")

    line = enrich.process_file(path, apply=False)

    assert line.startswith("WOULD-WRITE")
    assert path.read_text(encoding="utf-8") == before  # unchanged on disk


def test_apply_writes_and_second_run_is_noop(tmp_path):
    path = _write(tmp_path, "scholarx", _BOILERPLATE)

    line1 = enrich.process_file(path, apply=True)
    assert line1.startswith("WROTE")

    written = json.loads(path.read_text(encoding="utf-8"))
    assert "epistemic-answer" in [c["id"] for c in written["capabilities"]]

    line2 = enrich.process_file(path, apply=True)
    assert line2.startswith("OK") and "already enriched" in line2


def test_process_file_skips_invalid_json(tmp_path):
    agent_dir = tmp_path / "broken"
    agent_dir.mkdir()
    path = agent_dir / "a2a.json"
    path.write_text("{not valid json", encoding="utf-8")

    line = enrich.process_file(path, apply=True)
    assert line.startswith("SKIP")


def test_find_a2a_manifests_globs_one_level(tmp_path):
    _write(tmp_path, "agent-a", _BOILERPLATE)
    _write(tmp_path, "agent-b", _BOILERPLATE)
    (tmp_path / "not-an-agent-dir").mkdir()

    found = enrich.find_a2a_manifests(tmp_path)

    assert [p.parent.name for p in found] == ["agent-a", "agent-b"]


def test_main_agents_root_dry_run_reports_without_writing(tmp_path, capsys):
    path = _write(tmp_path, "agent-a", _BOILERPLATE)
    before = path.read_text(encoding="utf-8")

    rc = enrich.main(["--agents-root", str(tmp_path)])

    assert rc == 0
    assert path.read_text(encoding="utf-8") == before
    out = capsys.readouterr().out
    assert "would be updated" in out


def test_main_requires_exactly_one_of_agents_root_or_file(tmp_path):
    with pytest.raises(SystemExit):
        enrich.main([])
    with pytest.raises(SystemExit):
        enrich.main(
            ["--agents-root", str(tmp_path), "--file", str(tmp_path / "a2a.json")]
        )
