"""Tests for the ``workspace_config`` doctor check + its loader-side validator.

The check (CONCEPT:AU-OS.deployment.os-4) validates ``workspace.yml`` through the same loader the
bootstrap/ingestion code uses. These tests build a manifest in ``tmp_path`` (never
the real root file) and assert: a valid manifest passes, a malformed one is flagged
fail, and a structurally-coherent-but-incomplete one warns.
"""

from __future__ import annotations

from pathlib import Path

from agent_utilities.core.workspace_config import validate_workspace_yml
from agent_utilities.deployment import doctor as D

VALID = """\
name: "Test Workspace"
path: "/tmp/ws"
description: "A test workspace."
repositories:
  - url: "https://github.com/Example/root-repo.git"
    description: "root repo"
subdirectories:
  agent-packages:
    description: "core"
    repositories:
      - url: "https://github.com/Example/agent-utilities.git"
        description: "the hub"
    subdirectories:
      skills:
        description: "skills"
        repositories:
          - url: "https://github.com/Example/universal-skills.git"
            description: "skills repo"
graph:
  enabled: true
"""


def _write(tmp_path: Path, body: str) -> str:
    p = tmp_path / "workspace.yml"
    p.write_text(body, encoding="utf-8")
    return str(p)


# ── the pure validator ──────────────────────────────────────────────────────
def test_valid_manifest_validates_clean(tmp_path):
    rep = validate_workspace_yml(_write(tmp_path, VALID))
    assert rep["found"] and rep["parsed"]
    assert rep["errors"] == []
    assert rep["warnings"] == []
    # root + agent-packages + skills = 3 repositories[*].url across the tree.
    assert rep["repo_count"] == 3


def test_missing_url_is_an_error(tmp_path):
    body = """\
name: "W"
path: "/tmp/ws"
description: "d"
repositories:
  - description: "no url here"
"""
    rep = validate_workspace_yml(_write(tmp_path, body))
    assert rep["parsed"]
    assert any("url" in e for e in rep["errors"])


def test_malformed_yaml_is_an_error(tmp_path):
    rep = validate_workspace_yml(_write(tmp_path, "repositories: [unclosed\n"))
    assert rep["found"]
    assert rep["errors"] and not rep["parsed"]


def test_top_level_list_is_an_error(tmp_path):
    rep = validate_workspace_yml(_write(tmp_path, "- a\n- b\n"))
    assert rep["found"] and not rep["parsed"]
    assert any("mapping" in e for e in rep["errors"])


def test_non_mapping_subdirectory_is_an_error(tmp_path):
    body = """\
name: "W"
path: "/tmp/ws"
description: "d"
subdirectories:
  broken: "not a mapping"
"""
    rep = validate_workspace_yml(_write(tmp_path, body))
    assert any("broken" in e for e in rep["errors"])


def test_missing_optionals_warn_not_error(tmp_path):
    body = """\
repositories:
  - url: "https://github.com/Example/r.git"
"""
    rep = validate_workspace_yml(_write(tmp_path, body))
    assert rep["errors"] == []  # url is present + well-formed → no errors
    # missing name/description/path + repo missing description → advisory warnings
    assert rep["warnings"]


def test_missing_file_is_not_found(tmp_path):
    rep = validate_workspace_yml(str(tmp_path / "does-not-exist.yml"))
    assert rep["found"] is False
    assert rep["errors"] == []


# ── the doctor check (reuses the validator) ─────────────────────────────────
def test_doctor_check_ok_for_valid(tmp_path, monkeypatch):
    path = _write(tmp_path, VALID)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.coverage.find_workspace_manifest",
        lambda: Path(path),
    )
    res = D._check_workspace_config()
    assert res["name"] == "workspace_config"
    assert res["status"] == "ok"
    assert "3 repositories" in res["detail"]


def test_doctor_check_fail_for_malformed(tmp_path, monkeypatch):
    body = """\
name: "W"
path: "/tmp/ws"
repositories:
  - description: "no url"
"""
    path = _write(tmp_path, body)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.coverage.find_workspace_manifest",
        lambda: Path(path),
    )
    res = D._check_workspace_config()
    assert res["status"] == "fail"
    assert res["remediation"]


def test_doctor_check_skips_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.coverage.find_workspace_manifest",
        lambda: None,
    )
    # No XDG file expected in the test sandbox either; force the fallback to miss.
    monkeypatch.setattr(
        "agent_utilities.core.workspace_config.get_workspace_yml_path",
        lambda: tmp_path / "nope.yml",
    )
    res = D._check_workspace_config()
    assert res["status"] == "skip"


def test_workspace_config_in_registry():
    assert "workspace_config" in D.CHECKS
    assert D.CHECKS["workspace_config"] is D._check_workspace_config
