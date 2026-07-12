"""CONCEPT:AU-OS.observability.run-wide-correlation-id — Unified dev-lifecycle CLI tests (namespace isolation + run-scoped token mint)."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pytest

from agent_utilities import cli
from agent_utilities.security.run_token import decode_token, validate_token

pytestmark = pytest.mark.concept(id="AU-OS.observability.run-wide-correlation-id")


def test_runtime_dir_isolated_by_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_UTILITIES_RUNTIME_DIR", str(tmp_path))
    a = cli.runtime_dir("alpha")
    b = cli.runtime_dir("beta")
    assert a != b
    assert a.name == "alpha" and b.name == "beta"


def test_status_reports_components(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_UTILITIES_RUNTIME_DIR", str(tmp_path))
    st = cli.status("default")
    assert st["namespace"] == "default"
    assert set(st["components"]) == set(cli.COMPONENTS)
    assert all(
        c["running"] is False for c in st["components"].values()
    )  # nothing started


def test_run_mints_scoped_token(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_UTILITIES_RUNTIME_DIR", str(tmp_path))
    out = cli.run("ns1", "researcher", "find X", project="proj")
    decoded = decode_token(out["tool_token"])
    assert decoded.run_id == "run:ns1:researcher"
    assert decoded.project == "proj"
    # token is scoped to the artifact/proxy/runs endpoints, not unlimited
    assert (
        validate_token(out["tool_token"], endpoint="/api/artifacts/*").run_id
        == "run:ns1:researcher"
    )


def test_main_status_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AGENT_UTILITIES_RUNTIME_DIR", str(tmp_path))
    rc = cli.main(["--namespace", "qa", "--json", "status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["namespace"] == "qa"


def test_main_run_emits_token(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AGENT_UTILITIES_RUNTIME_DIR", str(tmp_path))
    rc = cli.main(["--json", "run", "agentA", "do the thing"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "tool_token" in out and out["agent"] == "agentA"


def test_install_skills_imports_current_universal_skills_bridge_path(monkeypatch):
    """Regression: ``_install_skills`` must import ``universal_skills.core.skill_installer``
    directly. The old ``universal_skills.core.skill_installer.scripts`` path (pre
    commit b9d23f77) no longer exists — ``skill_installer`` is now a flat
    backward-compat re-export module, not a package with a ``scripts``
    submodule — so that import always raised ``ModuleNotFoundError`` and
    silently disabled ``agent-utilities install-skills``. This proves the fixed
    import path resolves and is actually used (not the stale one), by
    injecting a fake bridge module at the current path and asserting
    ``_install_skills`` drives it end-to-end instead of falling into the
    "not installed" fallback.
    """
    fake_target = Path("/tmp/fake-claude-skills")
    fake_installer = types.SimpleNamespace(
        TOOL_PATHS={"claude": fake_target},
        detect_present_tools=lambda: {},
        install_skills=lambda *a, **kw: True,
    )
    fake_pkg = types.ModuleType("universal_skills")
    fake_core = types.ModuleType("universal_skills.core")
    fake_core.skill_installer = fake_installer
    monkeypatch.setitem(sys.modules, "universal_skills", fake_pkg)
    monkeypatch.setitem(sys.modules, "universal_skills.core", fake_core)
    monkeypatch.setitem(
        sys.modules, "universal_skills.core.skill_installer", fake_installer
    )

    args = argparse.Namespace(
        tool="claude",
        path=None,
        skills="",
        group=None,
        no_graphs=False,
        force=False,
        symlink=False,
        layer="all",
    )
    out = cli._install_skills(args)
    assert "error" not in out
    assert out["installed"] == {"claude": str(fake_target)}


def test_context_glossary_present():
    from pathlib import Path

    ctx = Path(__file__).resolve().parents[2] / "docs" / "CONTEXT.md"
    text = ctx.read_text()
    # glossary defines the new ubiquitous-language terms for the adapted features
    for term in (
        "Live Artifact",
        "Run-Scoped Token",
        "Pre-Emit Gate",
        "Adapter",
        "Provider Proxy",
    ):
        assert f"**{term}**" in text
