"""CONCEPT:OS-5.11 — Unified dev-lifecycle CLI tests (namespace isolation + run-scoped token mint)."""

from __future__ import annotations

import json

import pytest

from agent_utilities import cli
from agent_utilities.security.run_token import decode_token, validate_token

pytestmark = pytest.mark.concept(id="OS-5.11")


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
