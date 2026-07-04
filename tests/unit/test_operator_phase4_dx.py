"""Phase-4 DX: gotchas-in-KG, governed self-deploy, scaffolder/validate scripts.

CONCEPT:AU-KG.ingest.gotcha-feedback-capture (gotchas), OS-5.50 (self-deploy), OS-5.51 (invisible coordination).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agent_utilities.deployment.self_deploy import execute_redeploy, plan_redeploy
from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService
from agent_utilities.knowledge_graph.retrieval.code_context import build_code_context

REPO = Path(__file__).resolve().parents[2]


# ── KG-2.140 gotchas-in-KG ────────────────────────────────────────────────────
class _GotchaBackend:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props


@pytest.mark.concept("AU-KG.ingest.gotcha-feedback-capture")
def test_record_gotcha_pins_node_with_normalized_path():
    backend = _GotchaBackend()
    svc = FeedbackService(backend=backend)
    res = svc.record_correction(
        "gotcha",
        "/au/agent_utilities/mcp/kg_server.py",
        corrected_value="_get_engine() hangs in a one-off host process",
    )
    assert res.applied
    node = next(iter(backend.nodes.values()))
    assert node["type"] == "Gotcha"
    # path normalized off the /au mount
    assert node["path"].startswith(
        "/home/apps/workspace/agent-packages/agent-utilities/"
    )
    assert "hangs" in node["note"]


@pytest.mark.concept("AU-KG.ingest.gotcha-feedback-capture")
def test_code_context_surfaces_gotchas_in_how():
    _CANON = "/home/apps/workspace/agent-packages/agent-utilities/x.py"

    class _Engine:
        def query_cypher(self, cypher, params):
            if "c.name = $tok" in cypher:
                return [
                    {
                        "id": "code:" + _CANON + "::f",
                        "name": "f",
                        "file_path": _CANON,
                        "line": 10,
                        "language": "python",
                        "kind": "function",
                    }
                ]
            if "g:Gotcha" in cypher:
                return [
                    {
                        "note": "regenerate the manifest with PYTHONPATH set",
                        "severity": "warn",
                    }
                ]
            return []

    res = build_code_context(_Engine(), query="how does f work", intent="how")
    assert "gotchas" in res["used_primitives"]
    assert "⚠️ GOTCHA" in res["answer"]
    assert "PYTHONPATH" in res["answer"]


# ── OS-5.51 governed self-deploy ──────────────────────────────────────────────
@pytest.mark.concept("AU-OS.deployment.os-2")
def test_plan_redeploy_is_safe_and_complete():
    plan = plan_redeploy("graph-os")
    assert plan["service"] == "graph-os"
    assert "restart" in plan and "verify" in plan and "rollback" in plan


@pytest.mark.concept("AU-OS.deployment.os-2")
def test_execute_redeploy_dry_run_by_default():
    res = execute_redeploy("graph-os")
    assert res["status"] == "planned" and res["executed"] is False


@pytest.mark.concept("AU-OS.deployment.os-2")
def test_execute_redeploy_confirm_blocks_without_restart_mechanic(monkeypatch):
    # Fake an allowing policy so we reach the restart-mechanic gate.
    import agent_utilities.orchestration.action_policy as ap

    class _Decision:
        decision = "allow"

    monkeypatch.setattr(
        ap,
        "get_action_policy",
        lambda engine=None: type("P", (), {"decide": lambda self, req: _Decision()})(),
    )
    # No restart_fn provided → must refuse (host restart stays human-gated).
    res = execute_redeploy("graph-os", confirm=True)
    assert res["executed"] is False and res["status"] == "blocked"
    # With a restart_fn + healthy check → deploys.
    res2 = execute_redeploy(
        "graph-os", confirm=True, restart_fn=lambda s: None, health_fn=lambda s: True
    )
    assert res2["status"] == "deployed" and res2["healthy"] is True


# ── OS-5.51-dx scripts smoke ──────────────────────────────────────────────────
@pytest.mark.concept("AU-OS.deployment.os-2")
@pytest.mark.parametrize(
    "script",
    ["validate_change.py", "scaffold_graph_action.py", "reserve_concepts_hook.py"],
)
def test_dx_scripts_have_help(script):
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # --help exits 0 for argparse scripts; reserve_concepts_hook has no argparse
    # so it treats --help as a filename and exits 0 (no markers found).
    assert res.returncode == 0
