"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating — workspace mutating-action gate wired to the fleet ActionPolicy (OS-5.24)."""

from __future__ import annotations

from agent_utilities.runtime import DevWorkspace, LocalWorkspace, action_policy_gate
from agent_utilities.runtime.events import CmdRunAction, ErrorObservation


def _active_engine(monkeypatch) -> None:
    """An active process engine the policy can audit its decision on."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from tests.unit.fleet_autonomy_fakes import FakeEngine

    engine = FakeEngine()
    monkeypatch.setattr(IntelligenceGraphEngine, "get_active", lambda: engine)


async def test_default_policy_allows_sandboxed_workspace_cmd(monkeypatch):
    # Shipped default policy maps workspace.* to the auto tier (sandbox = boundary).
    _active_engine(monkeypatch)
    gate = action_policy_gate()
    allowed, _ = gate("workspace.cmd")
    assert allowed

    ws = DevWorkspace(LocalWorkspace(), run_id="pol1", policy_gate=gate)
    async with ws:
        obs = await ws.act(CmdRunAction(command="echo ok"))
    assert obs.kind == "cmd_output"
    assert "ok" in obs.stdout


async def test_operator_override_can_forbid_shell(tmp_path):
    policy_file = tmp_path / "policy.yml"
    policy_file.write_text(
        "version: 1\n"
        "defaults: {tier: approval_required}\n"
        "rules:\n"
        "  - {kind: workspace.cmd, target: '*', tier: forbidden}\n"
        "  - {kind: workspace.write, target: '*', tier: auto}\n"
    )
    from agent_utilities.orchestration.action_policy import ActionPolicy

    gate = action_policy_gate(ActionPolicy(policy_path=str(policy_file)))
    allowed, _ = gate("workspace.cmd")
    assert not allowed

    ws = DevWorkspace(LocalWorkspace(), run_id="pol2", policy_gate=gate)
    async with ws:
        denied = await ws.act(CmdRunAction(command="echo nope"))
        assert isinstance(denied, ErrorObservation)
        assert "denied by policy" in denied.message


def test_default_gate_fails_closed_without_an_active_engine(monkeypatch):
    """No engine means no audited receipt, so even the auto tier cannot act."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    monkeypatch.setattr(IntelligenceGraphEngine, "get_active", lambda: None)
    allowed, reason = action_policy_gate()("workspace.cmd")
    assert not allowed
    assert "receipt unavailable" in reason


def test_default_gate_binds_the_engine_active_at_decision_time(monkeypatch):
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    gate = action_policy_gate()
    monkeypatch.setattr(IntelligenceGraphEngine, "get_active", lambda: None)
    assert gate("workspace.cmd")[0] is False
    _active_engine(monkeypatch)
    assert gate("workspace.cmd")[0] is True
