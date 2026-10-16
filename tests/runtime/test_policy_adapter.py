"""CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating — workspace mutating-action gate wired to the fleet ActionPolicy (OS-5.24)."""

from __future__ import annotations

from agent_utilities.runtime import DevWorkspace, LocalWorkspace, action_policy_gate
from agent_utilities.runtime.events import CmdRunAction, ErrorObservation


class _AuditEngine:
    """Minimal audit sink: an allowing decision is only effect-authorizing when
    its ``ActionDecision`` audit node was durably written (policy receipt)."""

    def __init__(self) -> None:
        self.nodes: dict[str, str] = {}

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = node_type

    def query_cypher(self, query, params=None):
        return []  # empty ledger: nothing rate-limited or blast-radius bound


async def test_default_policy_allows_sandboxed_workspace_cmd():
    # Shipped default policy maps workspace.* to the auto tier (sandbox = boundary).
    from agent_utilities.orchestration.action_policy import ActionPolicy

    engine = _AuditEngine()
    gate = action_policy_gate(ActionPolicy(engine=engine))
    allowed, _ = gate("workspace.cmd")
    assert allowed
    assert "ActionDecision" in engine.nodes.values()  # receipt-backed allow

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
