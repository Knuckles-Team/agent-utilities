"""Exercise reviewed action effects through the owning catalog generator."""

from types import SimpleNamespace

from scripts.gen_capability_power import build_cpd


def generated_action_items(tool_name: str, actions: list[str]) -> dict:
    descriptor = build_cpd(
        tool=SimpleNamespace(name=tool_name, tags=set(), parameters={}, description=""),
        action_tool_routes={},
        ledger={},
        ledger_path=None,
        ledger_live=False,
        manifest_by_tool={tool_name: actions},
        engine_domains={},
        mining_actions=(),
        graphlearn_actions=(),
        deep_mining_actions=(),
        generated_at="2026-01-01T00:00:00Z",
    )
    return {item["action"]: item for item in descriptor.does}
