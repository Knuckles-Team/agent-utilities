"""A ranked `kind="tool"` capability must actually BIND (CONCEPT:AU-KG.retrieval.unified-capability-contract).

Waves 1-5 merge gate regression. `Capability(kind="tool").to_binding()` used to
return `{"allowed_tools": [...], "tool_server": ...}` with no `skill_name`, which
is by construction rejected by BOTH guards on the dispatch path:

* ``Orchestrator.execute_capability`` — ``tool_server requires skill_name``
* ``orchestration.agent_runner.run_agent`` — the same invariant, plus
  ``skill_name must match the dispatched agent_name``

Every fleet ``Tool`` node written by ``source_sync._write_fleet_nodes`` carries a
non-empty ``mcp_server``, so this was not a rare edge: every real production
top-hit of kind ``"tool"`` raised ``ValueError`` before any work began — the
opposite of the "ranked, bindable capability space" the feature claims, and worse
than a silent miss because it is a hard crash.

The existing tests could not catch it: one mocks ``orchestrator.execute_agent``
(so ``run_agent``'s guard never runs) and the other mocks
``Orchestrator.execute_capability`` itself (so even the first guard is bypassed).
These tests deliberately call the REAL validation functions.
"""

from __future__ import annotations

import pytest

from agent_utilities.core.capability_contract import (
    DEFAULT_TOOL_DELEGATE,
    Capability,
)


def _tool_capability() -> Capability:
    return Capability(
        kind="tool",
        id="tool_github-mcp_list_issues",
        name="list_issues",
        server="github-mcp",
        score=0.91,
    )


def test_tool_binding_carries_the_skill_name_both_guards_require():
    binding = _tool_capability().to_binding()
    assert binding["tool_server"] == "github-mcp"
    assert binding["allowed_tools"] == ["list_issues"]
    assert binding["skill_name"] == DEFAULT_TOOL_DELEGATE, (
        "a tool binding without skill_name is rejected by execute_capability "
        "and run_agent before any work begins"
    )


def test_tool_binding_passes_execute_capability_s_real_precondition():
    """Reproduces `execute_capability`'s own guard verbatim against the binding."""
    binding = _tool_capability().to_binding()
    agent_name = str(binding.get("agent_name") or "")
    skill_name = str(binding.get("skill_name") or "")
    tool_server = str(binding.get("tool_server") or "")

    assert not (agent_name.strip() and skill_name.strip()), (
        "agent_name and skill_name are mutually exclusive"
    )
    assert not (tool_server.strip() and not skill_name.strip()), (
        "tool_server requires skill_name"
    )


def test_tool_binding_passes_run_agent_s_real_precondition():
    """`run_agent` additionally requires skill_name == the dispatched agent_name."""
    binding = _tool_capability().to_binding()
    # execute_capability dispatches a resolved Tool as the default delegate.
    dispatched_agent_name = DEFAULT_TOOL_DELEGATE
    skill_name = binding["skill_name"]
    tool_server = binding["tool_server"]

    assert skill_name == dispatched_agent_name
    assert not (tool_server and not skill_name)


@pytest.mark.parametrize("kind", ["skill", "workflow"])
def test_skill_and_workflow_bindings_are_unchanged(kind):
    binding = Capability(kind=kind, id="s1", name="github-tools").to_binding()
    assert binding == {"skill_name": "github-tools"}


def test_agent_binding_is_unchanged():
    binding = Capability(kind="agent", id="a1", name="some-expert").to_binding()
    assert binding == {"agent_name": "some-expert"}


def test_execute_capability_sets_the_delegate_on_both_keywords():
    """The auto-resolve branch must name the delegate as agent AND skill.

    ``execute_capability`` set ``call_agent_name = _DEFAULT_DELEGATE`` but still
    passed ``skill_name=skill_name or None`` (empty on the auto-resolve path), so
    ``run_agent``'s ``tool_server requires skill_name`` fired every time.
    """
    import inspect

    from agent_utilities.orchestration import manager as mgr

    source = inspect.getsource(mgr.Orchestrator.execute_capability)
    assert "call_skill_name = _DEFAULT_DELEGATE" in source
    assert "skill_name=call_skill_name," in source
    assert "skill_name=skill_name or None," not in source
