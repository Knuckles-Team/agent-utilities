"""EG's pre-tool risk is advisory -- attached to the verdict, never obeyed."""

from __future__ import annotations

import pytest

from agent_utilities.security.tool_guard import PermissionPolicy, Rule
from tests.unit.decide.fakes import FakeTransport, abstained, acted


@pytest.mark.spec("AU-CONTROL-R005")
def test_eg_s_risk_verdict_is_attached_but_never_changes_the_decision(
    eg: FakeTransport,
) -> None:
    eg.answer = acted("deny")
    decision = PermissionPolicy(rules=[Rule("allow", tool="*")]).decide(
        None, "read_file", {}
    )
    assert decision.verdict == "allow", "a score never removes authority"
    assert decision.advisory is not None and decision.advisory.startswith("deny")
    assert eg.requests[0]["question"]["safety"] == "security"


@pytest.mark.spec("AU-CONTROL-R005")
def test_an_abstention_attaches_nothing(eg: FakeTransport) -> None:
    eg.answer = abstained()
    decision = PermissionPolicy().decide(None, "delete_repo", {})
    assert decision.verdict == "deny" and decision.advisory is None
