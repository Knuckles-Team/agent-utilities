"""EH-038: EG's pre-tool risk is advisory -- attached to the verdict, never obeyed."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from agent_utilities import decide
from agent_utilities.security.tool_guard import PermissionPolicy, Rule
from tests.unit.decide.fakes import FakeTransport, abstained, acted, runner


@pytest.fixture
def eg() -> Iterator[FakeTransport]:
    transport = FakeTransport()
    token = decide.use_runner(runner(transport))
    yield transport
    decide._RUNNER.reset(token)


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


def test_an_abstention_attaches_nothing(eg: FakeTransport) -> None:
    eg.answer = abstained()
    decision = PermissionPolicy().decide(None, "delete_repo", {})
    assert decision.verdict == "deny" and decision.advisory is None
