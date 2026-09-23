"""EH-041/042/043: connector decisions are evaluate-only and never authorize."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from agent_utilities import decide
from agent_utilities.decide.consumers.connectors import (
    connector_tool,
    propose_writeback,
    triage_playbook,
)
from tests.unit.decide.fakes import FakeTransport, abstained, acted, runner

PLAYBOOKS = {
    "default": object(),
    "servicenow": object(),
    "servicenow:critical": object(),
}


@pytest.fixture
def eg() -> Iterator[FakeTransport]:
    transport = FakeTransport()
    token = decide.use_runner(runner(transport))
    yield transport
    decide._RUNNER.reset(token)


def test_triage_decides_among_registered_playbooks(eg: FakeTransport) -> None:
    eg.answer = acted("servicenow", digest="sha256:" + "0" * 63 + "2")
    assert triage_playbook("servicenow", "critical", PLAYBOOKS) == "servicenow"
    eg.answer = abstained(digest="sha256:" + "0" * 63 + "1")
    assert triage_playbook("servicenow", "critical", PLAYBOOKS) == "servicenow:critical"
    assert eg.op_names() == [], "evaluate-only: outside the sample nothing is logged"


def test_connector_tool_choice_falls_back_to_the_connector_s_pick(
    eg: FakeTransport,
) -> None:
    eg.answer = abstained()
    assert connector_tool("jira", ["search", "get"], "get") == "get"
    eg.answer = acted("search")
    assert connector_tool("jira", ["search", "get"], "get") == "search"


def test_a_writeback_is_only_ever_proposed(eg: FakeTransport) -> None:
    proposals = {"close-incident": {"server": "servicenow-mcp", "tool": "t"}}
    eg.answer = acted("close-incident")
    assert propose_writeback(proposals) == (
        "close-incident",
        proposals["close-incident"],
    )
    eg.answer = abstained()
    assert propose_writeback(proposals) == ("no_write", None)
    assert eg.requests[-1]["question"]["safety"] == "write_back"
