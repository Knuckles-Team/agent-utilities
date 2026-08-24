"""Tests for BackgroundAgentSpawner (CONCEPT:AU-AHE.evaluation.backtest-harness).

Regression coverage for the descriptionription alias typo in
``_check_context_shifts``'s Cypher query.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import MagicMock

from agent_utilities.harness.background_spawner import BackgroundAgentSpawner


def test_check_context_shifts_synthesizes_team_from_description_value():
    """Regression test for the descriptionription alias typo.

    ``_check_context_shifts`` reads ``r.get("description", "")`` off each
    returned row and only synthesizes a response team when that value is
    truthy (``if desc:``). If the Cypher query aliases ``e.description``
    to anything other than ``"description"``, ``desc`` is always empty,
    the truthiness check always fails, and the high-impact-event
    response path silently never fires. This fake backend derives the
    row key from the *actual* query text so the test fails if the alias
    in the source drifts from the key the consumer reads.
    """

    def fake_execute(query, params=None):
        if "MATCH (e:Event)" in query:
            alias_match = re.search(r"e\.description AS (\w+)", query)
            assert alias_match, "query must alias e.description"
            alias = alias_match.group(1)
            return [{"event_id": "evt-1", alias: "Disk usage critical on r820"}]
        # The follow-up "mark resolved" write.
        return []

    mock_backend = MagicMock()
    mock_backend.execute.side_effect = fake_execute

    fake_engine = MagicMock()
    fake_engine.backend = mock_backend

    spawner = BackgroundAgentSpawner(engine=fake_engine)
    spawner.orchestrator = MagicMock()
    mock_team = MagicMock()
    mock_team.team_id = "team-xyz"
    spawner.orchestrator.synthesize_team.return_value = mock_team

    asyncio.run(spawner._check_context_shifts())

    # Proves the description VALUE actually reached the team-synthesis
    # call (as the `query` kwarg), not just that the alias string in
    # the Cypher query changed.
    assert spawner.orchestrator.synthesize_team.call_count == 1
    _, kwargs = spawner.orchestrator.synthesize_team.call_args
    assert kwargs["query"] == "Disk usage critical on r820"
    assert kwargs["domain"] == "background_operations"
