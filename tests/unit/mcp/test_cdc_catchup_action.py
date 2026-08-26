"""``graph_ingest(action="cdc_catchup")`` is a live caller of the CA-21
envelope-source registry (CONCEPT:AU-KG.ingest.debezium-changeenvelope).

Proves the wiring end to end at the MCP dispatch layer: the action looks up
``debezium_envelope.get_envelope_source("cdc")`` and invokes it with the
resolved engine — the SAME registered handler
``source_sync._DELTA_HANDLERS["cdc"]`` will call once CA-22 lands (ordered
CA-21 -> CA-22), so this action and that future path never drift. The REST
twin is automatic: ``graph_ingest`` already has a ``/graph/ingest`` entry in
``kg_server.ACTION_TOOL_ROUTES`` (the single tool<->REST parity map every
other action-routed tool goes through), so no separate REST route is needed
or added here.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.ingestion import debezium_envelope
from agent_utilities.mcp import kg_server
from agent_utilities.security.brain_context import ActorContext, ActorType


def _register_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.write_ingest_tools import register_write_ingest_tools

    mcp = FastMCP("test")
    register_write_ingest_tools(mcp)


def _session() -> GraphSession:
    actor = ActorContext(
        actor_id="test-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="test-tenant",
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy",
        audience="test-audience",
    )


@pytest.fixture(autouse=True)
def _reset_state():
    saved_session = kg_server._PROCESS_SESSION
    kg_server._PROCESS_SESSION = _session()
    yield
    kg_server._PROCESS_SESSION = saved_session


async def test_cdc_catchup_dispatches_the_registered_cdc_handler_with_the_live_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_tools()
    sentinel_engine = object()
    # cdc_catchup carries no graph/connection selector (there is exactly one
    # CDC consumer, not a per-graph fan-out choice), so graph_ingest resolves
    # the engine through kg_server._get_engine()'s plain default path rather
    # than the explicit-graph registry _resolve_target_engines/
    # resolve_explicit_graph consult — patch that same function directly.
    monkeypatch.setattr(kg_server, "_get_engine", lambda: sentinel_engine)

    calls: list[dict] = []

    def _fake_handler(engine, *, mode, ids, client):
        calls.append({"engine": engine, "mode": mode, "ids": ids, "client": client})
        return {"status": "ok", "counts": {"succeeded": 1}}

    monkeypatch.setattr(debezium_envelope, "run_cdc_catchup", _fake_handler)
    monkeypatch.setitem(debezium_envelope._ENVELOPE_SOURCES, "cdc", _fake_handler)

    out = await kg_server._execute_tool("graph_ingest", action="cdc_catchup")
    payload = json.loads(out)

    assert payload == {"status": "ok", "counts": {"succeeded": 1}}
    assert len(calls) == 1
    assert calls[0]["engine"] is sentinel_engine
    assert calls[0]["mode"] == "delta"


async def test_cdc_catchup_full_mode_is_threaded_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())

    calls: list[dict] = []
    monkeypatch.setitem(
        debezium_envelope._ENVELOPE_SOURCES,
        "cdc",
        lambda engine, *, mode, ids, client: calls.append(mode) or {"status": "ok"},
    )

    await kg_server._execute_tool(
        "graph_ingest", action="cdc_catchup", corpus_name="full"
    )
    assert calls == ["full"]


def test_graph_ingest_has_a_rest_twin_for_cdc_catchup_to_ride() -> None:
    """cdc_catchup needs no separate REST route: graph_ingest's existing
    single tool<->REST parity entry already covers it."""
    assert kg_server.ACTION_TOOL_ROUTES["graph_ingest"] == "/graph/ingest"


async def test_cdc_catchup_reaches_the_real_debezium_flag_check_not_a_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine wiring test (CONCEPT:AU-AHE.evaluation.live-path-probe):
    drives the REAL MCP entrypoint through the REAL registered "cdc" handler
    (``debezium_envelope.run_cdc_catchup`` itself — nothing patched here) and
    observes the REAL ``kafka_adapter.debezium_consumer_enabled`` seam it
    calls. This proves entrypoint -> registry lookup -> run_cdc_catchup ->
    kafka_adapter is one live, reachable edge — not three separately-unit-
    tested islands whose connection is only asserted by a monkeypatched
    stand-in (the other tests in this file, which prove the DISPATCH shape
    with a fake handler, cannot prove this: they replace the very seam this
    test observes).
    """
    from agent_utilities.knowledge_graph.streams import kafka_adapter
    from tests.wiring import observe

    _register_tools()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: object())
    monkeypatch.delenv("KAFKA_CDC_DEBEZIUM_ENABLED", raising=False)

    with observe(kafka_adapter, "debezium_consumer_enabled") as seam:
        out = await kg_server._execute_tool("graph_ingest", action="cdc_catchup")

    seam.assert_called(
        why="graph_ingest(action='cdc_catchup') must reach the REAL "
        "debezium_envelope.run_cdc_catchup, which real-checks the "
        "KAFKA_CDC_DEBEZIUM_ENABLED flag before ever constructing a Kafka "
        "consumer"
    )
    payload = json.loads(out)
    # Flag unset -> the REAL run_cdc_catchup short-circuits with no Kafka
    # I/O attempted (proven by the seam call above returning False and
    # nothing downstream of it running) -- this is CA-21-W05's rollback
    # contract exercised through the live entrypoint, not asserted in
    # isolation.
    assert payload["status"] == "skipped"
    assert "KAFKA_CDC_DEBEZIUM_ENABLED" in payload["reason"]
