"""Native WorkItem contracts for the unified long-running Loop model."""

from __future__ import annotations

from contextlib import contextmanager

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.research.loops import (
    TERMINAL_STATUS,
    active_loops,
    claim_loop,
    mark_loop_status,
    prioritize_loop,
    submit_loop,
)
from agent_utilities.orchestration import work_item as wi
from agent_utilities.security.brain_context import ActorContext, ActorType
from tests.unit.orchestration.test_work_item import NativeEngine


class LoopEngine(NativeEngine):
    """Native WorkItem double plus the two bounded Concept intake queries."""

    def __init__(self, addressed: set[str] | None = None) -> None:
        super().__init__()
        self.addressed = set(addressed or ())

    def query_cypher(self, cypher, params=None):
        query = " ".join(cypher.split())
        if "ADDRESSED_BY" in query:
            return [{"id": node_id} for node_id in sorted(self.addressed)]
        if query.startswith("MATCH (c:Concept) RETURN"):
            rows = [
                {"id": node_id, **node}
                for node_id, node in self.nodes.items()
                if node.get("label") == "Concept"
            ]
            return rows[: int((params or {}).get("limit", len(rows)))]
        return super().query_cypher(cypher, params)


def _session() -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="opaque-subject",
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id="tenant-a",
            authenticated=True,
        ),
        tenant="tenant-a",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="tenant-a",
    )


@contextmanager
def _authority():
    with use_session(_session()):
        yield


def _submit(engine: LoopEngine, objective: str, **kwargs):
    with _authority():
        return submit_loop(engine, objective, **kwargs)


def test_submit_loop_keeps_definition_immutable_and_work_item_authoritative():
    engine = LoopEngine()
    loop = _submit(
        engine,
        "Make the build pass",
        kind="develop",
        validation_cmd="pytest -q",
    )

    definition = engine.nodes[loop["id"]]
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert definition["loop_kind"] == "develop"
    assert definition["validation_cmd"] == "pytest -q"
    assert "status" not in definition
    assert item is not None and item["status"] == "ready"
    assert loop["status"] == "submitted"


def test_active_loops_reads_status_only_from_work_items():
    engine = LoopEngine(addressed={"loop:research:done"})
    research = _submit(
        engine, "open question", loop_id="loop:research:active", kind="research"
    )
    _submit(engine, "answered", loop_id="loop:research:done", kind="research")
    develop = _submit(
        engine,
        "build",
        loop_id="loop:develop:active",
        kind="develop",
        validation_cmd="pytest",
    )
    finished = _submit(engine, "finished", loop_id="loop:develop:done", kind="develop")
    with _authority():
        assert claim_loop(engine, finished["id"])
        assert mark_loop_status(engine, finished["id"], "completed")

    with _authority():
        rows = {loop["id"]: loop for loop in active_loops(engine, limit=10)}
    assert research["id"] in rows
    assert develop["id"] in rows
    assert "loop:research:done" not in rows
    assert finished["id"] not in rows
    assert rows[develop["id"]]["status"] == "ready"


def test_claim_loop_uses_native_lease_and_lost_race_is_final():
    engine = LoopEngine()
    loop = _submit(engine, "one owner", kind="develop")
    with _authority():
        assert claim_loop(engine, loop["id"]) is True
        assert claim_loop(engine, loop["id"]) is False
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert item is not None and item["status"] == "leased"
    assert engine.native_calls == ["claim", "claim"]


def test_priority_updates_only_the_work_item_and_orders_intake():
    engine = LoopEngine()
    background = _submit(
        engine,
        "background",
        loop_id="loop:research:background",
        prio_bucket=3,
    )
    critical = _submit(
        engine,
        "critical",
        loop_id="loop:research:critical",
        prio_bucket=0,
    )
    with _authority():
        assert prioritize_loop(engine, background["id"], 1)
        ordered = active_loops(engine, limit=10)
    assert [row["id"] for row in ordered] == [critical["id"], background["id"]]
    assert "status" not in engine.nodes[background["id"]]


def test_terminal_status_is_definition_neutral():
    assert "completed" in TERMINAL_STATUS
    assert "running" not in TERMINAL_STATUS
