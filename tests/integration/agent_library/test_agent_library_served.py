"""Agent Library and WorkItem served-Python coverage (EH-249 / PA-08).

Drives the REAL ``epistemic-graph-server`` (the session ``tiny_engine``) over
the authenticated wire, the way AU's control plane reaches it.

Bodies are sent in the server's canonical form: every field, in Rust
declaration order, defaults included. The ``eg2.`` envelope MAC covers the
server's re-serialization of the typed ``Method``. At EG ``b43f33569`` the
generated ``send_agent_component_*`` senders emit ``exclude_none`` pydantic
dumps and fail that MAC ("Authentication failed"). That defect is filed with
EG (au-core CONTRACT-REQUEST R5), and these tests prove the served behaviour
independently of it.

The engine binds every authority field of the mutation context from the
verified carrier (``bind_agent_library_context``), so the request carries
well-formed placeholders. The assertions check that the engine replaced them.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.integration

_PLACEHOLDER = "caller-supplied-placeholder"
_PLACEHOLDER_PRINCIPAL = "principal:sha256:" + "a" * 64
_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64


def _context(tenant: str, expected_revision: int) -> dict[str, Any]:
    """``AgentLibraryMutationContext`` in declaration order."""
    return {
        "request_id": 0,
        "principal": _PLACEHOLDER_PRINCIPAL,
        "caller_principal": _PLACEHOLDER_PRINCIPAL,
        "attempt_nonce": "0" * 64,
        "tenant_id": tenant,
        "actor_scope": _PLACEHOLDER,
        "purpose_id": _PLACEHOLDER,
        "policy_revision": _PLACEHOLDER,
        "policy_digest": _PLACEHOLDER_DIGEST,
        "policy_decision_id": _PLACEHOLDER,
        "idempotency_key": _PLACEHOLDER,
        "expected_revision": expected_revision,
        "trace_id": None,
        "created_at_ms": 0,
    }


def _draft(tenant: str, component_id: str, kind: str, name: str) -> dict[str, Any]:
    """``AgentComponentDraft`` in declaration order."""
    return {
        "component_id": component_id,
        "kind": kind,
        "version": "1.0.0",
        "content_digest": "sha256:" + "1" * 64,
        "content_ref": None,
        "facts": {"facts": "opaque"},
        "provenance": {"origin": "native"},
        "summary": f"served coverage component {name}",
        "classification": [],
        "requires": [],
        "provides": [],
        "declared_capabilities": [],
        "required_capabilities": [],
        "declared_required_capabilities": [],
        "attributes": {"name": name},
        "tenant_id": tenant,
        "actor_scope": _PLACEHOLDER,
        "purpose_id": _PLACEHOLDER,
        "policy_digest": _PLACEHOLDER_DIGEST,
        "source_revision": "rev-1",
        "source_revision_digest": "sha256:" + "8" * 64,
    }


def _search(tenant: str, kinds: list[str], cursor: str | None = None) -> dict:
    """``AgentComponentSearchRequest`` in declaration order."""
    return {
        "op": "search",
        "request": {
            "tenant_id": tenant,
            "task": None,
            "capabilities": [],
            "kinds": kinds,
            "read_only": False,
            "limit": 64,
            "cursor": cursor,
        },
    }


class _Library:
    def __init__(self, client: Any, tenant: str) -> None:
        self.client = client
        self.tenant = tenant

    async def op(self, op: dict[str, Any], *, write: bool = False) -> Any:
        key = f"al:{uuid.uuid4().hex}" if write else None
        return await self.client._send(
            "AgentComponent", {"op": op}, None, idempotency_key=key
        )

    async def publish(self, draft: dict[str, Any]) -> Any:
        request = {
            "context": _context(self.tenant, 0),
            "component": draft,
            "evaluation_receipt_digest": None,
        }
        return await self.op({"op": "publish", "request": request}, write=True)

    async def retire(self, component_id: str, revision: int) -> Any:
        request = {
            "context": _context(self.tenant, revision),
            "component_id": component_id,
        }
        return await self.op({"op": "retire", "request": request}, write=True)

    async def read(self, verb: str, component_id: str, tenant: str | None = None):
        op = {
            "op": verb,
            "tenant_id": tenant or self.tenant,
            "component_id": component_id,
        }
        return await self.op(op)

    async def search_page(self, kinds: list[str]) -> Any:
        from epistemic_graph.generated.agent_component import AgentComponentSearchPage

        page = await self.op(_search(self.tenant, kinds))
        return AgentComponentSearchPage.model_validate(page)


async def _connect(socket_path: str) -> Any:
    from _test_engine import TEST_AUTH_SECRET, request_context
    from epistemic_graph.client import EpistemicGraphClient

    return await EpistemicGraphClient.connect(
        socket_path=socket_path,
        auth_secret=TEST_AUTH_SECRET,
        verified_context=request_context(),
    )


def test_agent_library_publish_search_retire_history_served(tiny_engine) -> None:
    from _test_engine import TEST_TENANT

    from agent_utilities.api.agent_control_adapters import candidates_from_entries

    suffix = uuid.uuid4().hex[:10]
    skill_id, card_id = f"skill:served-{suffix}", f"a2a:served-{suffix}"
    kinds = ["skill", "a2a_agent_card"]

    async def scenario() -> None:
        library = _Library(await _connect(tiny_engine), TEST_TENANT)
        await library.publish(_draft(TEST_TENANT, skill_id, "skill", "pr-review"))
        await library.publish(_draft(TEST_TENANT, card_id, "a2a_agent_card", "expert"))

        current = await library.read("current", skill_id)
        assert current["lifecycle"] == "published"
        assert current["tenant_id"] == TEST_TENANT
        # Authority fields are bound by the engine, never taken from the body.
        assert current["actor_scope"] != _PLACEHOLDER
        assert current["policy_digest"] != _PLACEHOLDER_DIGEST
        assert current["purpose_id"] == "agent-component:publish"

        page = await library.search_page(kinds)
        projected = {c.component_id: c for c in candidates_from_entries(page.entries)}
        assert projected[skill_id].kind == "skill"
        assert projected[skill_id].name == "pr-review"
        assert projected[card_id].kind == "agent"

        await library.retire(skill_id, int(current["entry_revision"]))
        retired = await library.read("current", skill_id)
        assert retired["lifecycle"] == "retired"
        remaining = {e.component_id for e in (await library.search_page(kinds)).entries}
        assert skill_id not in remaining
        assert card_id in remaining
        history = await library.read("history", skill_id)
        assert [row["lifecycle"] for row in history][-1] == "retired"
        assert len(history) >= 2

    asyncio.run(scenario())


def test_agent_library_refuses_cross_tenant_bodies_served(tiny_engine) -> None:
    from _test_engine import TEST_TENANT

    other = "tenant:someone-else"
    component_id = f"skill:foreign-{uuid.uuid4().hex[:10]}"

    async def scenario() -> None:
        library = _Library(await _connect(tiny_engine), other)
        with pytest.raises(RuntimeError, match="ACCESS_DENIED"):
            await library.publish(_draft(other, component_id, "skill", "x"))
        with pytest.raises(RuntimeError, match="ACCESS_DENIED"):
            await library.read("current", component_id)
        assert await library.read("current", component_id, TEST_TENANT) is None

    asyncio.run(scenario())


@pytest.fixture()
def process_engine_on_graph(engine_graph: Any):
    """AU's process engine bound to the fixture graph, with ``__control__`` ready.

    Mirrors ``test_adopt_a_native_work_item_acceptance``: production startup
    materializes the control graph before queue traffic.
    """
    from agent_utilities.api import current_session
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
    from agent_utilities.knowledge_graph.core.shard_topology import CONTROL_GRAPH_NAME

    if IntelligenceGraphEngine.get_active() is not None:
        pytest.fail("a process-owned IntelligenceGraphEngine is already active")
    session = current_session()
    assert session is not None
    GraphComputeEngine._ensure_local_session_graph(
        engine_graph._client, CONTROL_GRAPH_NAME, session
    )
    engine = IntelligenceGraphEngine(
        backend=EpistemicGraphBackend(graph_name=engine_graph.graph_name),
        defer_background_start=True,
    )
    try:
        yield engine, engine_graph.for_graph(CONTROL_GRAPH_NAME).client
    finally:
        IntelligenceGraphEngine._set_active_for_tests(None)


def test_worker_reads_and_fences_an_eg_native_submitted_work_item(
    process_engine_on_graph: Any,
) -> None:
    """The hosted dispatch path's worker contract on the real engine.

    ``EgWorkItemStore`` admits through EG's native ``SubmitWorkItem`` into the
    ``__control__`` WorkItem graph (as the hosted plane binds it). The AU
    dispatch worker then reads the sanitized task, claims the row, marks it
    running and renews its lease through ``work_durability``
    (``dispatch_work_item_id`` makes the row its own fence).
    """
    from agent_utilities.api import WorkItemSubmission, current_session
    from agent_utilities.api.agent_control_adapters import EgWorkItemStore
    from agent_utilities.knowledge_graph.core import work_durability as wi
    from agent_utilities.knowledge_graph.core.shard_topology import CONTROL_GRAPH_NAME
    from agent_utilities.orchestration.agent_dispatch_worker import (
        _admitted_task_description,
    )

    engine, control_client = process_engine_on_graph
    session = current_session()
    assert session is not None
    store = EgWorkItemStore(
        control_client,
        graph=CONTROL_GRAPH_NAME,
        authentication_method="workload_identity",
        policy_digest="policy:served",
        catalog_digest="catalog:served",
        model_digest="model:served",
    )
    item_id = f"wi:served-{uuid.uuid4().hex[:12]}"
    submission = WorkItemSubmission(
        work_item_id=item_id,
        idempotency_key=f"idem-{item_id}",
        kind="orchestrator_task",
        description="summarize the release notes for alice@example.com",
        metadata={"team": "red"},
    )
    from agent_utilities.api import use_session

    work_session = session.with_graph(CONTROL_GRAPH_NAME)
    body, description = store._submit_request(submission, work_session)
    with use_session(work_session):
        admitted = control_client.work_items.submit(body)
    assert admitted["created"] is True
    assert admitted["work_item_id"] == item_id

    row = wi.get_work_item(engine, item_id)
    assert row is not None, "EG-native WorkItem row is invisible to the AU worker"
    assert row["tenant"] == session.tenant
    assert _admitted_task_description(engine, item_id) == description
    assert "alice@example.com" not in description

    claim = wi.claim_specific(engine, item_id, token="worker:served")
    assert claim is not None, "the AU worker could not claim the EG-native item"
    assert wi.mark_running(engine, item_id, claim)
    assert wi.heartbeat(engine, item_id, claim), "the worker cannot renew its lease"
    running = wi.get_work_item(engine, item_id)
    assert running is not None and running["status"] == "running"
    # The terminal commit leg (``CommitWorkItemResult``) currently fails the
    # ``eg2.`` MAC for every caller at EG b43f33569; the pre-existing
    # ``test_adopt_a_native_work_item_acceptance`` reproduces it. Filed as EG
    # CONTRACT-REQUEST R5; the commit assertion belongs there, not here.
