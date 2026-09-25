"""Concrete EG/AU adapters behind AU's agent-control ports.

The fake EG client implements the documented native contract
(``work_items.submit``/``cancel`` plus the requested ``GetWorkItem``/
``ListWorkItems`` reads, ``supports`` negotiation, and the ``AgentComponent``
wire method), so these tests prove the adapters' mapping and fail-closed
behavior without a live engine.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.api import (
    AgentControlPlaneUnavailable,
    AgentExecutionRequest,
    AgentTaskDispatchRequest,
    AgentWorkItemNotCancelable,
    CapabilitySearchRequest,
    EgCapabilitySearch,
    EgWorkItemStore,
    GraphSession,
    HarnessAgentExecutor,
    SignedAgentDispatchReceipt,
    WorkItemCancelRequest,
    WorkItemGetRequest,
    WorkItemIdempotencyConflict,
    WorkItemListRequest,
    WorkItemPayloadTooLarge,
    WorkItemSubmission,
    compose_eg_agent_control_plane,
    current_session,
    use_session,
)
from agent_utilities.api.agent_control_adapters import (
    DESCRIPTION_METADATA_KEY,
    candidates_from_entries,
    request_context_for,
)
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext

_DIGESTS = {
    "policy_digest": "policy:v1",
    "catalog_digest": "catalog:v1",
    "model_digest": "model:v1",
}


def _session(graph: str = "tenant-test") -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="agent:adapter-test",
            actor_type=ActorType.AI_AGENT,
            tenant_id="tenant:test",
            authenticated=True,
        ),
        tenant="tenant:test",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph=graph,
        policy_version="policy:test",
        audience="graph-os",
    )


def _row(item_id: str = "wi:1", *, status: str = "ready", version: int = 3):
    return {
        "work_item_id": item_id,
        "kind": "orchestrator_task",
        "status": status,
        "input_ref": "au-task:sha256:abc",
        "metadata": {DESCRIPTION_METADATA_KEY: "stored body", "team": "red"},
        "version": version,
        "updated_at_ms": 1_758_430_800_000,
    }


class _WorkItems:
    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.cancel_status = "cancelled"
        self.conflict = False

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.conflict:
            raise RuntimeError("IDEMPOTENCY_CONFLICT: engine text with detail")
        self.submitted.append(request)
        item_id = request["work_item_id"]
        replayed = item_id in self.rows
        self.rows.setdefault(item_id, _row(item_id, version=1))
        return {
            "work_item_id": item_id,
            "status": "ready",
            "created": not replayed,
            "replayed": replayed,
        }

    async def cancel(self, **kwargs: Any) -> dict[str, Any]:
        self.cancelled.append(kwargs)
        return {"status": self.cancel_status, "changed_work_item_ids": []}

    async def get(self, *, tenant: str, work_item_id: str):
        assert tenant == "tenant:test"
        return self.rows.get(work_item_id)

    async def list(self, *, tenant: str, cursor, limit, kind):
        assert tenant == "tenant:test"
        return {"items": list(self.rows.values())[:limit], "next_cursor": "c2"}


class _Client:
    def __init__(self, *, served: tuple[str, ...] = ("GetWorkItem", "ListWorkItems")):
        self.work_items = _WorkItems()
        self.served = served
        self.sent: list[tuple[str, Any, Any]] = []
        self.claims: list[dict[str, Any]] = []
        self.search_entries: list[dict[str, Any]] = []

    async def supports(self, method: str) -> bool:
        return method in self.served

    @contextmanager
    def use_verified_context(self, claims):
        self.claims.append(claims)
        yield

    async def _send(self, method, params, graph, *, idempotency_key=None):
        self.sent.append((method, params, graph))
        return {"entries": self.search_entries, "next_cursor": None}


def _store(client: _Client) -> EgWorkItemStore:
    return EgWorkItemStore(client, authentication_method="oidc", **_DIGESTS)


def _submission(**overrides: Any) -> WorkItemSubmission:
    values: dict[str, Any] = {
        "work_item_id": "wi:1",
        "idempotency_key": "idem-1",
        "kind": "orchestrator_task",
        "description": "email alice@example.com about the release",
        "metadata": {"team": "red"},
    }
    values.update(overrides)
    return WorkItemSubmission(**values)


# --- WorkItem store --------------------------------------------------------


async def test_submit_sends_session_derived_context_and_sanitized_payload() -> None:
    client = _Client()
    result = await _store(client).submit(_submission(), session=_session())

    (sent,) = client.work_items.submitted
    context = sent["context"]
    assert context["tenant_id"] == "tenant:test"
    assert context["agent_id"] == "agent:adapter-test"
    assert context["graph"] == "tenant-test"
    assert context["authentication_method"] == "oidc"
    assert sorted(context["scopes"]) == ["kg:read", "kg:write"]
    body = sent["metadata"][DESCRIPTION_METADATA_KEY]
    assert "alice@example.com" not in body
    assert sent["input_ref"].startswith("au-task:sha256:")
    assert len(sent["command_digest"]) == 64
    assert result.created and not result.replayed
    assert result.item.description == body
    assert result.item.metadata == {"team": "red"}


async def test_submit_digest_is_stable_for_the_same_sanitized_payload() -> None:
    client = _Client()
    store = _store(client)
    await store.submit(_submission(), session=_session())
    await store.submit(_submission(), session=_session())
    first, second = client.work_items.submitted
    assert first["command_digest"] == second["command_digest"]
    assert first["context"]["request_id"] != second["context"]["request_id"]


async def test_replayed_admission_returns_the_authoritative_row() -> None:
    client = _Client()
    store = _store(client)
    await store.submit(_submission(), session=_session())
    replay = await store.submit(_submission(), session=_session())
    assert replay.replayed and not replay.created
    assert replay.item.description == "stored body"


async def test_idempotency_conflict_is_typed_and_hides_engine_text() -> None:
    client = _Client()
    client.work_items.conflict = True
    with pytest.raises(WorkItemIdempotencyConflict) as raised:
        await _store(client).submit(_submission(), session=_session())
    assert "engine text" not in str(raised.value)


async def test_reserved_metadata_and_oversize_payload_are_refused() -> None:
    store = _store(_Client())
    with pytest.raises(ValueError, match="reserved"):
        await store.submit(
            _submission(metadata={DESCRIPTION_METADATA_KEY: "x"}), session=_session()
        )
    with pytest.raises(WorkItemPayloadTooLarge):
        await store.submit(_submission(description="x" * 70_000), session=_session())


async def test_session_without_graph_fails_closed() -> None:
    with pytest.raises(AgentControlPlaneUnavailable, match="graph"):
        await _store(_Client()).submit(_submission(), session=_session(graph=""))


async def test_get_and_list_are_tenant_scoped_and_strip_reserved_keys() -> None:
    client = _Client()
    client.work_items.rows["wi:1"] = _row()
    store = _store(client)
    item = await store.get(WorkItemGetRequest(work_item_id="wi:1"), session=_session())
    assert item is not None and item.version == 3
    assert item.description == "stored body"
    assert DESCRIPTION_METADATA_KEY not in item.metadata
    assert (
        await store.get(WorkItemGetRequest(work_item_id="nope"), session=_session())
        is None
    )
    page = await store.list(WorkItemListRequest(limit=10), session=_session())
    assert [row.work_item_id for row in page.items] == ["wi:1"]
    assert page.next_cursor == "c2"


async def test_reads_fail_closed_when_the_engine_does_not_serve_them() -> None:
    store = _store(_Client(served=()))
    with pytest.raises(AgentControlPlaneUnavailable, match="GetWorkItem"):
        await store.get(WorkItemGetRequest(work_item_id="wi:1"), session=_session())
    with pytest.raises(AgentControlPlaneUnavailable, match="ListWorkItems"):
        await store.list(WorkItemListRequest(), session=_session())


async def test_unknown_row_status_fails_closed() -> None:
    client = _Client()
    client.work_items.rows["wi:1"] = _row(status="exploded")
    with pytest.raises(AgentControlPlaneUnavailable, match="unknown status"):
        await _store(client).get(
            WorkItemGetRequest(work_item_id="wi:1"), session=_session()
        )


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ("missing", None),
        ("cancelled", "row"),
        ("noop", "row"),
        ("in_flight", AgentWorkItemNotCancelable),
        ("not_cancellable", AgentWorkItemNotCancelable),
        ("mystery", AgentControlPlaneUnavailable),
    ],
)
async def test_cancel_maps_every_engine_status(status: str, outcome: Any) -> None:
    client = _Client()
    client.work_items.rows["wi:1"] = _row(status="cancelled")
    client.work_items.cancel_status = status
    request = WorkItemCancelRequest(work_item_id="wi:1")
    if isinstance(outcome, type):
        with pytest.raises(outcome):
            await _store(client).cancel(request, session=_session())
        return
    result = await _store(client).cancel(request, session=_session())
    assert (result is None) is (outcome is None)
    (call,) = client.work_items.cancelled
    assert call["tenant"] == "tenant:test"
    assert call["reason_ref"] == "au-cancel-reason:caller_cancelled"


async def test_store_requires_native_namespace_and_digests() -> None:
    store = EgWorkItemStore(object(), authentication_method="oidc", **_DIGESTS)
    with pytest.raises(AgentControlPlaneUnavailable, match="WorkItem namespace"):
        await store.submit(_submission(), session=_session())
    with pytest.raises(ValueError, match="policy"):
        EgWorkItemStore(
            _Client(),
            authentication_method="oidc",
            policy_digest=" ",
            catalog_digest="c",
            model_digest="m",
        )


async def test_store_routes_to_its_work_item_graph() -> None:
    client = _Client()
    store = EgWorkItemStore(
        client, authentication_method="oidc", graph="__control__", **_DIGESTS
    )
    await store.submit(_submission(), session=_session())
    (sent,) = client.work_items.submitted
    assert sent["context"]["graph"] == "__control__"
    assert list(sent)[:5] == [
        "schema_version",
        "context",
        "work_item_id",
        "idempotency_key",
        "command_digest",
    ]


def test_request_context_never_widens_the_carrier() -> None:
    session = _session()
    context = request_context_for(session, "workload_identity")
    claims = session.engine_verified_context()
    assert set(context["scopes"]) == set(claims["scopes"])
    assert context["expires_at_ms"] >= context["issued_at_ms"]


# --- Capability search -----------------------------------------------------


def _entry(component_id: str, kind: str, name: str | None = None) -> Any:
    return SimpleNamespace(
        component_id=component_id,
        kind=SimpleNamespace(value=kind),
        attributes={"name": name} if name else None,
    )


def test_candidates_preserve_eg_order_and_drop_unmapped_kinds() -> None:
    candidates = candidates_from_entries(
        [
            _entry("skill:review", "skill", "pr-review"),
            _entry("tool:x", "tool"),
            _entry("card:expert", "a2a_agent_card"),
        ]
    )
    assert [(c.kind, c.name) for c in candidates] == [
        ("skill", "pr-review"),
        ("agent", "card:expert"),
    ]
    assert candidates[0].score > candidates[1].score
    assert {c.source for c in candidates} == {"eg_agent_component"}


async def test_task_search_sends_the_typed_task_iri_to_eg() -> None:
    """EG serves typed task-capability search (AgentComponentSearchRequest.task):
    the closed task IRI, never the free text, reaches the one search call."""
    client = _Client()
    result = await EgCapabilitySearch(client).search(
        CapabilitySearchRequest(task="review a PR", task_iri="eg:task/review"),
        session=_session(),
    )
    assert result == ()
    assert len(client.sent) == 1
    method, params, _graph = client.sent[0]
    assert method == "AgentComponent"
    assert params["op"]["op"] == "search"
    assert params["op"]["request"]["task"] == "eg:task/review"
    assert "review a PR" not in repr(params)


async def test_free_text_is_never_sent_to_eg_as_a_task() -> None:
    client = _Client()
    result = await EgCapabilitySearch(client).search(
        CapabilitySearchRequest(task="review a PR"), session=_session()
    )
    assert result == ()
    assert client.sent == []


def test_task_iri_is_the_closed_native_vocabulary() -> None:
    with pytest.raises(ValueError):
        CapabilitySearchRequest.model_validate(
            {"task": "t", "task_iri": "eg:task/anything"}
        )


async def test_named_agent_walks_the_kind_scoped_catalog(monkeypatch) -> None:
    from epistemic_graph.generated import storage

    requests: list[Any] = []

    async def served(client, request, graph=None, *, idempotency_key=None):
        requests.append(request)
        from epistemic_graph.generated.agent_component import AgentComponentSearchPage

        return AgentComponentSearchPage(
            entries=[], next_cursor="c2" if len(requests) == 1 else None
        )

    monkeypatch.setattr(storage, "send_agent_component_search", served)
    result = await EgCapabilitySearch(_Client()).search(
        CapabilitySearchRequest(task="free text", agent_name="expert"),
        session=_session(),
    )
    assert result == ()
    assert [r.task for r in requests] == [None, None]
    assert requests[1].cursor == "c2"


async def test_task_search_sends_the_typed_request_when_served(monkeypatch) -> None:
    from epistemic_graph.generated import storage

    captured: dict[str, Any] = {}

    async def served(client, request, graph=None, *, idempotency_key=None):
        captured.update(request=request, graph=graph)
        from epistemic_graph.generated.agent_component import AgentComponentSearchPage

        return AgentComponentSearchPage(entries=[], next_cursor=None)

    monkeypatch.setattr(storage, "send_agent_component_search", served)
    result = await EgCapabilitySearch(_Client()).search(
        CapabilitySearchRequest(task="review a PR", limit=5, task_iri="eg:task/review"),
        session=_session(),
    )
    assert result == ()
    request = captured["request"]
    assert request.tenant_id == "tenant:test"
    assert request.task == "eg:task/review"
    assert request.limit == 5
    assert captured["graph"] == "tenant-test"


# --- Agent execution -------------------------------------------------------


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any], GraphSession | None]] = []

    async def execute_agent(self, agent_name: str, task: str, **options: Any) -> str:
        self.calls.append((agent_name, task, options, current_session()))
        return "done"


async def test_executor_binds_the_verified_session_for_the_run() -> None:
    runner = _Runner()
    session = _session()
    result = await HarnessAgentExecutor.in_process(runner).execute_agent(
        AgentExecutionRequest(
            agent_name="expert",
            task="t",
            allowed_tools=("a",),
            execution_mode="graph",
            max_steps=7,
        ),
        session=session,
    )
    ((name, task, options, bound),) = runner.calls
    assert (name, task) == ("expert", "t")
    assert bound is session
    assert options["allowed_tools"] == ["a"]
    assert options["max_steps"] == 7
    assert options["execution_mode"] == "graph"
    assert options["run_id"] == result.run_id
    assert callable(options["progress_sink"])
    assert result.output == "done"
    assert result.execution_mode == "graph"


async def test_executor_reraises_the_runtime_error_contract() -> None:
    class _Refusing:
        async def execute_agent(self, agent_name: str, task: str, **options: Any):
            raise LookupError("no authorized capability")

    with pytest.raises(LookupError, match="no authorized capability"):
        await HarnessAgentExecutor.in_process(_Refusing()).execute_agent(
            AgentExecutionRequest(agent_name="expert", task="t"), session=_session()
        )


async def test_executor_maps_an_unusable_harness_to_unavailable() -> None:
    from agent_utilities.layers.execution import default_registry

    executor = HarnessAgentExecutor(default_registry(_Runner()), harness="nope")
    with pytest.raises(AgentControlPlaneUnavailable, match="not registered"):
        await executor.execute_agent(
            AgentExecutionRequest(agent_name="expert", task="t"), session=_session()
        )


def test_executor_requires_a_runner() -> None:
    with pytest.raises(TypeError):
        HarnessAgentExecutor.in_process(object())


# --- Composition -----------------------------------------------------------


class _Dispatch:
    async def enqueue(self, request, *, session):
        return SignedAgentDispatchReceipt(job_id=request.job_id, accepted=True)


async def test_composed_plane_submits_through_eg_adapters(monkeypatch) -> None:
    from agent_utilities.api import agent_control_adapters

    async def search(self, request, *, session):
        return candidates_from_entries(
            [_entry("card:expert", "a2a_agent_card", "expert")]
        )

    monkeypatch.setattr(agent_control_adapters.EgCapabilitySearch, "search", search)
    client = _Client()
    session = _session()
    plane = compose_eg_agent_control_plane(
        client,
        session,
        runner=_Runner(),
        authentication_method="oidc",
        signed_dispatch=_Dispatch(),
        **_DIGESTS,
    )
    with use_session(session):
        result = await plane.submit_agent_task(
            AgentTaskDispatchRequest(
                work_item_id="wi:1",
                idempotency_key="idem-1",
                job_id="job-1",
                session_ref="s-1",
                task="do the thing",
            )
        )
    assert result.capability.name == "expert"
    assert result.admission.created
    assert result.dispatch is not None and result.dispatch.accepted
    assert client.claims, "every port call runs under the verified context"
