"""Hosted control plane (AU-1), signed tool allowlist (AU-2), process runtime
port (AU-3) and redacted run output (AU-5)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.api import (
    AgentControlPlaneUnavailable,
    AgentTaskDispatchRequest,
    CapabilityCandidate,
    GraphSession,
    ProcessRunOutputReader,
    QueueSignedAgentDispatch,
    RunOutput,
    RunOutputRequest,
    SignedAgentDispatchReceipt,
    SignedAgentDispatchRequest,
    WorkItemSnapshot,
    WorkItemSubmissionResult,
    compose_agent_control_plane,
    compose_hosted_agent_control_plane,
    use_session,
)
from agent_utilities.api import hosted_control_plane as hosted
from agent_utilities.api.runtime import AgentRuntime
from agent_utilities.orchestration import agent_dispatch_worker as worker
from agent_utilities.orchestration.agent_dispatch import (
    DISPATCH_CARRIER_VERSION,
    KIND_WORK_ITEM_TURN,
    AgentTurnEnvelope,
    DispatchCarrier,
    DispatchCarrierError,
)
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext

_SECRET = b"hosted-control-plane-test-secret"  # test-only HMAC key


def _session() -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="agent:hosted-test",
            actor_type=ActorType.AI_AGENT,
            tenant_id="tenant:test",
            authenticated=True,
        ),
        tenant="tenant:test",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="tenant-test",
        policy_version="policy:test",
        audience="graph-os",
    )


class _Client:
    work_items = object()

    @contextmanager
    def use_verified_context(self, claims):
        yield


# --- AU-2: signed allowlist -------------------------------------------------


def _mint(**overrides: Any) -> DispatchCarrier:
    values: dict[str, Any] = {
        "tenant": "tenant:test",
        "session_id": "s-1",
        "job_id": "job-1",
        "kind": KIND_WORK_ITEM_TURN,
        "payload_ref": "wi:1",
        "agent_name": "expert",
        "allowed_tools": ("search", "read"),
        "secret": _SECRET,
    }
    values.update(overrides)
    return DispatchCarrier.mint(**values)


def test_carrier_signs_the_tool_allowlist() -> None:
    carrier = _mint()
    assert carrier.version == DISPATCH_CARRIER_VERSION == 2
    assert carrier.allowed_tools == ("search", "read")
    binding = dict(
        tenant="tenant:test",
        session_id="s-1",
        job_id="job-1",
        kind=KIND_WORK_ITEM_TURN,
        payload_ref="wi:1",
        agent_name="expert",
        secret=_SECRET,
    )
    carrier.verify(**binding, allowed_tools=("search", "read"))
    with pytest.raises(DispatchCarrierError, match="binding"):
        carrier.verify(**binding, allowed_tools=("search", "read", "shell"))
    with pytest.raises(DispatchCarrierError, match="binding"):
        carrier.verify(**binding, allowed_tools=None)
    widened = carrier.model_copy(update={"allowed_tools": ("search", "read", "shell")})
    with pytest.raises(DispatchCarrierError, match="signature"):
        widened.verify(**binding, allowed_tools=("search", "read", "shell"))


@pytest.mark.parametrize(
    "tools", [tuple(f"t{i}" for i in range(65)), ("a", "a"), (" ",), ("x" * 257,)]
)
def test_carrier_refuses_invalid_allowlists(tools: tuple[str, ...]) -> None:
    with pytest.raises(DispatchCarrierError):
        _mint(allowed_tools=tools)


def test_contract_bounds_the_allowlist() -> None:
    with pytest.raises(ValueError):
        SignedAgentDispatchRequest(
            job_id="j",
            work_item_id="w",
            session_ref="s",
            agent_name="a",
            allowed_tools=tuple(f"t{i}" for i in range(65)),
        )


# --- AU-2: worker enforcement -----------------------------------------------


def test_work_item_turn_is_its_own_fence() -> None:
    envelope = AgentTurnEnvelope(
        job_id="job-1", session_id="s", kind=KIND_WORK_ITEM_TURN, payload_ref="wi:9"
    )
    assert worker.dispatch_work_item_id(envelope) == "wi:9"
    legacy = AgentTurnEnvelope(job_id="job-1", session_id="s")
    assert worker.dispatch_work_item_id(legacy) == "workitem:dispatch:job-1"


def test_worker_runs_admitted_description_with_signed_allowlist(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core import work_durability
    from agent_utilities.orchestration import manager

    calls: list[dict[str, Any]] = []

    class _Orchestrator:
        def __init__(self, engine):
            pass

        async def execute_agent(self, **kwargs):
            calls.append(kwargs)
            return "answer"

    monkeypatch.setattr(manager, "Orchestrator", _Orchestrator)
    monkeypatch.setattr(
        work_durability,
        "get_work_item",
        lambda engine, item_id: {"metadata": {"au:description": "sanitized task"}},
    )
    envelope = AgentTurnEnvelope(
        job_id="job-1",
        session_id="s",
        kind=KIND_WORK_ITEM_TURN,
        payload_ref="wi:9",
        agent_name="expert",
        allowed_tools=("search",),
    )
    lease = SimpleNamespace(require_current=lambda: None)
    assert worker._execute_work_item_turn(envelope, object(), lease) == "completed"
    (call,) = calls
    assert call["task"] == "sanitized task"
    assert call["allowed_tools"] == ["search"]
    assert call["run_id"] == "job-1"


def test_worker_refuses_an_item_without_a_description(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core import work_durability

    monkeypatch.setattr(work_durability, "get_work_item", lambda engine, item_id: None)
    with pytest.raises(work_durability.WorkItemBackendUnavailable):
        worker._admitted_task_description(object(), "wi:missing")


# --- AU-1: hosted dispatch + composition ------------------------------------


class _Queue:
    def __init__(self, accept: bool = True) -> None:
        self.items: list[dict[str, Any]] = []
        self.accept = accept

    def put_if_below(self, item, max_depth):
        self.items.append(item)
        return self.accept


async def test_queue_dispatch_publishes_a_signed_work_item_turn(monkeypatch) -> None:
    from agent_utilities.orchestration import agent_dispatch

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", _SECRET.decode())
    monkeypatch.setattr(agent_dispatch, "dispatch_queue_depth", lambda q: 0)
    queue = _Queue()
    receipt = await QueueSignedAgentDispatch(queue).enqueue(
        SignedAgentDispatchRequest(
            job_id="job-1",
            work_item_id="wi:1",
            session_ref="s-1",
            agent_name="expert",
            allowed_tools=("search",),
        ),
        session=_session(),
    )
    assert receipt.accepted
    (item,) = queue.items
    envelope = AgentTurnEnvelope.from_item(item)
    assert envelope.kind == KIND_WORK_ITEM_TURN
    assert envelope.payload_ref == "wi:1"
    assert envelope.tenant == "tenant:test"
    assert envelope.allowed_tools == ("search",)
    envelope.authenticate_carrier()


async def test_queue_dispatch_full_queue_fails_closed(monkeypatch) -> None:
    from agent_utilities.core.config import config
    from agent_utilities.orchestration import agent_dispatch

    monkeypatch.setenv("AGENT_UTILITIES_TOKEN_SECRET", _SECRET.decode())
    monkeypatch.setattr(
        agent_dispatch,
        "dispatch_queue_depth",
        lambda q: int(config.agent_dispatch_max_depth),
    )
    with pytest.raises(AgentControlPlaneUnavailable, match="admission bound"):
        await QueueSignedAgentDispatch(_Queue()).enqueue(
            SignedAgentDispatchRequest(
                job_id="j", work_item_id="w", session_ref="s", agent_name="a"
            ),
            session=_session(),
        )


class _Search:
    async def search(self, request, *, session):
        return [
            CapabilityCandidate(
                kind="agent", name="expert", component_id="c", score=1.0, source="t"
            )
        ]


class _Store:
    def __init__(self) -> None:
        self.cancelled: list[Any] = []

    async def submit(self, request, *, session):
        item = WorkItemSnapshot(
            work_item_id=request.work_item_id,
            kind=request.kind,
            status="ready",
            version=1,
            updated_at_ms=1,
        )
        return WorkItemSubmissionResult(item=item, created=True, replayed=False)

    async def cancel(self, request, *, session):
        self.cancelled.append(request)
        return None


class _Dispatch:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.requests: list[SignedAgentDispatchRequest] = []

    async def enqueue(self, request, *, session):
        self.requests.append(request)
        if self.fail:
            raise AgentControlPlaneUnavailable("queue down")
        return SignedAgentDispatchReceipt(job_id=request.job_id, accepted=True)


def _task(**overrides: Any) -> AgentTaskDispatchRequest:
    values: dict[str, Any] = {
        "work_item_id": "wi:1",
        "idempotency_key": "k",
        "job_id": "job-1",
        "session_ref": "s-1",
        "task": "do it",
        "allowed_tools": ("search",),
    }
    values.update(overrides)
    return AgentTaskDispatchRequest(**values)


async def test_allowlist_flows_to_dispatch_and_failed_dispatch_cancels() -> None:
    session = _session()
    store = _Store()
    ok = _Dispatch(fail=False)
    plane = compose_agent_control_plane(
        _Client(),
        session,
        capability_search=_Search(),
        work_item_store=store,
        signed_dispatch=ok,
    )
    with use_session(session):
        await plane.submit_agent_task(_task())
    assert ok.requests[0].allowed_tools == ("search",)
    assert store.cancelled == []

    failing = _Dispatch(fail=True)
    plane = compose_agent_control_plane(
        _Client(),
        session,
        capability_search=_Search(),
        work_item_store=store,
        signed_dispatch=failing,
    )
    with use_session(session), pytest.raises(AgentControlPlaneUnavailable):
        await plane.submit_agent_task(_task())
    (cancel,) = store.cancelled
    assert cancel.work_item_id == "wi:1"
    assert cancel.reason == "dispatch_admission_failed"


def test_hosted_composition_binds_every_port_from_two_inputs() -> None:
    plane = compose_hosted_agent_control_plane(_Client(), _session())
    assert isinstance(plane._work_item_store, hosted.EgWorkItemStore)
    assert isinstance(plane._signed_dispatch, QueueSignedAgentDispatch)
    assert isinstance(plane._run_output, ProcessRunOutputReader)
    assert plane._capability_search is not None
    assert plane._agent_executor is not None
    names = {d.name for d in plane.operation_descriptors}
    assert "get_run_output" in names


def test_admission_digests_are_stable_hex() -> None:
    first = hosted.admission_digests(_session())
    assert first == hosted.admission_digests(_session())
    assert all(len(d) == 64 for d in first)
    assert hosted.authentication_method_for(_session()) == "oidc"


async def test_hosted_execution_fails_closed_without_a_runtime(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", classmethod(lambda cls: None)
    )
    with pytest.raises(AgentControlPlaneUnavailable, match="runtime is not open"):
        hosted.process_engine()


# --- AU-5: run output --------------------------------------------------------


def test_run_output_is_mapped_redacted_and_bounded() -> None:
    assert hosted.run_output_from_trace("r", {"status": "not_found"}) is None
    done = hosted.run_output_from_trace(
        "r", {"status": "completed", "result_preview": "mail bob@example.com"}
    )
    assert done is not None and done.status == "succeeded"
    assert "bob@example.com" not in done.output
    failed = hosted.run_output_from_trace("r", {"status": "timeout"})
    assert failed is not None and failed.status == "failed"
    odd = hosted.run_output_from_trace("r", {"status": "degraded"})
    assert odd is not None and odd.status == "unknown"
    long = hosted.run_output_from_trace(
        "r", {"status": "completed", "result_preview": "x" * 100_050}
    )
    assert long is not None and long.truncated and len(long.output) == 100_000


async def test_run_output_reads_under_the_callers_session(monkeypatch) -> None:
    from agent_utilities.api import current_session
    from agent_utilities.orchestration import manager

    seen: list[Any] = []

    class _Orchestrator:
        def __init__(self, engine):
            pass

        def get_run_trace(self, run_id):
            seen.append(current_session())
            return {"status": "completed", "result_preview": "ok"}

    monkeypatch.setattr(manager, "Orchestrator", _Orchestrator)
    monkeypatch.setattr(hosted, "process_engine", lambda: object())
    session = _session()
    plane = compose_agent_control_plane(
        _Client(), session, run_output=ProcessRunOutputReader()
    )
    with use_session(session):
        result = await plane.get_run_output(RunOutputRequest(run_id="job-1"))
    assert result == RunOutput(run_id="job-1", status="succeeded", output="ok")
    assert seen == [session]


# --- AU-3: runtime port --------------------------------------------------------


def test_runtime_views_and_drain() -> None:
    events: list[Any] = []

    class _Compute:
        def for_graph(self, graph):
            events.append(("for_graph", graph))
            return SimpleNamespace(client=f"client:{graph}")

        def drain(self, timeout_s):
            events.append(("drain", timeout_s))
            return SimpleNamespace(timed_out=False, active_requests=0)

        def close(self):
            events.append(("close",))

    engine = SimpleNamespace(
        graph_compute=_Compute(),
        start_background_daemons=lambda: events.append(("daemons",)),
        start_task_workers=lambda count: events.append(("workers", count)),
    )
    runtime = AgentRuntime(engine, "host")
    assert runtime.graph_client("g1") == "client:g1"
    runtime.start_background_daemons()
    runtime.start_task_workers(2)
    result = runtime.drain_and_close(5.0)
    assert result.timed_out is False and result.active_requests == 0
    assert events[-2:] == [("drain", 5.0), ("close",)]


def test_open_process_runtime_reuses_the_active_engine(monkeypatch) -> None:
    from agent_utilities.api import runtime as runtime_port
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    active = object()
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", classmethod(lambda cls: active)
    )
    opened = runtime_port.open_process_runtime(
        role="client", defer_background_start=True
    )
    assert opened.engine is active
