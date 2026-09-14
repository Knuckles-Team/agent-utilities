"""Governed AU host adapter for agent-webui contact delivery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import types
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.messaging.models import SendResult
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext
from agent_utilities.server.webui_contact_delivery import (
    WebUIContactDelivery,
    build_webui_contact_delivery,
)
from agent_utilities.server.webui_contact_governance import (
    contact_delivery_factory_kwargs,
)


class _ContactDeliveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    success: bool
    receipt: str | None = None


@dataclass(frozen=True)
class _Submission:
    name: str = "Ada Example"
    email: str = "ada@example.invalid"
    subject: str = "A bounded question"
    message: str = "Please contact me about Graph OS."


@dataclass(frozen=True)
class _Request:
    submission: _Submission = _Submission()
    destination: str = "telegram:support-inbox"
    retention_days: int = 0
    actor_reference: str = hashlib.sha256(b"webui-contact-test").hexdigest()
    idempotency_key: str = "contactreq_" + "b" * 32


@pytest.fixture(autouse=True)
def _contact_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = types.ModuleType("agent_webui.contact_delivery")
    contract.ContactDeliveryResult = _ContactDeliveryResult
    contract.load_contact_delivery_config = lambda: SimpleNamespace(
        destination="telegram:support-inbox", retention_days=0
    )
    monkeypatch.setitem(sys.modules, "agent_webui.contact_delivery", contract)


@pytest.fixture
def session() -> GraphSession:
    actor = ActorContext(
        actor_id="webui-contact-test",
        actor_type=ActorType.HUMAN,
        roles=("user",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:write"}),
        policy_version="current",
        audience="graph-runtime",
    )


def _negative_claim() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "claimed": False,
        "reason": "empty",
        "work_item_id": None,
        "kind": None,
        "payload_ref": None,
        "lease_holder_ref": None,
        "lease_epoch": None,
        "fencing_token": None,
        "lease_expires_at_ms": None,
        "attempt": None,
        "max_attempts": None,
        "tenant_in_flight": None,
        "changed_work_item_ids": [],
    }


class _ControlAuthority:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.commit_status: str | None = None
        self.raise_on_read = False
        self.raise_on_cas = False
        self.deny_claim = False

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if self.raise_on_read:
            raise RuntimeError("private authority detail")
        node = self.nodes.get(str((params or {}).get("id") or ""))
        if node is None:
            return []
        if "ContactRateLimit" in query:
            return [
                {
                    "window_start": node.get("window_start"),
                    "count": node.get("count"),
                    "attempt_ids": node.get("attempt_ids"),
                }
            ]
        return [dict(node)]

    def create_node_if_absent(
        self, node_id: str, *, properties: dict[str, Any]
    ) -> bool:
        if self.raise_on_cas:
            raise RuntimeError("private CAS detail")
        with self.lock:
            if node_id in self.nodes:
                return False
            self.nodes[node_id] = {"id": node_id, **properties}
            return True

    def compare_and_set_node_fields(
        self,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
    ) -> bool:
        with self.lock:
            node = self.nodes.get(node_id)
            if node is None or any(
                node.get(key) != value for key, value in conditions.items()
            ):
                return False
            node.update(updates)
            return True

    def claim_work_item(self, request: Any) -> dict[str, Any]:
        if self.deny_claim:
            return _negative_claim()
        with self.lock:
            node = self.nodes.get(str(request.work_item_id))
            if node is None or node.get("status") != "ready":
                return _negative_claim()
            epoch = int(node.get("lease_epoch") or 0) + 1
            attempt = int(node.get("attempt") or 0) + 1
            node.update(
                status="leased",
                lease_owner=request.worker_ref,
                lease_epoch=epoch,
                fencing_token=epoch,
                lease_expires_at=(request.now_ms + request.lease_ms) / 1000.0,
                attempt=attempt,
            )
            return {
                "schema_version": "1",
                "claimed": True,
                "reason": "claimed",
                "work_item_id": request.work_item_id,
                "kind": node["kind"],
                "payload_ref": node["payload_ref"],
                "lease_holder_ref": request.worker_ref,
                "lease_epoch": epoch,
                "fencing_token": epoch,
                "lease_expires_at_ms": request.now_ms + request.lease_ms,
                "attempt": attempt,
                "max_attempts": node["max_attempts"],
                "tenant_in_flight": 1,
                "changed_work_item_ids": [request.work_item_id],
            }

    def commit_work_item_result(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.commit_status is not None:
                return {"status": self.commit_status}
            node = self.nodes.get(request["work_item_id"])
            if node is None:
                return {"status": "missing"}
            if node.get("status") in {
                "succeeded",
                "failed",
                "cancelled",
                "dead_letter",
            }:
                return {"status": "noop"}
            node.update(
                status=request["outcome"],
                result_ref=request.get("result_ref"),
                error_ref=request.get("error_ref"),
                lease_owner=None,
                lease_expires_at=None,
            )
            return {"status": "committed"}


class _Engine:
    def __init__(self, authority: _ControlAuthority | None = None) -> None:
        if authority is not None:
            self._work_item_engine = authority


class _MessagingService:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None
        self.error: BaseException | None = None
        self.success = True

    async def send(self, *args: Any, **kwargs: Any) -> SendResult:
        self.calls.append((args, kwargs))
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return SendResult(success=self.success, message_id="provider-private-id")


async def _run_sync(operation: Any) -> Any:
    return operation()


async def _thread_sync(operation: Any) -> Any:
    return await asyncio.to_thread(operation)


def _adapter(
    authority: _ControlAuthority,
    service: _MessagingService,
) -> WebUIContactDelivery:
    return WebUIContactDelivery(
        _Engine(authority),
        fixed_destination="telegram:support-inbox",
        sync_runner=_run_sync,
        messaging_service=service,  # type: ignore[arg-type]
        clock=lambda: 1_700_000_001.0,
    )


def test_factory_declares_capabilities_only_for_complete_delivery_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    authority = _ControlAuthority()
    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda _cls: _Engine(authority)),
    )
    adapter = build_webui_contact_delivery(_run_sync)
    assert adapter is not None
    assert adapter.supports_atomic_idempotency is True
    assert adapter.supports_shared_rate_limit is True

    monkeypatch.setattr(
        sys.modules["agent_webui.contact_delivery"],
        "load_contact_delivery_config",
        lambda: SimpleNamespace(destination="telegram:support-inbox", retention_days=1),
    )
    assert build_webui_contact_delivery(_run_sync) is None


def test_capabilities_stay_disabled_without_native_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda _cls: _Engine()),
    )
    assert build_webui_contact_delivery(_run_sync) is None


@pytest.mark.asyncio
async def test_success_replays_stable_receipt_without_resending_and_persists_no_pii(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    adapter = _adapter(authority, service)

    with use_session(session):
        first = await adapter(_Request())  # type: ignore[arg-type]
        replay = await adapter(_Request())  # type: ignore[arg-type]

    assert first.success is True
    assert first.receipt is not None and first.receipt.startswith("contact_")
    assert replay == first
    assert len(service.calls) == 1
    args, kwargs = service.calls[0]
    assert args[:2] == ("telegram", "support-inbox")
    assert kwargs["persist_outbound"] is False
    assert kwargs["policy_runner"] is _run_sync
    assert kwargs["source"] == "webui_contact"

    persisted = json.dumps(authority.nodes, sort_keys=True)
    for private_value in (
        "Ada Example",
        "ada@example.invalid",
        "A bounded question",
        "Please contact me about Graph OS.",
        "support-inbox",
        _Request().idempotency_key,
    ):
        assert private_value not in persisted


@pytest.mark.asyncio
async def test_concurrent_duplicate_never_invokes_provider_twice(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    service.started = asyncio.Event()
    service.release = asyncio.Event()
    adapter = _adapter(authority, service)

    with use_session(session):
        first_task = asyncio.create_task(adapter(_Request()))  # type: ignore[arg-type]
        await service.started.wait()
        concurrent = await adapter(_Request())  # type: ignore[arg-type]
        service.release.set()
        first = await first_task
        replay = await adapter(_Request())  # type: ignore[arg-type]

    assert first.success is True
    assert concurrent == _ContactDeliveryResult(success=False)
    assert replay == first
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_same_key_with_different_content_is_unknown_and_never_resent(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    adapter = _adapter(authority, service)

    with use_session(session):
        assert (await adapter(_Request())).success is True  # type: ignore[arg-type]
        mismatch = await adapter(  # type: ignore[arg-type]
            replace(_Request(), submission=replace(_Submission(), subject="Changed"))
        )

    assert mismatch == _ContactDeliveryResult(success=False)
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_commit_ambiguity_is_unknown_and_replay_never_resends(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    authority.commit_status = "conflict"
    service = _MessagingService()
    adapter = _adapter(authority, service)

    with use_session(session):
        first = await adapter(_Request())  # type: ignore[arg-type]
        replay = await adapter(_Request())  # type: ignore[arg-type]

    assert first == replay == _ContactDeliveryResult(success=False)
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_provider_failure_is_terminal_unknown_and_replay_never_resends(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    service.success = False
    adapter = _adapter(authority, service)

    with use_session(session):
        first = await adapter(_Request())  # type: ignore[arg-type]
        replay = await adapter(_Request())  # type: ignore[arg-type]

    assert first == replay == _ContactDeliveryResult(success=False)
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_shared_limiter_allows_only_five_new_items_per_actor_window(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    adapter = _adapter(authority, service)

    with use_session(session):
        results = [
            await adapter(  # type: ignore[arg-type]
                replace(_Request(), idempotency_key=f"contactreq_{index:032x}")
            )
            for index in range(6)
        ]

    assert [result.success for result in results] == [True] * 5 + [False]
    assert len(service.calls) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attempt_ids", "count"),
    [
        ([f"workitem:webui_contact:{index:064x}" for index in range(6)], 6),
        ([f"workitem:webui_contact:{1:064x}"] * 2, 2),
        ([f"workitem:webui_contact:{1:064x}"], 2),
        (["unbounded-or-invalid"], 1),
    ],
)
async def test_malformed_limiter_state_fails_closed_without_transition(
    session: GraphSession,
    attempt_ids: list[str],
    count: int,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    adapter = _adapter(authority, service)

    with use_session(session):
        assert (await adapter(_Request())).success is True  # type: ignore[arg-type]
        rate_node = next(
            node
            for node in authority.nodes.values()
            if node["id"].startswith("contact_rate:")
        )
        rate_node.update(
            window_start=1_699_999_920, count=count, attempt_ids=attempt_ids
        )
        result = await adapter(  # type: ignore[arg-type]
            replace(_Request(), idempotency_key="contactreq_" + "c" * 32)
        )

    assert result == _ContactDeliveryResult(success=False)
    assert len(service.calls) == 1
    assert rate_node["attempt_ids"] == attempt_ids


@pytest.mark.asyncio
async def test_valid_limiter_state_resets_in_place_after_window_rollover(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    now = [1_700_000_001.0]
    adapter = WebUIContactDelivery(
        _Engine(authority),
        fixed_destination="telegram:support-inbox",
        sync_runner=_run_sync,
        messaging_service=service,  # type: ignore[arg-type]
        clock=lambda: now[0],
    )

    with use_session(session):
        assert (await adapter(_Request())).success is True  # type: ignore[arg-type]
        now[0] += 61.0
        second = await adapter(  # type: ignore[arg-type]
            replace(_Request(), idempotency_key="contactreq_" + "c" * 32)
        )

    rate_nodes = [
        node
        for node in authority.nodes.values()
        if node["id"].startswith("contact_rate:")
    ]
    assert second.success is True
    assert len(rate_nodes) == 1
    assert rate_nodes[0]["count"] == 1
    assert len(rate_nodes[0]["attempt_ids"]) == 1
    assert len(service.calls) == 2


@pytest.mark.asyncio
async def test_distinct_key_cas_race_admits_only_five_shared_attempts(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    adapter = WebUIContactDelivery(
        _Engine(authority),
        fixed_destination="telegram:support-inbox",
        sync_runner=_thread_sync,
        messaging_service=service,  # type: ignore[arg-type]
        clock=lambda: 1_700_000_001.0,
    )

    with use_session(session):
        results = await asyncio.gather(
            *(
                adapter(  # type: ignore[arg-type]
                    replace(_Request(), idempotency_key=f"contactreq_{index:032x}")
                )
                for index in range(6)
            )
        )

    assert sum(result.success for result in results) == 5
    assert len(service.calls) == 5


@pytest.mark.asyncio
async def test_limiter_cas_exception_fails_closed_without_second_send(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    adapter = _adapter(authority, service)

    with use_session(session):
        assert (await adapter(_Request())).success is True  # type: ignore[arg-type]
        authority.raise_on_cas = True
        result = await adapter(  # type: ignore[arg-type]
            replace(_Request(), idempotency_key="contactreq_" + "c" * 32)
        )

    assert result == _ContactDeliveryResult(success=False)
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_failed_claim_and_fixed_destination_mismatch_never_send(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    authority.deny_claim = True
    service = _MessagingService()
    adapter = _adapter(authority, service)

    with use_session(session):
        unclaimed = await adapter(_Request())  # type: ignore[arg-type]
        rerouted = await adapter(  # type: ignore[arg-type]
            replace(_Request(), destination="slack:other-channel")
        )

    assert unclaimed == rerouted == _ContactDeliveryResult(success=False)
    assert service.calls == []


@pytest.mark.asyncio
async def test_actor_reference_must_match_authenticated_server_context(
    session: GraphSession,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    adapter = _adapter(authority, service)

    with use_session(session):
        result = await adapter(  # type: ignore[arg-type]
            replace(_Request(), actor_reference="a" * 64)
        )

    assert result == _ContactDeliveryResult(success=False)
    assert authority.nodes == {}
    assert service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["no_authority", "read_error", "retention"])
async def test_missing_authority_read_failure_and_retention_fail_closed(
    session: GraphSession,
    failure: str,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    request = _Request()
    engine: Any = _Engine(authority)
    if failure == "read_error":
        authority.raise_on_read = True
    elif failure == "retention":
        request = replace(request, retention_days=1)
    adapter = WebUIContactDelivery(
        engine,
        fixed_destination="telegram:support-inbox",
        sync_runner=_run_sync,
        messaging_service=service,  # type: ignore[arg-type]
    )
    if failure == "no_authority":
        del engine._work_item_engine

    with use_session(session):
        result = await adapter(request)  # type: ignore[arg-type]

    assert result == _ContactDeliveryResult(success=False)
    assert service.calls == []


@pytest.mark.asyncio
async def test_factory_uses_the_exact_engine_that_passed_capability_preflight(
    monkeypatch: pytest.MonkeyPatch,
    session: GraphSession,
) -> None:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    original_authority = _ControlAuthority()
    replacement_authority = _ControlAuthority()
    original_engine = _Engine(original_authority)
    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda _cls: original_engine),
    )
    adapter = build_webui_contact_delivery(_run_sync)
    assert adapter is not None
    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda _cls: _Engine(replacement_authority)),
    )

    with use_session(session):
        result = await adapter(_Request())  # type: ignore[arg-type]

    assert result == _ContactDeliveryResult(success=False)
    assert original_authority.nodes
    assert replacement_authority.nodes == {}


@pytest.mark.asyncio
async def test_provider_exception_is_sanitized_from_result_and_log(
    session: GraphSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    authority = _ControlAuthority()
    service = _MessagingService()
    service.error = RuntimeError("provider leaked ada@example.invalid")
    adapter = _adapter(authority, service)

    with use_session(session):
        result = await adapter(_Request())  # type: ignore[arg-type]

    assert result == _ContactDeliveryResult(success=False)
    assert "ada@example.invalid" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert len(service.calls) == 1


def test_contact_factory_kwargs_support_old_and_new_webui_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def old_factory(agent: object, *, workspace_helpers: object) -> object:
        return agent, workspace_helpers

    def new_factory(
        agent: object,
        *,
        workspace_helpers: object,
        contact_delivery: object | None = None,
    ) -> object:
        return agent, workspace_helpers, contact_delivery

    monkeypatch.setattr(
        "agent_utilities.server.webui_contact_delivery.build_webui_contact_delivery",
        lambda _runner: sentinel,
    )
    old_kwargs = contact_delivery_factory_kwargs(old_factory, _run_sync)
    new_kwargs = contact_delivery_factory_kwargs(new_factory, _run_sync)

    assert old_kwargs == {}
    assert old_factory(object(), workspace_helpers={}, **old_kwargs)
    assert new_kwargs == {"contact_delivery": sentinel}
    assert new_factory(object(), workspace_helpers={}, **new_kwargs)[-1] is sentinel


def test_both_production_webui_factories_inject_the_contact_adapter() -> None:
    import agent_utilities.server.app as app_module
    import agent_utilities.server.webui_co_service as service_module

    for module in (app_module, service_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "contact_delivery_factory_kwargs(" in source
        assert "**contact_kwargs" in source
