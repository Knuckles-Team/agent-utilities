"""Queue-only agent dispatch and WorkItem authority contracts."""

from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_utilities.orchestration import agent_dispatch
from agent_utilities.orchestration.agent_dispatch import (
    AGENT_TURNS_TOPIC,
    KIND_GOAL_LOOP,
    AgentTurnEnvelope,
)
from agent_utilities.orchestration.agent_dispatch_worker import (
    WorkItemLeaseGuard,
    WorkItemLeaseLost,
    worker_token,
)


def test_envelope_round_trip_carries_references_only() -> None:
    envelope = AgentTurnEnvelope(
        session_id="session-ref",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-ref",
        tenant="tenant-ref",
        prio_bucket=1,
    )
    restored = AgentTurnEnvelope.from_item(envelope.to_item())
    assert restored == envelope
    assert restored.payload_ref == "goal-ref"
    assert not hasattr(restored, "payload")


def test_job_id_keeps_full_uuid_entropy() -> None:
    job_id = AgentTurnEnvelope(session_id="session-ref").job_id
    assert re.fullmatch(r"dispatch-[0-9a-f]{32}", job_id)


def test_session_partition_key_precedes_tenant() -> None:
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )

    first = AgentTurnEnvelope(session_id="session-ref", tenant="tenant-a").to_item()
    second = AgentTurnEnvelope(session_id="session-ref", tenant="tenant-b").to_item()
    assert partition_key_for(first) == partition_key_for(second)


def test_sqlite_is_queue_transport_not_inline_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        agent_dispatch, "_sqlite_queue_path", lambda: str(tmp_path / "dispatch.db")
    )
    queue = agent_dispatch.create_dispatch_queue(
        SimpleNamespace(
            task_queue_backend="sqlite",
            queue_backend="sqlite",
            state_db_uri=None,
        )
    )
    queue.put(AgentTurnEnvelope(session_id="session-ref").to_item())
    assert queue.get_queue_size() == 1


def test_kafka_dispatch_uses_fixed_topic_and_group(monkeypatch) -> None:
    captured: dict = {}

    class FakeKafkaQueue:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.kafka_queue_backend.KafkaQueueBackend",
        FakeKafkaQueue,
    )
    agent_dispatch.create_dispatch_queue(
        SimpleNamespace(
            task_queue_backend="kafka",
            queue_backend="kafka",
            state_db_uri=None,
            kafka_bootstrap_servers="broker.invalid:9092",
            agent_turns_partitions=8,
        )
    )
    assert captured["tasks_topic"] == AGENT_TURNS_TOPIC
    assert captured["consumer_group"] == "agent-dispatch"
    assert captured["partitions"] == 8


def test_worker_identity_is_opaque_and_process_stable() -> None:
    first = worker_token()
    assert first == worker_token()
    assert re.fullmatch(r"worker:[0-9a-f]{32}", first)
    assert "/" not in first and "\\" not in first and "@" not in first


def test_dispatch_module_has_no_inline_or_parallel_task_authority() -> None:
    assert not hasattr(agent_dispatch, "resolve_dispatch_backend")
    assert not hasattr(agent_dispatch, "dispatch_queue_enabled")


def test_dispatch_lease_defaults_meet_published_rto() -> None:
    from agent_utilities.core.config import AgentConfig

    contract = yaml.safe_load(
        (Path(__file__).parents[2] / "scripts/scale/workload_contract.yml").read_text()
    )
    config = AgentConfig()
    assert config.agent_dispatch_renew_interval_s < config.agent_dispatch_claim_ttl_s
    assert config.agent_dispatch_claim_ttl_s <= contract["availability"]["rto_seconds"]

    guard = WorkItemLeaseGuard(
        object(),
        "workitem-ref",
        {"work_item_id": "workitem-ref"},
        lease_ttl_s=config.agent_dispatch_claim_ttl_s,
    )
    assert guard.heartbeat_interval_s == config.agent_dispatch_renew_interval_s


def test_dispatch_depth_probe_fails_closed() -> None:
    class BrokenQueue:
        def get_queue_size(self):
            raise ConnectionError("unavailable")

    with pytest.raises(ConnectionError):
        agent_dispatch.dispatch_queue_depth(BrokenQueue())


def test_dispatch_rejects_before_workitem_when_queue_is_full(monkeypatch) -> None:
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.queue_backend import MemoryQueueBackend

    queue = MemoryQueueBackend()
    queue.put({"job_id": "already-queued"})
    monkeypatch.setattr(config, "agent_dispatch_max_depth", 1)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.resolve_session",
        lambda **_kwargs: SimpleNamespace(tenant="tenant-ref"),
    )
    with pytest.raises(agent_dispatch.DispatchQueueFull):
        agent_dispatch.enqueue_agent_turn(
            AgentTurnEnvelope(session_id="session-ref"), queue=queue
        )


def test_lease_guard_heartbeats_during_long_execution(monkeypatch) -> None:
    calls: list[float] = []

    def renew(*_args, **_kwargs):
        calls.append(time.monotonic())
        return True

    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_dispatch_worker._fence_still_valid",
        renew,
    )
    guard = WorkItemLeaseGuard(
        object(),
        "workitem-ref",
        {"work_item_id": "workitem-ref"},
        lease_ttl_s=0.09,
        heartbeat_interval_s=0.01,
    )
    with guard:
        time.sleep(0.035)
    assert len(calls) >= 3


def test_lease_guard_rejects_side_effect_after_fence_loss(monkeypatch) -> None:
    outcomes = iter((True, False))
    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_dispatch_worker._fence_still_valid",
        lambda *_args, **_kwargs: next(outcomes),
    )
    guard = WorkItemLeaseGuard(
        object(),
        "workitem-ref",
        {"work_item_id": "workitem-ref"},
        lease_ttl_s=10.0,
    )
    guard.start()
    try:
        with pytest.raises(WorkItemLeaseLost):
            guard.side_effect(lambda: pytest.fail("stale effect executed"))
    finally:
        guard.close()
