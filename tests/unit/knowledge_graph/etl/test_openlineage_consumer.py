"""Unit tests for the OpenLineage RunEvent consumer (CA-25, DEC-CA-05).

(CONCEPT:AU-KG.ingest.openlineage-consumer)

Covers CA-25's consumer offset semantics and feature flag. Pure mapping
and dataset identity tests live in EG's ``test_openlineage_derivation.py``.
``OpenLineageKafkaConsumer.drain_once`` advances offsets (unlike CA-21's Debezium consumer, EVERY record in a batch
is committed regardless of per-record outcome).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent_utilities.knowledge_graph.etl import openlineage_consumer as module
from agent_utilities.knowledge_graph.etl.openlineage_consumer import (
    OpenLineageKafkaConsumer,
    openlineage_consumer_enabled,
)

pytestmark = pytest.mark.concept("AU-KG.ingest.openlineage-consumer")


# ── shared fixture builders ─────────────────────────────────────────────────


def _dataset(
    *,
    namespace: str = "iceberg://lakehouse/sales",
    name: str = "orders",
    snapshot: str = "snap-42",
) -> dict[str, Any]:
    facets: dict[str, Any] = {}
    if snapshot:
        facets["version"] = {"datasetVersion": snapshot}
    return {"namespace": namespace, "name": name, "facets": facets}


def _run_event(
    *,
    run_id: str = "01977c9e-0000-7000-8000-000000000001",
    job_name: str = "etl-orders",
    job_namespace: str = "spark",
    event_type: str = "COMPLETE",
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    parent_run_id: str = "",
) -> dict[str, Any]:
    run: dict[str, Any] = {"runId": run_id}
    if parent_run_id:
        run["facets"] = {"parent": {"run": {"runId": parent_run_id}}}
    return {
        "eventType": event_type,
        "run": run,
        "job": {"namespace": job_namespace, "name": job_name},
        "inputs": inputs if inputs is not None else [],
        "outputs": outputs if outputs is not None else [],
    }


# ── openlineage_consumer_enabled: the cast=bool trap ───────────────────────


def test_openlineage_consumer_enabled_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(module.KAFKA_OPENLINEAGE_CONSUMER_ENABLED_ENV, raising=False)
    assert openlineage_consumer_enabled() is False


def test_openlineage_consumer_enabled_string_false_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact bug class the program flags: cast=bool would make
    bool("false") True. setting()'s inferred to_boolean cast must not."""
    monkeypatch.setenv(module.KAFKA_OPENLINEAGE_CONSUMER_ENABLED_ENV, "false")
    assert openlineage_consumer_enabled() is False


def test_openlineage_consumer_enabled_string_true_is_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(module.KAFKA_OPENLINEAGE_CONSUMER_ENABLED_ENV, "true")
    assert openlineage_consumer_enabled() is True


# ── OpenLineageKafkaConsumer.drain_once: fire-and-forget offset semantics ──


class _FakeRecord:
    def __init__(self, value: bytes | None, offset: int) -> None:
        self.value = value
        self.offset = offset


class _FakeTopicPartition:
    def __init__(self, topic: str, partition: int = 0) -> None:
        self.topic = topic
        self.partition = partition

    def __hash__(self) -> int:
        return hash((self.topic, self.partition))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _FakeTopicPartition)
            and self.topic == other.topic
            and self.partition == other.partition
        )


class _FakeConsumer:
    def __init__(self, batches: list[dict[Any, list[_FakeRecord]]]) -> None:
        self._batches = list(batches)
        self.commits: list[dict[Any, int]] = []

    async def getmany(self, *, timeout_ms: int = 0, max_records: int | None = None):
        return self._batches.pop(0) if self._batches else {}

    async def commit(self, offsets: dict[Any, int]) -> None:
        self.commits.append(dict(offsets))

    async def stop(self) -> None:
        return None


def _wire(event: dict[str, Any]) -> bytes:
    return json.dumps(event).encode("utf-8")


def _batch(*values: bytes) -> tuple[dict, Any]:
    tp = _FakeTopicPartition(module.OPENLINEAGE_TOPIC)
    return {tp: [_FakeRecord(v, offset=i) for i, v in enumerate(values)]}, tp


class _NullEngine:
    """Not None, but exposes no callable add_node — record_openlineage_run_event
    returns None for it (failed count), same as a real infra failure."""


def test_drain_once_quarantines_and_still_commits() -> None:
    event = _run_event()
    event["job"] = {}  # missing job.name -> quarantined
    batch, tp = _batch(_wire(event))
    consumer = OpenLineageKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))
    result = asyncio.run(consumer.drain_once(engine=None))
    assert result["counts"] == {"succeeded": 0, "quarantined": 1, "failed": 0}
    assert consumer._consumer.commits == [{tp: 1}]


def test_drain_once_malformed_json_counts_failed_and_still_commits() -> None:
    batch, tp = _batch(b"not-json{{{")
    consumer = OpenLineageKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))
    result = asyncio.run(consumer.drain_once(engine=None))
    assert result["counts"] == {"succeeded": 0, "quarantined": 0, "failed": 1}
    # Fire-and-forget (DEC-CA-05): unlike CA-21's Debezium consumer, this
    # commits through a bad record rather than stalling the batch on it.
    assert consumer._consumer.commits == [{tp: 1}]


@pytest.mark.parametrize("payload", [b"[]", b"null", b"42"])
def test_drain_once_non_object_json_counts_failed_and_commits(payload: bytes) -> None:
    batch, tp = _batch(payload)
    consumer = OpenLineageKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))
    result = asyncio.run(consumer.drain_once(engine=None))
    assert result["counts"] == {"succeeded": 0, "quarantined": 0, "failed": 1}
    assert consumer._consumer.commits == [{tp: 1}]


def test_drain_once_write_failure_does_not_stop_the_batch() -> None:
    """Two valid, mappable events; the graph write itself fails for both
    (engine has no add_node) — both still counted failed, both still
    committed, neither blocks the other (fire-and-forget contract)."""
    events = [
        _run_event(run_id=f"01977c9e-0000-7000-8000-00000000000{i}") for i in (1, 2)
    ]
    batch, tp = _batch(*(_wire(e) for e in events))
    consumer = OpenLineageKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))
    result = asyncio.run(consumer.drain_once(engine=_NullEngine()))
    assert result["counts"] == {"succeeded": 0, "quarantined": 0, "failed": 2}
    assert consumer._consumer.commits == [{tp: 2}]


def test_drain_once_requires_connection() -> None:
    consumer = OpenLineageKafkaConsumer(config=None)
    with pytest.raises(RuntimeError):
        asyncio.run(consumer.drain_once(engine=None))


class _FakeGraphEngine:
    """Minimal add_node/link_nodes double — end-to-end drain_once -> engine
    write proof (mirrors tests/unit/knowledge_graph/test_etl_lineage.py's
    _FakeEngine, duplicated locally to keep this test module self-contained).
    """

    def __init__(self) -> None:
        self.nodes: list[tuple[str, str, dict]] = []
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes.append((node_id, str(node_type), dict(properties or {})))

    def link_nodes(self, s, t, rel):
        self.edges.append((s, t, str(rel)))


def test_drain_once_success_writes_activity_and_entities_through_lineage() -> None:
    event = _run_event(
        run_id="01977c9e-0000-7000-8000-000000000099",
        outputs=[_dataset(name="orders", snapshot="snap-42")],
    )
    batch, tp = _batch(_wire(event))
    consumer = OpenLineageKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))
    engine = _FakeGraphEngine()

    result = asyncio.run(consumer.drain_once(engine=engine))

    assert result["counts"] == {"succeeded": 1, "quarantined": 0, "failed": 0}
    assert consumer._consumer.commits == [{tp: 1}]
    activity_nodes = [n for n in engine.nodes if n[2].get("kind") == "openlineage_run"]
    assert len(activity_nodes) == 1
    entity_nodes = [n for n in engine.nodes if n[2].get("kind") == "dataset"]
    assert entity_nodes[0][0] == "iceberg://lakehouse/sales/orders@snap-42"
    assert any(e[2] == "was_generated_by" for e in engine.edges)
