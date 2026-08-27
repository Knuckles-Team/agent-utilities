"""Unit tests for the OpenLineage RunEvent consumer (CA-25, DEC-CA-05).

(CONCEPT:AU-KG.ingest.openlineage-consumer)

Covers CA-25's acceptance gates: the mapping layer's quarantine rules
(unmapped dataset namespace, missing run.runId/job.name, unmapped
eventType), the ``iceberg://<catalog>/<ns>/<table>@<snapshot>`` dataset
naming rule, and ``OpenLineageKafkaConsumer.drain_once``'s fire-and-forget
offset semantics (unlike CA-21's Debezium consumer, EVERY record in a batch
is committed regardless of per-record outcome).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent_utilities.knowledge_graph.etl import openlineage_consumer as module
from agent_utilities.knowledge_graph.etl.openlineage_consumer import (
    MalformedLineageDataset,
    MappedRunEvent,
    OpenLineageKafkaConsumer,
    QuarantinedLineageEvent,
    dataset_entity_id,
    map_openlineage_event,
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


# ── dataset_entity_id ───────────────────────────────────────────────────────


def test_dataset_entity_id_builds_iceberg_uri() -> None:
    assert dataset_entity_id(_dataset()) == "iceberg://lakehouse/sales/orders@snap-42"


def test_dataset_entity_id_missing_version_facet_raises() -> None:
    with pytest.raises(MalformedLineageDataset):
        dataset_entity_id(_dataset(snapshot=""))


def test_dataset_entity_id_non_iceberg_namespace_raises() -> None:
    with pytest.raises(MalformedLineageDataset):
        dataset_entity_id(_dataset(namespace="postgres://db/public"))


def test_dataset_entity_id_missing_name_raises() -> None:
    with pytest.raises(MalformedLineageDataset):
        dataset_entity_id(_dataset(name=""))


# ── map_openlineage_event: happy path + RunTrace-correlation extraction ────


def test_map_openlineage_event_valid_creates_mapped_event() -> None:
    event = _run_event(
        inputs=[_dataset(name="raw_orders", snapshot="snap-7")],
        outputs=[_dataset(name="orders", snapshot="snap-42")],
        parent_run_id="01977c9e-0000-7000-8000-00000000ffff",
    )
    mapped = map_openlineage_event(event)
    assert isinstance(mapped, MappedRunEvent)
    assert mapped.run_id == "01977c9e-0000-7000-8000-000000000001"
    assert mapped.job_name == "etl-orders"
    assert mapped.job_namespace == "spark"
    assert mapped.event_type == "COMPLETE"
    assert mapped.activity_status == "completed"
    assert mapped.input_dataset_ids == ("iceberg://lakehouse/sales/raw_orders@snap-7",)
    assert mapped.output_dataset_ids == ("iceberg://lakehouse/sales/orders@snap-42",)
    assert mapped.parent_run_id == "01977c9e-0000-7000-8000-00000000ffff"


def test_map_openlineage_event_no_datasets_is_valid() -> None:
    mapped = map_openlineage_event(_run_event(event_type="START"))
    assert isinstance(mapped, MappedRunEvent)
    assert mapped.activity_status == "running"
    assert mapped.input_dataset_ids == ()
    assert mapped.output_dataset_ids == ()


@pytest.mark.parametrize(
    "event_type,expected_status",
    [
        ("START", "running"),
        ("RUNNING", "running"),
        ("COMPLETE", "completed"),
        ("ABORT", "aborted"),
        ("FAIL", "failed"),
        ("OTHER", "unknown"),
    ],
)
def test_map_openlineage_event_status_table(
    event_type: str, expected_status: str
) -> None:
    mapped = map_openlineage_event(_run_event(event_type=event_type))
    assert isinstance(mapped, MappedRunEvent)
    assert mapped.activity_status == expected_status


# ── map_openlineage_event: quarantine paths (acceptance gate 2) ───────────


def test_map_openlineage_event_missing_run_id_quarantined() -> None:
    event = _run_event()
    event["run"] = {}
    mapped = map_openlineage_event(event)
    assert isinstance(mapped, QuarantinedLineageEvent)
    assert "run.runId" in mapped.reason


def test_map_openlineage_event_missing_job_name_quarantined() -> None:
    event = _run_event()
    event["job"] = {"namespace": "spark"}
    mapped = map_openlineage_event(event)
    assert isinstance(mapped, QuarantinedLineageEvent)
    assert "job.name" in mapped.reason


def test_map_openlineage_event_unmapped_event_type_quarantined() -> None:
    mapped = map_openlineage_event(_run_event(event_type="RESURRECT"))
    assert isinstance(mapped, QuarantinedLineageEvent)
    assert "eventType" in mapped.reason


def test_map_openlineage_event_malformed_dataset_quarantines_whole_event() -> None:
    """A single unmapped-namespace dataset quarantines the WHOLE event — no
    partial mapping of the datasets that DID resolve (acceptance gate 2)."""
    event = _run_event(
        inputs=[_dataset(namespace="postgres://db/public", name="raw", snapshot="")],
        outputs=[_dataset(name="orders", snapshot="snap-42")],
    )
    mapped = map_openlineage_event(event)
    assert isinstance(mapped, QuarantinedLineageEvent)


def test_map_openlineage_event_correctly_namespaced_event_maps_normally() -> None:
    """The paired positive case for gate 2: a correctly-namespaced event maps
    to a normal MappedRunEvent, not a quarantine."""
    event = _run_event(outputs=[_dataset(name="orders", snapshot="snap-42")])
    mapped = map_openlineage_event(event)
    assert isinstance(mapped, MappedRunEvent)
    assert mapped.output_dataset_ids == ("iceberg://lakehouse/sales/orders@snap-42",)


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
