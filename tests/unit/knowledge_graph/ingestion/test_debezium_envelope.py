"""Debezium → ChangeEnvelope mapping + envelope-source registry (CA-21).

(CONCEPT:AU-KG.ingest.debezium-changeenvelope)

Covers the CA-21 lane's acceptance gates: c/u/d/r op mapping, unmapped-table
quarantine (and the SAME record mapping cleanly once a matching
``connector_manifest.yml`` resource exists), idempotency-key stability
across redelivery (both at the mapping layer and, per acceptance gate 4,
through the REAL ``ingest_envelope`` boundary — a second commit of the
identical Debezium record reports ``status="skipped"``, not a duplicate
node), and checkpoint/LSN monotonicity. Also covers the
:mod:`~agent_utilities.knowledge_graph.streams.kafka_adapter`
``DebeziumKafkaConsumer``'s fail-closed offset-commit semantics: an offset
only advances past a record whose commit was verified (success/skipped/
quarantine), never past one that failed or was rejected.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    reset_session,
    set_session,
)
from agent_utilities.knowledge_graph.ingestion import debezium_envelope as module
from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
    _position_advances,
    _typed_position,
)
from agent_utilities.knowledge_graph.ontology import connector_manifest_gate
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
)
from agent_utilities.knowledge_graph.streams.kafka_adapter import (
    DebeziumKafkaConsumer,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext
from tests.unit.knowledge_graph.ingestion.test_native_envelope_ingest import _Compute

pytestmark = pytest.mark.concept("AU-KG.ingest.debezium-changeenvelope")


# ── shared record builders ──────────────────────────────────────────────────


def _debezium_record(
    *,
    op: str = "c",
    db: str = "ca51pilot",
    schema: str = "public",
    table: str = "orders",
    lsn: int = 26652968,
    pk: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    before: dict[str, Any] | None = None,
    ts_ms: int = 1787771494084,
) -> dict[str, Any]:
    """The exact ``{"key": ..., "value": ...}`` shape a real Kafka Connect
    JSON-converter delivery decodes to — mirrors a live message consumed
    directly off the CA-51 pilot connector's ``cdc.ca51pilot.public.orders``
    topic, 2026-08-26 (see this lane's report for the raw bytes)."""
    pk = pk if pk is not None else {"id": 1}
    return {
        "key": pk,
        "value": {
            "before": before,
            "after": after if after is not None else {**pk, "customer": "acme-corp"},
            "source": {
                "version": "3.0.0.Final",
                "connector": "postgresql",
                "db": db,
                "schema": schema,
                "table": table,
                "lsn": lsn,
                "ts_ms": ts_ms,
            },
            "op": op,
            "ts_ms": ts_ms + 400,
        },
    }


@pytest.fixture
def manifest_with_orders(tmp_path, monkeypatch):
    """Writes a real ``connector_manifest.yml`` mapping ``orders`` -> ``Order``
    under a temp ``ca51pilot/`` package dir, and points
    :func:`connector_manifest_gate.find_connector_manifest` at it — gate 3's
    "after adding a matching manifest resource" demonstration, using the
    EXISTING manifest catalog mechanism (not a second one)."""
    pkg_dir = tmp_path / "ca51pilot"
    pkg_dir.mkdir()
    manifest_path = pkg_dir / "connector_manifest.yml"
    manifest_path.write_text(
        "connector: ca51pilot\n"
        "resources:\n"
        "- name: orders\n"
        "  label: Order\n"
        "provenance:\n"
        "  integrity:\n"
        '    hash: "test"\n'
        "schema_mappings:\n"
        "  orders:\n"
        "    ontology_class: Order\n"
    )
    # Prove the file actually parses under the real manifest schema before
    # relying on it (a broken fixture would otherwise just look like an
    # unrelated quarantine).
    import yaml

    ConnectorManifest.model_validate(yaml.safe_load(manifest_path.read_text()))

    def _find(source: str, *, agents_root=None):
        return manifest_path if source == "ca51pilot" else None

    monkeypatch.setattr(connector_manifest_gate, "find_connector_manifest", _find)
    return manifest_path


@pytest.fixture
def no_manifest(monkeypatch):
    """Ensures ``db`` resolves to no manifest at all (the default/unonboarded
    state — this is what the LIVE CA-51 pilot table looks like today, since
    "ca51pilot" has no entry in the fleet connector-manifest catalog)."""
    monkeypatch.setattr(
        connector_manifest_gate,
        "find_connector_manifest",
        lambda source, *, agents_root=None: None,
    )


# ── gate: c/u/d/r op mapping ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "op,expected_operation",
    [("c", "upsert"), ("u", "upsert"), ("r", "upsert"), ("d", "delete")],
)
def test_op_mapping(manifest_with_orders, op, expected_operation) -> None:
    before = {"id": 1, "customer": "acme-corp"} if op == "d" else None
    record = _debezium_record(op=op, before=before)
    result = module.map_debezium_event(record)
    assert isinstance(result, ChangeEnvelope), result
    assert result.operation == expected_operation
    assert result.connector == "cdc"
    assert result.source_instance == "ca51pilot.orders"
    assert result.typed_payload["node_type"] == "Order"
    assert result.checkpoint == "26652968"
    assert result.provenance["lsn"] == "26652968"
    assert result.event_time == "2026-08-26T19:11:34.084Z"


def test_unmapped_op_quarantines(manifest_with_orders) -> None:
    record = _debezium_record(op="t")  # Debezium truncate — not in DEC-CA-03's table
    result = module.map_debezium_event(record)
    assert isinstance(result, module.QuarantinedRecord)
    assert result.db == "ca51pilot"
    assert result.table == "orders"
    assert "unmapped" in result.reason


# ── gate 3: unmapped table quarantines; the SAME record maps cleanly once a
# matching manifest resource exists ─────────────────────────────────────────


def test_unmapped_table_quarantines_then_maps_after_manifest_added(
    no_manifest, monkeypatch, tmp_path
) -> None:
    record = _debezium_record()

    quarantined = module.map_debezium_event(record)
    assert isinstance(quarantined, module.QuarantinedRecord)
    assert quarantined.db == "ca51pilot"
    assert quarantined.table == "orders"
    assert "not onboarded" in quarantined.reason

    # Now add the matching manifest resource and re-feed the IDENTICAL record.
    pkg_dir = tmp_path / "ca51pilot"
    pkg_dir.mkdir()
    manifest_path = pkg_dir / "connector_manifest.yml"
    manifest_path.write_text(
        "connector: ca51pilot\n"
        "resources:\n"
        "- name: orders\n"
        "provenance:\n"
        "  integrity:\n"
        '    hash: "test"\n'
        "schema_mappings:\n"
        "  orders:\n"
        "    ontology_class: Order\n"
    )
    monkeypatch.setattr(
        connector_manifest_gate,
        "find_connector_manifest",
        lambda source, *, agents_root=None: (
            manifest_path if source == "ca51pilot" else None
        ),
    )

    mapped = module.map_debezium_event(record)
    assert isinstance(mapped, ChangeEnvelope), mapped
    assert mapped.typed_payload["node_type"] == "Order"


def test_missing_db_or_table_quarantines() -> None:
    record = _debezium_record()
    record["value"]["source"]["table"] = ""
    result = module.map_debezium_event(record)
    assert isinstance(result, module.QuarantinedRecord)
    assert "missing source.db/source.table" in result.reason


def test_missing_key_quarantines_never_guesses_an_id(manifest_with_orders) -> None:
    record = _debezium_record(pk={})
    record["key"] = {}
    result = module.map_debezium_event(record)
    assert isinstance(result, module.QuarantinedRecord)
    assert "primary key" in result.reason


def test_missing_row_for_op_quarantines(manifest_with_orders) -> None:
    record = _debezium_record(op="d", before=None)  # delete with no `before` row
    result = module.map_debezium_event(record)
    assert isinstance(result, module.QuarantinedRecord)
    assert "missing Debezium 'before' row" in result.reason


def test_composite_key_is_deterministic(manifest_with_orders) -> None:
    record = _debezium_record(pk={"tenant": "t1", "id": 5})
    record["value"]["after"] = {"tenant": "t1", "id": 5, "customer": "acme"}
    result = module.map_debezium_event(record)
    assert isinstance(result, ChangeEnvelope)
    assert result.source_object_id == "id=5|tenant=t1"


# ── gate 4: idempotency-key stability + a real double-commit is a no-op ────


def test_idempotency_key_stable_across_redelivery(manifest_with_orders) -> None:
    record = _debezium_record()
    first = module.map_debezium_event(record)
    second = module.map_debezium_event(record)
    assert isinstance(first, ChangeEnvelope) and isinstance(second, ChangeEnvelope)
    assert first.idempotency_key == second.idempotency_key
    assert first.idempotency_key  # never empty


def test_different_lsn_changes_the_idempotency_key(manifest_with_orders) -> None:
    """(db, table, pk, lsn) is the full idempotency tuple (DEC-CA-03) — a
    genuinely new version of the SAME row must not collide with the prior
    one's dedup key."""
    first = module.map_debezium_event(_debezium_record(op="u", lsn=1))
    second = module.map_debezium_event(_debezium_record(op="u", lsn=2))
    assert isinstance(first, ChangeEnvelope) and isinstance(second, ChangeEnvelope)
    assert first.idempotency_key != second.idempotency_key


@pytest.fixture
def _native_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_PROFILE", "dev")
    actor = ActorContext(
        actor_id="ca21-fixture",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="fixture-tenant",
        authenticated=True,
    )
    token = set_session(
        GraphSession(
            actor=actor,
            tenant="fixture-tenant",
            scopes=frozenset({"kg:read", "kg:write"}),
            graph="fixture-graph",
            policy_version="fixture-policy",
            audience="fixture-audience",
        )
    )
    try:
        yield
    finally:
        reset_session(token)


def test_real_ingest_envelope_dedupes_a_replayed_debezium_record(
    manifest_with_orders, _native_profile
) -> None:
    """Acceptance gate 4, against the REAL native-atomic write boundary (not
    a proxy assertion on the mapping layer alone): feed the identical
    Debezium record through map_debezium_event -> ingest_envelope TWICE.
    One commit; the second reports status='skipped', never a duplicate
    node — ingest_envelope's own existing idempotency contract."""
    from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
        ingest_envelope,
    )

    engine = _Compute("graph-ca21")
    record = _debezium_record()

    envelope = module.map_debezium_event(record, tenant="fixture-tenant")
    assert isinstance(envelope, ChangeEnvelope)

    first = ingest_envelope(engine, envelope)
    assert first["status"] == "success", first

    replay_envelope = module.map_debezium_event(record, tenant="fixture-tenant")
    replay = ingest_envelope(engine, replay_envelope)
    assert replay["status"] == "skipped", replay
    assert len(engine.client.changes.applied) == 1


# ── checkpoint/LSN monotonicity ─────────────────────────────────────────────


def test_checkpoint_never_regresses_across_two_envelopes(manifest_with_orders) -> None:
    older = module.map_debezium_event(_debezium_record(op="u", lsn=100))
    newer = module.map_debezium_event(_debezium_record(op="u", lsn=200))
    assert isinstance(older, ChangeEnvelope) and isinstance(newer, ChangeEnvelope)
    prior_position = _typed_position(older.checkpoint, content=False)
    next_position = _typed_position(newer.checkpoint, content=False)
    assert _position_advances(next_position, prior_position)
    # And the reverse (an out-of-order/duplicate redelivery) must NOT look
    # like an advance — this is envelope_ingest's own existing monotonic-
    # cursor enforcement, exercised here against real Debezium-derived
    # checkpoints rather than a synthetic string.
    assert not _position_advances(prior_position, next_position)


# ── registry ─────────────────────────────────────────────────────────────


def test_register_and_get_envelope_source_round_trip() -> None:
    def _handler(engine, *, mode="delta", ids=None, client=None):
        return {"status": "ok"}

    module.register_envelope_source("unit-test-source", _handler)
    try:
        assert module.get_envelope_source("unit-test-source") is _handler
        assert "unit-test-source" in module.list_envelope_sources()
    finally:
        module._ENVELOPE_SOURCES.pop("unit-test-source", None)


def test_cdc_handler_is_self_registered() -> None:
    assert module.get_envelope_source("cdc") is module.run_cdc_catchup


def test_get_envelope_source_unknown_returns_none() -> None:
    assert module.get_envelope_source("does-not-exist") is None


# ── consumer fail-closed offset-commit semantics ────────────────────────────


class _FakeRecord:
    def __init__(self, key: bytes | None, value: bytes | None, offset: int) -> None:
        self.key = key
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


def _fake_records_batch(*rows: tuple[bytes | None, bytes | None]) -> dict:
    tp = _FakeTopicPartition("cdc.ca51pilot.public.orders")
    return {tp: [_FakeRecord(k, v, offset=i) for i, (k, v) in enumerate(rows)]}, tp


def _wire(record: dict[str, Any]) -> tuple[bytes, bytes]:
    import json

    return (
        json.dumps(record["key"]).encode("utf-8"),
        json.dumps(record["value"]).encode("utf-8"),
    )


def test_drain_once_quarantines_and_still_commits_the_offset(
    no_manifest,
) -> None:
    batch, tp = _fake_records_batch(_wire(_debezium_record()))
    consumer = DebeziumKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))
    result = asyncio.run(consumer.drain_once(engine=None))
    assert result["counts"]["quarantined"] == 1
    assert result["status"] == "ok"
    assert consumer._consumer.commits == [{tp: 1}]


class _FakeIngestEngine:
    """Marker object — drain_once's success/failure path is driven entirely
    by the monkeypatched ``ingest_envelope`` below, so this engine is never
    actually inspected."""


def test_drain_once_commits_through_success_and_stops_before_a_failure(
    manifest_with_orders, monkeypatch
) -> None:
    rec_a = _debezium_record(lsn=1)
    rec_b = _debezium_record(lsn=2)
    rec_c = _debezium_record(lsn=3)
    batch, tp = _fake_records_batch(_wire(rec_a), _wire(rec_b), _wire(rec_c))
    consumer = DebeziumKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))

    calls: list[str] = []

    def _fake_ingest_envelope(engine, envelope):
        calls.append(envelope.checkpoint)
        if envelope.checkpoint == "2":
            return {"status": "failed", "reason": "injected failure"}
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        _fake_ingest_envelope,
    )

    result = asyncio.run(consumer.drain_once(engine=_FakeIngestEngine()))

    # Record 0 (lsn=1) succeeds; record 1 (lsn=2) fails -> STOP. Record 2
    # (lsn=3) is never even attempted.
    assert calls == ["1", "2"]
    assert result["status"] == "failed"
    assert result["counts"] == {
        "succeeded": 1,
        "skipped": 0,
        "quarantined": 0,
        "failed": 1,
    }
    # The offset committed is exactly ONE past the last VERIFIED record
    # (offset 0, the lsn=1 success) — never past the failed record at
    # offset 1, and never the fetcher's already-buffered offset 2.
    assert consumer._consumer.commits == [{tp: 1}]


def test_drain_once_malformed_payload_is_rejected_not_silently_skipped() -> None:
    tp = _FakeTopicPartition("cdc.ca51pilot.public.orders")
    batch = {tp: [_FakeRecord(b'{"id":1}', b"not-json{{{", offset=0)]}
    consumer = DebeziumKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))

    result = asyncio.run(consumer.drain_once(engine=None))

    assert result["status"] == "failed"
    assert result["counts"]["failed"] == 1
    assert consumer._consumer.commits == []  # never committed past the bad record


def test_drain_once_requires_connection() -> None:
    consumer = DebeziumKafkaConsumer(config=None)
    with pytest.raises(RuntimeError):
        asyncio.run(consumer.drain_once(engine=None))


def test_debezium_default_topic_pattern_matches_the_live_pilot_topic() -> None:
    from agent_utilities.knowledge_graph.streams.kafka_adapter import (
        _DEBEZIUM_DEFAULT_TOPIC_PATTERN,
    )

    assert _DEBEZIUM_DEFAULT_TOPIC_PATTERN.match("cdc.ca51pilot.public.orders")
    assert not _DEBEZIUM_DEFAULT_TOPIC_PATTERN.match("ca51pilot.public.orders")
    assert not _DEBEZIUM_DEFAULT_TOPIC_PATTERN.match("openlineage.events")


# ── CA-21-W05 migration/rollback: KAFKA_CDC_DEBEZIUM_ENABLED ────────────────


def test_run_cdc_catchup_skips_without_touching_kafka_when_flag_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAFKA_CDC_DEBEZIUM_ENABLED", raising=False)

    connect_attempted = False

    class _ExplodingConsumer(DebeziumKafkaConsumer):
        async def connect(self):  # pragma: no cover - must never be reached
            nonlocal connect_attempted
            connect_attempted = True
            raise AssertionError("must not connect to Kafka when the flag is unset")

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.streams.kafka_adapter.DebeziumKafkaConsumer",
        _ExplodingConsumer,
    )

    result = module.run_cdc_catchup(engine=None)

    assert result["status"] == "skipped"
    assert "KAFKA_CDC_DEBEZIUM_ENABLED" in result["reason"]
    assert connect_attempted is False


def test_run_cdc_catchup_honors_the_flag_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_CDC_DEBEZIUM_ENABLED", "true")
    tp = _FakeTopicPartition("cdc.ca51pilot.public.orders")
    consumer = DebeziumKafkaConsumer(
        config=None, consumer=_FakeConsumer([{tp: []}])
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.streams.kafka_adapter.DebeziumKafkaConsumer",
        lambda config: consumer,
    )

    result = module.run_cdc_catchup(engine=None)

    assert result["status"] == "ok"
    assert result["counts"]["quarantined"] == 0


def test_run_cdc_catchup_explicit_client_bypasses_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator/test-injected client is a deliberate choice — the flag
    only gates the "build a real one" path."""
    monkeypatch.delenv("KAFKA_CDC_DEBEZIUM_ENABLED", raising=False)
    tp = _FakeTopicPartition("cdc.ca51pilot.public.orders")
    consumer = DebeziumKafkaConsumer(
        config=None, consumer=_FakeConsumer([{tp: []}])
    )

    result = module.run_cdc_catchup(engine=None, client=consumer)

    assert result["status"] == "ok"
