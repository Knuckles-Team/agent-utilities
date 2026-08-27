"""``apply_envelope`` + ``EgCdcKafkaConsumer`` (CA-24-W03/W04).

Covers: basic upsert, out-of-order/duplicate ``seq`` rejection (P3's
negative case), marking resolution from ``permissioning`` (never the
envelope — see ``indexer.py``'s module doc on the DEC-CA-03 field gap),
tombstone with a known and an unknown object type, an OpenSearch write
failure NOT advancing the consumer offset, and a quarantined/rejected
record still advancing it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.knowledge_graph.ontology import permissioning
from agent_utilities.knowledge_graph.search.client import OpenSearchClient
from agent_utilities.knowledge_graph.search.indexer import (
    EgCdcKafkaConsumer,
    apply_envelope,
    derive_content,
    resolve_object_type,
)
from agent_utilities.knowledge_graph.search.tests.conftest import (
    FakeOpenSearch,
    make_client,
    require_doc,
)

pytestmark = pytest.mark.concept("AU-KG.retrieval.opensearch-cdc-indexer")


def _envelope(**overrides: Any) -> dict[str, Any]:
    base = {
        "seq": 1,
        "graph": "acme",
        "op": "upsert",
        "node_id": "n1",
        "edge_id": None,
        "before": None,
        "after": {"type": "Person", "name": "Ada"},
        "ts": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return base


# ── resolve_object_type / derive_content ────────────────────────────────────


def test_resolve_object_type_prefers_type_key() -> None:
    assert resolve_object_type({"type": "Person", "node_type": "Other"}) == "Person"


def test_resolve_object_type_falls_back_to_node_type() -> None:
    assert resolve_object_type({"node_type": "Person"}) == "Person"


def test_resolve_object_type_none_when_absent() -> None:
    assert resolve_object_type({"name": "Ada"}) is None
    assert resolve_object_type(None) is None
    assert resolve_object_type({"type": "  "}) is None


def test_derive_content_joins_scalars_excludes_type_keys() -> None:
    content = derive_content(
        {"type": "Person", "name": "Ada", "age": 30, "tags": ["x"]}
    )
    assert "Person" not in content
    assert "Ada" in content
    assert "30" in content


# ── apply_envelope: upsert ──────────────────────────────────────────────────


def test_apply_upsert_basic(marking_authority) -> None:
    client = make_client()
    result = apply_envelope(client, _envelope())
    assert result["status"] == "applied"
    assert result["index"] == "kg-acme-person"
    doc = require_doc(client, "kg-acme-person", "n1")
    assert doc["node_id"] == "n1"
    assert doc["node_type"] == "Person"
    assert doc["tenant"] == "acme"
    assert doc["updated_lsn"] == 1
    assert doc["marking"] == []
    assert doc["properties"] == {"type": "Person", "name": "Ada"}


def test_apply_upsert_creates_index_with_explicit_keyword_mapping(
    marking_authority,
) -> None:
    """Regression for a REAL failure hit against the live CA-50 cluster
    (2026-08-26): OpenSearch's dynamic mapping maps a bare string field to
    `text`, which broke a `terms` aggregation on `node_id` and would have
    made a `term` DLS filter on `marking` match analyzed tokens rather than
    the exact marking name. `ensure_index` must be called with the explicit
    `doc_shape.INDEX_MAPPINGS` (keyword for node_id/node_type/tenant/marking)
    on first write, not left to dynamic inference."""
    from agent_utilities.knowledge_graph.search import doc_shape

    class _RecordingClient(FakeOpenSearch):
        def __init__(self) -> None:
            super().__init__()
            self.create_calls: list[tuple[str, Any]] = []

        def index(self, index: str, id: str, body: Any, refresh: Any = False):  # noqa: A002
            return super().index(index, id, body, refresh)

    client = OpenSearchClient(client=_RecordingClient())
    real_create = client.raw.indices.create
    seen: list[dict] = []

    def _spy_create(index: str, body: Any = None):
        seen.append({"index": index, "body": body})
        return real_create(index, body)

    client.raw.indices.create = _spy_create  # type: ignore[method-assign]

    apply_envelope(client, _envelope())

    assert len(seen) == 1
    assert seen[0]["index"] == "kg-acme-person"
    assert seen[0]["body"]["mappings"] == doc_shape.INDEX_MAPPINGS
    assert doc_shape.INDEX_MAPPINGS["properties"]["marking"] == {"type": "keyword"}
    assert doc_shape.INDEX_MAPPINGS["properties"]["node_id"] == {"type": "keyword"}


def test_apply_upsert_resolves_marking_from_permissioning_never_the_envelope(
    marking_authority,
) -> None:
    permissioning.apply_marking("n1", "restricted", tenant="acme")
    client = make_client()
    # A bogus "marking" key on the envelope must be IGNORED — the real wire
    # envelope never carries one at all (see indexer.py's module doc); this
    # proves the code path doesn't accidentally trust it if present.
    envelope = _envelope(marking="public-should-be-ignored")
    result = apply_envelope(client, envelope)
    assert result["marking"] == ["restricted"]
    doc = require_doc(client, "kg-acme-person", "n1")
    assert doc["marking"] == ["restricted"]


def test_apply_upsert_quarantines_when_type_unresolvable(marking_authority) -> None:
    result = apply_envelope(make_client(), _envelope(after={"name": "no-type"}))
    assert result["status"] == "quarantined"


@pytest.mark.parametrize(
    "envelope_overrides",
    [
        {"seq": None},
        {"graph": ""},
        {"node_id": ""},
        {"op": "delete"},  # eg's real wire vocabulary is upsert|tombstone, not "delete"
    ],
)
def test_apply_upsert_quarantines_malformed_envelope(
    marking_authority, envelope_overrides: dict[str, Any]
) -> None:
    result = apply_envelope(make_client(), _envelope(**envelope_overrides))
    assert result["status"] == "quarantined"


def test_apply_upsert_rejects_out_of_order_seq(marking_authority) -> None:
    client = make_client()
    apply_envelope(client, _envelope(seq=5, after={"type": "Person", "name": "Ada"}))
    stale = apply_envelope(
        client, _envelope(seq=3, after={"type": "Person", "name": "Stale"})
    )
    assert stale["status"] == "rejected_stale"
    doc = require_doc(client, "kg-acme-person", "n1")
    assert doc["properties"]["name"] == "Ada"  # never overwritten


def test_apply_upsert_rejects_redelivered_same_seq(marking_authority) -> None:
    """Per the lane's Authority/invariants section (the binding contract:
    "a message whose seq is not greater than the currently-indexed seq for
    that node must be rejected") — a same-seq redelivery is rejected as a
    no-op, not silently re-applied. (The lane's own Scope bullet phrases
    this looser, as "last-write-wins on the SAME seq is fine" — resolved
    here in favor of the stricter, explicitly-titled Authority/invariants
    wording; flagged in the lane report.)"""
    client = make_client()
    apply_envelope(client, _envelope(seq=5, after={"type": "Person", "name": "Ada"}))
    dup = apply_envelope(
        client, _envelope(seq=5, after={"type": "Person", "name": "Ada"})
    )
    assert dup["status"] == "rejected_stale"


def test_apply_upsert_accepts_newer_seq(marking_authority) -> None:
    client = make_client()
    apply_envelope(client, _envelope(seq=1, after={"type": "Person", "name": "Ada"}))
    result = apply_envelope(
        client, _envelope(seq=2, after={"type": "Person", "name": "Ada2"})
    )
    assert result["status"] == "applied"
    doc = require_doc(client, "kg-acme-person", "n1")
    assert doc["properties"]["name"] == "Ada2"
    assert doc["updated_lsn"] == 2


def test_apply_upsert_opensearch_read_failure_is_failed(marking_authority) -> None:
    class _BrokenRead(FakeOpenSearch):
        def get(self, index: str, id: str) -> dict:  # noqa: A002
            raise RuntimeError("cluster unreachable")

    client = OpenSearchClient(client=_BrokenRead())
    result = apply_envelope(client, _envelope())
    assert result["status"] == "failed"


def test_apply_upsert_opensearch_write_failure_is_failed_not_quarantined(
    marking_authority,
) -> None:
    class _BrokenWrite(FakeOpenSearch):
        def index(self, *args: Any, **kwargs: Any) -> dict:
            raise RuntimeError("cluster unreachable")

    client = OpenSearchClient(client=_BrokenWrite())
    result = apply_envelope(client, _envelope())
    assert result["status"] == "failed"


# ── apply_envelope: tombstone ────────────────────────────────────────────────


def test_apply_tombstone_known_type_deletes(marking_authority) -> None:
    client = make_client()
    apply_envelope(client, _envelope(seq=1))
    result = apply_envelope(
        client,
        _envelope(
            seq=2, op="tombstone", after=None, before={"type": "Person", "name": "Ada"}
        ),
    )
    assert result["status"] == "applied"
    assert client.get_document("kg-acme-person", "n1") is None


def test_apply_tombstone_rejects_stale_seq(marking_authority) -> None:
    client = make_client()
    apply_envelope(client, _envelope(seq=5))
    result = apply_envelope(
        client,
        _envelope(
            seq=2, op="tombstone", after=None, before={"type": "Person", "name": "Ada"}
        ),
    )
    assert result["status"] == "rejected_stale"
    assert client.get_document("kg-acme-person", "n1") is not None


def test_apply_tombstone_unknown_type_falls_back_to_tenant_wildcard(
    marking_authority,
) -> None:
    client = make_client()
    apply_envelope(client, _envelope(seq=1))
    result = apply_envelope(
        client, _envelope(seq=2, op="tombstone", after=None, before=None)
    )
    assert result["status"] == "applied"
    assert client.get_document("kg-acme-person", "n1") is None


def test_apply_tombstone_absent_node_is_safe_noop(marking_authority) -> None:
    result = apply_envelope(
        make_client(),
        _envelope(node_id="ghost", op="tombstone", after=None, before=None),
    )
    assert result["status"] == "applied"


# ── EgCdcKafkaConsumer.drain_once: fail-closed offset semantics ────────────


class _FakeRecord:
    def __init__(self, value: bytes | None, offset: int) -> None:
        self.value = value
        self.offset = offset


class _FakeTopicPartition:
    def __init__(self, topic: str = "eg.cdc.acme", partition: int = 0) -> None:
        self.topic = topic
        self.partition = partition

    def __hash__(self) -> int:
        return hash((self.topic, self.partition))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeTopicPartition) and (
            self.topic,
            self.partition,
        ) == (
            other.topic,
            other.partition,
        )


class _FakeConsumer:
    def __init__(self, batches: list[dict]) -> None:
        self._batches = list(batches)
        self.commits: list[dict] = []

    async def getmany(self, *, timeout_ms: int = 0, max_records: int | None = None):
        return self._batches.pop(0) if self._batches else {}

    async def commit(self, offsets: dict) -> None:
        self.commits.append(dict(offsets))

    async def stop(self) -> None:
        return None


def _wire(envelope: dict[str, Any]) -> bytes:
    import json

    return json.dumps(envelope).encode("utf-8")


def test_drain_once_commits_through_applied_and_quarantined_records(
    marking_authority,
) -> None:
    tp = _FakeTopicPartition()
    batch = {
        tp: [
            _FakeRecord(_wire(_envelope(seq=1, node_id="n1")), offset=0),
            _FakeRecord(
                _wire(_envelope(seq=1, node_id="n2", after={"name": "no-type"})),
                offset=1,
            ),
        ]
    }
    consumer = EgCdcKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))
    client = make_client()

    result = asyncio.run(consumer.drain_once(client))

    assert result["status"] == "ok"
    assert result["counts"] == {
        "applied": 1,
        "rejected_stale": 0,
        "quarantined": 1,
        "failed": 0,
    }
    assert consumer._consumer.commits == [{tp: 2}]


def test_drain_once_stops_at_first_failure_never_commits_past_it(
    marking_authority,
) -> None:
    tp = _FakeTopicPartition()
    batch = {
        tp: [
            _FakeRecord(_wire(_envelope(seq=1, node_id="n1")), offset=0),
            _FakeRecord(_wire(_envelope(seq=1, node_id="n2")), offset=1),
            _FakeRecord(_wire(_envelope(seq=1, node_id="n3")), offset=2),
        ]
    }
    consumer = EgCdcKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))

    class _BreaksOnSecondWrite(FakeOpenSearch):
        def __init__(self) -> None:
            super().__init__()
            self._writes = 0

        def index(self, *args: Any, **kwargs: Any) -> dict:
            self._writes += 1
            if self._writes == 2:
                raise RuntimeError("cluster unreachable")
            return super().index(*args, **kwargs)

    client = OpenSearchClient(client=_BreaksOnSecondWrite())
    result = asyncio.run(consumer.drain_once(client))

    assert result["status"] == "failed"
    assert result["counts"]["applied"] == 1
    assert result["counts"]["failed"] == 1
    # offset committed exactly ONE past record 0 (the sole verified write) —
    # never past the failed record 1, and record 2 is never even attempted.
    assert consumer._consumer.commits == [{tp: 1}]
    assert client.get_document("kg-acme-person", "n3") is None


def test_drain_once_malformed_payload_is_failed_not_silently_skipped(
    marking_authority,
) -> None:
    tp = _FakeTopicPartition()
    batch = {tp: [_FakeRecord(b"not-json{{{", offset=0)]}
    consumer = EgCdcKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))

    result = asyncio.run(consumer.drain_once(make_client()))

    assert result["status"] == "failed"
    assert result["counts"]["failed"] == 1
    assert consumer._consumer.commits == []


def test_eg_cdc_kafka_consumer_drain_once_requires_connection() -> None:
    consumer = EgCdcKafkaConsumer(config=None)
    with pytest.raises(RuntimeError):
        asyncio.run(consumer.drain_once(make_client()))


def test_drain_once_reports_watermark_from_last_applied_seq(marking_authority) -> None:
    tp = _FakeTopicPartition()
    batch = {
        tp: [
            _FakeRecord(_wire(_envelope(seq=1, node_id="n1")), offset=0),
            _FakeRecord(_wire(_envelope(seq=7, node_id="n2")), offset=1),
        ]
    }
    consumer = EgCdcKafkaConsumer(config=None, consumer=_FakeConsumer([batch]))
    result = asyncio.run(consumer.drain_once(make_client()))
    assert result["watermark"] == 7
