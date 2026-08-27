"""``rebuild_index`` (CA-24-W06): replay-from-offset-0 determinism (P3)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.search.rebuild import rebuild_index
from agent_utilities.knowledge_graph.search.tests.conftest import (
    make_client,
    require_doc,
)


def _stream() -> list[dict]:
    return [
        {
            "seq": 1,
            "graph": "acme",
            "op": "upsert",
            "node_id": "n1",
            "after": {"type": "Person", "name": "Ada"},
        },
        {
            "seq": 2,
            "graph": "acme",
            "op": "upsert",
            "node_id": "n2",
            "after": {"type": "Person", "name": "Bo"},
        },
        {
            "seq": 3,
            "graph": "acme",
            "op": "upsert",
            "node_id": "n1",
            "after": {"type": "Person", "name": "Ada Lovelace"},
        },
        {
            "seq": 4,
            "graph": "acme",
            "op": "tombstone",
            "node_id": "n2",
            "before": {"type": "Person", "name": "Bo"},
        },
        {
            "seq": 1,
            "graph": "other-tenant",
            "op": "upsert",
            "node_id": "x1",
            "after": {"type": "Person", "name": "Ignored"},
        },
    ]


def test_rebuild_index_from_records_is_deterministic(marking_authority) -> None:
    client = make_client()
    result_1 = rebuild_index("acme", records=_stream(), opensearch=client)
    snapshot_1 = dict(require_doc(client, "kg-acme-person", "n1"))

    result_2 = rebuild_index("acme", records=_stream(), opensearch=client)
    snapshot_2 = dict(require_doc(client, "kg-acme-person", "n1"))

    assert result_1["status"] == "ok"
    assert result_2["status"] == "ok"
    assert snapshot_1 == snapshot_2  # byte-identical (gate 4)
    assert result_1["counts"] == result_2["counts"]
    assert result_1["index_counts"] == result_2["index_counts"] == {"kg-acme-person": 1}


def test_rebuild_index_scopes_replay_to_the_named_graph(marking_authority) -> None:
    client = make_client()
    rebuild_index("acme", records=_stream(), opensearch=client)
    assert client.get_document("kg-other_tenant-person", "x1") is None


def test_rebuild_index_applies_tombstones(marking_authority) -> None:
    client = make_client()
    rebuild_index("acme", records=_stream(), opensearch=client)
    assert client.get_document("kg-acme-person", "n1") is not None
    assert client.get_document("kg-acme-person", "n2") is None  # tombstoned by seq=4


def test_rebuild_index_from_seq_skips_earlier_records(marking_authority) -> None:
    client = make_client()
    result = rebuild_index("acme", records=_stream(), opensearch=client, from_seq=3)
    # seq=1,2 skipped: n1's seq=1 upsert never happens, only seq=3 (also n1)
    # and seq=4 (tombstone of n2, which was never indexed under from_seq=3).
    doc = client.get_document("kg-acme-person", "n1")
    assert doc is not None
    assert doc["properties"]["name"] == "Ada Lovelace"
    assert result["counts"]["applied"] >= 1


def test_rebuild_index_drops_existing_index_first(marking_authority) -> None:
    client = make_client()
    rebuild_index("acme", records=_stream(), opensearch=client)
    # A stray doc that would NOT be reproduced by a fresh replay.
    client.ensure_index("kg-acme-person")
    client.index_document(
        "kg-acme-person", "stray", {"node_id": "stray", "updated_lsn": 0}
    )
    assert client.get_document("kg-acme-person", "stray") is not None

    rebuild_index("acme", records=_stream(), opensearch=client)
    assert client.get_document("kg-acme-person", "stray") is None


def test_rebuild_index_requires_exactly_one_source(marking_authority) -> None:
    with pytest.raises(ValueError):
        rebuild_index("acme", opensearch=make_client())
    with pytest.raises(ValueError):
        rebuild_index("acme", opensearch=make_client(), records=[], consumer=object())


def test_rebuild_index_drains_an_injected_consumer(marking_authority) -> None:
    class _OneShotConsumer:
        """Unconnected-consumer contract: rebuild_index owns connect/drain/
        disconnect (see rebuild.py's async-safety docstring) — the fake
        provides no-op connect/disconnect so it satisfies that surface."""

        def __init__(self, envelopes: list[dict]) -> None:
            self._envelopes = envelopes
            self._drained = False

        async def connect(self) -> None:
            return None

        async def disconnect(self) -> None:
            return None

        async def drain_once(self, opensearch) -> dict:
            from agent_utilities.knowledge_graph.search.indexer import apply_envelope

            if self._drained:
                return {
                    "status": "ok",
                    "counts": {
                        "applied": 0,
                        "rejected_stale": 0,
                        "quarantined": 0,
                        "failed": 0,
                    },
                }
            self._drained = True
            counts = {"applied": 0, "rejected_stale": 0, "quarantined": 0, "failed": 0}
            for envelope in self._envelopes:
                status = apply_envelope(opensearch, envelope)["status"]
                counts[status] += 1
            return {"status": "ok", "counts": counts}

    client = make_client()
    consumer = _OneShotConsumer(_stream())
    result = rebuild_index("acme", opensearch=client, consumer=consumer)
    assert result["status"] == "ok"
    assert client.get_document("kg-acme-person", "n1") is not None
