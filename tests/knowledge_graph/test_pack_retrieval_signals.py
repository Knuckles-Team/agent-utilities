from __future__ import annotations

"""Tests for pack-driven retrieval signals: recency + source-trust.

CONCEPT:EG-KG.compute.rust-native-training-loss — Pack-Driven Retrieval Signals
"""


from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.models.schema_pack import RecencyDecaySpec, SchemaPack
from agent_utilities.models.schema_packs import get_schema_pack

_PATCH = (
    "agent_utilities.knowledge_graph.retrieval.hybrid_retriever.create_embedding_model"
)


def _retriever(pack):
    from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    return HybridRetriever(MagicMock(), schema_pack=pack)


@patch(_PATCH, side_effect=Exception("no embed in test"))
class TestRecencyBoost:
    def test_newer_outranks_older(self, _m):
        r = _retriever(get_schema_pack("research-state"))
        now = datetime.now(UTC)
        newer = {
            "type": "document",
            "event_time": (now - timedelta(days=1)).isoformat(),
        }
        older = {
            "type": "document",
            "event_time": (now - timedelta(days=730)).isoformat(),
        }
        assert r._recency_boost(newer) > r._recency_boost(older)
        # Boost is always a positive amplification (>= 1.0), never a penalty.
        assert r._recency_boost(older) >= 1.0

    def test_missing_event_time_is_neutral(self, _m):
        r = _retriever(get_schema_pack("research-state"))
        assert r._recency_boost({"type": "document"}) == 1.0
        assert r._recency_boost({"type": "document", "event_time": "not-a-date"}) == 1.0

    def test_untracked_type_is_neutral(self, _m):
        r = _retriever(get_schema_pack("research-state"))
        # "agent" has no recency spec in the research pack.
        node = {"type": "agent", "event_time": datetime.now(UTC).isoformat()}
        assert r._recency_boost(node) == 1.0

    def test_as_of_reference_time(self, _m):
        pack = SchemaPack(
            name="t",
            recency_decay={
                "document": RecencyDecaySpec(half_life_days=10, coefficient=1.0)
            },
        )
        r = _retriever(pack)
        node = {"type": "document", "event_time": "2024-01-01T00:00:00+00:00"}
        # Evaluated as-of a date close to the event => high boost; far => low boost.
        near = r._recency_boost(node, as_of="2024-01-02T00:00:00+00:00")
        far = r._recency_boost(node, as_of="2024-12-31T00:00:00+00:00")
        assert near > far

    def test_core_pack_is_no_op(self, _m):
        r = _retriever(get_schema_pack("core"))
        node = {"type": "document", "event_time": datetime.now(UTC).isoformat()}
        assert r._recency_boost(node) == 1.0


@patch(_PATCH, side_effect=Exception("no embed in test"))
class TestSourceTrustBoost:
    def test_trusted_source_boosted(self, _m):
        r = _retriever(get_schema_pack("research-state"))
        assert r._source_trust_boost({"source": "peer_reviewed"}) == pytest.approx(1.3)
        assert r._source_trust_boost({"source": "arxiv"}) == pytest.approx(1.2)
        assert r._source_trust_boost({"source": "blog"}) == pytest.approx(0.7)

    def test_unknown_and_missing_source_neutral(self, _m):
        r = _retriever(get_schema_pack("research-state"))
        assert r._source_trust_boost({"source": "mystery"}) == 1.0
        assert r._source_trust_boost({}) == 1.0

    def test_core_pack_is_no_op(self, _m):
        r = _retriever(get_schema_pack("core"))
        assert r._source_trust_boost({"source": "arxiv"}) == 1.0
