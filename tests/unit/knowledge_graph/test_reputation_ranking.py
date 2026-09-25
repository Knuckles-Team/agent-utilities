"""Learned reliability fills the existing pure EG ranking request."""

import asyncio

import pytest

from agent_utilities.knowledge_graph.retrieval.reputation_ranking import (
    rank_with_reputation,
)


class Query:
    def __init__(self, rows):
        self.rows = rows
        self.sql_text = None
        self.candidates = None

    async def sql(self, query):
        self.sql_text = query
        return self.rows

    async def rank_by_provenance(self, candidates, weights=None):
        self.candidates = candidates
        return {"ranked": [row["id"] for row in candidates]}


class Client:
    def __init__(self, rows):
        self.query = Query(rows)


def test_visible_estimate_replaces_prior_and_missing_estimate_keeps_it():
    client = Client([{"subject": "source-a", "mean": 0.8}])
    candidates = [
        {
            "id": "doc-a",
            "source_id": "source-a",
            "similarity": 0.7,
            "freshness": 0.9,
            "source_reliability": 0.4,
        },
        {
            "id": "doc-b",
            "source_id": "source-b",
            "similarity": 0.6,
            "freshness": 0.9,
            "source_reliability": 0.5,
        },
    ]
    result = asyncio.run(rank_with_reputation(client, candidates))
    assert result == {"ranked": ["doc-a", "doc-b"]}
    assert [row["source_reliability"] for row in client.query.candidates] == [0.8, 0.5]
    assert all("source_id" not in row for row in client.query.candidates)
    assert "subject_kind = 'option'" in client.query.sql_text
    assert "source-a" in client.query.sql_text


def test_malformed_or_unavailable_reputation_fails_before_ranking():
    client = Client([{"subject": "other", "mean": 0.7}])
    candidate = {"id": "doc-a", "source_id": "source-a", "source_reliability": 0.4}
    with pytest.raises(ValueError, match="unexpected source"):
        asyncio.run(rank_with_reputation(client, [candidate]))
    assert client.query.candidates is None
    with pytest.raises(ValueError, match="probability"):
        asyncio.run(
            rank_with_reputation(
                Client([]), [{**candidate, "source_reliability": float("nan")}]
            )
        )
    with pytest.raises(ValueError, match="1..128"):
        asyncio.run(rank_with_reputation(Client([]), []))


def test_source_id_is_sql_escaped():
    client = Client([])
    asyncio.run(
        rank_with_reputation(
            client, [{"id": "doc", "source_id": "o'brien", "source_reliability": 0.5}]
        )
    )
    assert "'o''brien'" in client.query.sql_text
