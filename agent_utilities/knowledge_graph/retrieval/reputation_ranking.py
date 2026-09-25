"""Rank request-carried candidates using visible EG reputation estimates.

The engine's `reputation` relation applies the caller's decision-log visibility.
RankByProvenance stays pure compute; this adapter fills its source reliability
from those visible estimates and uses the stored value only as a prior.
"""

from __future__ import annotations

import math
from typing import Any, Protocol


class ReputationQuery(Protocol):
    async def sql(self, query: str) -> list[dict[str, Any]]: ...

    async def rank_by_provenance(
        self, candidates: list[dict[str, Any]], weights: dict[str, float] | None = None
    ) -> dict[str, Any]: ...


class ReputationClient(Protocol):
    @property
    def query(self) -> ReputationQuery: ...


def _literal(value: str) -> str:
    if not value or len(value) > 256 or "\x00" in value:
        raise ValueError("source id is empty, oversized, or contains NUL")
    return "'" + value.replace("'", "''") + "'"


def _prior(candidate: dict[str, Any]) -> float:
    value = candidate.get("source_reliability")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("candidate source_reliability must be a probability")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("candidate source_reliability must be a probability")
    return number


async def rank_with_reputation(
    client: ReputationClient,
    candidates: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fill learned reliability before one authenticated engine rank call.

    At most 128 candidates/source IDs are admitted. The visible relation is
    queried once; an unavailable or malformed answer fails rather than
    silently claiming a learned estimate.
    """
    if not 1 <= len(candidates) <= 128:
        raise ValueError("rank_with_reputation needs 1..128 candidates")
    source_ids = []
    for candidate in candidates:
        source = candidate.get("source_id", candidate.get("id"))
        if not isinstance(source, str):
            raise ValueError("candidate source id must be text")
        _literal(source)
        _prior(candidate)
        source_ids.append(source)
    literals = ", ".join(_literal(source) for source in sorted(set(source_ids)))
    rows = await client.query.sql(
        "SELECT subject, mean FROM reputation "
        "WHERE subject_kind = 'option' AND status = 'estimated' "
        f"AND subject IN ({literals})"
    )
    learned: dict[str, float] = {}
    for row in rows:
        subject, mean = row.get("subject"), row.get("mean")
        if not isinstance(subject, str) or subject not in source_ids:
            raise ValueError("reputation response contains an unexpected source")
        learned[subject] = _prior({"source_reliability": mean})
    ranked_input = []
    for candidate, source in zip(candidates, source_ids, strict=True):
        item = {key: value for key, value in candidate.items() if key != "source_id"}
        item["source_reliability"] = learned.get(source, _prior(candidate))
        ranked_input.append(item)
    return await client.query.rank_by_provenance(ranked_input, weights)
