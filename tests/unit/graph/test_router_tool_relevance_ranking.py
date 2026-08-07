#!/usr/bin/python
from __future__ import annotations

"""Regression tests for D-CDX-53: mixed legacy/canonical persisted Tool
``relevance_score`` values must be ranked on ONE canonical scale, not the
raw stored value.

Before the fix, ``expert_executor_step``'s dynamic-tool Cypher query ordered
directly on ``t.relevance_score`` in the database. A legacy row scored
``0.9`` (float, ``[0, 1]`` convention) and a canonical row scored ``50``
(int, ``[0, 100]`` convention) are semantically "0.9 is better than 0.5",
but raw numeric ordering ranks ``50 > 0.9`` — ~100x apart and inverted. Worse,
a DB-side ``LIMIT`` can truncate the better legacy tool out of the result
entirely before any normalization ever runs.

``_rank_tool_rows_by_relevance`` (agent_utilities/graph/_router_impl.py) is
the fix: it normalizes every candidate row through the exact same canonical
boundary as ``ToolNode``/``MCPToolInfo``
(``agent_utilities.models.tool_score.normalize_legacy_relevance_score``)
before comparing, so ranking is deterministic and scale-correct regardless
of which convention a given row was written under.
"""

from agent_utilities.graph._router_impl import _rank_tool_rows_by_relevance


def _row(name: str, score) -> dict:
    return {"name": name, "server": "srv", "relevance_score": score}


def test_legacy_float_and_canonical_int_rank_on_the_same_scale() -> None:
    """A legacy 0.9 (== canonical 90) must outrank a canonical 50 — proving
    the two conventions are compared on ONE scale, not two."""
    rows = [_row("canonical_mid", 50), _row("legacy_high", 0.9)]
    ranked = _rank_tool_rows_by_relevance(rows)
    assert [r["name"] for r in ranked] == ["legacy_high", "canonical_mid"]


def test_semantically_equal_scores_rank_together_not_100x_apart() -> None:
    """Legacy 0.5 and canonical 50 are the SAME quality tool under the two
    conventions; a canonical 51 (marginally better) must still outrank both,
    and the legacy 0.5 must not fall to the bottom just because 0.5 < 50 on
    the raw numeric scale."""
    rows = [
        _row("legacy_half", 0.5),
        _row("canonical_half", 50),
        _row("canonical_slightly_better", 51),
    ]
    ranked = _rank_tool_rows_by_relevance(rows)
    names = [r["name"] for r in ranked]
    assert names[0] == "canonical_slightly_better"
    # legacy_half and canonical_half are tied at 50 on the canonical scale —
    # both must rank ABOVE nothing else lower, and neither must be dropped.
    assert set(names[1:]) == {"legacy_half", "canonical_half"}


def test_pool_truncation_does_not_drop_the_best_legacy_tool() -> None:
    """Simulates what a raw DB-side ``ORDER BY relevance_score DESC LIMIT``
    would have discarded: many canonical-scored rows numerically larger than
    a raw legacy float, even though the legacy tool is semantically the
    best. The candidate pool passed in here represents the bounded
    name-ordered fetch (_TOOL_CANDIDATE_POOL_LIMIT) — ranking must still
    surface the legacy tool at the top once normalized."""
    rows = [_row(f"mediocre_{i}", 10) for i in range(10)]
    rows.append(_row("best_legacy", 0.95))  # canonical-equivalent: 95
    ranked = _rank_tool_rows_by_relevance(rows, limit=5)
    assert ranked[0]["name"] == "best_legacy"
    assert len(ranked) == 5


def test_limit_is_respected() -> None:
    rows = [_row(f"t{i}", i) for i in range(20)]
    ranked = _rank_tool_rows_by_relevance(rows, limit=5)
    assert len(ranked) == 5
    # Highest canonical ints (19..15) must be the top 5.
    assert {r["name"] for r in ranked} == {"t19", "t18", "t17", "t16", "t15"}


def test_corrupt_score_ranks_last_but_is_not_dropped() -> None:
    """An out-of-range/ambiguous stored value (negative, non-legacy
    fraction, bool, string) must never crowd out a well-scored tool, but it
    also must not silently vanish from the candidate list — it should still
    appear, just ranked as if its score were 0."""
    rows = [
        _row("good", 80),
        _row("corrupt_negative", -5),
        _row("corrupt_string", "50"),
        _row("corrupt_bool", True),
        _row("corrupt_fraction", 1.9),
    ]
    ranked = _rank_tool_rows_by_relevance(rows, limit=10)
    assert ranked[0]["name"] == "good"
    assert {r["name"] for r in ranked} == {r_["name"] for r_ in rows}


def test_missing_score_defaults_to_zero_rank() -> None:
    rows = [{"name": "no_score", "server": "srv"}, _row("has_score", 1)]
    ranked = _rank_tool_rows_by_relevance(rows)
    assert ranked[0]["name"] == "has_score"
    assert ranked[1]["name"] == "no_score"


def test_empty_and_none_rows_handled() -> None:
    assert _rank_tool_rows_by_relevance([]) == []
    assert _rank_tool_rows_by_relevance(None) == []
