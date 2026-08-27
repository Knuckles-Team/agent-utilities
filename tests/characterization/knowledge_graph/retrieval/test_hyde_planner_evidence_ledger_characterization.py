"""Characterization tests for ``build_evidence_ledger`` (CX-AU-08).

Pins observed behaviour of
``agent_utilities.knowledge_graph.retrieval.hyde_planner.build_evidence_ledger``
before refactor: the ACCEPT/REJECT threshold is inclusive (``score >=
accept_floor`` at exactly 0.38 ACCEPTs), ``content`` falls back to ``name``
then to ``""``, the numeric-token regex captures a leading ``$`` and
thousands separators but not a token immediately preceded by a word
character or another ``$`` (so ``"v2"`` and ``"$$5"`` are NOT captured as
plain numbers the second time), content is truncated to 280 chars in the
row but the FULL content still feeds numeric extraction, ``rank`` is the
enumerate index (not sorted by score), and ``accepted_numbers`` aggregates
numbers only from ACCEPTed rows in row order.

CX-AU-08 owns only ``agent_utilities/knowledge_graph/retrieval``; this test
file is added in isolation (commit 1) and must be byte-identical across
commit 2 (the refactor).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.retrieval.hyde_planner import (
    build_evidence_ledger,
)


def test_score_at_exactly_the_standard_threshold_is_accepted():
    nodes = [{"id": "1", "_score": 0.38, "content": "x"}]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["decision"] == "ACCEPT"
    assert ledger["accept_count"] == 1
    assert ledger["reject_count"] == 0


def test_score_just_below_threshold_is_rejected():
    nodes = [{"id": "1", "_score": 0.379999, "content": "x"}]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["decision"] == "REJECT"
    assert ledger["rows"][0]["reason"] == "near-miss"


def test_missing_score_defaults_to_zero_and_is_rejected():
    nodes = [{"id": "1", "content": "x"}]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["score"] == 0.0
    assert ledger["rows"][0]["decision"] == "REJECT"


def test_content_falls_back_to_name_then_empty_string():
    nodes = [
        {"id": "1", "_score": 1.0, "content": "the content"},
        {"id": "2", "_score": 1.0, "name": "the name"},
        {"id": "3", "_score": 1.0},
    ]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["content"] == "the content"
    assert ledger["rows"][1]["content"] == "the name"
    assert ledger["rows"][2]["content"] == ""


def test_content_key_wins_over_name_when_both_present():
    """Pins the fallback ORDER, not just presence: `content or name`, not
    the reverse -- a node carrying both fields must report `content`."""
    nodes = [{"id": "1", "_score": 1.0, "content": "the content", "name": "the name"}]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["content"] == "the content"


def test_content_truncated_to_280_chars_in_row():
    long_content = "a" * 400
    nodes = [{"id": "1", "_score": 1.0, "content": long_content}]
    ledger = build_evidence_ledger("q", nodes)
    assert len(ledger["rows"][0]["content"]) == 280


def test_numeric_extraction_captures_dollar_amounts_and_thousands_commas():
    nodes = [{"id": "1", "_score": 1.0, "content": "it costs $1,200.50 total"}]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["numbers"] == ["$1,200.50"]


def test_numeric_extraction_excludes_number_glued_to_a_word():
    """The regex has a negative lookbehind on \\w and $, so 'v2' does not
    yield a bare '2' token."""
    nodes = [{"id": "1", "_score": 1.0, "content": "schema v2 shipped"}]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["numbers"] == []


def test_numeric_extraction_still_scans_full_content_past_280_chars():
    long_content = "a" * 300 + " $99 at the end"
    nodes = [{"id": "1", "_score": 1.0, "content": long_content}]
    ledger = build_evidence_ledger("q", nodes)
    assert "$99" in ledger["rows"][0]["numbers"]


def test_rank_is_enumerate_index_not_score_sorted():
    nodes = [
        {"id": "low", "_score": 0.1, "content": "x"},
        {"id": "high", "_score": 0.9, "content": "y"},
    ]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["rank"] == 0
    assert ledger["rows"][0]["id"] == "low"
    assert ledger["rows"][1]["rank"] == 1
    assert ledger["rows"][1]["id"] == "high"


def test_event_time_passthrough():
    nodes = [{"id": "1", "_score": 1.0, "content": "x", "event_time": "2026-01-01"}]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["event_time"] == "2026-01-01"


def test_accepted_numbers_aggregate_only_from_accepted_rows_in_row_order():
    nodes = [
        {"id": "a", "_score": 0.5, "content": "$10"},  # ACCEPT
        {"id": "b", "_score": 0.0, "content": "$20"},  # REJECT
        {"id": "c", "_score": 0.9, "content": "$30 and $40"},  # ACCEPT
    ]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["accepted_ids"] == ["a", "c"]
    assert ledger["accepted_numbers"] == ["$10", "$30", "$40"]


def test_empty_nodes_yields_empty_ledger():
    ledger = build_evidence_ledger("q", [])
    assert ledger == {
        "query": "q",
        "rows": [],
        "accepted_ids": [],
        "accept_count": 0,
        "reject_count": 0,
        "accepted_numbers": [],
    }


def test_score_rounded_to_four_decimal_places_in_row():
    nodes = [{"id": "1", "_score": 0.123456789, "content": "x"}]
    ledger = build_evidence_ledger("q", nodes)
    assert ledger["rows"][0]["score"] == 0.1235
