"""Characterization tests for ``parse_hyde_plan`` (CX-AU-08).

Pins observed behaviour of
``agent_utilities.knowledge_graph.retrieval.hyde_planner.parse_hyde_plan``
before refactor, including edge cases not obvious from a casual read:

* the ``{.*}`` DOTALL regex is GREEDY -- it spans from the first ``{`` to the
  LAST ``}`` in the whole string, so two separate JSON-looking objects in one
  raw response collapse into one unparseable blob and fall through to the
  safe single-query fallback (rather than picking the first object),
* an invalid/unknown ``search_mode`` in the parsed JSON is silently replaced
  by the fallback mode, not rejected outright,
* comma-string ``keywords`` are split and stripped, dropping empty segments,
* an empty/whitespace-only ``vector_queries`` list in the parsed JSON is
  backfilled with ``original_query`` (the plan is never left with zero
  queries),
* ``mode_hint`` only changes the FALLBACK mode (used on parse failure or an
  invalid parsed mode) -- it does not override an explicitly valid
  ``search_mode`` present in the JSON,
* no ``{`` in ``raw`` at all raises internally and is swallowed by the
  ``except`` into the safe fallback.

CX-AU-08 owns only ``agent_utilities/knowledge_graph/retrieval``; this test
file is added in isolation (commit 1) and must be byte-identical across
commit 2 (the refactor).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.retrieval.hyde_planner import parse_hyde_plan


def test_valid_json_object_is_parsed():
    raw = '{"vector_queries": ["q1", "q2"], "keywords": ["a", "b"], "search_mode": "deep"}'
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.vector_queries == ["q1", "q2"]
    assert plan.keywords == ["a", "b"]
    assert plan.search_mode == "deep"


def test_json_embedded_in_prose_is_extracted():
    raw = 'Here is the plan:\n{"vector_queries": ["q1"]}\nThanks.'
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.vector_queries == ["q1"]


def test_comma_string_keywords_are_split_and_stripped():
    raw = '{"vector_queries": ["q1"], "keywords": " a ,b, , c "}'
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.keywords == ["a", "b", "c"]


def test_keywords_as_list_is_stringified_elementwise():
    raw = '{"vector_queries": ["q1"], "keywords": [1, "b"]}'
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.keywords == ["1", "b"]


def test_keywords_absent_defaults_to_empty_list():
    raw = '{"vector_queries": ["q1"]}'
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.keywords == []


def test_invalid_search_mode_falls_back_to_standard_by_default():
    raw = '{"vector_queries": ["q1"], "search_mode": "nonsense"}'
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.search_mode == "standard"


def test_invalid_search_mode_falls_back_to_mode_hint_when_given():
    raw = '{"vector_queries": ["q1"], "search_mode": "nonsense"}'
    plan = parse_hyde_plan(raw, original_query="orig", mode_hint="deep")
    assert plan.search_mode == "deep"


def test_valid_explicit_search_mode_is_not_overridden_by_mode_hint():
    raw = '{"vector_queries": ["q1"], "search_mode": "standard"}'
    plan = parse_hyde_plan(raw, original_query="orig", mode_hint="deep")
    assert plan.search_mode == "standard"


def test_search_mode_absent_uses_fallback_mode_standard():
    raw = '{"vector_queries": ["q1"]}'
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.search_mode == "standard"


def test_search_mode_absent_uses_fallback_mode_deep_hint():
    raw = '{"vector_queries": ["q1"]}'
    plan = parse_hyde_plan(raw, original_query="orig", mode_hint="deep")
    assert plan.search_mode == "deep"


def test_empty_vector_queries_in_json_is_backfilled_with_original():
    raw = '{"vector_queries": []}'
    plan = parse_hyde_plan(raw, original_query="the original query")
    assert plan.vector_queries == ["the original query"]


def test_whitespace_only_vector_queries_are_dropped_and_backfilled():
    raw = '{"vector_queries": ["   ", ""]}'
    plan = parse_hyde_plan(raw, original_query="fallback")
    assert plan.vector_queries == ["fallback"]


def test_no_json_object_at_all_falls_back_safely():
    raw = "not json at all"
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.vector_queries == ["orig"]
    assert plan.search_mode == "standard"


def test_malformed_json_falls_back_safely():
    raw = "{not valid json}"
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.vector_queries == ["orig"]


def test_fallback_respects_mode_hint_deep():
    raw = "garbage"
    plan = parse_hyde_plan(raw, original_query="orig", mode_hint="deep")
    assert plan.search_mode == "deep"


def test_two_json_objects_in_one_raw_collapse_to_unparseable_and_fallback():
    """The extraction regex is GREEDY: it spans first '{' to the LAST '}' in
    the string, so two separate JSON objects become one invalid blob rather
    than the first one being picked."""
    raw = 'plan A: {"vector_queries": ["q1"]} plan B: {"other": 1}'
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.vector_queries == ["orig"]


def test_non_dict_json_top_level_falls_back_safely():
    # No top-level '{' outside a nested object -> `.get` on int/list-like
    # scalars never even happens; this pins the "no match -> ValueError ->
    # fallback" path for a bracketed non-object payload.
    raw = "[1, 2, 3]"
    plan = parse_hyde_plan(raw, original_query="orig")
    assert plan.vector_queries == ["orig"]
