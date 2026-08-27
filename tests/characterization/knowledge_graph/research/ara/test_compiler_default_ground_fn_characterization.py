"""Characterization tests for the ``_ground`` closure returned by
``ARACompiler._default_ground_fn`` (CX-AU-09).

CCN 12 at time of writing (the nested closure, not the enclosing method --
lizard, the repo's own gate, charges a nested closure's branches to the
closure itself, not the enclosing def; see scripts/check_complexity.py's own
docstring). Because the closure has no importable name of its own, it is
exercised through ``ARACompiler._ground(artifact)`` -- the class's own
grounding entry point, which calls ``_default_ground_fn()`` and invokes the
returned closure once per claim when no ``ground_fn`` was injected. This
avoids importing the closure as a new test-only symbol while still driving
every branch of its actual logic (token extraction, the query_cypher
best-effort try/except, and the row-matching predicate).

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED ``compiler.py`` before any refactor commit, and must
not change during the refactor commit that follows.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara.artifact import (
    Claim,
    ResearchArtifact,
)
from agent_utilities.knowledge_graph.research.ara.compiler import ARACompiler


class _Engine:
    def __init__(self, concepts=None, raise_on_query=False):
        self._concepts = concepts if concepts is not None else []
        self._raise_on_query = raise_on_query
        self.queries: list[str] = []

    def query_cypher(self, q, params=None):
        self.queries.append(q)
        if self._raise_on_query:
            raise RuntimeError("backend unavailable")
        return self._concepts


def _artifact_with_claim(statement: str) -> ResearchArtifact:
    return ResearchArtifact(
        article_id="p",
        title="P",
        claims=[Claim(id="claim:p:0", statement=statement)],
    )


def test_matches_by_substring_token_over_length_four() -> None:
    eng = _Engine(
        concepts=[{"id": "concept:owl-reasoning", "name": "owl reasoning bridge"}]
    )
    art = _artifact_with_claim("we improve reasoning about claims")
    compiler = ARACompiler(eng)
    groundings = compiler._ground(art)
    assert groundings["claim:p:0"] == ["concept:owl-reasoning"]


def test_tokens_of_length_four_or_less_are_ignored() -> None:
    # "over" (4 chars) must NOT match "coverage" even though it's a substring,
    # because tokens with len <= 4 are excluded from the match set.
    eng = _Engine(concepts=[{"id": "concept:c1", "name": "test coverage report"}])
    art = _artifact_with_claim("this is over the line")
    compiler = ARACompiler(eng)
    groundings = compiler._ground(art)
    assert groundings == {}


def test_query_cypher_uses_the_supported_query_verbatim() -> None:
    eng = _Engine(concepts=[])
    art = _artifact_with_claim("anything statement")
    ARACompiler(eng)._ground(art)
    assert eng.queries == [
        "MATCH (c:Concept) RETURN c.id AS id, c.name AS name LIMIT 200"
    ]


def test_query_cypher_exception_degrades_to_no_groundings() -> None:
    eng = _Engine(raise_on_query=True)
    art = _artifact_with_claim("reasoning about claims")
    compiler = ARACompiler(eng)
    groundings = compiler._ground(art)
    assert groundings == {}


def test_rows_missing_id_or_non_dict_are_skipped() -> None:
    eng = _Engine(
        concepts=[
            "not-a-dict",
            {"name": "reasoning concept"},  # no id
            {"id": "concept:valid", "name": "reasoning concept"},
        ]
    )
    art = _artifact_with_claim("about reasoning things")
    compiler = ARACompiler(eng)
    groundings = compiler._ground(art)
    assert groundings["claim:p:0"] == ["concept:valid"]


def test_row_with_no_name_is_never_a_hit() -> None:
    eng = _Engine(concepts=[{"id": "concept:noname", "name": ""}])
    art = _artifact_with_claim("reasoning about things")
    compiler = ARACompiler(eng)
    groundings = compiler._ground(art)
    assert groundings == {}


def test_multiple_matching_rows_preserve_row_order_uncapped_by_this_layer() -> None:
    # de-duplication against the claim's own evidence_ids happens one layer up
    # in _ground, not inside this closure -- the closure itself returns every
    # matching row's id, in row order, even if that means duplicates.
    eng = _Engine(
        concepts=[
            {"id": "concept:a", "name": "reasoning first"},
            {"id": "concept:b", "name": "grounding second"},
            {"id": "concept:a", "name": "reasoning again"},
        ]
    )
    art = _artifact_with_claim("reasoning and grounding together")
    compiler = ARACompiler(eng)
    groundings = compiler._ground(art)
    assert groundings["claim:p:0"] == ["concept:a", "concept:b", "concept:a"]


def test_matching_is_case_insensitive_on_both_sides() -> None:
    eng = _Engine(concepts=[{"id": "concept:x", "name": "REASONING Engine"}])
    art = _artifact_with_claim("REASONING about claims")
    compiler = ARACompiler(eng)
    groundings = compiler._ground(art)
    assert groundings["claim:p:0"] == ["concept:x"]
