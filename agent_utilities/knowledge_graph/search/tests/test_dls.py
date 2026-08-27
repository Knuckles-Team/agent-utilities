"""DLS query rendering (CA-24-W05) + the binding-constraint projection/
aggregation negative control (`DEC-CA-09` review note / `CA-24` lane doc's
2026-08-25 amendment on `filter_commons_catalog`/BUG-PE-039)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.search import dls
from agent_utilities.knowledge_graph.search.tests.conftest import _match_query


def test_render_dls_query_empty_is_match_all() -> None:
    assert dls.render_dls_query([]) == {"match_all": {}}


def test_render_dls_query_single_marking() -> None:
    assert dls.render_dls_query(["restricted"]) == {
        "bool": {"must_not": [{"term": {"marking": "restricted"}}]}
    }


def test_render_dls_query_multiple_markings_uses_terms() -> None:
    query = dls.render_dls_query(["restricted", "secret", "restricted", ""])
    assert query == {
        "bool": {"must_not": [{"terms": {"marking": ["restricted", "secret"]}}]}
    }


def test_role_lacks_markings() -> None:
    lacking = dls.role_lacks_markings(
        role_tokens=["marking:internal", "kg:reader"],
        all_markings=["internal", "restricted", "secret"],
    )
    assert lacking == ["restricted", "secret"]


def test_render_dls_query_for_role() -> None:
    query = dls.render_dls_query_for_role(
        role_tokens=["marking:internal"], all_markings=["internal", "restricted"]
    )
    assert query == {"bool": {"must_not": [{"term": {"marking": "restricted"}}]}}


def test_hidden_from_actor() -> None:
    assert (
        dls.hidden_from_actor(["restricted"], role_tokens=["marking:internal"]) is True
    )
    assert (
        dls.hidden_from_actor(["restricted"], role_tokens=["marking:restricted"])
        is False
    )
    assert dls.hidden_from_actor([], role_tokens=[]) is False


def test_wrap_query_with_dls_no_restriction_returns_original_query() -> None:
    query = {"match": {"content": "ada"}}
    assert dls.wrap_query_with_dls(query, []) == query


def test_wrap_query_with_dls_composes_bool_filter() -> None:
    query = {"match": {"content": "ada"}}
    wrapped = dls.wrap_query_with_dls(query, ["restricted"])
    assert wrapped == {
        "bool": {
            "must": [{"match": {"content": "ada"}}],
            "filter": [{"bool": {"must_not": [{"term": {"marking": "restricted"}}]}}],
        }
    }


# ── the binding-constraint negative control: projection + aggregation ──────


def _fixture_docs() -> list[dict]:
    return [
        {
            "node_id": "n1",
            "node_type": "Person",
            "tenant": "acme",
            "marking": [],
            "content": "public one",
        },
        {
            "node_id": "n2",
            "node_type": "Person",
            "tenant": "acme",
            "marking": ["restricted"],
            "content": "secret two",
        },
    ]


def test_dls_query_still_hides_restricted_doc_under_a_projection() -> None:
    """A `RETURN t.id, t.name`-shaped projection is exactly the case
    `filter_commons_catalog`'s row-scan got wrong (BUG-PE-039): a fetched
    row missing its classification column looked unclassifiable. OpenSearch
    DLS does not scan fetched rows at all — the query DSL this module
    renders is merged into the query BEFORE execution, so it applies
    identically whether or not the caller also passed a `_source` filter.
    This test proves it against the same query-matching engine
    (`_match_query`) the fixture OpenSearch double uses, with a `_source`
    filter present on the search body (the closest a plain `dict`-query
    matcher can model "the caller projected fields")."""
    docs = _fixture_docs()
    dls_query = dls.render_dls_query(["restricted"])
    filtered_query = dls.wrap_query_with_dls({"match_all": {}}, ["restricted"])
    body = {
        "query": filtered_query,
        "_source": ["node_id"],  # a projection — must not defeat the filter
    }
    visible = [d for d in docs if _match_query(filtered_query, d)]
    assert [d["node_id"] for d in visible] == ["n1"]
    assert dls_query == {"bool": {"must_not": [{"term": {"marking": "restricted"}}]}}
    assert body["_source"] == [
        "node_id"
    ]  # the projection stays part of the real query body


def test_dls_query_still_hides_restricted_doc_under_aggregation_style_matching() -> (
    None
):
    """Same principle for an aggregation: the restricted doc must not be
    counted, exactly as it must not be returned as a hit."""
    docs = _fixture_docs()
    filtered_query = dls.wrap_query_with_dls({"match_all": {}}, ["restricted"])
    matching_count = sum(1 for d in docs if _match_query(filtered_query, d))
    assert matching_count == 1  # only the public doc contributes to the "aggregation"
