"""Canonical source-id naming schema + named-graph routing.

CONCEPT:AU-KG.ingest.source-id-naming-schema — every connector's ``source_system`` is built by
``make_source_id`` (``<system>[:<instance>][:<kind>]``, slugged) so the fleet partitions into
predictable ``urn:source:...`` named graphs and code never lands in the SPARQL default graph.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.sparql.source_partition import (
    SOURCE_GRAPH_PREFIX,
    default_graph_leak_labels,
    graph_uri_for,
    graph_uri_for_source,
    make_source_id,
    reset_default_graph_report,
    route_graph_uri,
    slug_part,
    source_of,
)


def test_make_source_id_hierarchy_and_slugging():
    assert make_source_id("leanix") == "leanix"
    assert make_source_id("gitlab", "gl.corp") == "gitlab:gl.corp"
    assert make_source_id("gitlab", "gl.corp", "code") == "gitlab:gl.corp:code"
    assert make_source_id("code", "Agent-Utilities") == "code:agent-utilities"
    # Non-slug chars (spaces, punctuation) collapse to '-'; colons can't smuggle a part.
    assert make_source_id("confluence", "Eng Wiki!") == "confluence:eng-wiki"
    assert make_source_id("gitlab", "a:b:c") == "gitlab:a-b-c"
    # Empty optional parts are dropped, not stringified.
    assert make_source_id("jira", "", None) == "jira"


def test_slug_part_keeps_dot_dash_underscore():
    assert slug_part("GL.Corp_1-x") == "gl.corp_1-x"
    assert slug_part("  Spaces Here  ") == "spaces-here"
    assert slug_part(":::") == ""


def test_named_graph_iri_round_trips_through_source_of():
    src = make_source_id("code", "agent-utilities")
    props = {"source_system": src}
    assert source_of(props) == "code:agent-utilities"
    assert graph_uri_for(props) == f"{SOURCE_GRAPH_PREFIX}code:agent-utilities"
    assert graph_uri_for_source(src) == "urn:source:code:agent-utilities"


def test_generic_sources_route_to_default_graph():
    # The engine's internal provenance tags must NOT create urn:source:* graphs.
    for generic in ("system", "internal", "kg", ""):
        assert source_of({"source_system": generic}) is None
        assert graph_uri_for({"source_system": generic}) is None
    # A real external source does route.
    assert graph_uri_for({"source_system": "leanix"}) == "urn:source:leanix"


def test_route_guard_counts_leak_labels_but_not_internal(monkeypatch):
    reset_default_graph_report()
    # A sourced node routes to its graph and is NOT counted as a default-graph write.
    assert route_graph_uri({"source_system": "code:au"}, "Code") == "urn:source:code:au"
    # An internal label with no source is expected in the default graph — not a leak.
    assert route_graph_uri({}, "Claim") is None
    # A first-class entity with no source IS a leak — counted by label.
    assert route_graph_uri({}, "Document") is None
    leaks = default_graph_leak_labels()
    assert leaks.get("Document") == 1
    assert "Claim" not in leaks  # internal, not a leak
    assert "Code" not in leaks  # was sourced
    reset_default_graph_report()


def test_strict_mode_rejects_unsourced_external_node(monkeypatch):
    reset_default_graph_report()
    monkeypatch.setenv("KG_STRICT_SOURCE_PARTITION", "true")
    # Strict: a non-internal label with no source fails loudly instead of leaking.
    with pytest.raises(ValueError, match="no source_system"):
        route_graph_uri({}, "Document")
    # Internal labels are still allowed through in strict mode.
    assert route_graph_uri({}, "Claim") is None
    reset_default_graph_report()


def test_coverage_doctor_flags_leaks_across_any_backend():
    from agent_utilities.knowledge_graph.backends.sparql.source_partition import (
        source_partition_coverage,
    )

    class _FakeBackend:
        def execute(self, q):
            # Code fully sourced; some Document unsourced (leak); Claim internal.
            return [
                {"label": "Code", "total": 100, "sourced": 100},
                {"label": "Document", "total": 10, "sourced": 6},
                {"label": "Claim", "total": 50, "sourced": 0},
            ]

    cov = source_partition_coverage(_FakeBackend())
    assert cov["supported"] is True
    assert cov["leaking"] is True
    assert cov["leaks"] == {"Document": 4}  # Code clean, Claim internal
    assert cov["by_label"]["Code"]["unsourced"] == 0

    # A backend with no execute() degrades gracefully.
    assert source_partition_coverage(object())["supported"] is False
