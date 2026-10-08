"""AU-CONTROL-R029: the ontology selects the sources; the report reads them live.

Two fake sources stand in for live systems: a supply-chain MCP server that
exposes suppliers and components, and a CMDB GraphQL API that exposes
services. Only metadata becomes graph facts; every entity row is read
through a virtual mapping at question time.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from agent_utilities.knowledge_graph.virtual_graph import (
    DiscoveredEntity,
    DiscoveredRelationship,
    MaterializationPolicy,
    MetadataContract,
    OperationAdapter,
    SourceConnection,
    TripleOntology,
    VirtualCatalog,
    VirtualMapping,
    cross_source_report,
    metadata_triples,
    select_sources,
    tbox_from_sparql,
)
from agent_utilities.knowledge_graph.virtual_graph.contracts import check_mapping
from agent_utilities.knowledge_graph.virtual_graph.federation import (
    answer_cross_source,
    install_cross_source,
)
from agent_utilities.knowledge_graph.virtual_graph.ontology import (
    ALT_LABEL,
    DOMAIN,
    LABEL,
    RANGE,
    SUBCLASS,
    TBOX_QUERIES,
)

EX = "http://example.org/ops#"
SUPPLIER, COMPONENT, SERVICE = EX + "Supplier", EX + "Component", EX + "Service"
SUPPLIED_BY, DEPENDS_ON = EX + "suppliedBy", EX + "dependsOn"
QUESTION = "How does the supply chain affect our services?"

TBOX = [
    (SUPPLIER, LABEL, "supplier"),
    (SUPPLIER, ALT_LABEL, "supply chain"),
    (COMPONENT, LABEL, "component"),
    (EX + "Chain", LABEL, "chain"),
    (SERVICE, LABEL, "service"),
    (SERVICE, ALT_LABEL, "services"),
    (EX + "CriticalService", SUBCLASS, SERVICE),
    (SUPPLIED_BY, DOMAIN, COMPONENT),
    (SUPPLIED_BY, RANGE, SUPPLIER),
    (DEPENDS_ON, DOMAIN, SERVICE),
    (DEPENDS_ON, RANGE, COMPONENT),
]

SCM_ROWS = {
    "list_suppliers": [
        {"supplier_id": "s1", "name": "Acme Chips", "risk": "high"},
        {"supplier_id": "s2", "name": "Bolt Works", "risk": "low"},
    ],
    "list_components": [
        {"part_no": "p1", "supplier_id": "s1", "title": "GPU"},
        {"part_no": "p2", "supplier_id": "s2", "title": "Rack"},
        {"part_no": "p3", "supplier_id": "s1", "title": "NIC"},
    ],
}
CMDB_ROWS = {
    "query_services": [
        {"svc": "billing", "component": "p1", "tier": "1"},
        {"svc": "search", "component": "p3", "tier": "2"},
        {"svc": "intranet", "component": "p9", "tier": "3"},
    ],
}


class Recorder:
    """A fake live source: answers operations and records every call."""

    def __init__(self, rows: Mapping[str, list[dict[str, Any]]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, operation: str, args: Mapping[str, Any]) -> list:
        self.calls.append((operation, dict(args)))
        rows = self.rows[operation]
        if "in" in args:
            return [r for r in rows if r.get(args["field"]) in set(args["in"])]
        return list(rows)


SCM = SourceConnection("scm", "mcp", "mcp://supply-chain/tools", ())
CMDB = SourceConnection("cmdb", "graphql", "graphql://cmdb/api", ("filter:in",))
SCM_META = MetadataContract(
    "scm",
    "2026-10",
    (
        DiscoveredEntity("supplier", "supplier_id", ("name", "risk"), "list_suppliers"),
        DiscoveredEntity(
            "component", "part_no", ("supplier_id", "title"), "list_components"
        ),
    ),
    (DiscoveredRelationship("component", "supplier_id", "supplier"),),
)
CMDB_META = MetadataContract(
    "cmdb",
    "v3",
    (DiscoveredEntity("service", "svc", ("component", "tier"), "query_services"),),
)
MAPPINGS = (
    VirtualMapping("m-supplier", "scm", "supplier", SUPPLIER, "supplier_id", (), True),
    VirtualMapping(
        "m-component",
        "scm",
        "component",
        COMPONENT,
        "part_no",
        ((SUPPLIED_BY, "supplier_id"),),
        True,
    ),
    VirtualMapping(
        "m-service",
        "cmdb",
        "service",
        SERVICE,
        "svc",
        ((DEPENDS_ON, "component"),),
        True,
    ),
)


def _catalog(
    mappings: Sequence[VirtualMapping] = MAPPINGS,
) -> tuple[VirtualCatalog, Recorder, Recorder]:
    scm, cmdb = Recorder(SCM_ROWS), Recorder(CMDB_ROWS)
    catalog = VirtualCatalog()
    catalog.register(SCM, SCM_META, OperationAdapter(SCM, SCM_META, scm))
    catalog.register(CMDB, CMDB_META, OperationAdapter(CMDB, CMDB_META, cmdb))
    for mapping in mappings:
        catalog.add_mapping(mapping)
    return catalog, scm, cmdb


def test_the_ontology_selects_both_sources_for_the_question() -> None:
    catalog, _, _ = _catalog()
    selection = select_sources(QUESTION, TripleOntology(TBOX), catalog)
    assert {c.class_iri for c in selection.concepts} == {SUPPLIER, SERVICE}
    assert EX + "Chain" not in {c.class_iri for c in selection.concepts}
    assert {r.predicate for r in selection.path} == {SUPPLIED_BY, DEPENDS_ON}
    chosen = {s.class_iri: s.mapping.source_id for s in selection.sources}
    assert chosen == {SUPPLIER: "scm", COMPONENT: "scm", SERVICE: "cmdb"}
    assert selection.complete


def test_the_report_joins_live_rows_across_sources_without_copying() -> None:
    catalog, scm, cmdb = _catalog()
    report = asyncio.run(cross_source_report(QUESTION, TripleOntology(TBOX), catalog))
    body = report.to_dict()
    assert body["complete"] and body["reason"] == "answered"
    pairs = sorted((row[SUPPLIER]["name"], row[SERVICE]["svc"]) for row in report.rows)
    assert pairs == [("Acme Chips", "billing"), ("Acme Chips", "search")]
    assert body["materialized_rows"] == 0
    assert {read["mode"] for read in body["reads"]} == {"virtual"}
    assert cmdb.calls == [
        ("query_services", {"field": "component", "in": ["p1", "p2", "p3"]})
    ]
    assert all(args == {} for _, args in scm.calls), "scm lacks filter:in"
    facts = [f for hop in body["path"] for f in hop["facts"]]
    assert [DEPENDS_ON, DOMAIN, SERVICE] in facts
    assert {s["contract_digest"][:7] for s in body["sources"]} == {"sha256:"}


def test_an_unapproved_mapping_leaves_the_class_uncovered_and_reads_nothing() -> None:
    unapproved = dataclasses.replace(MAPPINGS[2], approved=False)
    catalog, scm, cmdb = _catalog((*MAPPINGS[:2], unapproved))
    report = asyncio.run(cross_source_report(QUESTION, TripleOntology(TBOX), catalog))
    assert not report.complete and report.reason == "uncovered_classes"
    assert report.selection.uncovered == (SERVICE,)
    assert scm.calls == [] and cmdb.calls == []


def test_a_subclass_mapping_serves_its_superclass() -> None:
    critical = dataclasses.replace(
        MAPPINGS[2], mapping_id="m-critical", class_iri=EX + "CriticalService"
    )
    catalog, _, _ = _catalog((*MAPPINGS[:2], critical))
    selection = select_sources(QUESTION, TripleOntology(TBOX), catalog)
    assert {s.mapping.mapping_id for s in selection.sources} >= {"m-critical"}


def test_the_row_budget_marks_the_report_incomplete() -> None:
    catalog, _, _ = _catalog()
    report = asyncio.run(
        cross_source_report(QUESTION, TripleOntology(TBOX), catalog, row_budget=1)
    )
    assert not report.complete and report.reason == "row_budget_exceeded"
    assert report.rows == []


def test_a_single_concept_question_is_not_cross_source() -> None:
    catalog, _, _ = _catalog()
    selection = select_sources("list every supplier", TripleOntology(TBOX), catalog)
    assert selection.reason == "fewer_than_two_concepts"


def test_mappings_must_name_discovered_entities_and_fields() -> None:
    bad = VirtualMapping("m-x", "scm", "supplier", SUPPLIER, "vat_no", (), True)
    with pytest.raises(ValueError, match="undiscovered fields"):
        check_mapping(bad, SCM_META)
    ghost = VirtualMapping("m-y", "scm", "invoice", SUPPLIER, "id", (), True)
    with pytest.raises(ValueError, match="undiscovered entity"):
        check_mapping(ghost, SCM_META)


@pytest.mark.parametrize(
    "ref",
    [
        "postgres://admin:pw@db/x",  # sanitizer:ignore -- synthetic refusal fixture
        "https://api/x?token=abc",
        "not a ref",
    ],
)
def test_a_connection_refuses_credentials_and_raw_strings(ref: str) -> None:
    with pytest.raises(ValueError):
        SourceConnection("x", "sql", ref)


def test_only_metadata_is_materialized() -> None:
    triples = metadata_triples(CMDB, CMDB_META, MAPPINGS[2:])
    text = repr(triples)
    assert "billing" not in text and "p1" not in text
    assert any(o.endswith("#VirtualMapping") for _, _, o in triples)
    copyable = SourceConnection("lake", "iceberg", "iceberg://lake/ops", ("copy",))
    policy = MaterializationPolicy(hot_reads=10)
    assert policy.decide(MAPPINGS[0], copyable, reads=12) == "materialize"
    assert policy.decide(MAPPINGS[0], copyable, reads=3) == "virtual"
    assert policy.decide(MAPPINGS[0], CMDB, reads=500) == "virtual"


def test_the_tbox_comes_from_eg_sparql() -> None:
    answers = {
        "labels": [{"s": SUPPLIER, "p": LABEL, "o": "supplier"}],
        "ancestry": [{"s": EX + "CriticalService", "o": SERVICE}],
        "relations": [{"p": DEPENDS_ON, "d": SERVICE, "r": COMPONENT}],
    }
    asked: list[str] = []

    async def sparql(query: str) -> list[dict[str, Any]]:
        asked.append(query)
        kind = next(k for k, q in TBOX_QUERIES if q == query)
        return answers[kind]

    ontology = asyncio.run(tbox_from_sparql(sparql))
    assert len(asked) == 3 and "subClassOf>+" in asked[1]
    assert SERVICE in ontology.ancestors(EX + "CriticalService")
    assert ontology.relations[0].predicate == DEPENDS_ON


@pytest.fixture
def installed() -> Iterator[VirtualCatalog]:
    catalog, _, _ = _catalog()

    async def ontology() -> TripleOntology:
        return TripleOntology(TBOX)

    install_cross_source(catalog, ontology)
    yield catalog
    install_cross_source(None)


def test_the_installed_catalog_answers_and_ignores_non_cross_source(installed) -> None:
    report = asyncio.run(answer_cross_source(QUESTION))
    assert report is not None and report.complete
    assert asyncio.run(answer_cross_source("restart the gateway")) is None


def test_ask_routes_planning_and_cross_source_questions(installed) -> None:
    from agent_utilities.mcp.tools import intent_tools

    async def ask(text: str, **hints: Any) -> dict[str, Any] | None:
        return await intent_tools._question_route("ask", text, hints, "intent:x")

    report = asyncio.run(ask(QUESTION))
    assert report is not None
    assert report["routing"]["chosen_tool"] == "cross_source_report"
    assert report["result"]["kind"] == "cross_source_report"
    plan = asyncio.run(ask("How can I deploy the billing service?"))
    assert plan is not None and plan["result"]["kind"] == "task_plan"
    assert plan["routing"]["chosen_tool"] == "task_planner"
    assert asyncio.run(ask("How can I deploy it?", tool="graph_run")) is None
    assert asyncio.run(intent_tools._question_route("act", QUESTION, {}, "i")) is None


def test_dispatch_intent_returns_the_plan_for_a_bare_ask(installed) -> None:
    from agent_utilities.mcp.tools.intent_tools import dispatch_intent

    out = asyncio.run(dispatch_intent("ask", "What steps deploy the billing service?"))
    assert out["executed"] is True
    assert out["result"]["kind"] == "task_plan"
