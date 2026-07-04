"""Twenty CRM: extract + write-back (CONCEPT:AU-KG.ingest.enterprise-source-extractor)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors import twenty as tw_ext
from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback


class FakeTwenty:
    def __init__(self):
        self.created = []

    # GraphQL-style for companies, REST-style for people/opps (exercise both shapes)
    def get_companies(self):
        return {
            "data": {"companies": {"edges": [{"node": {"id": "c1", "name": "Acme"}}]}}
        }

    def get_people(self):
        return {
            "data": [
                {
                    "id": "p1",
                    "name": {"firstName": "Ada", "lastName": "L"},
                    "email": "ada@x.io",
                    "companyId": "c1",
                }
            ]
        }

    def get_opportunities(self):
        return {
            "data": [
                {
                    "id": "o1",
                    "name": "Big deal",
                    "stage": "NEW",
                    "company": {"id": "c1"},
                }
            ]
        }

    def create_company(self, name):
        self.created.append(("company", name))


def test_twenty_extract():
    batch = tw_ext.extract({"client": FakeTwenty()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["twcompany:c1"].type == "Customer"
    assert by_id["twperson:p1"].type == "Person"
    assert by_id["twperson:p1"].props["name"] == "Ada L"
    assert by_id["twopp:o1"].type == "SalesOrder"
    assert by_id["twcompany:c1"].props["domain"] == "twenty"
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("twperson:p1", "twcompany:c1", "BELONGS_TO") in triples
    assert ("twopp:o1", "twcompany:c1", "PLACED_BY") in triples


def test_twenty_sink_create(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeTwenty()
    out = run_writeback(
        "twenty",
        client=client,
        creations=[{"type": "Customer", "name": "Globex"}],
        dry_run=False,
    )
    assert out["created"] == 1
    assert client.created == [("company", "Globex")]


def test_twenty_sink_refused(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    out = run_writeback(
        "twenty",
        client=FakeTwenty(),
        creations=[{"type": "Customer", "name": "x"}],
        dry_run=False,
    )
    assert out["status"] == "refused"
