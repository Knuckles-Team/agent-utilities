"""Unit tests for the EA tool clients (CONCEPT:AU-KG.ingest.enterprise-source-extractor)."""

from __future__ import annotations

from agent_utilities.ecosystem import ea_clients
from agent_utilities.ecosystem.ea_clients import LeanixEAClient, get_leanix_client


def _client() -> LeanixEAClient:
    c = LeanixEAClient("https://demo.leanix.net", "tok", verify_ssl=False)
    c._bearer = "bearer"  # skip token exchange
    return c


def test_get_leanix_client_unconfigured(monkeypatch):
    monkeypatch.setattr(
        ea_clients, "setting", lambda k, d=None, cast=None: "" if d == "" else d
    )
    assert get_leanix_client() is None


def test_get_leanix_client_configured(monkeypatch):
    vals = {"LEANIX_URL": "https://demo.leanix.net", "LEANIX_TOKEN": "tok"}
    monkeypatch.setattr(
        ea_clients, "setting", lambda k, d=None, cast=None: vals.get(k, d)
    )
    client = get_leanix_client()
    assert isinstance(client, LeanixEAClient)
    assert client.base_url == "https://demo.leanix.net"


def test_relation_fields_from_data_model():
    c = _client()
    c._data_model = {
        "factSheets": {
            "Application": {
                "relations": {
                    "relApplicationToITComponent": {
                        "targetFactSheetType": "ITComponent"
                    },
                    "relApplicationToBusinessCapability": {},
                    "lifecycle": {},  # non-rel field must be filtered out
                }
            }
        }
    }
    fields = c._relation_fields("Application")
    assert "relApplicationToITComponent" in fields
    assert "relApplicationToBusinessCapability" in fields
    assert "lifecycle" not in fields


def test_factsheets_parses_nodes_and_filters_since(monkeypatch):
    c = _client()
    c._data_model = {"factSheets": {"Application": {"relations": {}}}}

    pages = {
        None: {
            "allFactSheets": {
                "pageInfo": {"hasNextPage": False, "endCursor": "z"},
                "edges": [
                    {
                        "node": {
                            "id": "a",
                            "name": "A",
                            "type": "Application",
                            "updatedAt": "2026-01-01",
                        }
                    },
                    {
                        "node": {
                            "id": "b",
                            "name": "B",
                            "type": "Application",
                            "updatedAt": "2026-06-01",
                        }
                    },
                ],
            }
        }
    }
    monkeypatch.setattr(c, "_gql", lambda q, v=None: pages[v.get("after")])

    # No watermark: both returned.
    all_fs = c.factsheets("Application")
    assert {fs["id"] for fs in all_fs} == {"a", "b"}

    # Watermark drops the older one.
    delta = c.factsheets("Application", since="2026-03-01")
    assert {fs["id"] for fs in delta} == {"b"}


def test_factsheets_ids_narrowing(monkeypatch):
    c = _client()
    c._data_model = {"factSheets": {"Application": {"relations": {}}}}
    monkeypatch.setattr(
        c,
        "_gql",
        lambda q, v=None: {
            "allFactSheets": {
                "pageInfo": {"hasNextPage": False},
                "edges": [
                    {"node": {"id": "a", "type": "Application"}},
                    {"node": {"id": "b", "type": "Application"}},
                ],
            }
        },
    )
    only = c.factsheets("Application", ids=["b"])
    assert [fs["id"] for fs in only] == ["b"]


def test_write_methods_build_expected_gql(monkeypatch):
    c = _client()
    calls: list[tuple[str, dict]] = []

    def fake_gql(query, variables=None):
        calls.append((query, variables or {}))
        if "createFactSheet" in query:
            return {"createFactSheet": {"factSheet": {"id": "new-1"}}}
        return {"updateFactSheet": {"factSheet": {"id": variables.get("id")}}}

    monkeypatch.setattr(c, "_gql", fake_gql)

    assert c.create_fact_sheet("Application", "Svc")["id"] == "new-1"

    rel = c.create_fact_sheet_relation("fs1", "relApplicationToITComponent", "it1")
    assert rel["id"] == "fs1"
    patch = calls[-1][1]["patches"][0]
    assert patch["op"] == "add"
    assert patch["path"] == "/relApplicationToITComponent/new_1"
    assert '"factSheetId": "it1"' in patch["value"]

    tag = c.add_tag("fs1", "tag-9")
    assert tag["id"] == "fs1"
    assert calls[-1][1]["patches"][0]["path"] == "/tags"
