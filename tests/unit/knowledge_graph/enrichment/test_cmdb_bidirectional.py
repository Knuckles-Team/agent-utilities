"""ServiceNow + ERPNext bidirectional: TRM/inventory read + CMDB/ERP write (KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors import erpnext as erp_ext
from agent_utilities.knowledge_graph.enrichment.extractors import servicenow as snow_ext
from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback


# ── ServiceNow extractor: TRM + risk + federation ────────────────────────────
class FakeSnowClient:
    def cmdb_cis(self):
        return [
            {
                "sys_id": "ci1",
                "name": "host01",
                "ci_class": "cmdb_ci_server",
                "model_id": "m1",
            }
        ]

    def cmdb_models(self):
        return [
            {
                "sys_id": "m1",
                "display_name": "PowerEdge R740",
                "manufacturer": "Dell",
                "end_of_life": "2027-01-01",
            }
        ]

    def assets(self):
        return [
            {
                "sys_id": "a1",
                "display_name": "Laptop-7",
                "install_status": "in_use",
                "model": "m1",
            }
        ]


def test_servicenow_trm_extract():
    batch = snow_ext.extract({"client": FakeSnowClient()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["ci:ci1"].type == "ConfigurationItem"
    assert by_id["snproduct:m1"].type == "TechnologyProduct"
    assert by_id["asset:a1"].type == "AssetInstance"
    # federation key everywhere
    assert by_id["ci:ci1"].props["externalToolId"] == "ci1"
    assert by_id["ci:ci1"].props["domain"] == "servicenow"
    # EOL product → risk lifecycle prop + a TechnologyRisk node
    assert by_id["snproduct:m1"].props["endOfLifeDate"] == "2027-01-01"
    assert any(n.type == "TechnologyRisk" for n in batch.nodes)
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("ci:ci1", "snproduct:m1", "INSTANCE_OF") in triples
    assert ("asset:a1", "snproduct:m1", "INSTANCE_OF") in triples


# ── ERPNext extractor: Asset/Warehouse + federation ──────────────────────────
class FakeFrappe:
    _D = {
        "Asset": [
            {
                "name": "AST-1",
                "asset_name": "Server A",
                "item_code": "ITEM-SRV",
                "status": "Submitted",
                "warehouse": "WH-Main",
            }
        ],
        "Warehouse": [{"name": "WH-Main", "warehouse_name": "Main"}],
        "Item": [{"name": "ITEM-SRV", "item_name": "Server", "actual_qty": 4}],
    }

    def get_list(self, doctype):
        return self._D.get(doctype, [])


def test_erpnext_asset_inventory_extract():
    batch = erp_ext.extract({"client": FakeFrappe()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["asset:AST-1"].type == "AssetInstance"
    assert by_id["warehouse:WH-Main"].type == "Warehouse"
    assert by_id["item:ITEM-SRV"].props["actual_qty"] == 4
    # federation stamp on every node
    assert by_id["asset:AST-1"].props["externalToolId"] == "AST-1"
    assert by_id["asset:AST-1"].props["domain"] == "erpnext"
    triples = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("asset:AST-1", "item:ITEM-SRV", "INSTANCE_OF") in triples
    assert ("asset:AST-1", "warehouse:WH-Main", "LOCATED_IN") in triples


# ── Write-back sinks ─────────────────────────────────────────────────────────
class FakeSnowApi:
    def __init__(self):
        self.created = []

    def create_cmdb_instance(self, className, attributes, source):
        self.created.append((className, attributes.get("name")))


class FakeErpApi:
    def __init__(self):
        self.created = []

    def create_document(self, doctype, data):
        self.created.append((doctype, data))


def test_servicenow_sink_dry_run_and_live(monkeypatch):
    out = run_writeback(
        "servicenow",
        client=FakeSnowApi(),
        creations=[{"type": "server", "name": "host42"}],
        dry_run=True,
    )
    assert out["proposals"][0]["op"] == "create_cmdb_instance"
    assert out["proposals"][0]["class"] == "cmdb_ci_server"

    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    api = FakeSnowApi()
    out = run_writeback(
        "servicenow",
        client=api,
        creations=[{"type": "service", "name": "billing-svc"}],
        dry_run=False,
    )
    assert out["created"] == 1
    assert api.created == [("cmdb_ci_service", "billing-svc")]


def test_servicenow_sink_refused_without_flag(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    out = run_writeback(
        "servicenow",
        client=FakeSnowApi(),
        creations=[{"type": "server", "name": "x"}],
        dry_run=False,
    )
    assert out["status"] == "refused"


def test_erpnext_sink_live_create(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    api = FakeErpApi()
    out = run_writeback(
        "erpnext",
        client=api,
        creations=[{"type": "assetinstance", "name": "Server A"}],
        dry_run=False,
    )
    assert out["created"] == 1
    assert api.created[0][0] == "Asset"
    assert api.created[0][1]["asset_name"] == "Server A"
