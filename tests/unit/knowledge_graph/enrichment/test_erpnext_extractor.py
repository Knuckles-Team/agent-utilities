"""ERPNext/Frappe source extractor tests (CONCEPT:KG-2.9)."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.enrichment.extractors.erpnext import extract
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)


class FakeFrappeClient:
    """Duck-typed Frappe client returning sample rows per doctype."""

    _DATA: dict[str, list[dict[str, Any]]] = {
        "Employee": [
            {"name": "HR-EMP-001", "employee_name": "Ada Lovelace", "department": "Engineering"},
            {"name": "HR-EMP-002", "employee_name": "Grace Hopper", "department": "Engineering"},
            {"name": "HR-EMP-003", "employee_name": "Solo Worker"},  # no department
        ],
        "Customer": [
            {"name": "CUST-001", "customer_name": "Acme Corp"},
            {"name": "CUST-002", "customer_name": "Globex"},
        ],
        "Sales Order": [
            {"name": "SO-0001", "customer": "CUST-001", "grand_total": 1500.0},
            {"name": "SO-0002", "customer": "CUST-002", "grand_total": 99.5},
        ],
        "Item": [
            {"name": "ITEM-WIDGET", "item_name": "Widget"},
            {"name": "ITEM-GADGET", "item_name": "Gadget"},
        ],
        "Issue": [
            {
                "name": "ISS-0001",
                "subject": "Login fails",
                "status": "Open",
                "priority": "High",
                "customer": "CUST-001",
            },
            {"name": "ISS-0002", "subject": "Slow page"},  # no customer
        ],
    }

    def get_list(self, doctype):
        return list(self._DATA.get(doctype, []))


class Config:
    def __init__(self, client):
        self.client = client


class FakeBackend:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, s, t, **props):
        self.edges.append((s, t, props.get("rel_type")))


def _run():
    return extract(Config(FakeFrappeClient()))


def test_source_is_registered():
    src = get_source("erpnext")
    assert src is not None
    assert src.extract is extract
    assert "ERPNext" in src.description


def test_employee_and_orgunit_nodes_and_member_of():
    batch = _run()
    by_id = {n.id: n for n in batch.nodes}

    emp = by_id["employee:HR-EMP-001"]
    assert emp.type == "Employee"
    assert emp.props["employee_name"] == "Ada Lovelace"
    assert emp.props["department"] == "Engineering"

    # OrgUnit synthesised once (deduped across two Engineering employees).
    org = by_id["orgunit:Engineering"]
    assert org.type == "OrgUnit"
    org_nodes = [n for n in batch.nodes if n.type == "OrgUnit"]
    assert len(org_nodes) == 1

    member_edges = [e for e in batch.edges if e.rel_type == "MEMBER_OF"]
    assert ("employee:HR-EMP-001", "orgunit:Engineering") in {
        (e.source, e.target) for e in member_edges
    }
    # Employee without a department emits no MEMBER_OF edge.
    assert all(e.source != "employee:HR-EMP-003" for e in member_edges)


def test_customer_and_item_nodes():
    batch = _run()
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["customer:CUST-001"].type == "Customer"
    assert by_id["item:ITEM-WIDGET"].type == "Item"


def test_sales_order_node_and_placed_by_edge():
    batch = _run()
    by_id = {n.id: n for n in batch.nodes}
    so = by_id["order:SO-0001"]
    assert so.type == "SalesOrder"
    assert so.props["grand_total"] == 1500.0

    placed_by = {(e.source, e.target) for e in batch.edges if e.rel_type == "PLACED_BY"}
    assert ("order:SO-0001", "customer:CUST-001") in placed_by
    assert ("order:SO-0002", "customer:CUST-002") in placed_by


def test_issue_maps_to_erpnext_issue_and_raised_by():
    batch = _run()
    by_id = {n.id: n for n in batch.nodes}

    issue = by_id["erpnextissue:ISS-0001"]
    assert issue.type == "ErpNextIssue"
    assert issue.props["subject"] == "Login fails"
    assert issue.props["status"] == "Open"

    raised_by = {
        (e.source, e.target) for e in batch.edges if e.rel_type == "RAISED_BY"
    }
    assert ("erpnextissue:ISS-0001", "customer:CUST-001") in raised_by
    # Issue without a customer emits no RAISED_BY edge.
    assert all(e.source != "erpnextissue:ISS-0002" for e in batch.edges)


def test_write_batch_persists_to_backend():
    batch = _run()
    backend = FakeBackend()
    n, e = write_batch(backend, batch)
    assert n == len(batch.nodes)
    assert e == len(batch.edges)
    assert backend.nodes["employee:HR-EMP-001"]["type"] == "Employee"
    assert backend.nodes["orgunit:Engineering"]["type"] == "OrgUnit"
    assert ("order:SO-0001", "customer:CUST-001", "PLACED_BY") in backend.edges
    assert ("employee:HR-EMP-001", "orgunit:Engineering", "MEMBER_OF") in backend.edges
