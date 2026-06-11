"""ERPNext/Frappe source extractor (CONCEPT:KG-2.9).

Self-registering source extractor that turns Frappe doctypes (Employee,
Customer, Sales Order, Item) into the uniform :class:`ExtractionBatch` of typed
:class:`GraphNode` + :class:`EnrichmentEdge` value objects, so HR/sales data
from an ERPNext instance flows into the KG through the single generic writer
with no edits to any shared hub file.

The injected ``config.client`` is duck-typed: it only needs a
``get_list(doctype) -> list[dict]`` method returning Frappe rows. No network or
ERPNext import happens at module import time — the client is supplied by the
caller, so this module is import-safe even when ERPNext is unavailable.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor


def _first(row: dict, *keys: str) -> Any:
    """Return the first present, non-empty value among ``keys`` (field tolerance)."""
    for key in keys:
        val = row.get(key)
        if val is not None and val != "":
            return val
    return None


def _get_client(config: Any) -> Any:
    """Extract the injected duck-typed client from ``config`` (attr or mapping)."""
    client = getattr(config, "client", None)
    if client is None and isinstance(config, dict):
        client = config.get("client")
    if client is None:
        raise ValueError("erpnext extractor requires config.client with get_list()")
    return client


def _get_list(client: Any, doctype: str) -> list[dict]:
    """Tolerantly fetch a doctype list; never raise for an absent/empty doctype."""
    try:
        rows = client.get_list(doctype)
    except Exception:
        return []
    return list(rows or [])


def extract(config: Any) -> ExtractionBatch:
    """Build an :class:`ExtractionBatch` from ERPNext/Frappe doctypes.

    Maps Employee/Customer/Sales Order/Item rows to typed nodes and emits
    ``PLACED_BY`` (SalesOrder→Customer) and ``MEMBER_OF`` (Employee→OrgUnit)
    edges, synthesising :class:`GraphNode` ``OrgUnit`` nodes for departments.
    """
    client = _get_client(config)
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    orgunits: dict[str, GraphNode] = {}

    # Employee -> Employee node (+ MEMBER_OF -> OrgUnit)
    for row in _get_list(client, "Employee"):
        name = _first(row, "name", "employee", "employee_name")
        if name is None:
            continue
        employee_name = _first(row, "employee_name", "name")
        department = _first(row, "department", "dept")
        node_id = f"employee:{name}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Employee",
                props={"employee_name": employee_name, "department": department},
            )
        )
        if department is not None:
            org_id = f"orgunit:{department}"
            if org_id not in orgunits:
                orgunits[org_id] = GraphNode(
                    id=org_id, type="OrgUnit", props={"name": department}
                )
            edges.append(
                EnrichmentEdge(source=node_id, target=org_id, rel_type="MEMBER_OF")
            )

    # Customer -> Customer node
    for row in _get_list(client, "Customer"):
        name = _first(row, "name", "customer", "customer_name")
        if name is None:
            continue
        customer_name = _first(row, "customer_name", "name")
        nodes.append(
            GraphNode(
                id=f"customer:{name}",
                type="Customer",
                props={"customer_name": customer_name},
            )
        )

    # Sales Order -> SalesOrder node (+ PLACED_BY -> Customer)
    for row in _get_list(client, "Sales Order"):
        name = _first(row, "name", "order")
        if name is None:
            continue
        grand_total = _first(row, "grand_total", "total")
        node_id = f"order:{name}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="SalesOrder",
                props={"grand_total": grand_total},
            )
        )
        customer = _first(row, "customer", "customer_name")
        if customer is not None:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"customer:{customer}",
                    rel_type="PLACED_BY",
                )
            )

    # Item -> Item node
    for row in _get_list(client, "Item"):
        name = _first(row, "name", "item_code", "item_name")
        if name is None:
            continue
        item_name = _first(row, "item_name", "name")
        nodes.append(
            GraphNode(
                id=f"item:{name}",
                type="Item",
                props={"item_name": item_name},
            )
        )

    # Issue -> ErpNextIssue node (ITSM helpdesk ticket; ServiceNow-incident
    # analogue, crosswalked to canonical :ApplicationEvent). Emits RAISED_BY ->
    # Customer when present, mirroring ServiceNow's assignee/CI linkage.
    for row in _get_list(client, "Issue"):
        name = _first(row, "name", "subject")
        if name is None:
            continue
        node_id = f"erpnextissue:{name}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="ErpNextIssue",
                props={
                    "subject": _first(row, "subject", "name"),
                    "status": _first(row, "status", "state"),
                    "priority": _first(row, "priority"),
                },
            )
        )
        customer = _first(row, "customer", "customer_name")
        if customer is not None:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"customer:{customer}",
                    rel_type="RAISED_BY",
                )
            )

    nodes.extend(orgunits.values())
    return ExtractionBatch(category="erpnext", nodes=nodes, edges=edges)


register_extractor("erpnext", extract, description="ERPNext/Frappe (HR/sales) → KG")
