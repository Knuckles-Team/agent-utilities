"""Odoo CRM source extractor (CONCEPT:KG-2.9).

Self-registering extractor that maps **Odoo** CRM records into the uniform
``ExtractionBatch`` shape (typed ``GraphNode`` + ``EnrichmentEdge``) so they
persist through the one generic writer with no edits to any shared hub file.

Mapping (Odoo record → GraphNode)::

    res.partner -> ``Customer`` id=odoo_customer:{id}
    crm.lead    -> ``Lead``     id=odoo_lead:{id}   (BELONGS_TO customer)

So Odoo joins the same ``crm`` cohort as Twenty (the cross-vendor CRM redundancy
the consolidation engine collapses). The Odoo client is **injected** (duck-typed)
via ``config["client"]`` and is expected to expose ``list_partners()`` /
``list_leads()``. All field access is tolerant; no network calls happen here.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "odoo"


def _get(record: Any, key: str, default: Any = None) -> Any:
    """Tolerant field access for dict records (or attr-style objects)."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _first(record: Any, *keys: str) -> Any:
    """Return the first present, non-empty value among ``keys``."""
    for key in keys:
        val = _get(record, key)
        if val is not None and val != "":
            return val
    return None


def _call(client: Any, name: str) -> list:
    """Call a client method if present, returning a list (tolerant)."""
    method = getattr(client, name, None)
    if not callable(method):
        return []
    try:
        result = method()
    except TypeError:
        try:
            result = method({})
        except Exception:
            return []
    except Exception:
        return []
    if isinstance(result, dict):
        result = (
            result.get("records") or result.get("items") or result.get("data") or []
        )
    return list(result) if result else []


def _rel_id(value: Any) -> Any:
    """Odoo many2one fields are ``[id, label]`` pairs — return the id."""
    if isinstance(value, list | tuple) and value:
        return value[0]
    return value


def extract(config: Any) -> ExtractionBatch:
    """Extract Odoo CRM customers + leads into a uniform ``ExtractionBatch``."""
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for rec in _call(client, "list_partners"):
        pid = _first(rec, "id", "name")
        if not pid:
            continue
        nodes.append(
            GraphNode(
                id=f"odoo_customer:{pid}",
                type="Customer",
                props={
                    "name": _first(rec, "display_name", "name"),
                    "capability": "crm",
                },
            )
        )

    for rec in _call(client, "list_leads"):
        lid = _first(rec, "id", "name")
        if not lid:
            continue
        node_id = f"odoo_lead:{lid}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Lead",
                props={
                    "name": _first(rec, "display_name", "name"),
                    "capability": "crm",
                },
            )
        )
        partner_ref = _rel_id(_first(rec, "partner_id", "partner"))
        if partner_ref:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"odoo_customer:{partner_ref}",
                    rel_type="BELONGS_TO",
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(CATEGORY, extract, description="Odoo CRM (customers/leads) → KG")
