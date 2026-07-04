"""Salesforce CRM source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Maps the Salesforce CRM into the uniform ExtractionBatch via SOQL: Account →
:Customer, Contact → :Person, Opportunity → :SalesOrder, with PLACED_BY edges.
Every node carries ``externalToolId`` (Salesforce Id) + ``domain="salesforce"``.
Client (``salesforce_agent.auth.get_client()``) is injected; tolerant.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "salesforce"
_DOMAIN = "salesforce"

# (SObject, node type, id prefix)
_OBJECTS = [
    ("Account", "Customer", "sfaccount"),
    ("Contact", "Person", "sfcontact"),
    ("Opportunity", "SalesOrder", "sfopp"),
]


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _records(client: Any, soql: str) -> list[dict]:
    q = getattr(client, "query", None)
    if not callable(q):
        return []
    try:
        res = q(soql)
    except Exception:  # noqa: BLE001
        return []
    if isinstance(res, dict):
        recs = res.get("records")
        return recs if isinstance(recs, list) else []
    return list(res) if isinstance(res, list) else []


def extract(config: Any) -> ExtractionBatch:
    """Extract Salesforce CRM objects into a uniform ExtractionBatch."""
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for sobject, label, prefix in _OBJECTS:
        soql = f"SELECT Id, Name, AccountId FROM {sobject} LIMIT 2000"
        for r in _records(client, soql):
            sid = r.get("Id")
            if not sid:
                continue
            node_id = f"{prefix}:{sid}"
            nodes.append(
                GraphNode(
                    id=node_id,
                    type=label,
                    props={
                        "name": r.get("Name"),
                        "externalToolId": str(sid),
                        "domain": _DOMAIN,
                    },
                )
            )
            acct = r.get("AccountId")
            if acct and sobject in ("Contact", "Opportunity"):
                edges.append(
                    EnrichmentEdge(
                        source=node_id,
                        target=f"sfaccount:{acct}",
                        rel_type="PLACED_BY"
                        if sobject == "Opportunity"
                        else "BELONGS_TO",
                    )
                )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="Salesforce CRM (accounts/contacts/opps) → KG"
)
