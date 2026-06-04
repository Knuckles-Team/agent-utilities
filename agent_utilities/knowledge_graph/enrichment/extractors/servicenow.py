"""ServiceNow ITSM source extractor (CONCEPT:KG-2.9).

Self-registering extractor that maps ServiceNow ITSM records — incidents,
changes, and CMDB configuration items — into the uniform ``ExtractionBatch``
shape (typed ``GraphNode`` + ``EnrichmentEdge``) so they persist through the one
generic writer with no edits to any shared hub file.

The ServiceNow client is **injected** (duck-typed) via ``config["client"]`` and
is expected to expose record-returning methods (``incidents()``, ``changes()``,
``cmdb_cis()``) that each yield a list of dict records with the usual
ServiceNow fields (``sys_id``, ``number``, ``short_description``, ``state``,
``priority``, ``cmdb_ci``, ``assigned_to``, ...). All field access is tolerant of
missing keys — this module performs **no network calls** itself.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_source

CATEGORY = "servicenow"


def _get(record: Any, key: str, default: Any = None) -> Any:
    """Tolerant field access for dict records (or attr-style objects)."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _ref(value: Any) -> str | None:
    """Normalise a ServiceNow reference field to a scalar string.

    Reference fields can arrive as a plain string or as a ``{"value": ...,
    "display_value": ...}`` dict. Returns ``None`` when empty.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value") or value.get("display_value")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _key(record: Any) -> str | None:
    """Stable identity for a record: prefer sys_id, fall back to number."""
    return _ref(_get(record, "sys_id")) or _ref(_get(record, "number"))


def _call(client: Any, name: str) -> list:
    """Call a client method if present, returning a list (tolerant)."""
    method = getattr(client, name, None)
    if not callable(method):
        return []
    result = method()
    return list(result) if result else []


def extract(config: Any) -> ExtractionBatch:
    """Extract ServiceNow ITSM records into a uniform ``ExtractionBatch``.

    ``config`` is a dict (or attr-style object) carrying an injected ``client``
    plus options. Incidents/changes become nodes linked to their affected CI
    (``AFFECTS``) and assignee (``ASSIGNED_TO``); CMDB CIs become
    ``ConfigurationItem`` nodes.
    """
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    # --- Incidents ---------------------------------------------------------
    for rec in _call(client, "incidents"):
        key = _key(rec)
        if not key:
            continue
        node_id = f"incident:{key}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Incident",
                props={
                    "number": _ref(_get(rec, "number")),
                    "short_description": _get(rec, "short_description"),
                    "state": _ref(_get(rec, "state")),
                    "priority": _ref(_get(rec, "priority")),
                },
            )
        )
        ci = _ref(_get(rec, "cmdb_ci"))
        if ci:
            edges.append(
                EnrichmentEdge(source=node_id, target=f"ci:{ci}", rel_type="AFFECTS")
            )
        assignee = _ref(_get(rec, "assigned_to"))
        if assignee:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"person:{assignee}",
                    rel_type="ASSIGNED_TO",
                )
            )

    # --- Changes -----------------------------------------------------------
    for rec in _call(client, "changes"):
        key = _key(rec)
        if not key:
            continue
        node_id = f"change:{key}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Change",
                props={
                    "number": _ref(_get(rec, "number")),
                    "short_description": _get(rec, "short_description"),
                    "state": _ref(_get(rec, "state")),
                    "priority": _ref(_get(rec, "priority")),
                },
            )
        )
        ci = _ref(_get(rec, "cmdb_ci"))
        if ci:
            edges.append(
                EnrichmentEdge(source=node_id, target=f"ci:{ci}", rel_type="AFFECTS")
            )
        assignee = _ref(_get(rec, "assigned_to"))
        if assignee:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"person:{assignee}",
                    rel_type="ASSIGNED_TO",
                )
            )

    # --- CMDB Configuration Items ------------------------------------------
    for rec in _call(client, "cmdb_cis"):
        key = _key(rec)
        if not key:
            continue
        nodes.append(
            GraphNode(
                id=f"ci:{key}",
                type="ConfigurationItem",
                props={
                    "name": _get(rec, "name"),
                    "short_description": _get(rec, "short_description"),
                    "state": _ref(_get(rec, "state")),
                },
            )
        )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_source(
    CATEGORY,
    extract,
    description="ServiceNow ITSM (incidents/changes/CMDB) → KG",
)
