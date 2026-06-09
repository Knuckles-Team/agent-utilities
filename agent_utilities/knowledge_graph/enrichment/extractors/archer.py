"""RSA Archer GRC source extractor (CONCEPT:KG-2.9).

Self-registering extractor that maps **RSA Archer** governance/risk/compliance
records into the uniform ``ExtractionBatch`` shape (typed ``GraphNode`` +
``EnrichmentEdge``) so they persist through the one generic writer with no edits
to any shared hub file.

Mapping (Archer record → GraphNode), with derived governance edges::

    risk     -> ``Risk``              id=archer_risk:{id}
    control  -> ``ComplianceControl`` id=archer_control:{id}  (MITIGATES risk)
    finding  -> ``Finding``           id=archer_finding:{id}  (AFFECTS control)

The Archer client is **injected** (duck-typed) via ``config["client"]`` and is
expected to expose ``list_risks()`` / ``list_controls()`` / ``list_findings()``.
All field access is tolerant and this module performs **no** network calls itself.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_source

CATEGORY = "archer"


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
        result = result.get("value") or result.get("items") or result.get("data") or []
    return list(result) if result else []


def extract(config: Any) -> ExtractionBatch:
    """Extract Archer GRC records into a uniform ``ExtractionBatch``.

    Risks/controls/findings become typed governance nodes; controls link
    ``MITIGATES`` their risk and findings link ``AFFECTS`` their control.
    """
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for rec in _call(client, "list_risks"):
        rid = _first(rec, "id", "Id", "name", "Name")
        if not rid:
            continue
        nodes.append(
            GraphNode(
                id=f"archer_risk:{rid}",
                type="Risk",
                props={
                    "name": _first(rec, "name", "Name", "Title"),
                    "capability": "grc",
                },
            )
        )

    for rec in _call(client, "list_controls"):
        cid = _first(rec, "id", "Id", "name", "Name")
        if not cid:
            continue
        node_id = f"archer_control:{cid}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="ComplianceControl",
                props={
                    "name": _first(rec, "name", "Name", "Title"),
                    "capability": "grc",
                },
            )
        )
        risk_ref = _first(rec, "riskId", "RiskId", "risk_id")
        if risk_ref:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"archer_risk:{risk_ref}",
                    rel_type="MITIGATES",
                )
            )

    for rec in _call(client, "list_findings"):
        fid = _first(rec, "id", "Id", "name", "Name")
        if not fid:
            continue
        node_id = f"archer_finding:{fid}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Finding",
                props={
                    "name": _first(rec, "name", "Name", "Title"),
                    "capability": "grc",
                },
            )
        )
        control_ref = _first(rec, "controlId", "ControlId", "control_id")
        if control_ref:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"archer_control:{control_ref}",
                    rel_type="AFFECTS",
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_source(
    CATEGORY,
    extract,
    description="RSA Archer GRC (risks/controls/findings) → KG",
)
