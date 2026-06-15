"""Cross-source inventory push (CONCEPT:KG-2.9).

The capstone of bidirectional enrichment: take the KG's reconciled technology
inventory (infra/topology nodes + LeanIX ITComponents/Applications + TRM products/
assets) and create, in a target CMDB/ERP, the items that don't yet exist there.

Reconciliation is the OWL layer's job — ``ALIGNED_WITH`` identity (the same
mechanism Camunda⇆ARIS⇆Egeria use) collapses an infra server, its LeanIX
ITComponent, and its CMDB CI into one identity. Here we simply skip any candidate
that is already represented in the target (its ``domain`` is the target, or it is
``ALIGNED_WITH`` a node that is) and propose the rest as creations through the
unified, fail-closed :func:`run_writeback`.
"""

from __future__ import annotations

import logging
from typing import Any

from .core import run_writeback

logger = logging.getLogger(__name__)

# KG node types that constitute the technology inventory pushed upstream.
INVENTORY_TYPES: tuple[str, ...] = (
    "Server",
    "Service",
    "Container",
    "HardwareNode",
    "ITComponent",
    "Application",
    "TechnologyProduct",
    "AssetInstance",
    "ConfigurationItem",
)


def collect_inventory_creations(
    backend: Any,
    target: str,
    *,
    node_types: tuple[str, ...] | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Inventory candidates not yet present in ``target`` → ``[{type,name}]``.

    Best-effort over the backend: a node is a candidate when it is named, of an
    inventory type, and its ``domain`` is not already the target. The
    ``ALIGNED_WITH`` cross-source-identity exclusion is applied when the backend
    can serve it (degrades to the domain check otherwise).
    """
    if backend is None:
        return []
    types = node_types or INVENTORY_TYPES
    rows: list[dict] = []
    try:
        rows = (
            backend.execute(
                "MATCH (n) WHERE n.name IS NOT NULL "
                "AND (n.domain IS NULL OR n.domain <> $t) "
                "RETURN n.type AS type, n.name AS name, n.id AS id LIMIT $limit",
                {"t": target, "limit": limit},
            )
            or []
        )
    except Exception:  # noqa: BLE001 - tolerant: no candidates rather than crash
        logger.debug("inventory candidate query failed", exc_info=True)
        return []

    type_set = {t.lower() for t in types}
    aligned = _aligned_to_target_ids(backend, target)
    creations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        ntype = str(r.get("type") or "")
        name = r.get("name")
        nid = str(r.get("id") or "")
        if ntype.lower() not in type_set or not name:
            continue
        if nid in aligned:  # already represented upstream via ALIGNED_WITH identity
            continue
        if name in seen:
            continue
        seen.add(name)
        creations.append({"type": ntype, "name": name, "node": nid})
    return creations


def _aligned_to_target_ids(backend: Any, target: str) -> set[str]:
    """Node ids ALIGNED_WITH a node already in ``target`` (best-effort)."""
    try:
        rows = backend.execute(
            "MATCH (n)-[:ALIGNED_WITH]-(m) WHERE m.domain = $t RETURN n.id AS id",
            {"t": target},
        )
        return {
            str(r["id"]) for r in (rows or []) if isinstance(r, dict) and r.get("id")
        }
    except Exception:  # noqa: BLE001 - alignment exclusion is optional
        return set()


def push_inventory(
    target: str,
    *,
    backend: Any = None,
    engine: Any = None,
    node_types: tuple[str, ...] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Collect the reconciled inventory and create the missing items in ``target``.

    Fail-closed + dry-run-first via :func:`run_writeback` (the target's
    ``*_ENABLE_WRITE`` gate still applies).
    """
    creations = collect_inventory_creations(backend, target, node_types=node_types)
    result = run_writeback(
        target, backend=backend, engine=engine, dry_run=dry_run, creations=creations
    )
    if isinstance(result, dict):
        result["inventory_candidates"] = len(creations)
    return result
