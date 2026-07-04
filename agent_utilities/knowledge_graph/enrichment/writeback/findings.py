"""KG findings → issue tickets (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

The inference→action loop: turn KG-derived risk findings (TRM TechnologyRisk nodes
— end-of-life, vulnerabilities, risk ratings) into filed tickets in an issue
tracker, fail-closed + dry-run-first via :func:`run_writeback`.
"""

from __future__ import annotations

import logging
from typing import Any

from .core import run_writeback

logger = logging.getLogger(__name__)


def collect_risk_findings(backend: Any, *, limit: int = 1000) -> list[dict[str, Any]]:
    """TechnologyRisk nodes → issue creations ``[{title, body, node}]``."""
    if backend is None:
        return []
    try:
        rows = (
            backend.execute(
                "MATCH (n) WHERE n.type = 'TechnologyRisk' "
                "RETURN n.id AS id, n.name AS name, n.riskRating AS rating, "
                "n.endOfLifeDate AS eol LIMIT $limit",
                {"limit": limit},
            )
            or []
        )
    except Exception:  # noqa: BLE001
        logger.debug("risk findings query failed", exc_info=True)
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict) or not r.get("id"):
            continue
        name = r.get("name") or r["id"]
        body_bits = []
        if r.get("rating"):
            body_bits.append(f"Risk rating: {r['rating']}")
        if r.get("eol"):
            body_bits.append(f"End of life: {r['eol']}")
        out.append(
            {
                "title": f"Technology risk: {name}",
                "body": "; ".join(body_bits) or "KG-detected technology risk.",
                "node": str(r["id"]),
            }
        )
    return out


def push_findings(
    target: str,
    *,
    backend: Any = None,
    engine: Any = None,
    project: dict[str, Any] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Collect KG risk findings and file them as issues in ``target``.

    ``project`` carries the tracker target (e.g. ``{"project_id": ...}`` for
    GitLab/Plane, ``{"owner":..., "repo":...}`` for GitHub). Fail-closed via
    :func:`run_writeback` (the target's ``*_ENABLE_WRITE`` gate applies).
    """
    creations = collect_risk_findings(backend)
    if project:
        for c in creations:
            c.update(project)
    result = run_writeback(
        target, backend=backend, engine=engine, dry_run=dry_run, creations=creations
    )
    if isinstance(result, dict):
        result["findings"] = len(creations)
    return result
