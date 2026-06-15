"""Backfeed KG-derived knowledge into LeanIX (CONCEPT:KG-2.9).

Closes the loop the other way: relationships the OWL reasoner *infers*, enrichment
attributes/tags the KG derives, and (optionally) fact sheets for entities the KG
discovered elsewhere are written **back** into LeanIX — the EA system-of-record.

Safety is first-class:

* **Fail-closed.** Live writes require ``LEANIX_ENABLE_WRITE`` to be set; without
  it the action runs in preview only.
* **Dry-run by default.** Every entry point defaults to ``dry_run=True``, returning
  the exact set of *proposed* writes so they can be reviewed before anything
  mutates the system-of-record.
* **Reversible provenance.** Inferred relations are tagged
  :data:`PROVENANCE_TAG` so an operator can find and undo agent-written links.
* **Resolution by federation key.** KG nodes map to LeanIX fact sheets via the
  ``externalToolId`` stamped at ingest (see the LeanIX extractor); a node without
  one is never silently written unless explicit auto-create is requested.

Transport is the single :class:`ea_clients.LeanixEAClient` — no second client.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

PROVENANCE_TAG = "agent-utilities:inferred"


@dataclass
class BackfeedResult:
    """Counts + proposals from a backfeed run."""

    relations_written: int = 0
    relations_skipped: int = 0
    enrichments_written: int = 0
    factsheets_created: int = 0
    errors: int = 0
    proposals: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "relations_written": self.relations_written,
            "relations_skipped": self.relations_skipped,
            "enrichments_written": self.enrichments_written,
            "factsheets_created": self.factsheets_created,
            "errors": self.errors,
            "proposals": self.proposals,
        }


def _edge_parts(edge: Any) -> tuple[str | None, str | None, str | None]:
    """Normalize an inferred edge (tuple or dict) to ``(source, rel_type, target)``."""
    if isinstance(edge, dict):
        src = edge.get("source") or edge.get("subject")
        rel = edge.get("rel_type") or edge.get("type") or edge.get("predicate")
        tgt = edge.get("target") or edge.get("object")
        return src, rel, tgt
    if isinstance(edge, list | tuple) and len(edge) == 3:
        return edge[0], edge[1], edge[2]
    return None, None, None


def build_factsheet_resolver(backend: Any) -> Callable[[str], str | None]:
    """KG node id → LeanIX GUID via the ``externalToolId`` federation key."""
    mapping: dict[str, str] = {}
    if backend is not None:
        try:
            rows = backend.execute(
                "MATCH (n) WHERE n.externalToolId IS NOT NULL AND n.domain = 'leanix' "
                "RETURN n.id AS id, n.externalToolId AS guid"
            )
            for r in rows or []:
                if isinstance(r, dict) and r.get("id") and r.get("guid"):
                    mapping[str(r["id"])] = str(r["guid"])
        except Exception:  # noqa: BLE001 - resolution is best-effort
            logger.debug("factsheet resolver: backend query failed", exc_info=True)

    def _resolve(node_id: str) -> str | None:
        return mapping.get(node_id)

    return _resolve


def build_relation_field_resolver(client: Any) -> Callable[[str], str | None]:
    """LPG rel_type → LeanIX relation field, inverted from the live metamodel."""
    inverse: dict[str, str] = {}
    try:
        from ..ontology.leanix_metamodel import compile_leanix_metamodel

        meta = client.meta_model() if client is not None else {}
        if meta:
            spec = compile_leanix_metamodel(meta)
            for rfield, (lpg, _target) in spec.relation_map.items():
                inverse[lpg] = rfield
    except Exception:  # noqa: BLE001 - degrade to no mapping
        logger.debug("relation field resolver: metamodel unavailable", exc_info=True)

    def _resolve(rel_type: str) -> str | None:
        return inverse.get(rel_type)

    return _resolve


def push_inferred_relations(
    edges: Iterable[Any],
    *,
    client: Any,
    resolve_factsheet: Callable[[str], str | None],
    relation_field_for: Callable[[str], str | None],
    dry_run: bool = True,
    result: BackfeedResult | None = None,
) -> BackfeedResult:
    """Write inferred relationships between existing fact sheets (idempotent)."""
    result = result or BackfeedResult()
    for edge in edges:
        src, rel, tgt = _edge_parts(edge)
        if not (src and rel and tgt):
            result.relations_skipped += 1
            continue
        src_guid = resolve_factsheet(src)
        tgt_guid = resolve_factsheet(tgt)
        rel_field = relation_field_for(rel)
        if not (src_guid and tgt_guid and rel_field):
            result.relations_skipped += 1
            continue
        if dry_run:
            result.proposals.append(
                {
                    "op": "create_relation",
                    "factSheet": src_guid,
                    "relation": rel_field,
                    "target": tgt_guid,
                    "provenance": PROVENANCE_TAG,
                }
            )
            continue
        try:
            client.create_fact_sheet_relation(src_guid, rel_field, tgt_guid)
            result.relations_written += 1
        except Exception:  # noqa: BLE001 - one failure never aborts the batch
            logger.debug("create_fact_sheet_relation failed", exc_info=True)
            result.errors += 1
    return result


def push_enrichment(
    items: Iterable[dict[str, Any]],
    *,
    client: Any,
    resolve_factsheet: Callable[[str], str | None],
    dry_run: bool = True,
    result: BackfeedResult | None = None,
) -> BackfeedResult:
    """Write derived attributes/tags onto existing fact sheets.

    Each item: ``{"node": <kg id>, "patches": [{op,path,value}...]}`` and/or
    ``{"node": <kg id>, "tag": <tag id>}``.
    """
    result = result or BackfeedResult()
    for item in items:
        node = item.get("node")
        guid = resolve_factsheet(node) if node else None
        if not guid:
            result.relations_skipped += 1
            continue
        patches = item.get("patches")
        tag = item.get("tag")
        if dry_run:
            result.proposals.append(
                {"op": "enrich", "factSheet": guid, "patches": patches, "tag": tag}
            )
            continue
        try:
            if patches:
                client.update_fact_sheet(guid, patches)
                result.enrichments_written += 1
            if tag:
                client.add_tag(guid, tag)
                result.enrichments_written += 1
        except Exception:  # noqa: BLE001
            logger.debug("enrichment write failed", exc_info=True)
            result.errors += 1
    return result


def push_creations(
    creations: Iterable[dict[str, Any]],
    *,
    client: Any,
    dry_run: bool = True,
    result: BackfeedResult | None = None,
) -> BackfeedResult:
    """Create new fact sheets for KG entities absent from LeanIX (highest risk).

    Each item: ``{"type": <FactSheetType>, "name": <str>}``.
    """
    result = result or BackfeedResult()
    for c in creations:
        fs_type = c.get("type")
        name = c.get("name")
        if not (fs_type and name):
            continue
        if dry_run:
            result.proposals.append(
                {"op": "create_fact_sheet", "type": fs_type, "name": name}
            )
            continue
        try:
            created = client.create_fact_sheet(fs_type, name)
            if created.get("id"):
                result.factsheets_created += 1
            else:
                result.errors += 1
        except Exception:  # noqa: BLE001
            logger.debug("create_fact_sheet failed", exc_info=True)
            result.errors += 1
    return result


def run_leanix_writeback(
    *,
    backend: Any = None,
    client: Any = None,
    inferences: Iterable[Any] | None = None,
    enrichments: Iterable[dict[str, Any]] | None = None,
    creations: Iterable[dict[str, Any]] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """The action core both the MCP tool and the REST route call.

    Fail-closed: a live (``dry_run=False``) write requires ``LEANIX_ENABLE_WRITE``.
    Returns a structured manifest (counts + proposals); never raises.
    """
    if client is None:
        from ...ecosystem.ea_clients import get_leanix_client

        client = get_leanix_client()
    if client is None:
        return {"status": "skipped", "reason": "no LeanIX client configured"}

    write_enabled = bool(setting("LEANIX_ENABLE_WRITE", False, cast=bool))
    if not dry_run and not write_enabled:
        return {
            "status": "refused",
            "reason": "LEANIX_ENABLE_WRITE not set; refusing live write to the system-of-record",
            "hint": "run with dry_run=true to preview the proposed writes",
        }

    result = BackfeedResult()
    resolve = build_factsheet_resolver(backend)
    rel_field_for = build_relation_field_resolver(client)

    if inferences:
        push_inferred_relations(
            inferences,
            client=client,
            resolve_factsheet=resolve,
            relation_field_for=rel_field_for,
            dry_run=dry_run,
            result=result,
        )
    if enrichments:
        push_enrichment(
            enrichments,
            client=client,
            resolve_factsheet=resolve,
            dry_run=dry_run,
            result=result,
        )
    if creations:
        push_creations(creations, client=client, dry_run=dry_run, result=result)

    out = result.as_dict()
    out["status"] = "completed"
    out["dry_run"] = dry_run
    out["write_enabled"] = write_enabled
    return out
