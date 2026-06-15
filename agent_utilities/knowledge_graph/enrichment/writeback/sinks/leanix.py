"""LeanIX write-back sink (CONCEPT:KG-2.9).

Backfeeds inferred relationships, enrichment, and new fact sheets into LeanIX —
fail-closed (``LEANIX_ENABLE_WRITE``), dry-run-first, reversible (inferred links
carry the provenance tag). Folds the former ``leanix_writeback`` module onto the
unified sink contract.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from ..core import (
    PROVENANCE_TAG,
    WritebackContext,
    WritebackResult,
    register_sink,
)

logger = logging.getLogger(__name__)


def _edge_parts(edge: Any) -> tuple[str | None, str | None, str | None]:
    if isinstance(edge, dict):
        return (
            edge.get("source") or edge.get("subject"),
            edge.get("rel_type") or edge.get("type") or edge.get("predicate"),
            edge.get("target") or edge.get("object"),
        )
    if isinstance(edge, list | tuple) and len(edge) == 3:
        return edge[0], edge[1], edge[2]
    return None, None, None


def build_relation_field_resolver(client: Any) -> Callable[[str], str | None]:
    """LPG rel_type → LeanIX relation field, inverted from the live metamodel."""
    inverse: dict[str, str] = {}
    try:
        from ....ontology.leanix_metamodel import compile_leanix_metamodel

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


def _push_relations(
    edges: Iterable[Any],
    *,
    client: Any,
    resolve: Callable[[str], str | None],
    rel_field_for: Callable[[str], str | None],
    dry_run: bool,
    result: WritebackResult,
) -> None:
    for edge in edges:
        src, rel, tgt = _edge_parts(edge)
        if not (src and rel and tgt):
            result.skipped += 1
            continue
        src_guid, tgt_guid, rel_field = resolve(src), resolve(tgt), rel_field_for(rel)
        if not (src_guid and tgt_guid and rel_field):
            result.skipped += 1
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
        except Exception:  # noqa: BLE001
            logger.debug("leanix create_fact_sheet_relation failed", exc_info=True)
            result.errors += 1


def _push_enrichment(
    items: Iterable[dict[str, Any]],
    *,
    client: Any,
    resolve: Callable[[str], str | None],
    dry_run: bool,
    result: WritebackResult,
) -> None:
    for item in items:
        node = item.get("node")
        guid = resolve(node) if node else None
        if not guid:
            result.skipped += 1
            continue
        patches, tag = item.get("patches"), item.get("tag")
        if dry_run:
            result.proposals.append(
                {"op": "enrich", "factSheet": guid, "patches": patches, "tag": tag}
            )
            continue
        try:
            if patches:
                client.update_fact_sheet(guid, patches)
                result.enriched += 1
            if tag:
                client.add_tag(guid, tag)
                result.enriched += 1
        except Exception:  # noqa: BLE001
            logger.debug("leanix enrichment write failed", exc_info=True)
            result.errors += 1


def _push_creations(
    creations: Iterable[dict[str, Any]],
    *,
    client: Any,
    dry_run: bool,
    result: WritebackResult,
) -> None:
    for c in creations:
        fs_type, name = c.get("type"), c.get("name")
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
                result.created += 1
            else:
                result.errors += 1
        except Exception:  # noqa: BLE001
            logger.debug("leanix create_fact_sheet failed", exc_info=True)
            result.errors += 1


class LeanixSink:
    """Write-back sink for LeanIX fact sheets."""

    domain = "leanix"
    enable_flag = "LEANIX_ENABLE_WRITE"

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = ops.get("client")
        if client is None:
            from .....ecosystem.ea_clients import get_leanix_client

            client = get_leanix_client()
        if client is None:
            result.skipped += 1
            return result

        resolve = ctx.resolver("leanix")
        rel_field_for = build_relation_field_resolver(client)
        if ops.get("inferences"):
            _push_relations(
                ops["inferences"],
                client=client,
                resolve=resolve,
                rel_field_for=rel_field_for,
                dry_run=dry_run,
                result=result,
            )
        if ops.get("enrichments"):
            _push_enrichment(
                ops["enrichments"],
                client=client,
                resolve=resolve,
                dry_run=dry_run,
                result=result,
            )
        if ops.get("creations"):
            _push_creations(
                ops["creations"], client=client, dry_run=dry_run, result=result
            )
        return result


register_sink(LeanixSink())
