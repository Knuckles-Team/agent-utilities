"""Capability write-back sink (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

Pushes minted/provisional ``BusinessCapability`` nodes back to EA tools (Archi
and/or LeanIX). Folds the former ``capability_writeback`` module; preserves the
EnrichmentPipeline injection point :func:`resolve_writeback_fn` (native, gated by
``KG_EA_WRITEBACK``) and exposes a unified sink for ``graph_writeback``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from agent_utilities.core.config import setting

from ...models import GraphNode
from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

WritebackFn = Callable[[list[GraphNode]], WritebackResult]


def _name_of(node: Any) -> str:
    props = node.props if isinstance(node, GraphNode) else (node.get("props") or node)
    nid = node.id if isinstance(node, GraphNode) else node.get("id")
    return str((props or {}).get("name") or nid)


def _props(node: Any) -> dict[str, Any]:
    return node.props if isinstance(node, GraphNode) else (node.get("props") or {})


def _type(node: Any) -> str:
    return node.type if isinstance(node, GraphNode) else str(node.get("type") or "")


def _should_push(node: Any) -> bool:
    """Only push provisional/code-derived capabilities (not mirrored upstream ones)."""
    if _type(node) != "BusinessCapability":
        return False
    p = _props(node)
    return bool(p.get("provisional") or p.get("derived_from"))


def _push_archi(client: Any, node: Any, result: WritebackResult) -> None:
    add_element = getattr(client, "add_element", None)
    if not callable(add_element):
        return
    try:
        add_element(
            type="Capability",
            name=_name_of(node),
            documentation=str(_props(node).get("summary") or ""),
            properties={"source": "agent-utilities"},
        )
        result.created += 1
    except Exception as exc:  # pragma: no cover - external transport
        logger.debug("Archi add_element failed: %s", exc)
        result.errors += 1


def _push_leanix(client: Any, node: Any, result: WritebackResult) -> None:
    create_fs = getattr(client, "create_fact_sheet", None)
    if callable(create_fs):
        try:
            create_fs("BusinessCapability", _name_of(node))
            result.created += 1
        except Exception as exc:  # pragma: no cover - external transport
            logger.debug("LeanIX create_fact_sheet failed: %s", exc)
            result.errors += 1
        return
    for method_name in ("postbusinesscapability", "create_business_capability"):
        method = getattr(client, method_name, None)
        if callable(method):
            try:
                method({"name": _name_of(node), "type": "BusinessCapability"})
                result.created += 1
            except Exception as exc:  # pragma: no cover
                logger.debug("LeanIX %s failed: %s", method_name, exc)
                result.errors += 1
            return


def push_capabilities(
    nodes: Iterable[Any],
    *,
    archi_client: Any | None = None,
    leanix_client: Any | None = None,
    existing_names: Iterable[str] | None = None,
    dry_run: bool = False,
) -> WritebackResult:
    """Push provisional/curated capabilities to the configured EA tools."""
    result = WritebackResult(target="capability")
    if archi_client is None and leanix_client is None:
        return result
    seen = {n.strip().lower() for n in (existing_names or []) if n}
    for node in nodes:
        if not _should_push(node):
            continue
        if _name_of(node).strip().lower() in seen:
            result.skipped += 1
            continue
        if dry_run:
            result.proposals.append({"op": "create_capability", "name": _name_of(node)})
            continue
        if archi_client is not None:
            _push_archi(archi_client, node, result)
        if leanix_client is not None:
            _push_leanix(leanix_client, node, result)
    return result


def make_writeback_fn(
    *,
    archi_client: Any | None = None,
    leanix_client: Any | None = None,
    existing_names: Iterable[str] | None = None,
) -> WritebackFn:
    """Build a one-arg writeback callable for injection into the pipeline."""
    names = list(existing_names or [])

    def _fn(nodes: list[GraphNode]) -> WritebackResult:
        return push_capabilities(
            nodes,
            archi_client=archi_client,
            leanix_client=leanix_client,
            existing_names=names,
        )

    return _fn


def _existing_capability_names(backend: Any) -> list[str]:
    if backend is None:
        return []
    try:
        rows = backend.execute(
            "MATCH (c:Capability) WHERE c.name IS NOT NULL RETURN c.name AS name LIMIT 5000"
        )
        return [
            str(r["name"])
            for r in (rows or [])
            if isinstance(r, dict) and r.get("name")
        ]
    except Exception:  # noqa: BLE001 - dedup is best-effort
        return []


def resolve_writeback_fn(
    backend: Any = None,
    *,
    archi_client: Any = None,
    leanix_client: Any = None,
) -> WritebackFn | None:
    """EnrichmentPipeline injection point; ``None`` when ``KG_EA_WRITEBACK`` is off."""
    if not setting("KG_EA_WRITEBACK", False):
        return None
    if archi_client is None and leanix_client is None:
        try:
            from .....ecosystem.ea_clients import get_archi_client, get_leanix_client

            archi_client = get_archi_client()
            leanix_client = get_leanix_client()
        except Exception:  # noqa: BLE001 - no EA clients configured
            logger.debug("KG_EA_WRITEBACK set but no EA clients available")
            return None
    if archi_client is None and leanix_client is None:
        return None
    return make_writeback_fn(
        archi_client=archi_client,
        leanix_client=leanix_client,
        existing_names=_existing_capability_names(backend),
    )


class CapabilitySink:
    """Write-back sink for minted capabilities (graph_writeback target=capability)."""

    domain = "capability"
    enable_flag = "KG_EA_WRITEBACK"

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        nodes = ops.get("nodes") or []
        archi_client = ops.get("archi_client")
        leanix_client = ops.get("leanix_client")
        if archi_client is None and leanix_client is None:
            try:
                from .....ecosystem.ea_clients import (
                    get_archi_client,
                    get_leanix_client,
                )

                archi_client = get_archi_client()
                leanix_client = get_leanix_client()
            except Exception:  # noqa: BLE001
                pass
        return push_capabilities(
            nodes,
            archi_client=archi_client,
            leanix_client=leanix_client,
            existing_names=_existing_capability_names(ctx.backend),
            dry_run=dry_run,
        )


register_sink(CapabilitySink())
