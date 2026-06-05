"""Push derived/curated capabilities back to EA tools (CONCEPT:KG-2.8).

Closes the loop for the bottom-up case: capabilities minted from code (or a
curated registry) by :mod:`realizes` can be written back to LeanIX and/or Archi
so the enterprise-architecture catalog is enriched from the codebases an
acquisition brought in.

Both clients are **duck-typed and optional** — pass whichever you have. Calls are
idempotent (skip capabilities already present upstream) and individually
tolerant (one failing client never aborts the batch). No network or EA-tool
import happens here; the caller injects ready clients.

    Archi   : ``client.add_element(type="Capability", name=..., documentation=...)``
    LeanIX  : ``client.postbusinesscapability({...})`` (reference-data API),
              falling back to ``create_business_capability(...)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from .models import GraphNode

logger = logging.getLogger(__name__)

WritebackFn = Callable[[list[GraphNode]], "WritebackResult"]


class WritebackResult:
    """Per-target counts of capabilities pushed (and skipped)."""

    def __init__(self) -> None:
        self.archi_pushed = 0
        self.leanix_pushed = 0
        self.skipped_existing = 0
        self.errors = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "archi_pushed": self.archi_pushed,
            "leanix_pushed": self.leanix_pushed,
            "skipped_existing": self.skipped_existing,
            "errors": self.errors,
        }


def _name_of(node: GraphNode) -> str:
    return str(node.props.get("name") or node.id)


def _should_push(node: GraphNode) -> bool:
    """Only push provisional/code-derived capabilities (not mirrored upstream ones)."""
    if node.type != "BusinessCapability":
        return False
    return bool(node.props.get("provisional") or node.props.get("derived_from"))


def _push_archi(client: Any, node: GraphNode, result: WritebackResult) -> None:
    add_element = getattr(client, "add_element", None)
    if not callable(add_element):
        return
    try:
        add_element(
            type="Capability",
            name=_name_of(node),
            documentation=str(node.props.get("summary") or ""),
            properties={"source": "agent-utilities", "kg_id": node.id},
        )
        result.archi_pushed += 1
    except Exception as exc:  # pragma: no cover - external client transport
        logger.debug("Archi add_element failed for %s: %s", node.id, exc)
        result.errors += 1


def _push_leanix(client: Any, node: GraphNode, result: WritebackResult) -> None:
    payload = {
        "name": _name_of(node),
        "description": str(node.props.get("summary") or ""),
        "type": "BusinessCapability",
    }
    for method_name in ("postbusinesscapability", "create_business_capability"):
        method = getattr(client, method_name, None)
        if callable(method):
            try:
                method(payload)
                result.leanix_pushed += 1
            except Exception as exc:  # pragma: no cover - external client transport
                logger.debug("LeanIX %s failed for %s: %s", method_name, node.id, exc)
                result.errors += 1
            return


def push_capabilities(
    nodes: Iterable[GraphNode],
    *,
    archi_client: Any | None = None,
    leanix_client: Any | None = None,
    existing_names: Iterable[str] | None = None,
) -> WritebackResult:
    """Push provisional/curated capabilities to the configured EA tools.

    ``existing_names`` (case-insensitive) suppresses capabilities already present
    upstream, keeping the operation idempotent across runs.
    """
    result = WritebackResult()
    if archi_client is None and leanix_client is None:
        return result
    seen = {n.strip().lower() for n in (existing_names or []) if n}

    for node in nodes:
        if not _should_push(node):
            continue
        if _name_of(node).strip().lower() in seen:
            result.skipped_existing += 1
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
    """Best-effort: current Capability node names (for idempotent writeback)."""
    if backend is None:
        return []
    try:
        rows = backend.execute(
            "MATCH (c:Capability) WHERE c.name IS NOT NULL RETURN c.name AS name LIMIT 5000"
        )
        return [r.get("name") for r in (rows or []) if isinstance(r, dict) and r.get("name")]
    except Exception:  # noqa: BLE001 - dedup is best-effort
        return []


def resolve_writeback_fn(
    backend: Any = None,
    *,
    archi_client: Any = None,
    leanix_client: Any = None,
) -> WritebackFn | None:
    """Build the pipeline writeback callable when EA writeback is enabled, else ``None``.

    CONCEPT:KG-2.8 — this is the **injection point** the ``EnrichmentPipeline`` calls. Gated by the
    ``KG_EA_WRITEBACK`` env flag (default off → returns ``None`` → no-op, no regression). When on, it
    resolves Archi/LeanIX clients (passed in, or best-effort from the EA bridge), seeds existing
    capability names for idempotency, and returns the callable the pipeline invokes on minted caps.
    """
    import os

    if os.environ.get("KG_EA_WRITEBACK", "0") not in ("1", "true", "True"):
        return None
    if archi_client is None and leanix_client is None:
        try:  # best-effort: a deployment may expose EA clients via the ecosystem bridge
            from ...ecosystem.ea_clients import get_archi_client, get_leanix_client

            archi_client = get_archi_client()
            leanix_client = get_leanix_client()
        except Exception:  # noqa: BLE001 - no EA clients configured → nothing to push to
            logger.debug("KG_EA_WRITEBACK set but no EA clients available; skipping writeback")
            return None
    if archi_client is None and leanix_client is None:
        return None
    return make_writeback_fn(
        archi_client=archi_client,
        leanix_client=leanix_client,
        existing_names=_existing_capability_names(backend),
    )
