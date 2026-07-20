"""ERPNext write-back sink (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Backfeeds the KG into ERPNext — creates inventory as Items/Assets, enriches
existing docs, and retires (disables) decommissioned ones — fail-closed
(``ERPNEXT_ENABLE_WRITE``), dry-run-first. Resolves KG nodes to Frappe doc names
via the ``externalToolId``/``domain="erpnext"`` federation key. Uses the
``erpnext-agent`` resource client (``create_document``/``update_document``).

ERPNext has no generic relation table, so inferred *relationships* are reported as
skipped (link them via document link-fields in a future, doctype-specific pass).
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

# KG node type → ERPNext doctype + the doctype's name field for created inventory.
_DOCTYPE_MAP: dict[str, tuple[str, str]] = {
    "technologyproduct": ("Item", "item_code"),
    "item": ("Item", "item_code"),
    "assetinstance": ("Asset", "asset_name"),
    "asset": ("Asset", "asset_name"),
    "server": ("Asset", "asset_name"),
    "hardwarenode": ("Asset", "asset_name"),
    "service": ("Asset", "asset_name"),
    "itcomponent": ("Asset", "asset_name"),
}


def _doctype_for(node_type: str) -> tuple[str, str]:
    return _DOCTYPE_MAP.get((node_type or "").lower(), ("Item", "item_code"))


def _extract_doc_name(res: Any, fallback: str | None = None) -> str | None:
    """Best-effort pull of the created Frappe doc ``name`` from the return shape."""
    if isinstance(res, str):
        return res or fallback
    for obj in (res, getattr(res, "data", None)):
        if isinstance(obj, dict):
            inner = obj.get("data")
            if isinstance(inner, dict) and inner.get("name"):
                return str(inner["name"])
            if obj.get("name"):
                return str(obj["name"])
    name = getattr(res, "name", None)
    return str(name) if name else fallback


class ErpNextSink:
    """Write-back sink for ERPNext Items & Assets."""

    domain = "erpnext"
    enable_flag = "ERPNEXT_ENABLE_WRITE"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from erpnext_agent.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001 - connector absent / unconfigured
            logger.debug("erpnext write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result
        resolve = ctx.resolver("erpnext")

        # creations — new Items / Assets.
        for c in ops.get("creations") or []:
            name = c.get("name")
            if not name:
                continue
            doctype, name_field = _doctype_for(c.get("type", ""))
            doctype = c.get("doctype") or doctype
            data = {name_field: name, **(c.get("attributes") or {})}
            if dry_run:
                result.proposals.append(
                    {"op": "create_document", "doctype": doctype, "name": name}
                )
                continue
            try:
                res = client.create_document(doctype, data)  # type: ignore[union-attr]  # client None-checked above
                result.created += 1
                ctx.stamp_external_id(
                    c.get("node"),
                    self.domain,
                    _extract_doc_name(res, fallback=name),
                    node_type=c.get("type", ""),
                )
            except Exception:  # noqa: BLE001
                logger.debug("erpnext create_document failed", exc_info=True)
                result.errors += 1

        # enrichments — update fields on existing docs.
        for item in ops.get("enrichments") or []:
            doc_name = resolve(item.get("node"))
            attrs = item.get("attributes") or {}
            doctype = item.get("doctype") or "Item"
            if not (doc_name and attrs):
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append(
                    {"op": "update_document", "doctype": doctype, "name": doc_name}
                )
                continue
            try:
                client.update_document(doctype, doc_name, attrs)  # type: ignore[union-attr]  # client None-checked above
                result.enriched += 1
            except Exception:  # noqa: BLE001
                logger.debug("erpnext update_document failed", exc_info=True)
                result.errors += 1

        # retirements — disable the doc.
        for item in ops.get("retirements") or []:
            doc_name = resolve(item.get("node"))
            doctype = item.get("doctype") or "Item"
            if not doc_name:
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append(
                    {"op": "retire", "doctype": doctype, "name": doc_name}
                )
                continue
            try:
                client.update_document(doctype, doc_name, {"disabled": 1})  # type: ignore[union-attr]  # client None-checked above
                result.retired += 1
            except Exception:  # noqa: BLE001
                logger.debug("erpnext retire failed", exc_info=True)
                result.errors += 1

        # ERPNext has no generic relation table — inferred relations are skipped.
        for _edge in ops.get("inferences") or []:
            result.skipped += 1

        return result


register_sink(ErpNextSink())
