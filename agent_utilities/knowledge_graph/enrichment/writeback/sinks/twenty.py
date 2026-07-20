"""Twenty CRM write-back sink (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Create/enrich CRM records (companies/people/opportunities) from KG-derived
knowledge — standard tier, fail-closed (``TWENTY_ENABLE_WRITE``), dry-run-first.
Resolves KG nodes to Twenty ids via ``externalToolId``/``domain="twenty"``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

# KG node type → (create method, update method)
_METHODS: dict[str, tuple[str, str]] = {
    "customer": ("create_company", "update_company"),
    "person": ("create_person", "update_person"),
    "salesorder": ("create_opportunity", "update_opportunity"),
}


def _extract_record_id(res: Any) -> str | None:
    """Best-effort pull of the created Twenty record id from the return shape."""
    if isinstance(res, str):
        return res or None
    for obj in (res, getattr(res, "data", None)):
        if isinstance(obj, dict):
            for key in ("id", "recordId"):
                if obj.get(key):
                    return str(obj[key])
            inner = obj.get("data")
            if isinstance(inner, dict) and inner.get("id"):
                return str(inner["id"])
    rid = getattr(res, "id", None)
    return str(rid) if rid else None


class TwentySink:
    domain = "twenty"
    enable_flag = "TWENTY_ENABLE_WRITE"
    risk_tier = "standard"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from twenty_mcp.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("twenty write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result
        resolve = ctx.resolver("twenty")

        for c in ops.get("creations") or []:
            name = c.get("name")
            create_m, _ = _METHODS.get(
                (c.get("type") or "").lower(), ("create_company", "")
            )
            if not name:
                continue
            if dry_run:
                result.proposals.append({"op": create_m, "name": name})
                continue
            try:
                method = getattr(client, create_m, None)
                if method is not None:
                    res = method(name)
                    result.created += 1
                    ctx.stamp_external_id(
                        c.get("node"),
                        self.domain,
                        _extract_record_id(res),
                        node_type=c.get("type", ""),
                    )
            except Exception:  # noqa: BLE001
                logger.debug("twenty %s failed", create_m, exc_info=True)
                result.errors += 1

        for item in ops.get("enrichments") or []:
            rid = resolve(item.get("node"))
            attrs = item.get("attributes") or {}
            _, update_m = _METHODS.get(
                (item.get("type") or "").lower(), ("", "update_company")
            )
            if not (rid and attrs):
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append({"op": update_m, "id": rid})
                continue
            try:
                getattr(client, update_m)(rid, attrs)
                result.enriched += 1
            except Exception:  # noqa: BLE001
                logger.debug("twenty %s failed", update_m, exc_info=True)
                result.errors += 1

        return result


register_sink(TwentySink())
