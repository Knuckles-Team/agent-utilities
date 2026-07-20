"""Salesforce CRM write-back sink (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Creates/enriches CRM records (Account/Contact/Opportunity) from KG-derived
knowledge — fail-closed (``SALESFORCE_ENABLE_WRITE``), dry-run-first. Resolves KG
nodes to Salesforce Ids via ``externalToolId``/``domain="salesforce"``. Tolerant
of the connector's create surface (``create_record``/``create``/``insert``).
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

_SOBJECT = {
    "customer": "Account",
    "person": "Contact",
    "salesorder": "Opportunity",
    "deal": "Opportunity",
}


class SalesforceSink:
    domain = "salesforce"
    enable_flag = "SALESFORCE_ENABLE_WRITE"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from salesforce_agent.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("salesforce write client unavailable", exc_info=True)
            return None

    def _create(self, client: Any, sobject: str, data: dict) -> bool:
        for name in ("create_record", "create", "insert"):
            method = getattr(client, name, None)
            if callable(method):
                method(sobject, data)
                return True
        return False

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result
        resolve = ctx.resolver("salesforce")

        for c in ops.get("creations") or []:
            name = c.get("name")
            if not name:
                continue
            sobject = c.get("sobject") or _SOBJECT.get(
                (c.get("type") or "").lower(), "Account"
            )
            if dry_run:
                result.proposals.append(
                    {"op": "create", "sobject": sobject, "name": name}
                )
                continue
            try:
                if self._create(
                    client, sobject, {"Name": name, **(c.get("attributes") or {})}
                ):
                    result.created += 1
                else:
                    result.errors += 1
            except Exception:  # noqa: BLE001
                logger.debug("salesforce create failed", exc_info=True)
                result.errors += 1

        for item in ops.get("enrichments") or []:
            sid = resolve(item.get("node"))
            attrs = item.get("attributes") or {}
            sobject = item.get("sobject") or "Account"
            if not (sid and attrs):
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append({"op": "update", "sobject": sobject, "id": sid})
                continue
            updater = getattr(client, "update_record", None) or getattr(
                client, "update", None
            )
            if not callable(updater):
                result.errors += 1
                continue
            try:
                updater(sobject, sid, attrs)
                result.enriched += 1
            except Exception:  # noqa: BLE001
                logger.debug("salesforce update failed", exc_info=True)
                result.errors += 1

        return result


register_sink(SalesforceSink())
