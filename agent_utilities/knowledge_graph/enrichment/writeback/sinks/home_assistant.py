"""Home Assistant write-back sink — KG-driven control (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

The KG→action arm for the homelab: call HA services from KG inferences (e.g. act
on a derived state grouping). Fail-closed (``HOMEASSISTANT_ENABLE_WRITE``),
dry-run-first. ``creations`` items: ``{domain, service, data}``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)


class HomeAssistantSink:
    domain = "homeassistant"
    enable_flag = "HOMEASSISTANT_ENABLE_WRITE"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from home_assistant_agent.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("home assistant write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result

        call = getattr(client, "call_service", None)
        for c in ops.get("creations") or []:
            svc_domain = c.get("domain")
            service = c.get("service")
            if not (svc_domain and service):
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append(
                    {"op": "call_service", "service": f"{svc_domain}.{service}"}
                )
                continue
            if not callable(call):
                result.errors += 1
                continue
            try:
                call(svc_domain, service, c.get("data") or {})
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("home assistant call_service failed", exc_info=True)
                result.errors += 1

        return result


register_sink(HomeAssistantSink())
