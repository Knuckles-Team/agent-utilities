"""Ansible Tower write-back sink — KG-governed remediation (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

The KG→action arm: launch Tower job templates from KG inferences (e.g. remediate
drift/risk). Fail-closed (``ANSIBLE_ENABLE_WRITE``), dry-run-first. ``creations``
items: ``{template_id, extra_vars}``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)


class AnsibleSink:
    domain = "ansible"
    enable_flag = "ANSIBLE_ENABLE_WRITE"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from ansible_tower_mcp.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("ansible write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result

        launch = getattr(client, "launch_job", None)
        for c in ops.get("creations") or []:
            template = c.get("template_id") or c.get("name")
            if not template:
                continue
            if dry_run:
                result.proposals.append({"op": "launch_job", "template": template})
                continue
            if not callable(launch):
                result.errors += 1
                continue
            try:
                launch(template, c.get("extra_vars") or {})
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("ansible launch_job failed", exc_info=True)
                result.errors += 1

        return result


register_sink(AnsibleSink())
