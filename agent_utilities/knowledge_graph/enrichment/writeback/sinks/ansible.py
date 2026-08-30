"""Ansible Tower write-back sink — KG-governed remediation (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

The KG→action arm: launch Tower job templates from KG inferences (e.g. remediate
drift/risk). Fail-closed (``ANSIBLE_ENABLE_WRITE``), dry-run-first. ``creations``
items: ``{template_id, extra_vars}``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import (
    WritebackClientMixin,
    WritebackContext,
    WritebackResult,
    register_sink,
)

logger = logging.getLogger(__name__)


class AnsibleSink(WritebackClientMixin):
    domain = "ansible"
    enable_flag = "ANSIBLE_ENABLE_WRITE"
    client_module = "ansible_tower_mcp"
    client_label = "ansible"

    def _launch_creation(
        self,
        launch: Any,
        creation: dict[str, Any],
        *,
        dry_run: bool,
        result: WritebackResult,
    ) -> None:
        """Handle one ``creations`` item -- the per-item body of :meth:`run`."""
        template = creation.get("template_id") or creation.get("name")
        if not template:
            return
        if dry_run:
            result.proposals.append({"op": "launch_job", "template": template})
            return
        if not callable(launch):
            result.errors += 1
            return
        try:
            launch(template, creation.get("extra_vars") or {})
            result.created += 1
        except Exception:  # noqa: BLE001
            logger.debug("ansible launch_job failed", exc_info=True)
            result.errors += 1

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
            self._launch_creation(launch, c, dry_run=dry_run, result=result)

        return result


register_sink(AnsibleSink())
