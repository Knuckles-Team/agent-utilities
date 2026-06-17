"""CISO Assistant write-back sink (CONCEPT:KG-2.111 / CISO-003).

Pushes KG-derived governance entities back into the intuitem **CISO Assistant**
GRC estate — the first-class write-back twin of the CISO Assistant harvest, on the
unified write-back surface. Standard tier, fail-closed
(``CISO_ASSISTANT_ENABLE_WRITE``), dry-run-first.

A KG node of a governance type (``Policy``/``Control``/``Asset``/``Finding``/
``SecurityException``/``Entity``) becomes the matching CISO object via the
generated ``api_*_create`` method. CISO requires every object to live in a folder
(domain); the operator supplies a default folder id in the op payload
(``folder``) or per-creation, otherwise the creation is reported as skipped rather
than guessed.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

# KG node type → (CISO Assistant create method, object label).
_CREATE = {
    "policy": ("api_policies_create", "policy"),
    "control": ("api_applied_controls_create", "applied control"),
    "appliedcontrol": ("api_applied_controls_create", "applied control"),
    "asset": ("api_assets_create", "asset"),
    "finding": ("api_findings_create", "finding"),
    "securityexception": ("api_security_exceptions_create", "security exception"),
    "entity": ("api_entities_create", "entity"),
}


class CisoAssistantSink:
    domain = "ciso_assistant"
    enable_flag = "CISO_ASSISTANT_ENABLE_WRITE"
    risk_tier = "standard"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from ciso_assistant_api.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("ciso_assistant write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result

        default_folder = ops.get("folder")
        for c in ops.get("creations") or []:
            name = c.get("name")
            kind = (c.get("type") or "").lower()
            spec = _CREATE.get(kind)
            if not name or spec is None:
                result.skipped += 1
                continue
            method_name, label = spec
            folder = c.get("folder") or default_folder
            if dry_run:
                result.proposals.append(
                    {"op": f"create_{label.replace(' ', '_')}", "name": name}
                )
                continue
            if not folder:
                # CISO objects must belong to a folder; never guess one.
                result.skipped += 1
                continue
            payload = {
                "name": name,
                "folder": folder,
                "description": c.get("description", "KG-derived governance object."),
            }
            if c.get("ref_id"):
                payload["ref_id"] = c["ref_id"]
            method = getattr(client, method_name, None)
            if not callable(method):
                result.skipped += 1
                continue
            try:
                res = method(body=payload)
                data = getattr(res, "data", res)
                if isinstance(data, dict) and (data.get("id") or data.get("uuid")):
                    result.created += 1
                else:
                    result.errors += 1
            except Exception:  # noqa: BLE001
                logger.debug("ciso_assistant create failed", exc_info=True)
                result.errors += 1

        return result


register_sink(CisoAssistantSink())
