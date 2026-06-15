"""Legal-peripherals write-back sink (CONCEPT:KG-2.9) — HIGH-STAKES / propose-only.

Emits KG-derived legal actions as **proposed** filings/draft documents (LLC
formation, EIN/SS-4 drafts, regulatory filings). ``risk_tier="high_stakes"``:
even with ``LEGAL_ENABLE_WRITE`` set it never auto-files — proposals route to the
approval queue and only an explicit ``approve`` (``_approved``) replays live via
the legal-peripherals client. Inbound (SoS/statute reads) stays in the connector;
this is the governed write twin.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)

# KG creation type → (proposal op, client method name).
_FILING = {
    "einapplication": ("draft_ein", "draft_ein_form"),
    "ein": ("draft_ein", "draft_ein_form"),
    "llcformationfiling": ("file_llc_formation", "file_llc_formation"),
    "llcformation": ("file_llc_formation", "file_llc_formation"),
    "regulatoryfiling": ("file_regulatory", "submit_filing"),
}


class LegalSink:
    domain = "legal"
    enable_flag = "LEGAL_ENABLE_WRITE"
    risk_tier = "high_stakes"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from legal_peripherals_mcp.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            logger.debug("legal write client unavailable", exc_info=True)
            return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        filings = ops.get("filings") or ops.get("creations") or []

        proposals = []
        for f in filings:
            ftype = (f.get("type") or "").lower()
            mapped = _FILING.get(ftype)
            if not mapped:
                result.skipped += 1
                continue
            op, method = mapped
            proposals.append({"op": op, "method": method, "details": f})
        if dry_run:
            result.proposals.extend(proposals)
            return result

        # Live path: only on an approved replay (high_stakes never auto-files).
        client = self._client(ops)
        if client is None:
            result.skipped += len(proposals)
            return result
        for p in proposals:
            fn = getattr(client, p["method"], None)
            if not callable(fn):
                result.skipped += 1
                continue
            try:
                fn(**{k: v for k, v in p["details"].items() if k != "type"})
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("legal filing %s failed", p["op"], exc_info=True)
                result.errors += 1

        return result


register_sink(LegalSink())
