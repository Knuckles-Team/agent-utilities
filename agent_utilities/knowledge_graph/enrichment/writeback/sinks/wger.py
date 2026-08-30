"""wger write-back sink (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Logs KG-derived wellness facts back into wger: weight entries, body
measurements, and workout sessions (the coaching-loop write twin of the wger
extractor). Standard tier, fail-closed (``WGER_ENABLE_WRITE``), dry-run-first.
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


class WgerSink(WritebackClientMixin):
    domain = "wger"
    enable_flag = "WGER_ENABLE_WRITE"
    risk_tier = "standard"
    client_module = "wger_agent"
    client_label = "wger"

    def _apply_weight_entry(
        self, client: Any, c: dict[str, Any], dry_run: bool, result: WritebackResult
    ) -> None:
        op = {
            "op": "create_weight_entry",
            "date": c.get("date"),
            "weight": c.get("value") or c.get("weight"),
        }
        if dry_run:
            result.proposals.append(op)
        else:
            client.create_weight_entry(op["date"], op["weight"])  # type: ignore[union-attr]  # client None-checked above
            result.created += 1

    def _apply_measurement(
        self, client: Any, c: dict[str, Any], dry_run: bool, result: WritebackResult
    ) -> None:
        op = {
            "op": "create_measurement",
            "category": c.get("category"),
            "date": c.get("date"),
            "value": c.get("value"),
        }
        if dry_run:
            result.proposals.append(op)
        else:
            client.create_measurement(  # type: ignore[union-attr]  # client None-checked above
                op["category"], op["date"], op["value"]
            )
            result.created += 1

    def _apply_body_or_weight(
        self, client: Any, c: dict[str, Any], dry_run: bool, result: WritebackResult
    ) -> None:
        kind = (c.get("kind") or "weight").lower()
        if kind == "weight":
            self._apply_weight_entry(client, c, dry_run, result)
        else:
            self._apply_measurement(client, c, dry_run, result)

    def _apply_session(
        self, client: Any, c: dict[str, Any], dry_run: bool, result: WritebackResult
    ) -> None:
        op = {
            "op": "create_workout_session",
            "routine": c.get("routine"),
            "date": c.get("date"),
            "impression": c.get("impression", "3"),
            "notes": c.get("notes", ""),
        }
        if dry_run:
            result.proposals.append(op)
        else:
            client.create_workout_session(  # type: ignore[union-attr]  # client None-checked above
                op["routine"],
                op["date"],
                impression=op["impression"],
                notes=op["notes"],
            )
            result.created += 1

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result

        for c in ops.get("creations") or []:
            ctype = (c.get("type") or "").lower()
            try:
                if ctype in ("bodymeasurement", "weightentry", "weight"):
                    self._apply_body_or_weight(client, c, dry_run, result)
                elif ctype in ("workoutsession", "session"):
                    self._apply_session(client, c, dry_run, result)
                else:
                    result.skipped += 1
            except Exception:  # noqa: BLE001
                logger.debug("wger write failed for %s", ctype, exc_info=True)
                result.errors += 1

        return result


register_sink(WgerSink())
