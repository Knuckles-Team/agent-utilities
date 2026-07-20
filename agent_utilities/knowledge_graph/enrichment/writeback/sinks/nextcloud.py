"""Nextcloud write-back sink (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Backfeeds the KG into Nextcloud — schedules calendar events from inferences
(e.g. a review when a TRM end-of-life approaches, or a change window from a
process) and publishes KG reports as files — fail-closed (``NEXTCLOUD_ENABLE_WRITE``),
dry-run-first. Uses the ``nextcloud-agent`` client (``create_calendar_event`` /
``write_file``).
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)


def _ics(uid: str, summary: str, start: str | None, end: str | None) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//agent-utilities//KG//EN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"SUMMARY:{summary}",
    ]
    if start:
        lines.append(f"DTSTART:{start}")
    if end:
        lines.append(f"DTEND:{end}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines)


class NextcloudSink:
    """Write-back sink for Nextcloud calendar + files."""

    domain = "nextcloud"
    enable_flag = "NEXTCLOUD_ENABLE_WRITE"

    def _client(self, ops: dict[str, Any]) -> Any | None:
        client = ops.get("client")
        if client is not None:
            return client
        try:
            from nextcloud_agent.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001 - connector absent / unconfigured
            logger.debug("nextcloud write client unavailable", exc_info=True)
            return None

    def _calendar_url(self, client: Any, ops: dict[str, Any]) -> str | None:
        if ops.get("calendar_url"):
            return ops["calendar_url"]
        try:
            cals = client.list_calendars() or []
            if cals:
                return cals[0].get("url") or cals[0].get("href")
        except Exception:  # noqa: BLE001
            return None
        return None

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result

        # creations — schedule calendar events.
        cal_url = None
        for c in ops.get("creations") or []:
            name = c.get("name")
            if not name:
                continue
            if dry_run:
                result.proposals.append(
                    {"op": "create_calendar_event", "summary": name}
                )
                continue
            cal_url = cal_url or self._calendar_url(client, ops)
            if not cal_url:
                result.skipped += 1
                continue
            try:
                uid = c.get("node") or name
                ics = _ics(str(uid), name, c.get("start"), c.get("end"))
                client.create_calendar_event(cal_url, ics)  # type: ignore[union-attr]  # client None-checked above
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("nextcloud create_calendar_event failed", exc_info=True)
                result.errors += 1

        # enrichments — publish a KG report/file ({path, content}).
        for item in ops.get("enrichments") or []:
            path = item.get("path")
            content = item.get("content")
            if not (path and content is not None):
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append({"op": "write_file", "path": path})
                continue
            try:
                client.write_file(path, content)  # type: ignore[union-attr]  # client None-checked above
                result.enriched += 1
            except Exception:  # noqa: BLE001
                logger.debug("nextcloud write_file failed", exc_info=True)
                result.errors += 1

        return result


register_sink(NextcloudSink())
