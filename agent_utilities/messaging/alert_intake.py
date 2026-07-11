"""HTTP alert-intake for the messaging daemon (CONCEPT:AU-ECO.messaging.alert-intake).

Lets external monitoring — uptime-kuma, Alertmanager, any webhook source — route
notifications THROUGH agent-utilities messaging instead of each tool configuring its own
Telegram/Mattermost notifier. POST a webhook here and it is delivered on the configured
default platform/channel via the SAME connected backend the daemon already runs, so alerts
inherit the one unified messaging stack (Universal-capability: built once in messaging, every
producer reuses it).

Opt-in + non-fatal: only started when ``MESSAGING_ALERT_INTAKE_PORT`` is set, and a failure
here never touches the inbound listeners (it runs as an independent task).

Payload shapes handled: a bare string, ``{"text"|"msg"|"message"|"content": ...}`` (uptime-kuma
sends ``msg``), and Alertmanager ``{"alerts": [...]}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiohttp import web

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


def _extract_text(body: Any) -> str:
    """Pull a human-readable message out of a webhook payload."""
    if isinstance(body, str):
        return body.strip() or "(empty alert)"
    if isinstance(body, dict):
        for key in ("text", "msg", "message", "content"):
            if body.get(key):
                return str(body[key])
        if isinstance(body.get("alerts"), list):
            lines = []
            for a in body["alerts"]:
                ann = a.get("annotations", {}) or {}
                lab = a.get("labels", {}) or {}
                lines.append(
                    f"[{a.get('status', '?')}] "
                    + (ann.get("summary") or ann.get("description") or lab.get("alertname", "alert"))
                )
            if lines:
                return "\n".join(lines)
    return json.dumps(body)[:1500]


async def _handle(request: web.Request) -> web.Response:
    engine = request.app["engine"]
    try:
        body: Any = await request.json()
    except Exception:
        body = await request.text()
    text = _extract_text(body)

    platform = setting("MESSAGING_DEFAULT_PLATFORM", "telegram")
    channel = setting("MESSAGING_DEFAULT_CHANNEL", "")
    if not channel:
        return web.json_response(
            {"ok": False, "error": "MESSAGING_DEFAULT_CHANNEL is unset — cannot route the alert"},
            status=503,
        )

    from agent_utilities.messaging.service import MessagingService

    svc = MessagingService.instance(engine)
    backend = await svc.get_backend(platform)
    if backend is None:
        return web.json_response(
            {"ok": False, "error": f"no connected messaging backend for platform {platform!r}"},
            status=503,
        )
    try:
        await backend.send_message(channel, text)
    except Exception as exc:  # noqa: BLE001 — surface the delivery failure to the caller
        logger.warning("alert-intake delivery failed (%s/%s): %s", platform, channel, exc)
        return web.json_response({"ok": False, "error": f"delivery failed: {exc}"}, status=502)
    return web.json_response({"ok": True, "platform": platform, "channel": channel})


async def serve_alert_intake(engine: Any, port: int) -> None:
    """Run the alert-intake HTTP server until cancelled. Never raises into the caller."""
    app = web.Application()
    app["engine"] = engine
    app.router.add_post("/alert", _handle)
    app.router.add_get("/health", lambda r: web.json_response({"ok": True}))
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        logger.info(
            "[CONCEPT:AU-ECO.messaging.alert-intake] messaging alert-intake listening on :%s "
            "(POST /alert -> %s/%s)",
            port,
            setting("MESSAGING_DEFAULT_PLATFORM", "telegram"),
            setting("MESSAGING_DEFAULT_CHANNEL", ""),
        )
        await asyncio.Event().wait()  # run until the task is cancelled
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — the intake is best-effort; never kill the daemon
        logger.exception("messaging alert-intake crashed (listeners keep running)")
    finally:
        with __import__("contextlib").suppress(Exception):
            await runner.cleanup()
