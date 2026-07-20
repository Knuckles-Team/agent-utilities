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
import hmac
import ipaddress
import json
import logging
from typing import Any

from aiohttp import web

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

_MAX_BODY_BYTES = 256 * 1024
_MAX_ALERT_CHARS = 4_000
_MAX_ALERTS = 100
_DELIVERY_TIMEOUT_SECONDS = 30
_MAX_CONCURRENT_DELIVERIES = 16


def _extract_text(body: Any) -> str:
    """Pull a human-readable message out of a webhook payload."""
    if isinstance(body, str):
        return (body.strip() or "(empty alert)")[:_MAX_ALERT_CHARS]
    if isinstance(body, dict):
        for key in ("text", "msg", "message", "content"):
            if body.get(key):
                return str(body[key])[:_MAX_ALERT_CHARS]
        if isinstance(body.get("alerts"), list):
            lines = []
            for a in body["alerts"][:_MAX_ALERTS]:
                if not isinstance(a, dict):
                    continue
                ann = a.get("annotations", {}) or {}
                lab = a.get("labels", {}) or {}
                if not isinstance(ann, dict):
                    ann = {}
                if not isinstance(lab, dict):
                    lab = {}
                lines.append(
                    f"[{a.get('status', '?')}] "
                    + (
                        ann.get("summary")
                        or ann.get("description")
                        or lab.get("alertname", "alert")
                    )
                )
            if lines:
                return "\n".join(lines)[:_MAX_ALERT_CHARS]
    return json.dumps(body, ensure_ascii=False, default=str)[:_MAX_ALERT_CHARS]


def _loopback_bind(host: str) -> bool:
    value = str(host or "").strip().lower()
    if value in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(value.strip("[]").split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _bearer_token(request: web.Request) -> str:
    scheme, separator, credential = request.headers.get("Authorization", "").partition(" ")
    if separator and scheme.lower() == "bearer":
        return credential.strip()
    return ""


async def _handle(request: web.Request) -> web.Response:
    supplied = _bearer_token(request)
    required = request.app["alert_intake_token"]
    if not supplied or not hmac.compare_digest(supplied, required):
        return web.json_response(
            {"ok": False, "error": "authentication required"}, status=401
        )

    deliveries: asyncio.Semaphore = request.app["alert_intake_deliveries"]
    if deliveries.locked():
        return web.json_response(
            {"ok": False, "error": "delivery capacity exhausted"}, status=429
        )

    engine = request.app["engine"]
    async with deliveries:
        try:
            body: Any = await request.json()
        except Exception:
            body = await request.text()
        text = _extract_text(body)

        platform = setting("MESSAGING_DEFAULT_PLATFORM", "telegram")
        channel = setting("MESSAGING_DEFAULT_CHANNEL", "")
        if not channel:
            return web.json_response(
                {"ok": False, "error": "alert destination is unavailable"}, status=503
            )

        from agent_utilities.messaging.service import MessagingService

        svc = MessagingService.instance(engine)
        backend = await svc.get_backend(platform)
        if backend is None:
            return web.json_response(
                {"ok": False, "error": "alert destination is unavailable"}, status=503
            )
        try:
            await asyncio.wait_for(
                backend.send_message(channel, text),
                timeout=_DELIVERY_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 — boundary is fail-closed + generic
            logger.warning(
                "alert-intake delivery failed (%s)", type(exc).__name__
            )
            return web.json_response(
                {"ok": False, "error": "alert delivery failed"}, status=502
            )
        return web.json_response({"ok": True})


async def serve_alert_intake(engine: Any, port: int) -> None:
    """Run the alert-intake HTTP server until cancelled. Never raises into the caller."""
    host = str(setting("MESSAGING_ALERT_INTAKE_HOST", "127.0.0.1") or "").strip()
    allow_remote = str(
        setting("MESSAGING_ALERT_INTAKE_ALLOW_REMOTE", "False") or "False"
    ).lower() in {"1", "true", "yes", "on"}
    if not _loopback_bind(host) and not allow_remote:
        logger.error("messaging alert-intake disabled: non-loopback bind is not approved")
        return

    token_ref = str(setting("MESSAGING_ALERT_INTAKE_TOKEN_REF", "") or "").strip()
    if not token_ref:
        logger.error("messaging alert-intake disabled: token reference is required")
        return
    try:
        from agent_utilities.security.secrets_client import create_secrets_client

        token = create_secrets_client().resolve_ref(token_ref)
    except Exception:  # noqa: BLE001 — never disclose provider/ref details
        token = None
    if not token:
        logger.error("messaging alert-intake disabled: token reference is unresolved")
        return

    app = web.Application(client_max_size=_MAX_BODY_BYTES)
    app["engine"] = engine
    app["alert_intake_token"] = str(token)
    app["alert_intake_deliveries"] = asyncio.Semaphore(_MAX_CONCURRENT_DELIVERIES)
    app.router.add_post("/alert", _handle)
    app.router.add_get("/health", lambda r: web.json_response({"ok": True}))
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info(
            "[CONCEPT:AU-ECO.messaging.alert-intake] messaging alert-intake enabled on port %s",
            port,
        )
        await asyncio.Event().wait()  # run until the task is cancelled
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — the intake is best-effort; never kill the daemon
        logger.exception("messaging alert-intake crashed (listeners keep running)")
    finally:
        with __import__("contextlib").suppress(Exception):
            await runner.cleanup()
