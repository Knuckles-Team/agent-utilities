#!/usr/bin/python
from __future__ import annotations

"""Ontology Action System — notification + webhook dispatch (CONCEPT:AU-KG.ontology.batch-actions-executor).

Provenance (Palantir AIP doc: *action-types/overview* — "Notifications" and
"Webhooks"): after an Action Type's edits are submitted it may notify recipients
and call external systems. This module makes that dispatch **real**, never a
silent no-op:

  - :class:`Notifier` is a registerable sink protocol; the default
    :class:`RecordingNotifier` appends a durable record (so even with no live
    channel wired the notification is journaled + auditable). Production code can
    register an :class:`Notifier` that forwards to e-mail/Slack/etc.
  - :func:`send_webhook` POSTs the payload via ``httpx`` when the dependency is
    importable; otherwise it returns a recorded *outbound* record marked
    ``transport="recorded"`` so the attempt is never lost.

Both return a plain ``dict`` outcome record that the executor stores on the
:class:`ActionInvocation.dispatches` list and that is mirrored into the audit
log — closing the Wire-First loop (a dispatch that happened is observable).
"""

import logging
import time
from typing import Any, Protocol, runtime_checkable

from .models import NotificationSpec, WebhookSpec

logger = logging.getLogger(__name__)

__all__ = [
    "Notifier",
    "RecordingNotifier",
    "send_notification",
    "send_webhook",
    "set_default_notifier",
    "get_default_notifier",
]


@runtime_checkable
class Notifier(Protocol):
    """A registerable notification sink. CONCEPT:AU-KG.ontology.batch-actions-executor.

    Implementations forward a rendered notification to a concrete channel and
    return an outcome record. Must never raise — failures are recorded.
    """

    def notify(self, spec: NotificationSpec, message: str) -> dict[str, Any]:
        """Deliver ``message`` for ``spec`` and return an outcome record."""
        ...  # ABSTRACT-OK


class RecordingNotifier:
    """Default :class:`Notifier` that journals notifications durably. CONCEPT:AU-KG.ontology.batch-actions-executor.

    With no external channel wired, a notification still must be *recorded* (not
    dropped). This sink keeps an in-memory, inspectable log and returns a
    delivered=False/transport="recorded" outcome so callers can observe that the
    notification was produced and is awaiting a live channel.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def notify(self, spec: NotificationSpec, message: str) -> dict[str, Any]:
        record = {
            "kind": "notification",
            "channel": spec.channel,
            "recipient": spec.recipient,
            "message": message,
            "delivered": False,
            "transport": "recorded",
            "timestamp": time.time(),
        }
        self.records.append(record)
        logger.debug(
            "Notification recorded: channel=%s recipient=%s",
            spec.channel,
            spec.recipient,
        )
        return record


# Module-level default notifier — registerable so a deployment can swap in a
# live channel without touching the executor.
_DEFAULT_NOTIFIER: Notifier = RecordingNotifier()


def set_default_notifier(notifier: Notifier) -> None:
    """Register the process-wide default :class:`Notifier`."""
    global _DEFAULT_NOTIFIER
    _DEFAULT_NOTIFIER = notifier


def get_default_notifier() -> Notifier:
    """Return the process-wide default :class:`Notifier`."""
    return _DEFAULT_NOTIFIER


def send_notification(
    spec: NotificationSpec,
    message: str,
    notifier: Notifier | None = None,
) -> dict[str, Any]:
    """Dispatch one notification through ``notifier`` (or the default). Never raises."""
    sink = notifier or _DEFAULT_NOTIFIER
    try:
        return sink.notify(spec, message)
    except Exception as exc:  # noqa: BLE001 — dispatch never blocks the action
        logger.warning("Notifier failed: %s", exc)
        return {
            "kind": "notification",
            "channel": spec.channel,
            "recipient": spec.recipient,
            "message": message,
            "delivered": False,
            "transport": "error",
            "error": str(exc),
            "timestamp": time.time(),
        }


def send_webhook(spec: WebhookSpec, payload: dict[str, Any]) -> dict[str, Any]:
    """POST ``payload`` to ``spec.url`` via httpx, else record the outbound attempt.

    Returns an outcome record (status / transport). When ``httpx`` is importable
    the request is dispatched for real (short timeout, errors captured); when it
    is not, the attempt is journaled with ``transport="recorded"`` so a webhook
    is never a silent no-op.
    """
    body = {**spec.payload, **payload}
    base = {
        "kind": "webhook",
        "url": spec.url,
        "method": spec.method,
        "payload": body,
        "timestamp": time.time(),
    }
    try:
        import httpx  # type: ignore
    except Exception:  # noqa: BLE001 — httpx optional; record instead of dropping
        logger.debug("httpx unavailable — recording webhook to %s", spec.url)
        return {**base, "delivered": False, "transport": "recorded"}
    try:
        resp = httpx.request(
            spec.method,
            spec.url,
            json=body,
            headers=spec.headers or None,
            timeout=5.0,
        )
        return {
            **base,
            "delivered": 200 <= resp.status_code < 300,
            "transport": "httpx",
            "status_code": resp.status_code,
        }
    except Exception as exc:  # noqa: BLE001 — network failure is captured, not raised
        logger.warning("Webhook POST to %s failed: %s", spec.url, exc)
        return {**base, "delivered": False, "transport": "httpx", "error": str(exc)}
