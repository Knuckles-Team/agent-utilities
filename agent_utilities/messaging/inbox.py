"""Durable inbound-message inbox + retry — nothing goes unanswered (CONCEPT:AU-ECO.messaging.durable-inbound-pending).

An inbound chat turn is recorded as a durable ``:InboundMessage`` node BEFORE the reply is
attempted, and marked ``answered`` only when a reply is actually delivered. A reaper in the
messaging daemon re-attempts still-pending messages (the engine was down, a transient error)
with bounded retries, so the "I saved your message" fallback becomes a real promise instead of
best-effort — a turn that fails mid-flight is found and answered when the system recovers.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 4
# Don't retry until the ORIGINAL attempt has had time to land (it usually does) — only a
# genuinely-stuck/failed turn is past this window when the reaper next runs.
RETRY_GRACE_S = 90.0


def _inbox_id(platform: Any, channel_id: Any, message_id: Any, text: str) -> str:
    raw = f"{platform}:{channel_id}:{message_id}:{text}"
    h = hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"inbound:{platform}:{channel_id}:{h}"


def record_inbound(
    engine: Any,
    *,
    platform: Any,
    channel_id: Any,
    message_id: Any,
    text: str,
    session: str,
) -> str | None:
    """Persist an inbound message as ``pending`` BEFORE the reply is attempted (idempotent by
    content), so a failed/lost turn can be found + retried. Best-effort; returns the inbox id."""
    add_node = getattr(engine, "add_node", None) if engine is not None else None
    if not callable(add_node):
        return None
    iid = _inbox_id(platform, channel_id, message_id, text or "")
    try:
        add_node(
            node_id=iid,
            node_type="InboundMessage",
            properties={
                "id": iid,
                "platform": str(platform),
                "channel_id": str(channel_id),
                "message_id": str(message_id or ""),
                "text": (text or "")[:4000],
                "session": session or "",
                "status": "pending",
                "received_at": datetime.now(UTC).isoformat(),
                "attempts": 0,
            },
        )
        return iid
    except Exception as e:  # noqa: BLE001 — durability is best-effort, never blocks the reply
        logger.debug("[ECO-4.83] inbox record failed: %s", e)
        return None


def _set(engine: Any, inbox_id: str, props: dict[str, Any]) -> None:
    add_node = getattr(engine, "add_node", None) if engine is not None else None
    if not callable(add_node) or not inbox_id:
        return
    try:
        add_node(
            node_id=inbox_id,
            node_type="InboundMessage",
            properties={"id": inbox_id, **props},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[ECO-4.83] inbox update failed: %s", e)


def mark_answered(engine: Any, inbox_id: str | None) -> None:
    """Mark an inbound message answered once a reply was actually delivered."""
    if inbox_id:
        _set(
            engine,
            inbox_id,
            {"status": "answered", "answered_at": datetime.now(UTC).isoformat()},
        )


def pending_unanswered(
    engine: Any, *, older_than_s: float = RETRY_GRACE_S, limit: int = 20
) -> list[dict[str, Any]]:
    """Inbound messages still ``pending``, past the grace window, under the retry cap."""
    qc = getattr(engine, "query_cypher", None) if engine is not None else None
    if not callable(qc):
        return []
    try:
        rows = (
            qc(
                "MATCH (m:InboundMessage {status: 'pending'}) "
                "RETURN m.id as id, m.platform as platform, m.channel_id as channel_id, "
                "m.text as text, m.session as session, m.message_id as message_id, "
                "m.received_at as received_at, m.attempts as attempts LIMIT $lim",
                {"lim": int(limit)},
            )
            or []
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[ECO-4.83] inbox query failed: %s", e)
        return []
    cutoff = datetime.now(UTC).timestamp() - older_than_s
    out: list[dict[str, Any]] = []
    for r in rows:
        ra = r.get("received_at")
        try:
            ts = datetime.fromisoformat(ra).timestamp() if ra else 0.0
        except Exception:  # noqa: BLE001
            ts = 0.0
        if ts and ts > cutoff:
            continue  # still inside the grace window — the first attempt may yet land
        out.append(r)
    return out


async def retry_unanswered(
    engine: Any, reply_send: Callable[[dict[str, Any]], Awaitable[bool]]
) -> int:
    """Reaper: re-attempt each unanswered inbound message. ``reply_send(msg) -> bool`` generates
    + delivers the reply for one message (the daemon provides it, since it owns the backends);
    returns whether it was delivered. Marks ``answered`` on success; bumps ``attempts`` and
    finally ``dead_letter`` at :data:`MAX_ATTEMPTS`. Returns the number answered this pass."""
    answered = 0
    for m in pending_unanswered(engine):
        attempts = int(m.get("attempts") or 0)
        if attempts >= MAX_ATTEMPTS:
            _set(engine, m["id"], {"status": "dead_letter", "attempts": attempts})
            continue
        try:
            ok = bool(await reply_send(m))
        except Exception as e:  # noqa: BLE001
            logger.debug("[ECO-4.83] inbox retry send failed: %s", e)
            ok = False
        if ok:
            mark_answered(engine, m["id"])
            answered += 1
        else:
            # Re-pass the FULL record (not just the counter): the durable backend replaces a
            # node's property blob on upsert, so a bare {attempts} write would wipe the
            # platform/channel/text the NEXT retry needs to re-send. (CONCEPT:AU-ECO.messaging.durable-inbound-pending)
            _set(
                engine,
                m["id"],
                {
                    "platform": m.get("platform"),
                    "channel_id": m.get("channel_id"),
                    "message_id": m.get("message_id"),
                    "text": m.get("text"),
                    "session": m.get("session"),
                    "received_at": m.get("received_at"),
                    "status": "pending",
                    "attempts": attempts + 1,
                    "last_retry_at": datetime.now(UTC).isoformat(),
                },
            )
    if answered:
        logger.info(
            "[CONCEPT:AU-ECO.messaging.durable-inbound-pending] inbox reaper answered %d previously-unanswered turn(s).",
            answered,
        )
    return answered
