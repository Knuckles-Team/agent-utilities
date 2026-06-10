"""CONCEPT:ORCH-1.39 — invoker↔spawned-agent message channel over the engine's native
Communication Channels (KG-2.0).

The epistemic-graph Tokio server runs in one shared process that both the invoking agent and
the spawned agent connect to over UDS, so routing messages through its `ChannelManager` gives
**cross-process, totally-ordered** delivery for free (~0.19 ms/op). This module is a thin,
sync wrapper around `engine.graph_compute._client.channels`.

Receive uses a **client-side cursor** (`since` = count already consumed) over `get_messages`,
so no engine change is required. (A server-side `since_seq` cursor is a deferred scale
optimization — see the ORCH-1.39 design; per-channel dialogues are small so the O(n) re-read
is fine for now.)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _channels(engine: Any) -> Any | None:
    """Reach the engine's channels sub-client (None if unavailable)."""
    client = getattr(getattr(engine, "graph_compute", None), "_client", None)
    return getattr(client, "channels", None)


def channel_id_for(session_id: str, run_id: str) -> str:
    """Stable P2P channel id for an invoker↔spawned-agent pair."""
    return f"orch:{session_id}:{run_id}"


def open_channel(
    engine: Any, session_id: str, run_id: str, *, invoker: str | None = None
) -> str | None:
    """Create (idempotently) the P2P channel and return its id, or None if unavailable."""
    ch = _channels(engine)
    if ch is None:
        return None
    cid = channel_id_for(session_id, run_id)
    invoker_member = invoker or f"invoker:{session_id}"
    agent_member = f"agent:{run_id}"
    try:
        ch.create(
            cid,
            channel_type="PeerToPeer",
            creator=invoker_member,
            initial_members=[invoker_member, agent_member],
        )
    except Exception as exc:  # noqa: BLE001 — already-exists / re-open is fine
        logger.debug("open_channel(%s): %s", cid, exc)
    return cid


def send(engine: Any, channel_id: str, sender: str, payload: str) -> bool:
    """Send a message; returns True on success."""
    ch = _channels(engine)
    if ch is None or not channel_id:
        return False
    try:
        ch.send_message(channel_id, sender, payload)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("channel send failed (%s): %s", channel_id, exc)
        return False


def receive(engine: Any, channel_id: str, since: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Return (new_messages_after `since`, new_cursor).

    `since` is the count already consumed by this caller; the returned cursor is the new total,
    to pass as `since` on the next call. Ordering is the engine's total order.
    """
    ch = _channels(engine)
    if ch is None or not channel_id:
        return [], since
    try:
        msgs = ch.get_messages(channel_id) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("channel receive failed (%s): %s", channel_id, exc)
        return [], since
    new = msgs[since:] if since < len(msgs) else []
    return new, len(msgs)


def close(engine: Any, channel_id: str) -> bool:
    """Close the channel (persists the engine-side summary imprint)."""
    ch = _channels(engine)
    if ch is None or not channel_id:
        return False
    try:
        ch.close(channel_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("channel close (%s): %s", channel_id, exc)
        return False
