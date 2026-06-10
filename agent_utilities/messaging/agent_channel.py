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

import contextlib
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


def _channels(engine: Any) -> Any | None:
    """Reach the engine's channels sub-client (None if unavailable)."""
    client = getattr(getattr(engine, "graph_compute", None), "_client", None)
    return getattr(client, "channels", None)


def channel_id_for(session_id: str, run_id: str) -> str:
    """Stable P2P channel id for an invoker↔spawned-agent pair."""
    return f"orch:{session_id}:{run_id}"


def _parse_channel_id(channel_id: str) -> tuple[str, str]:
    """Recover (session_id, run_id) from ``orch:{session_id}:{run_id}`` (best-effort)."""
    parts = channel_id.split(":")
    if len(parts) >= 3 and parts[0] == "orch":
        return parts[1], ":".join(parts[2:])
    return channel_id, ""


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
        # "Group" (not PeerToPeer): allows members to join after creation, so an invoker and a
        # spawned agent that connect at different times — and any tool-chosen sender label — can
        # participate. PeerToPeer locks membership at creation and rejects later joins/senders.
        ch.create(
            cid,
            channel_type="Group",
            creator=invoker_member,
            initial_members=[invoker_member, agent_member],
        )
    except Exception as exc:  # noqa: BLE001 — already-exists / re-open is fine
        logger.debug("open_channel(%s): %s", cid, exc)
    # Upsert the durable Session anchor ONCE here (re-adding the node on every durable send
    # resets its outgoing edges on the engine, dropping earlier HAS_MESSAGE links).
    add_node = getattr(engine, "add_node", None)
    if callable(add_node):
        with contextlib.suppress(Exception):
            add_node(
                f"session:{session_id}",
                "Session",
                properties={"id": f"session:{session_id}", "session_id": session_id},
            )
    return cid


def send(
    engine: Any, channel_id: str, sender: str, payload: str, *, durable: bool = False
) -> bool:
    """Send a message; returns True on success.

    CONCEPT:ORCH-1.39 (Phase 4) — when ``durable`` is set, dual-write the message as a
    ``Session -[:HAS_MESSAGE]-> AgentMessage`` node so the dialogue survives engine restart and
    is replayable via :func:`history` (the live channel is in-RAM; the graph node is the record).
    """
    ch = _channels(engine)
    if ch is None or not channel_id:
        return False
    # Ensure the sender is a member (Group channels admit late joins); idempotent — an
    # already-member join is a no-op we tolerate. Without this, send_message rejects unknown
    # senders ("not a member of channel").
    with contextlib.suppress(Exception):
        ch.join(channel_id, sender)
    try:
        ch.send_message(channel_id, sender, payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("channel send failed (%s): %s", channel_id, exc)
        return False
    if durable:
        _persist_message(engine, channel_id, sender, payload)
    return True


def _persist_message(engine: Any, channel_id: str, sender: str, payload: str) -> None:
    """Dual-write a durable AgentMessage node anchored to the Session (best-effort)."""
    add_node = getattr(engine, "add_node", None)
    add_edge = getattr(engine, "add_edge", None)
    if not callable(add_node) or not callable(add_edge):
        return
    sid, run_id = _parse_channel_id(channel_id)
    snode = f"session:{sid}"
    mid = f"msg:{sid}:{run_id}:{uuid.uuid4().hex[:8]}"
    with contextlib.suppress(Exception):
        add_node(
            mid,
            "AgentMessage",
            properties={
                "id": mid,
                "channel_id": channel_id,
                "session_id": sid,
                "run_id": run_id,
                "sender": sender,
                "payload": payload,
                "created_at": time.time(),
            },
        )
        # The Session anchor is upserted once at open_channel — re-adding it here would reset
        # its outgoing edges and drop earlier messages. Just link the new message.
        add_edge(snode, mid, "HAS_MESSAGE")


def history(engine: Any, channel_id: str) -> list[dict[str, Any]]:
    """Replay the durable message history for a channel (id-anchored Session traversal).

    Reliable regardless of engine restart — reads AgentMessage nodes (not the in-RAM channel),
    ordered by ``created_at``. Returns [] when nothing was persisted / engine unavailable.
    """
    query = getattr(engine, "query_cypher", None)
    if not callable(query):
        return []
    sid, _ = _parse_channel_id(channel_id)
    snode = f"session:{sid}"
    try:
        rows = query(
            "MATCH (s {id: $snode})-[:HAS_MESSAGE]->(m:AgentMessage) RETURN m",
            {"snode": snode},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("channel history failed (%s): %s", channel_id, exc)
        return []
    msgs: list[dict[str, Any]] = []
    for row in rows or []:
        node = row.get("m", row) if isinstance(row, dict) else row
        props = node.get("properties", node) if isinstance(node, dict) else {}
        if props.get("channel_id") == channel_id:
            msgs.append(props)
    msgs.sort(key=lambda m: m.get("created_at", 0))
    return msgs


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


_ELICIT_TAG = "__elicit__"


def send_elicitation(
    engine: Any, channel_id: str, prompt: str, *, sender: str = "agent", durable: bool = True
) -> bool:
    """CONCEPT:ORCH-1.39 (Phase 4) — a spawned agent asks its invoker (→ user) a question.

    Encodes a tagged JSON payload the invoker recognises on the channel; durable by default so
    the request is not lost if the invoker is not actively polling. Bridge with
    :func:`drain_to_elicitation_queue` on the invoker side.
    """
    import json

    payload = json.dumps({_ELICIT_TAG: True, "prompt": prompt})
    return send(engine, channel_id, sender, payload, durable=durable)


def drain_to_elicitation_queue(
    engine: Any, channel_id: str, queue: Any, since: int = 0
) -> int:
    """Forward any pending elicitation requests on the channel to an in-process queue.

    Cross-process → in-process bridge: the invoker polls its spawned agent's channel and, for
    each tagged elicitation message, ``queue.put_nowait(prompt)`` onto its existing
    ``AgentDeps.elicitation_queue``/``ApprovalManager`` — no UI change. Returns the new receive
    cursor (pass back as ``since`` next call).
    """
    import json

    msgs, cursor = receive(engine, channel_id, since=since)
    for m in msgs:
        payload = m.get("payload") if isinstance(m, dict) else None
        if not isinstance(payload, str):
            continue
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and data.get(_ELICIT_TAG) and queue is not None:
            with contextlib.suppress(Exception):
                queue.put_nowait(data.get("prompt", ""))
    return cursor


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
