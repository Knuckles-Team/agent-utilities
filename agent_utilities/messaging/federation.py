"""BusFederationRelay — forward AgentBus messages across hubs (CONCEPT:ECO-4.86).

The single-hub bus (``AgentBus``, ECO-4.84) connects every session pointed at one graph-os.
The federation relay is the *mesh* layer on top: independent hubs peer over A2A HTTP so a
message published on hub A reaches a subscriber on hub B. It is built strictly **above**
``AgentBus`` — it never changes send/receive semantics; it reads a delivered message group and
forwards it to peer hubs, and on the receiving side applies it once (idempotent by ``msg_group``).

Key properties:
- **Peer discovery is KG-native** — hubs register as A2A peers (``protocols/a2a.py``) carrying the
  ``agent-bus-hub`` capability, so the existing agent-card/peer-registry machinery is reused.
- **Dedup + loop-break** — a forwarded message keeps its origin ``msg_group`` (deterministic node
  ids ⇒ a re-forward is a no-op upsert) and is stamped ``federated_from``; the relay never
  re-forwards a message that already carries that stamp.
- **Marking-scoped** — only ``commons`` traffic crosses a hub boundary; ``private``/``org``-marked
  messages (KG-2.60) stay home.

CONCEPT:ECO-4.86 — BusFederationRelay: cross-hub forwarding for the agent bus
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

#: A2A capability tag that marks a peer as a federated bus hub.
HUB_CAPABILITY = "agent-bus-hub"
#: Scopes that must never cross a hub boundary (KG-2.60 markings).
_LOCAL_ONLY_SCOPES = {"private", "org"}


def hub_id() -> str:
    """This hub's stable id (``BUS_HUB_ID`` or the hostname) — the federation origin label."""
    return str(setting("BUS_HUB_ID", socket.gethostname()))


class BusFederationRelay:
    """Forwards bus messages to peer hubs and applies inbound forwards. CONCEPT:ECO-4.86."""

    _instance: BusFederationRelay | None = None

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine

    @classmethod
    def instance(cls, engine: Any = None) -> BusFederationRelay:
        if cls._instance is None:
            cls._instance = cls(engine=engine)
        elif engine is not None and cls._instance._engine is None:
            cls._instance._engine = engine
        return cls._instance

    def _bus(self) -> Any:
        from agent_utilities.messaging.bus import AgentBus

        # Bind to this relay's own engine (one engine per hub) rather than the global
        # singleton, so a process hosting more than one hub keeps them isolated.
        return AgentBus(self._engine) if self._engine is not None else AgentBus.instance()

    # ── Peer hub registry (KG-native via A2A, ECO-4.86) ──────────────
    def register_hub(self, name: str, url: str, *, auth: str = "none") -> str:
        """Register a peer bus hub (an A2A peer tagged with the hub capability)."""
        from agent_utilities.protocols.a2a import register_a2a_peer

        return register_a2a_peer(
            name,
            url,
            description="agent bus hub",
            capabilities=HUB_CAPABILITY,
            auth=auth,
        )

    def list_hubs(self) -> list[dict[str, str]]:
        """Peer hubs known to this hub (A2A peers carrying ``agent-bus-hub``)."""
        from agent_utilities.protocols.a2a import list_a2a_peers

        out: list[dict[str, str]] = []
        for peer in list_a2a_peers().peers:
            caps = peer.capabilities or ""
            cap_list = caps.split(",") if isinstance(caps, str) else list(caps)
            if HUB_CAPABILITY in [c.strip() for c in cap_list] and peer.url:
                out.append({"name": peer.name, "url": peer.url})
        return out

    # ── Outbound forward (ECO-4.86) ──────────────────────────────────
    def forward(self, group: str, *, scope: str = "commons") -> dict[str, Any]:
        """Forward every message in ``group`` to each peer hub (one HTTP call per hub).

        Skips groups that are marked local-only or that were themselves received from another
        hub (loop break). Returns per-hub delivery counts.
        """
        if scope in _LOCAL_ONLY_SCOPES:
            return {"ok": True, "forwarded": 0, "skipped": f"scope={scope}"}
        msgs = self._bus().group_messages(group)
        if not msgs:
            return {"ok": False, "error": f"no local messages for group {group}"}
        if any(m.get("federated_from") for m in msgs):
            return {"ok": True, "forwarded": 0, "skipped": "already_federated"}

        first = msgs[0]
        recipients = sorted({m.get("recipient") for m in msgs if m.get("recipient")})
        body = {
            "action": "federate_in",
            "group": group,
            "sender": first.get("sender", ""),
            "topic": first.get("topic", ""),
            "to": ",".join(recipients),
            "payload": first.get("payload", ""),
            "origin": hub_id(),
        }
        hubs = self.list_hubs()
        results: dict[str, Any] = {}
        for hub in hubs:
            results[hub["name"]] = self._post(hub["url"], body)
        return {"ok": True, "group": group, "hubs": len(hubs), "results": results}

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        try:
            resp = httpx.post(f"{url.rstrip('/')}/graph/bus", json=body, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            inner = data.get("result", data)
            return json.loads(inner) if isinstance(inner, str) else inner
        except Exception as exc:  # noqa: BLE001 — a peer being down must not break the others
            logger.warning("[ECO-4.86] forward to %s failed: %s", url, exc)
            return {"ok": False, "error": str(exc)}

    # ── Inbound apply (ECO-4.86) ─────────────────────────────────────
    def apply_inbound(
        self,
        *,
        group: str,
        sender: str,
        recipients: list[str],
        payload: str,
        topic: str,
        origin: str,
    ) -> dict[str, Any]:
        """Apply a message forwarded from ``origin`` (idempotent, loop-safe)."""
        bus = self._bus()
        if bus.group_exists(group):
            return {"ok": True, "applied": 0, "dedup": True}
        delivered = bus.deliver_federated(
            group=group,
            sender=sender,
            recipients=recipients,
            payload=payload,
            topic=topic,
            origin=origin,
        )
        return {"ok": True, "applied": len(delivered), "delivered": delivered}
