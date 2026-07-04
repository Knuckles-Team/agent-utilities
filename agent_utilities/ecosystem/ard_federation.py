"""CONCEPT:AU-ECO.interop.ard-federation-relay — ARD federation relay (cross-registry discovery).

ARD federation lets one registry's ``/search`` surface capabilities hosted by another:
a consumer hits us, and (in ``auto`` mode) we fan the query out to our peer registries,
merge their ranked results with ours, and return the union. This mirrors the agent-bus
:class:`messaging.federation.BusFederationRelay` exactly — peers are discovered KG-natively
as A2A peers carrying an ``ard-registry`` capability, so the existing agent-card/peer
registry machinery is reused rather than a second peer store.

Three modes (the ARD spec's federation modes):

* ``none`` — local results only (no fan-out).
* ``referrals`` — local results plus a ``referrals`` list naming peer registries that may
  hold matches (the consumer follows up itself; we do not fetch).
* ``auto`` — local results plus a concurrent fan-out to every peer's ``/search``, merged
  and re-ranked, de-duplicated by ``(publisher domain, resource id)``.

Loop-break: outbound peer requests are sent with ``federationMode="none"`` (so a peer
never re-fans-out — a structural ``max_depth = 1``) and carry a ``via`` chain stamped
with this registry's origin; an inbound ``/search`` that already lists our origin in
``via`` is served local-only. A peer being down is isolated (its fan-out errors to an
empty contribution) and never breaks the others or the local result.
"""

from __future__ import annotations

import json
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

#: A2A capability tag that marks a peer as a federated ARD registry.
REGISTRY_CAPABILITY = "ard-registry"
#: Max peers queried concurrently in an ``auto`` fan-out.
_FANOUT_CONCURRENCY = 8
#: Per-peer search timeout (seconds) — a slow peer is dropped, not allowed to stall.
_PEER_TIMEOUT_S = 8.0


def origin() -> str:
    """This registry's stable origin label (``ARD_PUBLISHER_DOMAIN`` or hostname)."""
    return str(setting("ARD_PUBLISHER_DOMAIN", default=socket.gethostname()))


def default_mode() -> str:
    """The configured default federation mode (``none``/``referrals``/``auto``)."""
    mode = str(setting("ARD_FEDERATION_MODE", default="none")).lower().strip()
    return mode if mode in {"none", "referrals", "auto"} else "none"


class ArdFederationRelay:
    """Fan an ARD ``/search`` out to peer registries and merge results. CONCEPT:AU-ECO.interop.ard-federation-relay."""

    # ── Peer registry (KG-native via A2A, mirrors BusFederationRelay) ────────
    def register_registry(self, name: str, url: str, *, auth: str = "none") -> str:
        """Register a peer ARD registry (an A2A peer tagged with the registry capability)."""
        from ..protocols.a2a import register_a2a_peer

        return register_a2a_peer(
            name,
            url,
            description="ARD resource registry",
            capabilities=REGISTRY_CAPABILITY,
            auth=auth,
        )

    def list_registries(self) -> list[dict[str, str]]:
        """Peer ARD registries known to us (A2A peers carrying ``ard-registry``)."""
        from ..protocols.a2a import list_a2a_peers

        out: list[dict[str, str]] = []
        for peer in list_a2a_peers().peers:
            caps = peer.capabilities or ""
            cap_list = caps.split(",") if isinstance(caps, str) else list(caps)
            if REGISTRY_CAPABILITY in [c.strip() for c in cap_list] and peer.url:
                out.append({"name": peer.name, "url": peer.url})
        return out

    # ── Federated search (ECO-4.97) ──────────────────────────────────────────
    def federated_search(
        self,
        query_text: str,
        *,
        types: list[str] | None = None,
        page_size: int = 5,
        mode: str | None = None,
        via: list[str] | None = None,
        multiplexer: Any = None,
        engine: Any = None,
    ) -> dict:
        """Run a local ARD search and, per ``mode``, merge peer-registry results.

        ``via`` is the federation chain (loop-break): if our origin is already present we
        serve local-only regardless of ``mode``. The returned envelope carries the merged,
        de-duplicated, re-ranked ``results`` plus the ``federationMode`` actually applied.
        """
        from . import ard_registry

        chain = list(via or [])
        local = ard_registry.ard_search(
            query_text,
            types=types,
            page_size=page_size,
            multiplexer=multiplexer,
            engine=engine,
        )
        results = list(local.get("results") or [])
        applied = (mode or default_mode()).lower().strip()

        # Loop-break: if we're already in the chain, never fan out again.
        if origin() in chain:
            applied = "none"

        if applied == "none":
            return {**local, "federationMode": "none", "via": chain + [origin()]}

        peers = self.list_registries()
        if applied == "referrals":
            return {
                **local,
                "results": results[:page_size],
                "federationMode": "referrals",
                "referrals": peers,
                "via": chain + [origin()],
            }

        # auto: concurrent fan-out to every peer, merged + de-duplicated.
        body = {
            "query": {"text": query_text, "filter": {"type": list(types or [])}},
            "pageSize": page_size,
            "federationMode": "none",  # peers must not re-fan-out (max_depth = 1)
            "via": chain + [origin()],
        }
        for peer_results in self._fanout(peers, body):
            results.extend(peer_results)

        merged = _dedupe(results)
        merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return {
            **local,
            "results": merged[:page_size],
            "federationMode": "auto",
            "peers": len(peers),
            "via": chain + [origin()],
        }

    def _fanout(self, peers: list[dict[str, str]], body: dict) -> list[list[dict]]:
        if not peers:
            return []
        workers = min(len(peers), _FANOUT_CONCURRENCY)
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ard-fed"
        ) as ex:
            return list(ex.map(lambda p: self._post(p["url"], body), peers))

    def _post(self, url: str, body: dict) -> list[dict]:
        import httpx

        try:
            resp = httpx.post(
                f"{url.rstrip('/')}/search", json=body, timeout=_PEER_TIMEOUT_S
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, str):
                data = json.loads(data)
            results = data.get("results") if isinstance(data, dict) else None
            return [r for r in (results or []) if isinstance(r, dict)]
        except Exception as exc:  # noqa: BLE001 — a down peer must not break the others
            logger.warning("[ECO-4.97] ARD fan-out to %s failed: %s", url, exc)
            return []


def _dedupe(results: list[dict]) -> list[dict]:
    """Keep the highest-scoring result per ``(publisher domain, resource id)``."""
    best: dict[tuple[str, str], dict] = {}
    for r in results:
        domain = str((r.get("publisher") or {}).get("domain", ""))
        key = (domain, str(r.get("id", "")))
        cur = best.get(key)
        if cur is None or r.get("score", 0.0) > cur.get("score", 0.0):
            best[key] = r
    return list(best.values())
