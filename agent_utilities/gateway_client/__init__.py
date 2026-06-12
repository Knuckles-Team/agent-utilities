"""``agent_utilities.gateway_client`` — the surface-facing gateway client SDK.

CONCEPT:ECO-4.37 — Surface Gateway Client SDK

The consolidation audit's top *surface-side* finding: every user-facing surface
(agent-webui, agent-terminal-ui, geniusbot) hand-rolls its own HTTP/SSE client
against the same gateway — ~800 lines of duplicated transport, with hardcoded
URLs, no auth, no retry, and no rate-limit handling.

This is the single shared client the Python surfaces strangle their bespoke
clients onto. It is built on :class:`agent_utilities.http.AsyncBaseApiClient`
(ECO-4.35) — so it inherits auth injection, 429 backoff, error mapping, and log
redaction for free — and adds typed methods for the gateway's REST surface plus a
Server-Sent-Events streaming helper for the agent ``/stream`` channel.

Sibling of the *fleet*-side library: :mod:`agent_utilities.http` consolidated the
~58 connector agents' API clients; this consolidates the surfaces' gateway clients.

Example::

    from agent_utilities.gateway_client import GatewayClient

    async with GatewayClient("http://gateway.arpa:9000", token=jwt) as gw:
        agents = await gw.list_agents()
        async for event in gw.stream("Summarize the payments service"):
            print(event.get("type"), event.get("content", ""))
"""

from __future__ import annotations

from agent_utilities.gateway_client.client import DEFAULT_GATEWAY_URL, GatewayClient

__all__ = ["DEFAULT_GATEWAY_URL", "GatewayClient"]
