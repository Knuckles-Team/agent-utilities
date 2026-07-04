"""Standalone messaging inbound daemon (CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs).

Runs the messaging ``InboundRouter`` in its OWN process — isolated from the KG host
daemon's CPU-bound maintenance (codebase ingestion, relevance sweeps, enrichment) so an
inbound message's reply is never starved by background work sharing the event loop/GIL.

It connects to the SHARED epistemic-graph engine **as a client** (the same graph the host
serves) for ingest/recall, so chat memory stays unified — only the heavy maintenance stays
in the host process. Combined with the non-blocking inbound path (ECO-4.72) and the reserved
interactive inference slot (ORCH-1.59), this is what makes the messaging agent reliably
responsive.

Console entry point: ``agent-utilities-messaging``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Any

logger = logging.getLogger(__name__)


async def _serve(engine: Any, platforms: list[str]) -> None:
    """Connect configured backends and run the InboundRouter (blocks on listeners)."""
    from agent_utilities.messaging.commands import command_specs
    from agent_utilities.messaging.router import InboundRouter, create_planner_handler
    from agent_utilities.messaging.service import MessagingService

    svc = MessagingService.instance(engine)
    router = InboundRouter()
    for pid in platforms:
        backend = await svc.get_backend(pid)
        if backend is None:
            continue
        svc.register_connected(backend)
        # Publish OUR universal command set (CONCEPT:AU-ECO.messaging.single-inbound-command-dispatcher) where the platform supports
        # a runtime menu; a no-op elsewhere.
        try:
            await backend.register_commands(command_specs("messaging"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("messaging: register_commands(%s) failed: %s", pid, exc)
        router.register_backend(backend)
    router.set_default_handler(await create_planner_handler(engine))
    logger.info("[CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs] messaging daemon serving backends %s", platforms)
    await router.start()  # blocks on the per-backend listener tasks


_FLEET_AUTH_KEYS = (
    "MCP_CLIENT_AUTH",
    "OIDC_CLIENT_ID",
    "OIDC_CLIENT_SECRET",
    "OIDC_AUDIENCE",
    "OIDC_TOKEN_URL",
)


def _load_fleet_auth() -> None:
    """Ensure OIDC client-credentials are in ``os.environ`` for fleet MCP auth (ECO-4.75).

    Source order (first hit wins; values never logged):
      1. Already in the environment (the deployed/genesis path: injected from OpenBao).
      2. OpenBao ``apps/mcp-multiplexer`` when this process can reach the vault.
      3. The local Claude MCP config (``~/.claude.json`` multiplexer ``env``) — the dev
         bridge, reusing the exact creds the workspace already runs the multiplexer with.
    Without these the spawned multiplexer cannot reach the jwt-protected fleet, so the chat
    agent can't load github/gitlab/etc. tools.
    """
    from agent_utilities.core.config import setting

    if setting("MCP_CLIENT_AUTH"):
        logger.info("[ECO-4.75] fleet auth: already in environment.")
        return

    # 2) OpenBao (source of truth) — only when reachable from this process.
    try:
        vurl = str(setting("OPENBAO_URL", "")).rstrip("/")
        vtok = str(setting("OPENBAO_TOKEN", ""))
        if vurl and vtok:
            import json as _json
            import urllib.request

            req = urllib.request.Request(
                f"{vurl}/v1/apps/data/mcp-multiplexer",
                headers={"X-Vault-Token": vtok},
            )
            with urllib.request.urlopen(req, timeout=6) as r:  # nosec B310 — controlled https fleet/telegram URL
                data = _json.loads(r.read()).get("data", {}).get("data", {})
            got = [k for k in _FLEET_AUTH_KEYS if data.get(k)]
            for k in got:
                os.environ[k] = str(data[k])
            if got:
                logger.info(
                    "[ECO-4.75] fleet auth: loaded from OpenBao apps/mcp-multiplexer (%s).",
                    got,
                )
                return
    except Exception as e:  # noqa: BLE001
        logger.debug("[ECO-4.75] OpenBao fleet-auth read skipped: %s", e)

    # 3) Dev bridge: the multiplexer entry in ~/.claude.json already carries these creds.
    try:
        import json as _json
        from pathlib import Path

        cfg = _json.loads((Path.home() / ".claude.json").read_text())

        def _find(o: Any) -> dict[str, Any] | None:
            if isinstance(o, dict):
                env = o.get("env")
                if isinstance(env, dict) and env.get("OIDC_CLIENT_ID"):
                    return env
                for v in o.values():
                    r = _find(v)
                    if r:
                        return r
            elif isinstance(o, list):
                for i in o:
                    r = _find(i)
                    if r:
                        return r
            return None

        env = _find(cfg) or {}
        got = [k for k in _FLEET_AUTH_KEYS if env.get(k)]
        for k in got:
            os.environ[k] = str(env[k])
        if got:
            logger.info(
                "[ECO-4.75] fleet auth: loaded from local Claude MCP config (%s).", got
            )
            return
    except Exception as e:  # noqa: BLE001
        logger.debug("[ECO-4.75] Claude-config fleet-auth read skipped: %s", e)

    logger.warning(
        "[ECO-4.75] fleet auth: no OIDC creds found (env/OpenBao/Claude-config). The chat "
        "agent's multiplexer may fail to reach jwt-protected fleet tools."
    )


def main() -> None:
    """Run the isolated messaging inbound daemon (CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Connect to the shared engine as a CLIENT — do NOT take the host role. KG maintenance
    # (sweeps/ingestion) stays in the gateway host daemon; this process only handles chat.
    os.environ.setdefault("KG_DAEMON_ROLE", "client")

    # CONCEPT:AU-ECO.messaging.make-fleet-credentials-present — make fleet credentials present in THIS process env so the MCP
    # multiplexer the chat agent spawns (and every nested agent graph_orchestrate spawns,
    # via _spawn_auth_headers) can authenticate to the jwt-protected fleet.
    _load_fleet_auth()

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.messaging.service import MessagingService

    engine = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine()
    platforms = MessagingService.instance(engine).configured_platforms()
    if not platforms:
        logger.info(
            "messaging daemon: no backend configured (set TELEGRAM_BOT_TOKEN). Exiting."
        )
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop: asyncio.Future[None] = loop.create_future()

    def _shutdown(*_a: Any) -> None:
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:  # pragma: no cover — non-Unix
            pass

    serve_task = loop.create_task(_serve(engine, platforms))
    logger.info("[CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs] isolated messaging daemon started.")
    try:
        loop.run_until_complete(stop)
    finally:
        serve_task.cancel()
        loop.run_until_complete(asyncio.gather(serve_task, return_exceptions=True))
        loop.close()


if __name__ == "__main__":  # pragma: no cover
    main()
