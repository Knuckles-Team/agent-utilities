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
    logger.info(
        "[CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs] messaging daemon serving backends %s",
        platforms,
    )
    await router.start()  # blocks on the per-backend listener tasks


def _validate_fleet_auth() -> None:
    """Load XDG AgentConfig and fail closed on incomplete outbound fleet auth."""
    from agent_utilities.core.config import load_config
    from agent_utilities.mcp.client_credentials import (
        outbound_auth_configuration_status,
        validate_outbound_auth_configuration,
    )

    load_config()
    status = outbound_auth_configuration_status()
    validate_outbound_auth_configuration()
    logger.info(
        "[ECO-4.75] fleet auth configuration ready (mode=%s).",
        status["mode"],
    )


def _mint_process_session() -> Any:
    """Mint this process's verified graph authority (mirrors ``ingest_worker.py``).

    The engine's tenant-routing (``resolve_routing_graph``) and its write-path
    guard (``BrainGuardedBackend._verified_actor``) both require an ambient
    :class:`~agent_utilities.security.brain_context.ActorContext` — Company
    Brain enforcement is mandatory (CONCEPT:AU-KG.compute.data-is-private-its),
    not opt-in. As a standalone process (not a served MCP/REST boundary) this
    daemon must mint its OWN process identity before touching the engine —
    exactly like ``knowledge_graph/ingest_worker.py`` and ``mcp/kg_server.py``'s
    ``_mint_process_session`` do. Configure exactly one of ``KG_AUTH_TOKEN_REF``
    or ``KG_IDENTITY_OAUTH2``.
    """
    from agent_utilities.core.config import config
    from agent_utilities.security.request_identity import (
        acquire_process_identity_token,
        mint_actor_from_token_sync,
        mint_graph_session,
    )

    token = acquire_process_identity_token(config)
    actor = mint_actor_from_token_sync(token)
    session = mint_graph_session(actor)
    session.engine_verified_context()
    logger.info("[ECO-4.75] messaging daemon: verified graph process authority minted.")
    return session


def main() -> None:
    """Run the isolated messaging inbound daemon (CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Connect to the shared engine as a CLIENT — do NOT take the host role. KG maintenance
    # (sweeps/ingestion) stays in the gateway host daemon; this process only handles chat.
    os.environ.setdefault("KG_DAEMON_ROLE", "client")

    # CONCEPT:AU-ECO.messaging.make-fleet-credentials-present — every process
    # consumes the same XDG AgentConfig declaration. Credential material remains
    # behind its runtime reference and is resolved only at the outbound request.
    _validate_fleet_auth()

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.messaging.service import MessagingService
    from agent_utilities.security.brain_context import use_actor

    session = _mint_process_session()

    # Scope the whole process to this session's actor — engine construction
    # (tenant-graph routing) and every subsequent graph write (provenance/
    # authority arbitration) both require it. asyncio Tasks created below copy
    # the ambient contextvars at creation time, so every inbound-message task
    # inherits this same authority for the life of the process.
    with use_actor(session.actor), use_session(session):
        engine = IntelligenceGraphEngine.get_or_create()
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

        tasks = [loop.create_task(_serve(engine, platforms))]
        # Optional HTTP alert-intake (CONCEPT:AU-ECO.messaging.alert-intake): route external
        # webhooks (uptime-kuma, Alertmanager, …) THROUGH the messaging stack so alerts inherit
        # the one unified Telegram/Mattermost/… delivery instead of each tool wiring its own
        # notifier. Opt-in via MESSAGING_ALERT_INTAKE_PORT; runs as an INDEPENDENT task so a
        # failure here never affects the inbound listeners.
        from agent_utilities.core.config import setting

        _intake_port = setting("MESSAGING_ALERT_INTAKE_PORT", "")
        if _intake_port:
            from agent_utilities.messaging.alert_intake import serve_alert_intake

            tasks.append(
                loop.create_task(serve_alert_intake(engine, int(_intake_port)))
            )
        logger.info(
            "[CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs] isolated messaging daemon started."
        )
        try:
            loop.run_until_complete(stop)
        finally:
            for _t in tasks:
                _t.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.close()


if __name__ == "__main__":  # pragma: no cover
    main()
