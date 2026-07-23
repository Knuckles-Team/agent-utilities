"""Standalone messaging inbound daemon (CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs).

Runs the messaging ``InboundRouter`` in its OWN process — isolated from the KG host
daemon's CPU-bound maintenance (codebase ingestion, relevance sweeps, enrichment) so an
inbound message's reply is never starved by background work sharing the event loop/GIL.

It connects to the SHARED epistemic-graph engine **as a client** (the same graph the host
serves) for ingest/recall, so chat memory stays unified — only the heavy maintenance stays
in the host process. Combined with the non-blocking inbound path (ECO-4.72) and the reserved
interactive inference slot (ORCH-1.59), this is what makes the messaging agent reliably
responsive.

Console entry point: ``agent-utilities-messaging``. The serving/shutdown body
(:func:`run_forever`) is also reused, unchanged, by the ``graph-os`` entrypoint's
in-process co-service supervisor
(:mod:`agent_utilities.mcp.co_service_supervisor`) when messaging is configured —
so the standalone process and the composed co-service share exactly one
implementation of "connect the configured backends and serve until told to stop."
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
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
        "[CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs] messaging serving backends %s",
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


def configured_platforms(engine: Any = None) -> list[str]:
    """Platforms that are installed AND have real credentials configured.

    A pure config read (delegates to :meth:`MessagingService.configured_platforms`,
    which only inspects :class:`~agent_utilities.messaging.registry.MessagingRegistry` —
    no engine access) so composition detection can call this BEFORE any engine or
    process identity exists (CONCEPT:AU-ECO.messaging.messaging-reach-service-governed).
    """
    from agent_utilities.messaging.service import MessagingService

    return MessagingService.instance(engine).configured_platforms()


def mint_process_identity() -> Any:
    """Mint this process's verified graph authority.

    Copies the established pattern from ``kg_server.py``/``ingest_worker.py``
    (``acquire_process_identity_token`` → ``mint_actor_from_token_sync`` →
    ``mint_graph_session`` → ``engine_verified_context()``). Previously this daemon
    called ``IntelligenceGraphEngine()``/``get_or_create()`` bare, with no process
    identity wired at all — every real (non-``AGENT_UTILITIES_TESTING``) engine
    construction requires a verified :class:`GraphSession` (CONCEPT:AU-OS.identity.authenticated-identity-enforcement),
    so the messaging daemon could never actually construct the engine in a real
    deployment. This is that missing wiring.
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
    logger.info("Verified messaging process authority minted")
    return session


def run_forever(engine: Any, platforms: list[str], stop_event: threading.Event) -> None:
    """Blocking body: connect the configured backends and serve until ``stop_event``.

    Owns its OWN event loop (asyncio state is not shareable across threads), so this
    is safe to call either directly from the main thread of the standalone
    ``agent-utilities-messaging`` process (:func:`main`, driven by OS signals) or from
    a dedicated supervisor thread inside the ``graph-os`` entrypoint (driven by the
    supervisor's own shutdown event) — the serving/shutdown logic is defined exactly
    once and shared by both callers.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop: asyncio.Future[None] = loop.create_future()

    def _watch_stop() -> None:
        stop_event.wait()
        if not loop.is_closed():
            loop.call_soon_threadsafe(lambda: stop.done() or stop.set_result(None))

    watcher = threading.Thread(
        target=_watch_stop, daemon=True, name="MessagingStopWatch"
    )
    watcher.start()

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

        tasks.append(loop.create_task(serve_alert_intake(engine, int(_intake_port))))
    logger.info(
        "[CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs] messaging serving started."
    )
    try:
        loop.run_until_complete(stop)
    finally:
        for _t in tasks:
            _t.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()


def main() -> None:
    """Run the isolated messaging inbound daemon as a standalone process."""
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

    session = mint_process_identity()
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor

    with use_actor(session.actor), use_session(session):
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        engine = IntelligenceGraphEngine.get_or_create()
        platforms = configured_platforms(engine)
        if not platforms:
            logger.info(
                "messaging daemon: no backend configured (set TELEGRAM_BOT_TOKEN). Exiting."
            )
            return

        stop_event = threading.Event()

        def _shutdown(*_a: Any) -> None:
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _shutdown)
            except (
                ValueError,
                OSError,
            ):  # pragma: no cover — not main thread / unsupported
                pass

        logger.info(
            "[CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs] isolated messaging daemon started."
        )
        run_forever(engine, platforms, stop_event)


if __name__ == "__main__":  # pragma: no cover
    main()
