"""Serve agent-webui inside the graph-os process as a supervised co-service.

CONCEPT:AU-OS.deployment.webui-co-service — agent-webui as a graph-os co-service

Why this exists
---------------
Every piece of this was already built and only the last wire was missing:

* ``agent-utilities[ag-ui]`` already declares ``agent-webui`` as an optional
  dependency, so the package is installable alongside graph-os with no new
  distribution work.
* :func:`agent_utilities.server.app.build_agent_app` already mounts
  ``agent_webui.server.create_agent_web_app`` at ``/`` when ``enable_web_ui``
  is set, alongside the gateway routers agent-webui would otherwise mount for
  itself — agent-webui is the frontend facade over those same routers.
* ``ENABLE_WEB_UI`` is already real config, and
  :func:`agent_utilities.mcp.co_service_supervisor.detect_composition` already
  reports it as part of the composition plan.

What was missing is the branch that actually starts it. That branch previously
declined on the premise that agent-webui is "a separate Node/Vite frontend,
not a Python asyncio task". That premise is stale: agent-webui ships a FastAPI
application factory and *serves* its built Vite bundle as SPA static files, so
it runs in-process like any other ASGI app.

The identity consequence is the point
-------------------------------------
A co-service runs on ``_authorized_background_thread``, inheriting the graph-os
process's verified actor and ``GraphSession`` for its whole lifetime. graph-os
is the principal the engine's signer registry already trusts, so a WebUI
request handled here can sign engine admission **as itself**
(:func:`agent_utilities.security.admission_authority.resolve_admission_authority`)
— which is the only pairing the engine accepts
(``verify_register_identity_signature`` requires ``signer == principal``). Run
as a separate deployment, agent-webui holds no signer entry at all, which is
why tenant admission failed for every sign-in.
"""

from __future__ import annotations

import logging
import threading

__all__ = ["run_web_ui"]

logger = logging.getLogger(__name__)

#: How often the serving loop checks the supervisor's stop event. Short enough
#: that shutdown is prompt, long enough that an idle co-service costs nothing.
_STOP_POLL_SECONDS = 0.5


def run_web_ui(
    stop_event: threading.Event,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Serve the WebUI dashboard until ``stop_event`` is set.

    A blocking ``run(stop_event)`` callable in the shape
    :meth:`~agent_utilities.mcp.co_service_supervisor.CoServiceSupervisor.start_service`
    expects, so the supervisor's bounded-restart policy applies unchanged.

    Raises:
        ImportError: if the ``ag-ui`` extra is not installed. Deliberately
            propagated rather than swallowed — a deployment that set
            ``ENABLE_WEB_UI`` asked for this, and silently serving nothing is
            the failure mode this whole change exists to remove.
    """

    import asyncio

    import uvicorn

    from agent_utilities.core.config import config
    from agent_utilities.server.app import build_agent_app

    bind_host = host or str(getattr(config, "host", None) or "127.0.0.1")
    bind_port = int(port or getattr(config, "port", None) or 8000)

    # Import here, not at module import: graph-os must start normally when the
    # `ag-ui` extra is absent, and only a deployment that asked for the WebUI
    # should ever pay this import.
    from agent_webui.server import create_agent_web_app  # noqa: F401

    app = build_agent_app(
        enable_web_ui=True,
        host=bind_host,
        port=bind_port,
    )

    # Uvicorn access records include the raw query string, which can carry user
    # searches and graph symbols — same redaction posture as agent-webui's own
    # standalone entrypoint.
    server = uvicorn.Server(
        uvicorn.Config(app, host=bind_host, port=bind_port, access_log=False)
    )
    # Signal handlers may only be installed from the main thread, and the
    # supervisor owns shutdown through ``stop_event`` regardless.
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

    async def _serve() -> None:
        task = asyncio.ensure_future(server.serve())
        try:
            while not stop_event.is_set() and not task.done():
                await asyncio.sleep(_STOP_POLL_SECONDS)
        finally:
            server.should_exit = True
            await task

    logger.info(
        "agent-webui co-service serving in-process on %s:%s", bind_host, bind_port
    )
    asyncio.run(_serve())
