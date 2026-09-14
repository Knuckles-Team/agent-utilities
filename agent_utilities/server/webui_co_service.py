"""Serve agent-webui inside the graph-os process as a supervised co-service.

CONCEPT:AU-OS.deployment.webui-co-service — agent-webui as a graph-os co-service

Why this exists
---------------
Every piece of this was already built and only the last wire was missing:

* ``agent-utilities[ag-ui]`` already declares ``agent-webui`` as an optional
  dependency, so the package is installable alongside graph-os with no new
  distribution work.
* the gateway routers the dashboard fronts (``agent_utilities.gateway.*``) are
  already served from this process — agent-webui is the frontend facade over
  them, which is why serving it here duplicates nothing.
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

#: Port the dashboard binds inside the graph-os process. It MUST NOT be
#: ``config.port``: that is graph-os's OWN listener (the MCP transport), so
#: reusing it makes the two co-services fight for one socket and whichever
#: loses crash-loops. A dedicated variable keeps them independent.
WEB_UI_PORT_ENV = "GRAPH_OS_WEBUI_PORT"
DEFAULT_WEB_UI_PORT = 8080

#: agent-webui refuses a non-loopback listener until the raw-query logging
#: decision is explicit. Mirrors ``agent_webui.server._ACCESS_LOG_POLICY_ENV``.
ACCESS_LOG_POLICY_ENV = "AGENT_WEBUI_ACCESS_LOG_POLICY"


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
    import os

    import uvicorn

    from agent_utilities.core.config import config

    bind_host = host or str(getattr(config, "host", None) or "0.0.0.0")  # noqa: S104
    if port:
        bind_port = int(port)
    else:
        raw = str(os.environ.get(WEB_UI_PORT_ENV, "") or "").strip()
        try:
            bind_port = int(raw) if raw else DEFAULT_WEB_UI_PORT
        except ValueError:
            raise RuntimeError(
                f"{WEB_UI_PORT_ENV}={raw!r} is not an integer port"
            ) from None

    own_port = getattr(config, "port", None)
    if own_port is not None and int(own_port) == bind_port:
        # Fail loudly rather than crash-loop against graph-os's own socket.
        raise RuntimeError(
            f"the dashboard cannot bind port {bind_port}: that is graph-os's own "
            f"listener. Set {WEB_UI_PORT_ENV} to a free port."
        )

    # An embedder OWNS the ASGI server, so it must make the WebUI's raw-query
    # logging decision explicitly: a non-loopback listener refuses to start
    # without one (`_resolve_access_log_policy`). We pass `access_log=False` to
    # uvicorn below, so `disabled` is the declaration that matches what this
    # server actually does -- the same choice agent-webui's own entrypoint
    # makes. `setdefault`, so an operator may select `redacted` instead and
    # supply a redacting access logger.
    os.environ.setdefault(ACCESS_LOG_POLICY_ENV, "disabled")

    # Import here, not at module import: graph-os must start normally when the
    # `ag-ui` extra is absent, and only a deployment that asked for the WebUI
    # should ever pay this import.
    from agent_webui.api_extensions import (
        _get_engine_bounded,
        _invoke_governed_helper,
    )
    from agent_webui.orchestrator_model import build_orchestrator_model
    from agent_webui.server import create_agent_web_app

    from agent_utilities.core.contextual_model import create_context_agent
    from agent_utilities.server.webui_contact_governance import (
        contact_delivery_factory_kwargs,
    )
    from agent_utilities.server.webui_mcp_delegation import (
        webui_mcp_delegation_helpers,
    )
    from agent_utilities.server.webui_voice_delegation import (
        webui_voice_delegation_helpers,
    )

    # Assemble exactly what agent-webui's own entrypoint assembles.
    #
    # NOT `server.app.build_agent_app`: that constructs au's ENTIRE server
    # application -- skills, ontology, A2A, embedding writes -- and mounts the
    # dashboard as one part of it. Measured in the live pod, that path had not
    # finished building after 32 minutes against a contended engine (16s
    # commits), so the listener never bound and the co-service looked hung. The
    # dashboard is a frontend facade over routers that are already served; it
    # needs the orchestrator-model agent and the delegation helpers, nothing
    # more. Same measurement, this path: 11 seconds to a built app.
    agent = create_context_agent(model=build_orchestrator_model(_get_engine_bounded))
    helpers = {
        **webui_mcp_delegation_helpers(),
        **webui_voice_delegation_helpers(),
    }
    contact_kwargs = contact_delivery_factory_kwargs(
        create_agent_web_app,
        lambda operation: _invoke_governed_helper(operation, deadline=10.0),
    )
    app = create_agent_web_app(
        agent,
        workspace_helpers=helpers,
        listener_host=bind_host,
        **contact_kwargs,
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
