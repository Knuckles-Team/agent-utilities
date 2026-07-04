#!/usr/bin/python

import logging

from agent_utilities.core.config import setting

from . import (
    build_system_prompt_from_workspace,
    create_agent_parser,
    create_agent_server,
    initialize_workspace,
    load_identity,
)

__version__ = "0.2.33"


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # CONCEPT:AU-KG.ingest.attaching-this-root-logger — self-ingest telemetry. Opt-in (default-off): ships our
    # own logs into the epistemic-graph engine obs store. Clean no-op when
    # AGENT_UTILITIES_SELF_INGEST/EPISTEMIC_GRAPH_OBS_ADDR are unset.
    try:
        from agent_utilities.observability.self_ingest import (
            install_self_ingest_logging,
        )

        install_self_ingest_logging()
    except Exception:  # noqa: BLE001 — telemetry must never block startup
        logging.getLogger(__name__).debug("self-ingest logging not installed")


def agent_server():
    """Main agent execution server.

    CONCEPT:AU-ORCH.session.unified-agent-entrypoint — Unified Agent Entrypoint
    """
    parser = create_agent_parser()
    args = parser.parse_args()

    setup_logging(args.debug)

    if args.workspace:
        from agent_utilities.core import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = args.workspace
        logging.info(f"Workspace override set to: {args.workspace}")

    initialize_workspace()
    meta = load_identity()

    agent_name = setting("DEFAULT_AGENT_NAME", meta.get("name", "AI Agent"))
    system_prompt = setting(
        "AGENT_SYSTEM_PROMPT",
        meta.get("content") or build_system_prompt_from_workspace(),
    )

    print(f"{agent_name} v{__version__}")
    if args.debug:
        logging.getLogger().debug("Debug mode enabled")

    create_agent_server(
        provider=args.provider,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        mcp_url=args.mcp_url,
        mcp_config=args.mcp_config,
        custom_skills_directory=args.custom_skills_directory,
        debug=args.debug,
        host=args.host,
        port=args.port,
        enable_web_ui=args.web,
        enable_terminal_ui=args.terminal,
        ssl_verify=not args.insecure,
        name=agent_name,
        system_prompt=system_prompt,
        enable_otel=args.otel,
        otel_endpoint=args.otel_endpoint,
        otel_headers=args.otel_headers,
        otel_public_key=args.otel_public_key,
        otel_secret_key=args.otel_secret_key,
        otel_protocol=args.otel_protocol,
        workspace=args.workspace,
        enable_web_logs=args.web_logs,
    )


if __name__ == "__main__":
    agent_server()
