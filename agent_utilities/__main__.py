#!/usr/bin/python

import os
import logging

from . import (
    build_system_prompt_from_workspace,
    create_agent_parser,
    create_agent_server,
    initialize_workspace,
    load_identity,
)

logger = logging.getLogger(__name__)

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


def agent_server():
    parser = create_agent_parser()
    args = parser.parse_args()

    setup_logging(args.debug)

    if args.workspace:
        from . import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = args.workspace
        logging.info(f"Workspace override set to: {args.workspace}")

    initialize_workspace()
    meta = load_identity()

    agent_name = os.getenv("DEFAULT_AGENT_NAME", meta.get("name", "AI Agent"))
    system_prompt = os.getenv(
        "AGENT_SYSTEM_PROMPT",
        meta.get("content") or build_system_prompt_from_workspace(),
    )

    logger.info(f"{agent_name} v{__version__}")
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
    )


if __name__ == "__main__":
    agent_server()
