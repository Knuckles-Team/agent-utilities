#!/usr/bin/env python3
"""
Test script for end-to-end integration with the 7 target MCP servers.
This script demonstrates:
1. Connecting to specific MCP servers.
2. Executing dynamic agents against their lightest/safest tools.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure agent_utilities is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.orchestration.agent_runner import run_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("test_mcp_servers")

WORKSPACE_DIR = Path("/home/apps/workspace")
SCRATCH_DIR = Path(__file__).resolve().parents[1] / "scratch"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "docs" / "examples"


async def main():
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_BASE_URL", "http://10.0.0.18:1234/v1")
    os.environ.setdefault("LITE_LLM_MODEL_ID", "qwen/qwen3.5-9b")

    # Required MCP tools env vars
    os.environ.setdefault("DOCKER_HOST", "unix:///var/run/docker.sock")
    os.environ.setdefault(
        "TUNNEL_INVENTORY_PATH", str(EXAMPLES_DIR / "example_tunnel_inventory.yaml")
    )
    os.environ.setdefault("REPOSITORY_MANAGER_WORKSPACE", str(WORKSPACE_DIR))

    logger.info("Initializing Intelligence Graph Engine...")
    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        import networkx as nx

        from agent_utilities.core.paths import ensure_dirs
        from agent_utilities.knowledge_graph.backends import create_backend

        ensure_dirs()
        backend = create_backend(backend_type="ladybug")
        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=backend)

    # 1. Toolkit Ingestion for example_mcp_config.json
    mcp_config_path = EXAMPLES_DIR / "example_mcp_config.json"
    logger.info(f"Ingesting MCP Config: {mcp_config_path}")
    toolkit_result = await engine.ingest_agent_toolkit([str(mcp_config_path)])
    logger.info(f"Toolkit Ingestion Result: {toolkit_result}")

    # 2. Dynamic Agent Execution Tests
    test_cases = [
        {
            "server": "repository-manager",
            "task": "Can you use the rm_workspace tool to list the available actions for the workspace?",
        },
        {
            "server": "scholarx",
            "task": "Can you use the sx_info tool to list the categories?",
        },
        {
            "server": "container-manager-mcp",
            "task": "Can you list all docker images, list all running containers, get the logs for one of the running containers, show the volumes, and show the networks using your tools?",
        },
        {
            "server": "audio-transcriber",
            "task": "Can you describe the capabilities of the transcribe_audio tool?",
        },
        {
            "server": "tunnel-manager",
            "task": "Can you list the active tunnels from the inventory using your tools?",
        },
        {
            "server": "systems-manager",
            "task": "Can you get the system memory and CPU stats?",
        },
        {
            "server": "data-science-mcp",
            "task": "Can you describe the iris dataset using the describe_dataset tool?",
        },
    ]

    for tc in test_cases:
        logger.info(f"--- Testing Server: {tc['server']} ---")
        try:
            result = await run_agent(
                agent_name=tc["server"], task=tc["task"], max_steps=5, engine=engine
            )
            logger.info(f"Execution Result for {tc['server']}:")
            logger.info(result)
        except Exception as e:
            logger.error(f"Failed testing {tc['server']}: {e}")
        logger.info("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
