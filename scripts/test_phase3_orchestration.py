#!/usr/bin/env python3
"""
Test script for Phase 3 of the Graph Orchestration Validation.
This script demonstrates:
1. Running dynamic agent execution with specialized skills (c4-architecture).
2. Running concurrent/parallel multi-agent execution across different MCP domains (scholarx and systems-manager).
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
logger = logging.getLogger("test_phase3_orchestration")

WORKSPACE_DIR = Path("/home/apps/workspace")
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "docs" / "examples"


async def main():
    # Setup LLM configs
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_BASE_URL", "http://vllm.arpa/v1")
    os.environ.setdefault("LITE_LLM_MODEL_ID", "qwen/qwen3.5-9b")

    # Required MCP tools env vars
    os.environ.setdefault("DOCKER_HOST", "unix:///var/run/docker.sock")
    os.environ.setdefault("REPOSITORY_MANAGER_WORKSPACE", str(WORKSPACE_DIR))

    # Ensure docs/architecture directory exists for C4 skill output
    architecture_dir = Path(__file__).resolve().parents[1] / "docs" / "architecture"
    architecture_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured architecture directory exists at: {architecture_dir}")

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

    # Ingest the target toolsets from the configuration file
    mcp_config_path = EXAMPLES_DIR / "example_mcp_config.json"
    logger.info(f"Ingesting MCP Config: {mcp_config_path}")
    toolkit_result = await engine.ingest_agent_toolkit([str(mcp_config_path)])
    logger.info(f"Toolkit Ingestion Result: {toolkit_result}")

    # Define Phase 3 Test Cases
    test_cases = [
        {
            "name": "Test Case 1: Workspace Analysis & C4 Architecture Diagram Generation",
            "agent": "general",
            "task": (
                "Can you analyze the current workspace using the repository-manager tools, "
                "and then create an architectural C4 diagram of it using the `c4-architecture` skill?"
            ),
        },
        {
            "name": "Test Case 2: Multi-Agent Parallel Research & System Diagnostic",
            "agent": "general",
            "task": (
                "Can you fetch and summarize academic categories using `scholarx-mcp` tools, "
                "while concurrently retrieving host CPU/memory stats using `systems-manager`?"
            ),
        },
    ]

    for tc in test_cases:
        logger.info("\n========================================================")
        logger.info(f"Starting {tc['name']}")
        logger.info("========================================================")
        try:
            result = await run_agent(
                agent_name=tc["agent"], task=tc["task"], max_steps=20, engine=engine
            )
            logger.info(f"\n[SUCCESS] Execution result for {tc['name']}:")
            logger.info(result)
        except Exception as e:
            logger.error(
                f"\n[FAILURE] Execution failed for {tc['name']}: {e}", exc_info=True
            )
        logger.info("=" * 56)


if __name__ == "__main__":
    asyncio.run(main())
