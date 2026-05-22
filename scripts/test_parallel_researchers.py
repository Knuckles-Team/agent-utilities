#!/usr/bin/env python3
"""
Test script for Parallel Multi-Agent Research using Pydantic-Graph orchestration.

This script tests:
1. Spawning a parent orchestrator agent with the `graph-os` MCP server.
2. The parent orchestrator autonomously uses `graph_orchestrate` to dispatch 2 parallel sub-agents.
3. The sub-agents share identical toolsets (`scholarx`, `graph-os`) but focus on distinct topics (Topic A vs. Topic B).
4. Sub-agents use `graph_ingest` or `graph_write` to autonomously mutate the Knowledge Graph with their findings.
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
logger = logging.getLogger("test_parallel_researchers")

WORKSPACE_DIR = Path("/home/apps/workspace")
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "docs" / "examples"


async def main():
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_BASE_URL", "http://10.0.0.18:1234/v1")
    os.environ.setdefault("LITE_LLM_MODEL_ID", "qwen/qwen3.5-9b")

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

    # Ingest the MCP toolkits so the engine knows about graph-os and scholarx
    mcp_config_path = EXAMPLES_DIR / "example_mcp_config.json"
    logger.info(f"Ingesting MCP Config: {mcp_config_path}")
    await engine.ingest_agent_toolkit([str(mcp_config_path)])

    logger.info("=== Starting Parallel Researcher Orchestration Test ===")

    task_prompt = (
        "I need you to use the `graph_orchestrate` tool to spawn TWO parallel sub-agents. "
        "They both must be given the `scholarx` and `graph-os` toolsets. "
        "Sub-agent 1 should research 'Quantum Computing in Finance' using ScholarX, and then use graph_write to store its findings. "
        "Sub-agent 2 should research 'Multi-Agent Orchestration' using ScholarX, and then use graph_write to store its findings. "
        "Ensure you dispatch them correctly and await their consensus or completion."
    )

    logger.info(f"Dispatching parent agent with task: {task_prompt}")

    # We run the parent agent using the "graph-os" capability identity so it has access to graph_orchestrate natively.
    # Note: run_agent natively resolves the agent name to the toolsets mapped in the Knowledge Graph.
    result = await run_agent(agent_name="graph-os", task=task_prompt, max_steps=10)

    logger.info("=== Parallel Researcher Execution Result ===")
    logger.info(f"\n{result}\n")
    logger.info("==========================================")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception:
        logger.exception("Fatal error running parallel researchers test")
        sys.exit(1)
