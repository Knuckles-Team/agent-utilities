#!/usr/bin/env python3
"""
Test script for Native Skill Execution using Pydantic-Graph orchestration.

This script tests:
1. Spawning an agent mapped directly to a native agent-skill (`agent-utilities-evolution`).
2. Ensuring 1:1 traceability for prompt JSON blueprints and native tool distribution from `agent-utilities/tools/*`.
3. Validating that native execution paths bypass MCP wrapping and execute Python capabilities directly.
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
logger = logging.getLogger("test_native_skills")


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

    logger.info("=== Starting Native Skill Execution Test ===")

    task_prompt = (
        "I need you to run a wiring sweep over the current ecosystem using your native tools. "
        "Just do a high-level traceability check and report back. "
        "This is an evolution audit to ensure concept mapping is intact."
    )

    logger.info(f"Dispatching native skill agent with task: {task_prompt}")

    # We run the agent using the native "agent-utilities-evolution" capability identity.
    # The orchestration layer should detect it as a skill/template, load the prompt.json,
    # and bind the python tools from agent_utilities/tools/ without MCP.
    result = await run_agent(
        agent_name="agent-utilities-evolution", task=task_prompt, max_steps=5
    )

    logger.info("=== Native Skill Execution Result ===")
    logger.info(f"\n{result}\n")
    logger.info("=====================================")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.exception("Fatal error running native skills test")
        sys.exit(1)
