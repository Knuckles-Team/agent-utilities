#!/usr/bin/env python3
"""Test script for end-to-end LM Studio orchestration and Agent Toolkit ingestion.

This tests:
1. Globbing for skill directories and ingesting them natively (ECO-4.10).
2. Dual-ingestion: Triggering codebase ingestion for the skill repos.
3. Ingesting a test mcp_config.json.
4. Verifying persistence via Cypher queries.
5. Dynamically creating and running an agent via ORCH-1.21 using LM Studio.
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
logger = logging.getLogger("test_orchestration")

WORKSPACE_DIR = Path("/home/apps/workspace")
SKILLS_DIR = WORKSPACE_DIR / "agent-packages" / "skills"
SCRATCH_DIR = Path(__file__).resolve().parents[1] / "scratch"


async def main():
    # Force use of local LM studio if not configured otherwise
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_BASE_URL", "http://vllm.arpa/v1")
    os.environ.setdefault("LITE_LLM_MODEL_ID", "qwen/qwen3.5-9b")

    logger.info("Initializing Intelligence Graph Engine...")
    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        import networkx as nx

        from agent_utilities.core.paths import ensure_dirs, kg_db_path
        from agent_utilities.knowledge_graph.backends import create_backend

        ensure_dirs()
        str(kg_db_path())
        backend = create_backend(backend_type="ladybug")
        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=backend)

    # 1. Discover Skill Directories
    logger.info("Discovering skill directories...")
    skill_paths = []
    for repo in ["universal-skills", "skill-graphs"]:
        repo_path = SKILLS_DIR / repo
        if not repo_path.exists():
            continue
        for skill_md in repo_path.rglob("SKILL.md"):
            skill_paths.append(str(skill_md.parent))

    logger.info(f"Found {len(skill_paths)} skill directories.")

    # 2. Toolkit Ingestion (ECO-4.10)
    mcp_config_path = SCRATCH_DIR / "test_mcp_config.json"
    sources = skill_paths + [str(mcp_config_path)]

    logger.info("Ingesting Agent Toolkit (skills + mcp_config)...")
    toolkit_result = await engine.ingest_agent_toolkit(sources)
    logger.info(f"Toolkit Ingestion Result: {toolkit_result}")

    # 3. Dual-Ingestion: Trigger codebase ingestion for the skill repos
    logger.info("Triggering codebase ingestion for skill repositories...")
    for repo in ["universal-skills", "skill-graphs"]:
        repo_path = SKILLS_DIR / repo
        if repo_path.exists():
            job_id = engine.submit_task(
                target_path=str(repo_path),
                is_codebase=True,
                task_type="codebase",
                provenance={"agent_id": "test_script"},
            )
            logger.info(f"Submitted codebase ingestion job {job_id} for {repo}")

    # 4. Verify Persistence via Cypher
    logger.info("Verifying persistence in KG...")
    if engine.backend:
        results = engine.backend.execute(
            "MATCH (r:CallableResource) RETURN r.name AS name, r.resource_type AS rtype LIMIT 10"
        )
        for res in results:
            logger.info(f"Found Capability: [{res['rtype']}] {res['name']}")

    # 5. Dynamic Agent Execution (ORCH-1.21)
    # We execute "test-agent" which should resolve via our fallback or dynamically
    # Since we ingested "test-server", we can invoke it by name or request a capability.
    logger.info("Executing Dynamic Agent against LM Studio...")
    # Passing an agent name that uses the skills. E.g., we can use 'test-server' as the agent name
    # to test the MCP server resolution, or a generic name.
    result = await run_agent(
        agent_name="repository-manager",
        task="Can you use the rm_workspace tool to list the available actions for the workspace?",
        max_steps=5,
        engine=engine,
    )

    logger.info("--- Execution Result ---")
    logger.info(result)
    logger.info("------------------------")


if __name__ == "__main__":
    asyncio.run(main())
