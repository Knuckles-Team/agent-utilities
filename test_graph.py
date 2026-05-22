import asyncio

import networkx as nx

from agent_utilities.core.config import config
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.workflows.runner import WorkflowRunner

# Filter out gemma-4-e2b because it fails to load on LM Studio
if config.chat_models:
    config.chat_models = [m for m in config.chat_models if m.id != "google/gemma-4-e2b"]


async def main():
    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        engine = IntelligenceGraphEngine(graph=nx.MultiDiGraph())

    runner = WorkflowRunner(max_steps_per_agent=10)
    res = await runner.execute_by_name(
        workflow_name="x_posts_to_kg_ingestion",
        engine=engine,
        task="https://x.com/i/status/2057129225593741768",
    )
    print("Graph execution finished. Result:", res)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
