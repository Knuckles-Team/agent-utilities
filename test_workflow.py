import asyncio

from agent_utilities.mcp.kg_server import _get_engine
from agent_utilities.workflows.runner import WorkflowRunner


async def main():
    engine = _get_engine()
    runner = WorkflowRunner(max_steps_per_agent=5)

    try:
        wf_result = await runner.execute_by_name(
            workflow_name="x_posts_to_kg_ingestion",
            engine=engine,
            task="https://x.com/i/status/2057129225593741768",
        )
        print("Result:", wf_result.to_dict())
    except Exception:
        import traceback

        traceback.print_exc()


asyncio.run(main())
