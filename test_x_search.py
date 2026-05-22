import asyncio
import logging

from agent_utilities.models import AgentDeps
from agent_utilities.tools.x_search_tool import browse_x_post

logging.basicConfig(level=logging.INFO)


async def main():
    AgentDeps()
    try:
        print("Testing browse_x_post...")
        res = await browse_x_post(
            None, url="https://x.com/i/status/2057129225593741768"
        )
        print("BROWSE RESULT:")
        print(res)
    except Exception as e:
        print("ERROR:")
        print(e)


if __name__ == "__main__":
    asyncio.run(main())
