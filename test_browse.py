import asyncio

from agent_utilities.tools.x_search_tool import x_search


async def main():
    try:
        # Mocking the RunContext
        class MockDeps:
            config = {}

        class MockUsage:
            def request_tokens(self, *args, **kwargs):
                pass

        from pydantic_ai import RunContext
        from pydantic_ai.usage import Usage

        # In pydantic_ai 0.0.18+, context requires model and usage
        ctx = RunContext(
            deps=MockDeps(),
            retry=0,
            tool_name="x_search",
            prompt="test",
            model=None,
            usage=Usage(),
        )

        res = await x_search(
            ctx,
            "Detailed content of the X post status ID 2057129225593741768 canonical URL https://x.com/i/status/2057129225593741768",
        )
        print("Result:", res)
    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
