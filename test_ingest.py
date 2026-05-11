import asyncio
from pathlib import Path


async def test_kg_ingest():
    # Write a test document
    test_file = Path("test_doc.md")
    test_file.write_text(
        "# Test Document\n\nThis is a generic document for vector ingestion testing."
    )

    from agent_utilities.mcp.kg_server import _build_server

    args, mcp, _ = _build_server()

    # Run the tool via its wrapper
    res = await mcp.call_tool("kg_ingest", {"target_path": str(test_file.absolute())})
    print("kg_ingest result:", res)


if __name__ == "__main__":
    asyncio.run(test_kg_ingest())
