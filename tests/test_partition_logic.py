import pytest
from agent_utilities.mcp_agent_manager import partition_tools
from agent_utilities.models import MCPToolInfo


@pytest.mark.asyncio
async def test_partition_tools_no_splitting():
    # Create 25 tools with the same tag
    tools = []
    for i in range(25):
        tools.append(
            MCPToolInfo(
                name=f"tool_{i}",
                description=f"Description {i}",
                tag="docker",
                mcp_server="portainer",
            )
        )

    partitions = await partition_tools(tools)

    # NEW BEHAVIOR: should NOT be split
    assert "docker" in partitions
    assert len(partitions["docker"]) == 25
    assert "docker_1" not in partitions
    assert "docker_2" not in partitions


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_partition_tools_no_splitting())
