import logging
import asyncio
import httpx
from pathlib import Path
from datetime import UTC, datetime

# We use a testserver for direct ASGI interaction
GATEWAY_URL = "http://testserver"

async def ingest_target(target: Path, client: httpx.AsyncClient):
    print(f"Requesting ingestion for: {target}")
    try:
        response = await client.post(
            f"{GATEWAY_URL}/api/graph/ingest",
            json={"target_path": str(target)},
            timeout=300.0  # Allow 5 minutes per ingestion target
        )
        if response.status_code == 200:
            data = response.json()
            if "nodes" in data:
                print(f"  -> Added {data.get('nodes')} nodes, {data.get('edges')} edges")
            elif "chunks" in data:
                print(f"  -> Added {data.get('chunks')} document chunks")
            else:
                print(f"  -> Success: {data.get('result')}")
        else:
            print(f"  -> Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"  -> Request failed: {e}")

async def get_stats(client: httpx.AsyncClient):
    try:
        response = await client.post(
            f"{GATEWAY_URL}/api/graph/query",
            json={"cypher": "MATCH (n) RETURN n.type AS type, count(*) AS count ORDER BY count DESC LIMIT 50"}
        )
        if response.status_code == 200:
            return response.json().get("result", [])
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Request failed: {e}"

async def main():
    import logging
    logging.getLogger("agent_utilities").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Read the paths to ingest
    # We will expand discovery here to cover all MCP servers, skills, and tools
    workspace = Path("/home/apps/workspace")
    agent_packages = workspace / "agent-packages"
    
    # 1. MCP Servers (agents)
    agents_dir = agent_packages / "agents"
    agent_paths = [p for p in agents_dir.iterdir() if p.is_dir() and not p.name.startswith(".")] if agents_dir.exists() else []
    
    # 2. Universal Skills
    skills_dir = agent_packages / "skills" / "universal-skills" / "universal_skills"
    skill_paths = [p for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith(".")] if skills_dir.exists() else []
    
    # 3. Read from scratch/paths.txt as well
    paths_txt = workspace / "scratch" / "paths.txt"
    txt_paths = []
    if paths_txt.exists():
        with open(paths_txt) as f:
            txt_paths = [Path(line.strip()) for line in f if line.strip()]

    # Combine all targets, filtering out non-existent ones
    all_targets = set(agent_paths + skill_paths + txt_paths)
    valid_targets = [t for t in all_targets if t.exists()]
    
    print(f"Discovered {len(valid_targets)} valid targets for ingestion.")

    from agent_utilities.server.app import build_agent_app
    from httpx import ASGITransport
    app = build_agent_app()
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # Get Initial Stats
        print("Initial Stats:", await get_stats(client))

        for target in valid_targets:
            await ingest_target(target, client)

        # Get Final Stats
        print("Final Stats:", await get_stats(client))

if __name__ == "__main__":
    asyncio.run(main())
