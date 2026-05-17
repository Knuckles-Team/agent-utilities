import json
from agent_utilities.mcp.kg_server import _get_engine

def test():
    engine = _get_engine()
    tasks = engine.list_tasks()
    print("Progress Stats:", json.dumps(tasks.get("progress_stats", {}), indent=2))
    
    # Check if there are any failed tasks
    failed = tasks.get("failed", [])
    print(f"Failed count: {len(failed)}")
    if failed:
        print("Sample failed:", json.dumps(failed[:5], indent=2))

if __name__ == "__main__":
    test()
