
from fastapi.testclient import TestClient
from agent_utilities.server import create_agent_server
import uvicorn
from unittest.mock import MagicMock
import os

def test_docs_exists():
    # Mock uvicorn.run to capture the app
    original_run = uvicorn.run
    uvicorn.run = MagicMock()

    # Mock environment
    os.environ["OPENAI_API_KEY"] = "sk-dummy"

    # Create the server
    try:
        create_agent_server(name="TestAgent")
    except SystemExit:
        pass

    # Get the app
    app = uvicorn.run.call_args[0][0]
    client = TestClient(app)

    # Check /openapi.json
    response = client.get("/openapi.json")
    print(f"OpenAPI Status: {response.status_code}")
    if response.status_code == 200:
        spec = response.json()
        print(f"Title: {spec.get('info', {}).get('title')}")
        print("Paths:")
        for path in spec.get("paths", {}):
            print(f"  - {path}")

    uvicorn.run = original_run

if __name__ == "__main__":
    test_docs_exists()
