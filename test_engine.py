import subprocess
import time

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

# Start server
p = subprocess.Popen(
    [
        "cargo",
        "run",
        "--bin",
        "epistemic-graph-server",
        "--",
        "--socket-path",
        "/tmp/test_eg.sock",
    ],
    cwd="/home/apps/workspace/agent-packages/epistemic-graph",
)
time.sleep(2)

try:
    engine = GraphComputeEngine(backend_type="rust")
    engine._client._socket_path = (
        "/tmp/test_eg.sock"  # Oops, graph compute doesn't take socket path directly!
    )
    # Actually wait, graph compute uses GRAPH_SERVICE_SOCKET
except Exception as e:
    print(e)
finally:
    p.terminate()
