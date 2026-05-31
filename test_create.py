import uuid
import asyncio
from epistemic_graph.client import EpistemicGraphClient, SyncEpistemicGraphClient

graph_name = "test_" + uuid.uuid4().hex[:12]
client = SyncEpistemicGraphClient.connect(graph_name=graph_name)
try:
    client.tenants.create(graph_name)
    print("Created!")
except Exception as e:
    print("Exception during create:", repr(e))

try:
    client.nodes.add("test_node", {"type": "test"})
    print("Added node!")
except Exception as e:
    print("Exception during add:", repr(e))
