import uuid
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

try:
    g = GraphComputeEngine(graph_name="test_foo")
    print("Graph created successfully")
    g.add_node("test", {"foo": "bar"})
    print("Node added successfully")
except Exception as e:
    print(f"Exception: {e}")
