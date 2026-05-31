from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
import os
import json

os.environ["AGENT_UTILITIES_TESTING"] = "true"

print("Init engine")
engine = GraphComputeEngine(backend_type="epistemic_graph")

print("Add node")
engine.add_node("TEST", name="TEST_NAME")

print("List nodes")
nodes = engine._get_all_nodes()
print("Nodes:", nodes)

props = engine._client.nodes.properties("TEST")
print("Props:", type(props), props)
