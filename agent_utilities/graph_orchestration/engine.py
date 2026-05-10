"""Graph Orchestration Core."""

from ..graph.agent_orchestrator import AgentOrchestrator


class GraphOrchestrationCore:
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
