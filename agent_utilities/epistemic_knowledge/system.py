"""Epistemic Knowledge System."""

import networkx as nx

from ..knowledge_graph.core.engine import IntelligenceGraphEngine


class EpistemicKnowledgeSystem:
    def __init__(self):
        self.graph_engine = IntelligenceGraphEngine(graph=nx.MultiDiGraph())
