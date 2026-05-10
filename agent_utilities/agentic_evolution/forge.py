"""Agentic Evolution Forge."""

from ..harness.agentic_harness import AgenticHarness


class AgenticEvolutionForge:
    def __init__(self):
        self.harness = AgenticHarness(registry_path="mock")
