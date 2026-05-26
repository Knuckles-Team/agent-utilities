#!/usr/bin/python
"""Knowledge Graph Subpackage.

This subpackage implements Unified Graph Intelligence for the agent OS,
combining agent registries, tools, codebase structure, and long-term memory.
"""


def __getattr__(name: str):
    if name in ("CodeNode", "MemoryNode", "PipelineConfig", "RegistryGraphMetadata"):
        from ..models.knowledge_graph import (
            CodeNode as cn,
        )
        from ..models.knowledge_graph import (
            MemoryNode as mn,
        )
        from ..models.knowledge_graph import (
            PipelineConfig as pc,
        )
        from ..models.knowledge_graph import (
            RegistryGraphMetadata as rgm,
        )

        mapping = {
            "CodeNode": cn,
            "MemoryNode": mn,
            "PipelineConfig": pc,
            "RegistryGraphMetadata": rgm,
        }
        return mapping[name]

    if name in ("IntelligenceGraphEngine", "RegistryGraphEngine"):
        from .core.engine import (
            IntelligenceGraphEngine as ige,
        )
        from .core.engine import (
            RegistryGraphEngine as rge,
        )

        mapping = {
            "IntelligenceGraphEngine": ige,
            "RegistryGraphEngine": rge,
        }
        return mapping[name]

    if name in ("IntelligencePipeline", "RegistryPipeline"):
        from .pipeline import (
            IntelligencePipeline as ip,
        )
        from .pipeline import (
            RegistryPipeline as rp,
        )

        mapping = {
            "IntelligencePipeline": ip,
            "RegistryPipeline": rp,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineConfig",
    "RegistryGraphMetadata",
    "MemoryNode",
    "CodeNode",
    "IntelligencePipeline",
    "RegistryPipeline",
    "IntelligenceGraphEngine",
    "RegistryGraphEngine",
]
