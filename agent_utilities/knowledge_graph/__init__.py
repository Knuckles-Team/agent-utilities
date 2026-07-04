#!/usr/bin/python
"""Knowledge Graph Subpackage.

This subpackage implements Graph Intelligence for the agent OS,
combining agent registries, tools, codebase structure, and long-term memory.
"""


def __getattr__(name: str):
    from typing import Any

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

        mapping_models: dict[str, Any] = {
            "CodeNode": cn,
            "MemoryNode": mn,
            "PipelineConfig": pc,
            "RegistryGraphMetadata": rgm,
        }
        return mapping_models[name]

    if name == "IntelligenceGraphEngine":
        from .core.engine import (
            IntelligenceGraphEngine as ige,
        )

        return ige

    if name == "tenant_graph_name":
        # Tenant → named graph naming discipline (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw): the one
        # helper facade/backends use so tenant-scoped operations land on
        # per-tenant named graphs (and therefore one HRW shard) consistently.
        from .core.shard_topology import tenant_graph_name as tgn

        return tgn

    if name in ("IntelligencePipeline", "RegistryPipeline"):
        from .pipeline import (
            IntelligencePipeline as ip,
        )
        from .pipeline import (
            RegistryPipeline as rp,
        )

        mapping_pipelines: dict[str, Any] = {
            "IntelligencePipeline": ip,
            "RegistryPipeline": rp,
        }
        return mapping_pipelines[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineConfig",
    "tenant_graph_name",
    "RegistryGraphMetadata",
    "MemoryNode",
    "CodeNode",
    "IntelligencePipeline",
    "RegistryPipeline",
    "IntelligenceGraphEngine",
]
