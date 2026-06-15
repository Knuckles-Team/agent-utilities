"""Knowledge Distillation Engine.

CONCEPT:KG-2.2 — Knowledge Distillation Engine

Provides IdeaBlock-compatible structured knowledge ingestion, semantic
deduplication via LSH + cosine similarity, and iterative LLM-driven merging —
all natively integrated with the Knowledge Graph, OWL ontology, and tiered
memory systems.

Derived from the Blockify Agentic Data Optimization architecture but
implemented as pure Pydantic models using our existing LLM and embedding
infrastructure.

Public API::

    from agent_utilities.knowledge_graph.distillation import (
        DistillationEngine,
        KnowledgeDeduplicator,
        LSHIndex,
    )
"""

from .deduplicator import KnowledgeDeduplicator
from .distillation_engine import DistillationEngine
from .lsh_index import LSHIndex
from .physical_distiller import PhysicalDistillationEngine
from .skill_graph_distiller import SkillGraphDistiller
from .skill_graph_importer import import_skill_graph_pack
from .skill_synthesizer import (
    ConnectorSkillDistiller,
    DistillReport,
    SkillCandidate,
)
from .trading_curator import (
    build_knowledge_nodes,
    classify_trading_concept,
    organize_trading_knowledge,
)

__all__ = [
    "ConnectorSkillDistiller",
    "DistillReport",
    "DistillationEngine",
    "KnowledgeDeduplicator",
    "LSHIndex",
    "PhysicalDistillationEngine",
    "SkillCandidate",
    "SkillGraphDistiller",
    "build_knowledge_nodes",
    "classify_trading_concept",
    "import_skill_graph_pack",
    "organize_trading_knowledge",
]
