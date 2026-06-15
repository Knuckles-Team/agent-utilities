"""KG enrichment pipeline (CONCEPT:KG-2.8).

Turns raw ingestion into a deeply-typed, cross-linkable epistemic graph. Phase 1
ships the Code/Test vertical slice: the epistemic-graph Rust engine computes AST
+ test-quality metrics, this layer maps them into typed entities, classifies
"needs work", and writes through the single ``GraphBackend`` interface.

Strategy: ``.specify/design/kg-2.8-enrichment-interlinking/strategy.md``.
"""

from .cards import CapabilityCard, generate_symbol_cards, make_llm_fn
from .classify import TestThresholds, classify_test, test_needs_work
from .distill import (
    EnhancementCandidate,
    SpecDraft,
    distill_specs,
    gather_enhancement_candidates,
    what_specs_could_we_build,
    write_spec_drafts,
)
from .execute import (
    execute_agent_spec,
    execute_team_spec,
    make_capability_search,
    persist_as_runnable,
)
from .extractors.document import extract_document
from .features import cluster_features, make_community_fn, resolve_call_edges
from .models import (
    CodeEntity,
    Concept,
    Document,
    EnrichmentEdge,
    ExtractionBatch,
    ExtractionResult,
    Feature,
    GraphNode,
    TestEntity,
)
from .orchestration import (
    AgentSpec,
    PromptSpec,
    TeamSpec,
    WorkflowSpec,
    agent_to_batch,
    prompt_to_batch,
    team_to_batch,
    workflow_to_batch,
)
from .patterns import detect_patterns
from .pipeline import EnrichmentPipeline, EnrichmentSummary, make_parse_fn
from .pydantic_graph import (
    discover_pydantic_graph,
    propose_workflow_evolution,
    pydantic_graph_to_workflow,
)
from .query import (
    code_by_pattern,
    how_implemented,
    list_features,
    tests_needing_work,
)
from .materialize import (
    materialize_source,
    resolve_source_client,
)
from .registry import (
    discover_extractors,
    list_sources,
    register_extractor,
    write_batch,
)
from .semantic import (
    embed_and_store,
    find_related,
    link_concepts_to_code,
    make_embed_fn,
    make_search_fn,
)
from .synthesize import (
    evolve_prompts,
    persist_synthesis,
    synthesize_agent,
    synthesize_team,
)

__all__ = [
    "AgentSpec",
    "CapabilityCard",
    "CodeEntity",
    "Concept",
    "Document",
    "EnhancementCandidate",
    "EnrichmentEdge",
    "EnrichmentPipeline",
    "EnrichmentSummary",
    "ExtractionBatch",
    "ExtractionResult",
    "Feature",
    "GraphNode",
    "PromptSpec",
    "SpecDraft",
    "TeamSpec",
    "TestEntity",
    "TestThresholds",
    "WorkflowSpec",
    "agent_to_batch",
    "discover_extractors",
    "discover_pydantic_graph",
    "evolve_prompts",
    "execute_agent_spec",
    "execute_team_spec",
    "make_capability_search",
    "materialize_source",
    "persist_as_runnable",
    "resolve_source_client",
    "list_sources",
    "persist_synthesis",
    "prompt_to_batch",
    "propose_workflow_evolution",
    "pydantic_graph_to_workflow",
    "register_extractor",
    "synthesize_agent",
    "synthesize_team",
    "team_to_batch",
    "workflow_to_batch",
    "write_batch",
    "classify_test",
    "cluster_features",
    "code_by_pattern",
    "detect_patterns",
    "distill_specs",
    "embed_and_store",
    "extract_document",
    "find_related",
    "gather_enhancement_candidates",
    "generate_symbol_cards",
    "how_implemented",
    "link_concepts_to_code",
    "list_features",
    "make_community_fn",
    "make_embed_fn",
    "make_llm_fn",
    "make_parse_fn",
    "make_search_fn",
    "resolve_call_edges",
    "test_needs_work",
    "tests_needing_work",
    "what_specs_could_we_build",
    "write_spec_drafts",
]
