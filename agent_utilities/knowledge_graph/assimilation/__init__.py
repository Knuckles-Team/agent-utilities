#!/usr/bin/python
"""Graph-native assimilation engine (CONCEPT:KG-2.7).

The graph-compute middle of the ecosystem-evolution pipeline: turn an Evidence/
Capability Knowledge Graph of Sources + Features + Concepts into deduped,
gap-analysed, synergy-bundled, ranked features — using graph operations
(embedding similarity, community detection, PageRank) rather than re-reading the
corpus with an LLM. See `.specify/specs/ecosystem-evolution/`.

Concept: assimilation
"""

from .breadth_ingest import (
    BreadthReport,
    ProjectManifest,
    classify_project,
    discover_concepts,
    discover_projects,
    organize_libraries,
    run_breadth_ingest,
)
from .concept_matcher import ConceptMatcher, FeatureMatch, Match, MatchReport
from .dedup import DedupReport, dedup_features
from .gap_analysis import GapReport, is_closed, open_features
from .ingest import (
    IngestReport,
    canonical_source_id,
    content_fingerprint,
    enrich_concepts,
    ingest_concepts,
    ingest_conversations,
    ingest_documents,
)
from .ledger import (
    CloseOutReport,
    close_out,
    ledger_state,
    promote_feature_ledger,
    record_feature,
    set_status,
)
from .pilot import PilotReport, run_pilot, summarize
from .plan_synthesis import (
    PlanProposal,
    hydrate_feature,
    synthesize_plan_for_feature,
    synthesize_plans,
)
from .synergy import (
    RankedFeature,
    SynergyBundle,
    SynergyReport,
    rank_features,
    synergy_bundles,
)

__all__ = [
    "DedupReport",
    "dedup_features",
    "GapReport",
    "open_features",
    "is_closed",
    "SynergyBundle",
    "SynergyReport",
    "RankedFeature",
    "synergy_bundles",
    "rank_features",
    "CloseOutReport",
    "record_feature",
    "set_status",
    "close_out",
    "promote_feature_ledger",
    "ledger_state",
    "IngestReport",
    "canonical_source_id",
    "content_fingerprint",
    "ingest_documents",
    "ingest_conversations",
    "ingest_concepts",
    "enrich_concepts",
    "ConceptMatcher",
    "Match",
    "FeatureMatch",
    "MatchReport",
    "PlanProposal",
    "hydrate_feature",
    "synthesize_plan_for_feature",
    "synthesize_plans",
    "ProjectManifest",
    "BreadthReport",
    "classify_project",
    "discover_projects",
    "discover_concepts",
    "organize_libraries",
    "run_breadth_ingest",
    "PilotReport",
    "run_pilot",
    "summarize",
]
