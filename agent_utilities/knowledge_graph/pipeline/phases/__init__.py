from .centrality import centrality_phase
from .communities import communities_phase
from .community_reports import community_reports_phase
from .embedding import embedding_phase
from .external_graphs import external_graphs_phase
from .knowledge_base import knowledge_base_phase
from .memory import memory_phase
from .observability import decision_evolution_phase, experience_distillation_phase
from .parse import parse_phase
from .registry import registry_phase
from .scan import scan_phase
from .shacl_gate import shacl_gate_phase
from .sync import sync_phase
from .validate import validate_phase

# RF-031 (2026-09-22): resolve_phase/mro_phase/reference_phase deleted -- they
# wrote a self-contained SYMBOL/calls_raw/depends_on_raw/inherits_from subgraph
# nothing outside this closed loop (their own tests) ever read. See
# parse.py's module docstring for the full evidence.
PHASES = [
    memory_phase,
    scan_phase,
    parse_phase,
    registry_phase,
    communities_phase,
    community_reports_phase,
    centrality_phase,
    embedding_phase,
    shacl_gate_phase,
    sync_phase,
    external_graphs_phase,
    knowledge_base_phase,
    validate_phase,
    experience_distillation_phase,
    decision_evolution_phase,
]

# Bulk-ingest "structural" profile (CONCEPT:AU-KG.query.vendor-agnostic-traversal — throughput). Per-artifact runs
# extract only the local symbol graph (dependency-closed: memory→scan→parse→embedding).
# The expensive GLOBAL phases — registry (loads the whole discovery registry),
# communities, centrality, and sync — are deferred to a SINGLE
# end-of-run enrichment pass over the full accumulated graph (and offloaded to the Rust
# epistemic-graph compute layer where possible) instead of being re-run per artifact.
STRUCTURAL_PHASES = [
    memory_phase,
    scan_phase,
    parse_phase,
    embedding_phase,
]


def select_phases(profile: str | None) -> list:
    """Return the phase list for an ingest profile ('structural' | 'full'/None)."""
    if (profile or "").strip().lower() == "structural":
        return STRUCTURAL_PHASES
    return PHASES
