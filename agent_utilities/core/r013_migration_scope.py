"""Typed migration-scope manifest for AU-SEMANTIC-R013 (first slice).

Enumerates the core reasoning modules in scope for AU-SEMANTIC-R013
(reasoning and graph-analytics duplicates deleted only with proven EG
parity). This first slice (AU-SEMANTIC-R013.1) covers the named core
reasoning modules; the remaining named modules (topological, spectral,
synergy, analogy, hypergraph, blast-radius, natural-language-query,
hydration, world-model, ownership-claim, fingerprint) and the
``kg/maintenance``, ``id_management``, ``argumentation`` and ``actions``
packages are out of scope here and land as later ``AU-SEMANTIC-R013.n``
children. This manifest is the single typed source of truth those later
children and the per-method parity reports build against, rather than
re-deriving the file list from the requirement prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Repository-relative paths of the AU-SEMANTIC-R013.1 migration-scope slice.
R013_MODULE_PATHS: tuple[str, ...] = (
    "agent_utilities/knowledge_graph/core/formal_reasoning_core.py",
    "agent_utilities/knowledge_graph/core/graph_primitives.py",
    "agent_utilities/knowledge_graph/core/inference_engine.py",
    "agent_utilities/knowledge_graph/core/reasoner.py",
    "agent_utilities/knowledge_graph/core/semantic_subsumption.py",
)


@dataclass(frozen=True)
class MigrationScopeEntry:
    """One module in an AU-SEMANTIC migration requirement's scope."""

    requirement_id: str
    path: str


def validate_migration_scope(
    repo_root: Path, paths: tuple[str, ...] = R013_MODULE_PATHS
) -> None:
    """Fail loud if a listed migration-scope path is absent from the checkout.

    Raises ``ValueError`` naming every missing path so the manifest cannot
    silently drift from reality -- and so no deletion of these modules can
    be claimed against a manifest entry that no longer matches the
    checkout -- before the AU-SEMANTIC-R013 per-method parity reports and
    deletion land.
    """
    missing = [p for p in paths if not (repo_root / p).is_file()]
    if missing:
        raise ValueError(
            f"AU-SEMANTIC-R013 migration scope missing from checkout: {missing}"
        )
