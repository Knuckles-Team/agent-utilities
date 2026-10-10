"""Typed migration-scope manifest for AU-SEMANTIC-R018 (first slice).

Enumerates the deterministic-derivation modules in scope for
AU-SEMANTIC-R018 (deterministic derivation modules move to EG). This first
slice (AU-SEMANTIC-R018.1) covers the named ``kg/enrichment`` derivation
modules; ``kg/assimilation/**``, ``kg/etl/**``, ``kg/streams/**`` and the
time-series portion of ``kg/memory/**`` are out of scope here and land as
later ``AU-SEMANTIC-R018.n`` children. This manifest is the single typed
source of truth those later children and the forbidden-module census build
against, rather than re-deriving the file list from the requirement prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Repository-relative paths of the AU-SEMANTIC-R018.1 migration-scope slice.
R018_MODULE_PATHS: tuple[str, ...] = (
    "agent_utilities/knowledge_graph/enrichment/semantic.py",
    "agent_utilities/knowledge_graph/enrichment/relation_projection.py",
    "agent_utilities/knowledge_graph/enrichment/materialize.py",
    "agent_utilities/knowledge_graph/enrichment/git_coupling.py",
    "agent_utilities/knowledge_graph/enrichment/graph_collapse.py",
    "agent_utilities/knowledge_graph/enrichment/provenance.py",
)


@dataclass(frozen=True)
class MigrationScopeEntry:
    """One module in an AU-SEMANTIC migration requirement's scope."""

    requirement_id: str
    path: str


def validate_migration_scope(
    repo_root: Path, paths: tuple[str, ...] = R018_MODULE_PATHS
) -> None:
    """Fail loud if a listed migration-scope path is absent from the checkout.

    Raises ``ValueError`` naming every missing path so the manifest cannot
    silently drift from reality before the AU-SEMANTIC-R018 move and
    forbidden-module census land.
    """
    missing = [p for p in paths if not (repo_root / p).is_file()]
    if missing:
        raise ValueError(
            f"AU-SEMANTIC-R018 migration scope missing from checkout: {missing}"
        )
