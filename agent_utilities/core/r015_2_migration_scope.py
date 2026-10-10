"""Typed migration-scope manifest for AU-SEMANTIC-R015.2 (first slice).

Enumerates the ``kg/core/source_sync.py`` module and its named adapters that
are removed from AU once ``resolve_workflow_derivation_client()`` is wired to
EG's shipped ``workflow_derivation`` op (tracked by ``EG-REPO-INGEST-R002``).
This first slice (AU-SEMANTIC-R015.2.1) covers the concretely located
modules; the removal of the call sites and the wiring of the real EG call are
blocked on that producer op shipping and land as the later
``AU-SEMANTIC-R015.2.n`` child. This manifest is the single typed source of
truth that later child builds against, rather than re-deriving the file list
from the requirement prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Repository-relative paths of the AU-SEMANTIC-R015.2.1 migration-scope slice.
R015_2_MODULE_PATHS: tuple[str, ...] = (
    "agent_utilities/knowledge_graph/core/source_sync.py",
    "agent_utilities/knowledge_graph/governance_import.py",
    "agent_utilities/ecosystem/ea_clients.py",
    "agent_utilities/knowledge_graph/core/connection_profiler.py",
    "agent_utilities/knowledge_graph/core/data_prep_runtime.py",
    "agent_utilities/knowledge_graph/core/source_resolver.py",
)


@dataclass(frozen=True)
class MigrationScopeEntry:
    """One module in an AU-SEMANTIC migration requirement's scope."""

    requirement_id: str
    path: str


def validate_migration_scope(
    repo_root: Path, paths: tuple[str, ...] = R015_2_MODULE_PATHS
) -> None:
    """Fail loud if a listed migration-scope path is absent from the checkout.

    Raises ``ValueError`` naming every missing path so the manifest cannot
    silently drift from reality before the AU-SEMANTIC-R015.2 cutover and
    deletion land.
    """
    missing = [p for p in paths if not (repo_root / p).is_file()]
    if missing:
        raise ValueError(
            f"AU-SEMANTIC-R015.2 migration scope missing from checkout: {missing}"
        )
