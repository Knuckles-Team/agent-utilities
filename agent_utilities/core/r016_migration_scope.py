"""Typed migration-scope manifest for AU-SEMANTIC-R016.

Enumerates the AU modules in scope for AU-SEMANTIC-R016 (document, session
and feed source ingestion moving to the connector SDK). This is the single
typed source of truth later R016 children (the SDK-side copy, then the AU
importer switch and deletion) and any forbidden-module census build against,
rather than re-deriving the file list from the requirement prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Repository-relative paths of the AU-SEMANTIC-R016 migration scope.
R016_MODULE_PATHS: tuple[str, ...] = (
    "agent_utilities/knowledge_graph/core/conversation_ingestion.py",
    "agent_utilities/knowledge_graph/ingestion/web_fetch.py",
    "agent_utilities/knowledge_graph/ingestion/package_install_ingest.py",
    "agent_utilities/automation/worldmodel_pipeline.py",
    "agent_utilities/automation/feed_sources.py",
    "agent_utilities/automation/file_watcher.py",
    "agent_utilities/sdd/watcher.py",
)


@dataclass(frozen=True)
class MigrationScopeEntry:
    """One module in an AU-SEMANTIC migration requirement's scope."""

    requirement_id: str
    path: str


def validate_migration_scope(
    repo_root: Path, paths: tuple[str, ...] = R016_MODULE_PATHS
) -> None:
    """Fail loud if a listed migration-scope path is absent from the checkout.

    Raises ``ValueError`` naming every missing path so the manifest cannot
    silently drift from reality before the AU-SEMANTIC-R016 move lands.
    """
    missing = [p for p in paths if not (repo_root / p).is_file()]
    if missing:
        raise ValueError(
            f"AU-SEMANTIC-R016 migration scope missing from checkout: {missing}"
        )
