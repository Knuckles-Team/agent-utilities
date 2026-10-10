"""Typed migration-scope manifest for AU-SEMANTIC-R020.

Enumerates the AU modules in scope for AU-SEMANTIC-R020 (the ontology
object model moving to EG). This is the single typed source of truth the
later R020 children (the EG-side copy, then the AU importer switch and
deletion) and any forbidden-module census build against, rather than
re-deriving the file list from the requirement prose.

Excluded by the requirement's own carve-outs: the files retained for the
AU-SEMANTIC-R021 emitter cut (``value_types.py``, ``interfaces.py``,
``manifest_compiler.py``, ``leanix_metamodel.py``), the SDK manifest files
(``connector_manifest.py``, ``connector_manifest_gate.py`` and the
``connector_manifests/`` data tree), and ``sync_conflict.py`` (moves to the
SDK instead of EG).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Repository-relative paths of the AU-SEMANTIC-R020 migration scope.
R020_MODULE_PATHS: tuple[str, ...] = (
    "agent_utilities/knowledge_graph/ontology/classification_claims.py",
    "agent_utilities/knowledge_graph/ontology/derived_properties.py",
    "agent_utilities/knowledge_graph/ontology/document_processing.py",
    "agent_utilities/knowledge_graph/ontology/edits/__init__.py",
    "agent_utilities/knowledge_graph/ontology/edits/ledger.py",
    "agent_utilities/knowledge_graph/ontology/edits/revert.py",
    "agent_utilities/knowledge_graph/ontology/edits/writeback.py",
    "agent_utilities/knowledge_graph/ontology/finance_objects.py",
    "agent_utilities/knowledge_graph/ontology/functions/__init__.py",
    "agent_utilities/knowledge_graph/ontology/functions/objects.py",
    "agent_utilities/knowledge_graph/ontology/functions/registry.py",
    "agent_utilities/knowledge_graph/ontology/functions/runtime.py",
    "agent_utilities/knowledge_graph/ontology/indexing/__init__.py",
    "agent_utilities/knowledge_graph/ontology/indexing/funnel.py",
    "agent_utilities/knowledge_graph/ontology/indexing/staleness.py",
    "agent_utilities/knowledge_graph/ontology/links.py",
    "agent_utilities/knowledge_graph/ontology/object_path.py",
    "agent_utilities/knowledge_graph/ontology/object_set.py",
    "agent_utilities/knowledge_graph/ontology/ops_causal_crosswalk.py",
    "agent_utilities/knowledge_graph/ontology/permissioning.py",
    "agent_utilities/knowledge_graph/ontology/permissioning_external_sync.py",
    "agent_utilities/knowledge_graph/ontology/property_types.py",
    "agent_utilities/knowledge_graph/ontology/repository_provenance.py",
    "agent_utilities/knowledge_graph/ontology/research_objects.py",
    "agent_utilities/knowledge_graph/ontology/schema_graph.py",
    "agent_utilities/knowledge_graph/ontology/style_lint.py",
)


@dataclass(frozen=True)
class MigrationScopeEntry:
    """One module in an AU-SEMANTIC migration requirement's scope."""

    requirement_id: str
    path: str


def validate_migration_scope(
    repo_root: Path, paths: tuple[str, ...] = R020_MODULE_PATHS
) -> None:
    """Fail loud if a listed migration-scope path is absent from the checkout.

    Raises ``ValueError`` naming every missing path so the manifest cannot
    silently drift from reality before the AU-SEMANTIC-R020 move lands.
    """
    missing = [p for p in paths if not (repo_root / p).is_file()]
    if missing:
        raise ValueError(
            f"AU-SEMANTIC-R020 migration scope missing from checkout: {missing}"
        )
