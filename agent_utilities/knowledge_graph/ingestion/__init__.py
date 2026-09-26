"""Ingestion Module.

CONCEPT:AU-KG.ingest.ingestion-engine — Ingestion Engine

Single entrypoint for all data ingestion into the Knowledge Graph.
Content-typed adaptors handle codebase, document, social, SPARQL,
skill, MCP server, policy, event stream, and prompt ingestion.
"""


from epistemic_graph.ingestion.evidence_model import (
    Artifact,
    Fragment,
    FragmentKind,
)

from .change_envelope import OPERATIONS, ChangeEnvelope, Operation
from .engine import ContentType, IngestionEngine, IngestionManifest, IngestionResult
from .governed_documentation import (
    DOCUMENTATION_MAPPING_VERSION,
    DOCUMENTATION_SCHEMA_VERSION,
    DocumentationEvidence,
    DocumentationLifecycle,
    DocumentationProjectionBatch,
    DocumentationProjectionError,
    DocumentationSource,
    GovernedDocumentationProjection,
    GovernedDocumentationProjector,
    extract_documentation,
    project_markdown,
    rebuild_documentation,
)

__all__ = [
    "ContentType",
    "IngestionEngine",
    "IngestionManifest",
    "IngestionResult",
    "ChangeEnvelope",
    "Operation",
    "OPERATIONS",
    "Artifact",
    "Fragment",
    "FragmentKind",
    "DOCUMENTATION_MAPPING_VERSION",
    "DOCUMENTATION_SCHEMA_VERSION",
    "DocumentationEvidence",
    "DocumentationLifecycle",
    "DocumentationProjectionBatch",
    "DocumentationProjectionError",
    "DocumentationSource",
    "GovernedDocumentationProjection",
    "GovernedDocumentationProjector",
    "extract_documentation",
    "project_markdown",
    "rebuild_documentation",
]
