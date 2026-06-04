"""Ingestion Module.

CONCEPT:KG-2.7 — Ingestion Engine

Single entrypoint for all data ingestion into the Knowledge Graph.
Content-typed adaptors handle codebase, document, social, SPARQL,
skill, MCP server, policy, event stream, and prompt ingestion.
"""

from .engine import ContentType, IngestionEngine, IngestionManifest, IngestionResult

__all__ = [
    "ContentType",
    "IngestionEngine",
    "IngestionManifest",
    "IngestionResult",
]
