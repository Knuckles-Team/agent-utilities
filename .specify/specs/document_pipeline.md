# Document Pipeline Specification

## Overview
The Document Pipeline provides a tightly-wired system for managing documents across three storage layers: Document DB, Vector DB (via vector-mcp), and Knowledge Graph. It uses a unified ID system to track synchronization status across all systems.

## User Stories
- **As a System Agent**, I want to ingest documents so that they are chunked, embedded, and mapped in the knowledge graph simultaneously.
- **As a System Agent**, I want to update documents so that changes cascade across the document store, vector store, and knowledge graph without inconsistencies.
- **As an Administrator**, I want to perform soft/hard deletions so that stale or retracted information is properly expunged from all storage layers.

## Functional Requirements
- **FR-001 (Unified ID System)**: The system MUST generate and track unified IDs in the format `doc_{uuid}` for documents, `doc_{uuid}_chunk_{index}` for chunks, and `doc_{uuid}_entity_{type}_{index}` for extracted entities.
- **FR-002 (Ingestion Pipeline)**: The pipeline MUST support atomic insertion into Document DB, Vector DB, and Knowledge Graph, with rollback mechanisms on failure.
- **FR-003 (Update Pipeline)**: The pipeline MUST regenerate embeddings when content changes and cascade updates to vector embeddings and knowledge graph.
- **FR-004 (Deletion Pipeline)**: The pipeline MUST support both soft and hard deletes, cascading deletions across all storage layers.
- **FR-005 (Cleanup Manager)**: The cleanup manager MUST support configurable retention policies and scheduled automated cleanups for soft-deleted documents.

## Success Criteria
- **Atomic Operations**: Ingestion/Update failures must result in zero orphaned records in any downstream DB.
- **Zero Drift**: Synchronization status in the `UnifiedIDRegistry` must match the actual storage layer state exactly.
- **Database Agnosticism**: Operations must work consistently across SQLite, PostgreSQL, and MongoDB document backends.

## Edge Cases
- Document fails embedding generation (Vector DB offline).
- Document DB runs out of space during insertion.
- Soft-deleted document is queried before cleanup process runs.

## Data Model (Draft)
- `UnifiedIDRegistry` (Entity)
- `DocumentIngestionPipeline` (ProcessFlow)
- `DocumentStorageBackend` (Abstraction)
