# Knowledge Base (KB) Layer

The KB layer is an LLM-maintained personal wiki system built directly into the Knowledge Graph.

## Overview

The KB layer transforms raw documents (PDF, Markdown, HTML, DOCX) into a structured graph of **Articles**, **Concepts**, and **Facts**. It replaces external tools like Obsidian with a graph-native, agent-queryable alternative.

## Components

- **Parser (`parser.py`)**: Uses `LlamaIndex` and `SimpleDirectoryReader` to chunk documents and extract metadata.
- **Extractor (`extractor.py`)**: A Pydantic AI specialist that distills chunks into validated `Article` and `Fact` models.
- **Ingestion (`ingestion.py`)**: The engine that projects extracted data into the graph, handles deduplication, and maintains the `KBIndex`.

## Data Hierarchy

- `KnowledgeBase`: A named namespace (e.g., `kb:agent-utilities-docs`).
- `Article`: A compiled markdown wiki page.
- `KBConcept`: Key terminology or entities (linked via `ABOUT`).
- `KBFact`: Atomic, verifiable claims (linked via `CITES` to sources).
- `RawSource`: The original file or URL content.

## Maintenance

- **Health Checks**: Run `run_kb_health_check` to identify contradictions or gaps in the knowledge base.
- **Archiving**: Low-importance articles are automatically compressed to summary-only nodes after 180 days.
- **Sync**: Use `update_knowledge_base` for incremental re-ingestion of changed files.
