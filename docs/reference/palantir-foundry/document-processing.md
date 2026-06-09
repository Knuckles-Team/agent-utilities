# Ontology Document Processing

> Source: <https://www.palantir.com/docs/foundry/ontology/document-processing/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.

## Pipeline stages

1. **Media import & reference extraction** — PDFs uploaded as media sets; a "Get Media References" board retrieves references from media-set datasets, producing structured media-reference objects.
2. **Text extraction** — a "Text Extraction" board converts document content to raw text, enabling semantic operations on previously inaccessible unstructured content.

## Chunking strategy

Breaking larger passages into semantically distinct units addresses two constraints: embedding-model max-token limits, and the fact that smaller segments are more semantically distinct during retrieval.

### Configuration (Chunk String board)
- **Target size** — configurable character threshold (e.g. ~256 chars).
- **Separators** — multi-priority hierarchy for intelligent split points.
- **Overlap** — configurable context preservation between adjacent chunks (e.g. 20 chars).

### Non-code chunking pipeline
| Stage | Board | Transformation |
|---|---|---|
| 1 | Chunk String | Text array with overlap |
| 2 | Explode Array with Position | Array elements → individual rows + position metadata |
| 3 | Field Extraction | Isolate position + chunk values into columns |
| 4 | String Concatenation | Unique chunk ids (`original_id` + position) |
| 5 | Column Pruning | Remove intermediate columns |

### Output structure
Each chunk row carries `object_id` (source-document ref for linking), `chunk_id`, `chunk` (text), and `embedding` (vector, generated downstream).

## Integration with semantic search

- **Embedding generation** — chunked text vectors created via the semantic-search pipeline.
- **Object materialization** — chunks become Ontology objects with bidirectional link types to the source document, embedding + text properties, and full-text/semantic indexing.
- **Presentation** — search results surface alongside the rendered source PDF for source-of-truth cross-validation.

## Advanced processing

For sophisticated chunking (sliding windows, semantic boundaries, hierarchical chunking) use Python/TypeScript functions in code repositories rather than Pipeline Builder boards.

## Design principle

Documents are treated as **decomposable knowledge units** where chunk granularity directly impacts downstream retrieval precision/recall.
