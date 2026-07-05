---
name: kg-document-tree
description: >-
  Vectorless reasoning-tree retrieval over a single document's section tree (PageIndex-style
  map-then-fetch). Build the per-document table-of-contents tree, view the text-free structure,
  fetch cited char/page ranges, or navigate the tree by relevance. Use for long single
  documents — "find the section about…", "give me the structure of this doc", "fetch that
  range", "which section covers…" — where similar != relevant and an embedder's recall
  ceiling hurts. Complements kg-search's vector/community retrieval, does not replace it.
license: MIT
tags: [graph-os, retrieval, document]
tier: core
wraps: [graph_document_tree]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-document-tree

Fronts the **`graph_document_tree`** verb — reasoning-tree (vectorless) retrieval over a
document's section tree (CONCEPT:AU-KG.retrieval.section-tree /
CONCEPT:AU-KG.retrieval.tree-navigation; distills PageIndex). Where `kg-search` is
similarity-first (overlap chunks → ANN/BM25 + rerank, bounded by the embedder's recall
ceiling), this walks a per-document table-of-contents tree and returns **cited
`start..end` char (and page) ranges** — superior for long single documents.

## Actions
- **`build`** — build (and optionally persist) the section tree from inline `text` or a
  stored `document_id`. `thin` collapses tiny sections; `summarize` computes a text-free
  per-node map. Returns the structure.
- **`structure`** — the text-free table-of-contents map for a `document_id`
  (= PageIndex `get_document_structure`). Read this first — it is token-cheap.
- **`content`** — fetch section bodies for cited `ranges` (e.g. `"96..208,300..420"`)
  (= PageIndex `get_page_content`).
- **`retrieve`** — walk the tree by relevance for a `query` and return sections with cited
  ranges. `use_llm=true` tries LLM tree navigation before the lexical walk.

## Invoke
- **MCP:** `load_tools(tools=["graph_document_tree"])`, then call it.
- **REST twin:** `POST /graph/document-tree` with a JSON body.

## Example
```
graph_document_tree(action="build", document_id="doc:abc", text="# Manual\n...")
graph_document_tree(action="structure", document_id="doc:abc")
graph_document_tree(action="retrieve", document_id="doc:abc",
                    query="how does billing handle refunds", top_k=3)
graph_document_tree(action="content", document_id="doc:abc", ranges="96..208")
```

Route long single-document queries here; use `kg-search` for cross-document semantic recall.
