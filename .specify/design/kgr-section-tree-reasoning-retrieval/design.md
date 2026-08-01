# Design Document: A per-document reasoning tree, navigated by title/summary relevance — "vectorless" retrieval for the queries an embedder's recall ceiling hurts most

CONCEPT:AU-KG.retrieval.section-tree ·
CONCEPT:AU-KG.retrieval.tree-navigation

> `agent_utilities/knowledge_graph/ontology/document_processing.py:90-97`
> (the graph schema), `agent_utilities/knowledge_graph/retrieval/
> hierarchical_document_retriever.py` (the navigator).

## Decision — materialize the document's own heading structure as graph nodes, then walk it by relevance instead of flattening to similarity-ranked chunks

`CONCEPT:AU-KG.retrieval.section-tree`

`document_processing.py:90-97` materializes one `Section` node per
heading/TOC entry, linked to its parent `Document` (`HAS_SECTION`/
`SECTION_OF`) and to nested child sections (`HAS_SUBSECTION`), so the tree can
be reconstructed from the graph rather than re-parsed from the source text on
every read. This is the schema half of PageIndex's "vectorless" reasoning-tree
RAG (`pageindex/retrieve.py`'s `get_document_structure`); the retrieval half
is the pointer below.

**The rejected alternative**, stated from the retrieval side
(`hierarchical_document_retriever.py:9-19`): the existing
`HybridRetriever` is similarity-first — overlap chunks → ANN/BM25 + rerank —
"bounded by the embedder's recall ceiling." For a long single document (a
manual, a contract, a spec) where "similar ≠ relevant," flat chunk-similarity
retrieval is exactly the case an embedder's recall ceiling hurts most: the
chunk containing the answer may not be lexically or semantically *similar* to
the query even though it is structurally the right place to look (e.g. "what
does section 4.2 say" or a value buried three headings deep under an
unrelated-sounding title). Persisting the section tree as graph nodes is what
makes navigating it — rather than re-deriving it from raw text per query —
cheap enough to use routinely.

### Pointer — `CONCEPT:AU-KG.retrieval.tree-navigation`

`hierarchical_document_retriever.py:6-64`. The retriever that walks the
persisted section tree: it scores nodes on their *title + summary* (the
text-free map, not the section's full body), prunes irrelevant subtrees with
a beam walk, and returns surviving sections with their cited
`char_start..char_end` (and `page` for PDFs) ranges. Two navigators are
offered, with an explicit fallback order: the **lexical beam walk** (default,
`LexicalRelevanceScorer`, no model/network, always available) and **LLM
navigation** (opt-in — hand the text-free structure to an `llm_fn` and let it
pick relevant `node_id`s, mirroring PageIndex's `query_agent`), which falls
back to the lexical walk on absence or error. The docstring is explicit that
this **complements, not replaces** vector/community retrieval: "route long
single-document queries... here" — a routing decision (`STRATEGY_NAME =
"hierarchical_document"`), not a wholesale replacement of the similarity-first
path for the (majority) case where similarity ranking works fine.

## Risk Assessment

- **Blast Radius**: `document_processing.py` (`rebuild_section_tree`,
  `SECTION_NODE_TYPE`), `hierarchical_document_retriever.py`,
  `reasoning_reranker.py` (shares `LexicalRelevanceScorer`).
- **Backward Compatible**: Yes — a new retrieval strategy alongside `hybrid`/
  `semantic`, selected explicitly, not a default-path change.
- **Known weak point**: the beam walk scores on title + summary only — a
  section whose *title* is uninformative relative to its actual content
  (common in poorly-structured source documents) can be pruned before the
  walk ever reads its body, which the lexical-similarity path over raw chunks
  would not have missed.
