---
name: graph-query-and-explanation
description: >-
  Answer bounded read-only questions over Graph-OS with grounded evidence. Use
  for natural-language or multi-step data questions, Cypher or UQL reads, local
  or federated semantic search, code navigation, document trees, tables, metrics,
  geospatial lookups, epistemic answers, or cited explanations. For a
  hypothesis-driven multi-source study, mining, learning, or causal evaluation,
  use graph-research-and-analysis.
---

# Graph query and explanation

Turn a question into the smallest safe read plan, execute it, and return an
answer that separates evidence from inference.

## Choose the route

1. Use `graph_ask` for an ordinary natural-language question.
2. Use `ask_data` for a bounded multi-step data question that needs query
   planning, execution, and correction.
3. Use `graph_search` for local discovery and `graph_federated_search` across
   registered external graph references; add `graph_search_synthesis` only when
   the user wants a synthesized answer over several results.
4. Use `graph_query` for an explicit read-only Cypher or UQL query. Use
   `nl_query` when the request needs query planning but no query was supplied.
5. Use `graph_code` or `graph_code_nav` for symbols, callers, definitions, and
   impact. Use `graph_document_tree` for document structure.
6. Use `graph_analyze`, `graph_explain`, `graph_evaluate`, or `graph_epistemic`
   when the answer needs diagnosis, confidence, disagreement, or provenance.
7. Use `graph_table`, `graph_promql`, or `graph_gis` for tabular, metric, or
   spatial reads.

If the task spans several routes or needs independent verification, delegate
through `graph_agents` and require a final evidence synthesis. Keep a
single bounded lookup direct.

## Workflow

### 1. Frame the question

- State the target, scope, time boundary, and requested output.
- Identify whether the user supplied a query or expects natural-language
  planning.
- Treat ambiguous entity names as search terms before using them as identifiers.

### 2. Retrieve narrowly

- Start with the most selective operation and a bounded result count.
- Keep reads tenant- and graph-scoped when that context is available.
- Request only fields needed for the answer.
- Do not turn a read request into a write, ingestion, or workflow execution.

Example direct plan:

```text
graph_search(query="sample service dependency", mode="hybrid", top_k=8)
graph_query(cypher="MATCH (s:Service {id: $id})-[:DEPENDS_ON]->(d) RETURN d.id", params='{"id":"sample-service"}')
```

### 3. Cross-check important claims

- Confirm a semantic match with a structural query when feasible.
- Use `graph_epistemic` or `graph_evaluate` when sources conflict.
- Preserve returned citations, node identifiers, confidence, and time context.
- Label any conclusion not directly returned by a source as an inference.

### 4. Report

Return:

1. the concise answer;
2. the supporting evidence or citations;
3. uncertainty, conflicts, and material gaps;
4. the query or retrieval route used when reproducibility matters.

Use an economy model for bounded retrieval, filtering, and formatting. Escalate
only when evidence conflicts, causal judgment is requested, or synthesis spans
multiple domains.

## Guardrails

- Keep Cypher and UQL read-only.
- Parameterize identifiers and values; never interpolate untrusted text into a
  query.
- Do not claim that an empty result proves absence outside the queried scope.
- Do not expose hidden credentials, tokens, or unrelated tenant data.
- Stop and ask for scope when two interpretations would produce materially
  different answers.
