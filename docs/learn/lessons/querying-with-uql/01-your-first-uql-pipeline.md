# Your First UQL Pipeline

In the [previous lesson](../ontology-model-101/01-interfaces-object-types-and-links.md)
you learned the *shape* of the graph — interfaces, object types, and links.
This lesson teaches you to **ask it questions** using UQL, the engine's native
cross-modal query language.

## What UQL is

Most graph databases give you one lens: a graph-traversal language, or a
vector search, or full-text — pick one. UQL composes graph traversal, vector
similarity, lexical (BM25) search, bi-temporal time travel, federation across
external sources, and epistemic belief/evidence reasoning **in a single
pipelined query**, all executed against one consistent snapshot. It's the text
surface for the exact same structured plan the platform's own query planner
builds internally — there's no separate "toy" query language and "real" one
underneath.

## The pipeline model

Every UQL query has the same shape:

```
source |> stage |> stage |> ...
```

`source` seeds an ordered set of candidate rows (an id, plus an optional
score). Each `|>`-separated **stage** takes that row set and produces a new
one — filter it, re-rank it, traverse from it, whatever. Think Unix pipes, but
each stage understands graph structure, vectors, and text instead of bytes.

**The one rule that never has an exception: every query starts with `MATCH`.**

```uql
MATCH (:Label) [WHERE ...]
```

`REASON`, `FOREIGN`, `AS OF`, `TEXT`, `RANK BY`, and every other stage keyword
can only appear *after* `|>`, never as the very first token. (There's a subtle
executor-level rule where a stage fed an *empty* row set effectively re-seeds
itself — which is how you'll sometimes see `REASON` or `AS OF` acting
source-like right after a `MATCH` that matched nothing on purpose — but the
text always opens with `MATCH`.)

## Your first query: MATCH + WHERE

The simplest possible query selects everything of one type:

```uql
MATCH (:Document)
```

Add a filter with `WHERE` — comparisons (`>`, `<`, `=`) joined with `AND`
(there's no `OR` or `NOT` at this stage — compose those with separate queries
or a later stage instead):

```uql
MATCH (:Event) WHERE level > 3
```

## Traversal and ranking

`TRAVERSE` walks graph edges a bounded number of hops:

```uql
MATCH (:Doc) |> TRAVERSE -[:CITES]->{1,2}
```

That's "every `Doc`, then follow `CITES` edges 1 to 2 hops out."

`RANK BY ~[...]` re-orders the current row set by cosine similarity to a
literal embedding-space query vector — **not** a fixed "axis 0 = relevance,
axis 1 = recency" scheme; the vector's dimensionality just has to match
whatever embeddings the bound semantic store actually uses:

```uql
MATCH (:Doc) |> TRAVERSE -[:CITES]->{1,2} |> RANK BY ~[0.1, 0.9, 0.0] |> LIMIT 10
```

`LIMIT <k>` is a terminal, order-respecting top-k — almost every query ends
with one.

## Worked examples

These are real, grounded examples (not simplified toys) — the same ones the
`graph-query-and-explanation` skill's UQL reference documents:

```uql
# Filter, then traverse, then rank, then cap.
MATCH (:Doc) WHERE year > 2024 |> TRAVERSE -[:CITES]->{1,2} |> RANK BY ~[1.0, 0.0, 0.0, 0.0] |> LIMIT 10

# Federated + bi-temporal + windowed + traversal, composed in one line.
MATCH (:Event) WHERE level > 3 |> FOREIGN "peer-west" |> AS OF @1700000000 |> WINDOW 1 h |> TRAVERSE -[:CAUSED]->{1,2} |> LIMIT 10

# Fuse a vector rank and a lexical search, then re-rank by distance from a node.
MATCH (:Doc) |> FUSE [RANK BY ~[1.0, 0.0]] [TEXT "graph databases"] [RERANK NODE_DISTANCE FROM "n1"] |> LIMIT 5

# OWL-reasoner-inferred membership (note the empty MATCH — REASON re-seeds it).
MATCH (:NoSuchLabel) |> REASON <http://ex/Device> |> RANK BY ~[0.2, 0.4, -0.1] |> LIMIT 5

# Epistemic: evidence for a claim, as believed at a point in time.
MATCH (:Claim) |> EVIDENCE FOR "c1" |> BELIEF AS OF @1700000000 |> LIMIT 10
```

## Where to go from here

- Run a UQL query for real via `engine_query(action='uql', text='MATCH (:Document) |> LIMIT 5')`
  — note the keyword is `text`, not `query`. An unsupported clause in your
  build returns a clean `{"error": ...}`, never a silently wrong answer.
- Don't want to hand-write UQL yet? `graph_ask`/`nl_query` translate a plain
  natural-language question into a query (Cypher, SQL, or UQL) for you, and
  return the generated query alongside the answer so you can learn from it.
- The full grammar — every clause, every `Op`, and ~20 more worked examples —
  lives in `references/uql-reference.md` in the `graph-query-and-explanation`
  skill.
- SPARQL and SHACL are separate, standards-based surfaces for RDF/OWL data —
  see the web UI's **SPARQL & SHACL** view if you'd rather query the ontology
  itself that way.

That's the whole mental model: pick a `MATCH`, pipe it through stages with
`|>`, end with `LIMIT`. Everything else in UQL is just more stage keywords
layered onto that same shape.
