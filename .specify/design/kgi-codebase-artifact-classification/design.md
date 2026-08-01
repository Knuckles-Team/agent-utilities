# Design Document: A codebase ingest classifies the WHOLE tree, not just source files

CONCEPT:AU-KG.ingest.over-same-tree-fan · CONCEPT:AU-KG.ingest.deterministic-classifier · CONCEPT:AU-KG.ingest.writes-go

> `agent_utilities/knowledge_graph/ingestion/repo_classifier.py` (the router),
> `agent_utilities/knowledge_graph/ingestion/engine.py` (`_ingest_codebase` /
> `_route_classified_artifacts`, the caller), pinned by
> `tests/unit/knowledge_graph/test_repo_classification.py`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.ingest.self-tool-surface` | graph-os ingesting its own MCP tool surface — a different discovery-mechanism decision that happens to share `ingestion/engine.py` module locality | 0.30 | KG |
| `AU-KG.ingest.big-repo-structural-split` / `AU-KG.ingest.subtask-routing-key` | splits a repo's SOURCE files across shard writers for parallel commit throughput — orthogonal axis (size, not file-type) | 0.35 | KG |
| `AU-KG.ingest.mine-git-history-files` | a second thing a codebase ingest ALSO does, but mined from git log, not the file tree | 0.25 | KG |

### Extension Analysis

- **Primary Extension Point**: `repo_classifier._classify_file`'s precedence
  ladder (`repo_classifier.py:170-212`) and the bucket lists on
  `RepoClassification`.
- **Extension Strategy**: augment — a new artifact kind is a new precedence
  rule + a new bucket + a fan-out branch in `_route_classified_artifacts`; the
  walk and the "route to the existing adaptor" shape are unchanged.
- **New Concept Required?**: No — this document is the home for the three
  markers above.

## Decision — one deterministic walk classifies EVERY file by type; non-code artifacts fan out to their EXISTING native adaptors

`CONCEPT:AU-KG.ingest.over-same-tree-fan` · `CONCEPT:AU-KG.ingest.deterministic-classifier`

A repository is not just source code. `repo_classifier.py:5-9` states the
problem directly: a real repo also carries agent skills, system prompts,
spec/SDD docs and ordinary markdown, which the old structural pipeline
(source-extension-only) silently dropped on the floor. `classify_repo`
(`repo_classifier.py:215-255`) walks the tree exactly once and assigns every
file to a native `ContentType` — `Skill`/`Spec`/`Prompt`/`Config`/`Document`/
`Code` — via a **precedence ladder, not an LLM**: skill directories (a
`SKILL.md`) claim their whole subtree first, then `.specify/`/`*.spec.md`,
then prompt markers, then the two known config filenames, then document
extensions, and code last (`repo_classifier.py:20-33`, `170-212`). The engine
calls this from `_ingest_codebase` immediately AFTER the structural code pass
succeeds (`engine.py:1772-1785`) and fans each non-code bucket out to the
*existing* per-type adaptor in `_route_classified_artifacts`
(`engine.py:2046-2071`) — Skill → `ContentType.SKILL`, Prompt →
`ContentType.PROMPT`, Document → `ContentType.DOCUMENT`, Config → the
`mcp_config.json` subset only, Spec → an inline `Spec` node (no separate SPEC
adaptor exists). The docstring is explicit that this is "a router over
[the existing adaptors], not a new ingest engine" (`repo_classifier.py:17-18`),
and `_route_classified_artifacts`'s own docstring repeats it
(`engine.py:2056-2057`).

**The rejected alternative is an LLM-based / fuzzy classifier.** The module
docstring rules it out explicitly: "There is **no LLM** here: genuinely
ambiguous files fall through to a conservative default... rather than being
guessed" (`repo_classifier.py:14-16`). An LLM classifier would have been
non-deterministic (a re-ingest of the same tree could route a borderline file
differently, breaking the content-hash delta's assumption that a file's
identity is stable) and would have added a paid LLM call to every file in
every ingested repo just to answer a question a handful of extension/path/
content-sniff rules answers for free. The one place a "soft" signal is used
at all — `_sniff_prompt_json` peeking at up to 4096 bytes for mustache
placeholders or known prompt keys (`repo_classifier.py:149-167`) — is capped,
cheap, and reported with `confidence < 1.0` rather than treated as certain.

**A second rejected alternative is a dedicated ingest engine per artifact
type.** Skills, prompts and documents already had adaptors reachable through
`self.ingest`; building parallel tree-walk/parse logic for each inside the
codebase path would have duplicated that machinery and let it drift. The
classifier is explicitly scoped to *routing*, never re-implementing ingestion
(`repo_classifier.py:17-18`).

The cost accepted: a file that matches no rule (a lockfile, an image, an
unrecognized `.json`) is silently left unclassified rather than guessed
(`repo_classifier.py:212`) — correct by design, but means classification
coverage is bounded by how complete the precedence ladder is, not by "did we
try hard enough" on any one file.

### Pointer — CONCEPT:AU-KG.ingest.writes-go

`engine.py:2278-2281`. A specific consequence of routing Documents through the
classifier rather than ingesting them centrally: each routed document's own
adaptor already ran its own inline enrichment (concepts extracted per-file),
but its text is *also* bubbled up onto the parent codebase result's
`enrichable` list — `result.enrichable.extend(res.enrichable)` — so the
central enrichment seam processes the whole repo's documentation in ONE pass
(the canonical-fact layer) instead of the classifier's fan-out silently
bypassing it. This is deliberately asymmetric: Skill/Prompt/Config results are
**not** bubbled the same way (`engine.py:2293-2295`, comment: they already ran
their own inline enrichment pass on the declaration text and have no
document-length prose worth a second central pass). `test_repo_classification.py`
pins this via `_FakeManifest`, whose docstring calls out the delta-ledger as a
no-op double for exactly this codepath (`test_repo_classification.py:165-171`).

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/repo_classifier.py`,
  `agent_utilities/knowledge_graph/ingestion/engine.py`
  (`_ingest_codebase`/`_route_classified_artifacts`).
- **Backward Compatible**: Yes — `metadata["classify"]=False` opts a caller out
  entirely, and a routing failure is caught and logged, never failing the
  structural code ingest that already succeeded (`engine.py:1780-1784`).
- **Known weak point**: the precedence ladder is a fixed, hand-maintained list
  of extensions/paths/keys. A new artifact convention (a new prompt-file
  naming scheme, a new spec-doc suffix) silently falls through to
  "unclassified" until someone adds a rule — there is no telemetry today that
  surfaces a large unclassified-file count as a signal something needs a new
  rule.
