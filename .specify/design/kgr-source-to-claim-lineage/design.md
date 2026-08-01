# Design Document: A citation's provenance chain is walked from the evidence spine's own readers, never re-derived — and an unresolvable citation is a defect, never a silently thin answer

CONCEPT:AU-KG.retrieval.source-to-claim-lineage ·
CONCEPT:AU-KG.retrieval.mandatory-evidence-citation

> `agent_utilities/knowledge_graph/retrieval/lineage.py`,
> `agent_utilities/knowledge_graph/ingestion/evidence_spine.py` (the reused
> readers), `agent_utilities/mcp/tools/audit_tools.py`.

## Decision — `resolve_lineage` reuses the evidence spine's own readers verbatim; nothing here re-derives evidence

`CONCEPT:AU-KG.retrieval.source-to-claim-lineage`

`lineage.py:6-33` (Track 9 of the universal-ingestion program) states the
chain directly: given a fact/claim's cited fragment, walk backward through
the evidence spine — `claim → Fragment → Artifact → ChangeEnvelope fields →
pack version` — so an operator or eval gate can answer "what backs this, and
is it still true" for any retrieval result, "not just trust the citation
blindly." The docstring is explicit about the implementation constraint:
"**Nothing here re-derives evidence.** It reuses the evidence spine's own
readers verbatim" — `load_fragments` for the fragment reader,
`citation_status` for the four-outcome drift check
(current/moved/stale/lost), and the stored `Artifact` node's OWN properties
(`envelope_id`, `source_version`, `schema_version`,
`ontology_mapping_version`) rather than re-deriving them from the payload,
"mirroring `Artifact.from_envelope`'s own rule that governance/revision come
off the envelope, not the content."

**The rejected alternative**: a lineage resolver that re-parses or
re-computes provenance from the artifact's raw content each time it is asked
— which would drift from whatever `evidence_spine.py`'s ingestion-time logic
actually decided the first time, and would duplicate the drift-detection
logic `citation_status` already owns. Reusing the readers verbatim means
there is exactly one place that decides what a citation's provenance is,
whether the question is asked at ingest time or at query time.

## Pointer — `CONCEPT:AU-KG.retrieval.mandatory-evidence-citation`

`lineage.py:29-33,168-185`. When a citation cannot resolve to any evidence at
all (`citation_status` reports `lost`), `resolve_lineage` raises
`LineageNotFoundError` — the docstring names the principle directly:
"mandatory evidence citation... means an unresolvable citation is a defect to
surface, never a silently degraded/empty answer." **The rejected
alternative**, spelled out in the exception's own docstring: collapsing a
lost citation to "an empty/default lineage record" — which would let a
caller silently treat "there genuinely is no trail" the same as "a legitimate
trail that happens to be thin," destroying exactly the distinction the whole
lineage walk exists to preserve. Every other failure mode in the module
(unreachable engine, absent artifact row) degrades softly to "nothing found"
— it is specifically the *lost citation* outcome that is promoted to a hard
raise, because that is the one outcome mandatory-evidence-citation forbids
silently passing through.

## Risk Assessment

- **Blast Radius**: `lineage.py`, `evidence_spine.py`, `mcp/tools/
  audit_tools.py`, `observability/gateway_metrics.py`
  (`RETRIEVAL_CITATION_RESOLUTION`).
- **Backward Compatible**: Yes — `resolve_lineage` is a new read path over
  existing evidence-spine data; it does not change how fragments/artifacts
  are written.
- **Known weak point**: `LineageNotFoundError` is raised only when
  `citation_status` reports `lost` for the ARTIFACT id resolved from the
  fragment; if fragment→artifact resolution itself silently returns an empty
  string (engine unreachable, fragment id not found), the walk degrades soft
  through `_load_artifact_row`'s empty-dict path rather than raising — so an
  infrastructure failure and a genuinely lost citation are not perfectly
  distinguishable from the exception type alone.
