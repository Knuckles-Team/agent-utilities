# Design Document: One source-agnostic manifest compiler generalizes the LeanIX-only pipeline — fail-closed on integrity/signature before anything is written

CONCEPT:AU-KG.ontology.connector-manifest-compiler ·
CONCEPT:AU-KG.ontology.supply-chain-integrity

> `agent_utilities/knowledge_graph/ontology/manifest_compiler.py`,
> `agent_utilities/knowledge_graph/ontology/ontology_integrity.py`.

## Decision — generalize LeanIX's compile/export/apply pipeline into one source-agnostic compiler over `ConnectorManifest`, with LeanIX as its first (regression-tested) caller

`manifest_compiler.py:1-18` states the generalization directly:
`compile_manifest`/`export_manifest_ttl`/`apply_manifest` generalize
`ontology.leanix_metamodel`'s LeanIX-only
`compile_leanix_metamodel`/`export_leanix_ttl`/`apply_leanix_metamodel` into a
pipeline over any `ConnectorManifest`. `manifest_from_leanix_spec` makes
LeanIX "the **first caller** of this generalized compiler (proved lossless by
the golden-file test in `tests/`), **without touching the existing production
`sync_leanix_ontology` entry point**." Every transformation is stated as
"pure, deterministic, zero LLM calls" — a direct structural projection of the
manifest's declared fields, not an LLM-assisted generation step.

**The rejected alternative is generalizing by directly rewriting the
production LeanIX path** — riskier, because a bug in the generalization would
break the one live production ontology sync in the process of proving the
abstraction. Instead the new compiler is proved lossless against LeanIX via a
golden-file test WITHOUT the production entry point ever routing through it,
so the generalized path can mature independently before (if ever) LeanIX
itself is cut over.

### Pointer — `CONCEPT:AU-KG.ontology.supply-chain-integrity`

`manifest_compiler.py:194-238`, `ontology_integrity.py:1-50`. `apply_manifest`
is explicitly "Fail-closed: the `provenance.integrity` hash and `signature`
are re-verified against `trusted_signers` **BEFORE anything is written** — a
bad/missing/unsigned manifest raises, it is never silently skipped"
(`manifest_compiler.py:205-207`). The re-verification is real, not
trust-on-first-use: `apply_manifest` recomputes the canonical hash of the
freshly-generated turtle and compares it against
`manifest.provenance.integrity.hash`, raising `SignatureVerificationError` on
mismatch with a message naming the two real causes ("the manifest was
hand-edited after signing, or the compiler is non-deterministic") — then
separately re-verifies the Ed25519 signature against `trusted_signers`.
`ontology_integrity.py:1-12` states the guarantee this exists for: a
generated/reconciled `ontology_<source>.ttl` is "exactly the graph a trusted
party produced" via a canonical, serialization-order-invariant hash (an
RDF-canonicalization, URDNA2015-equivalent) plus a signature whose public key
is independently pinned and verifiable without the private signing secret.
**The rejected alternative is trusting a manifest's declared hash/signature at
face value** — the obvious shortcut, and the one that would let a
hand-tampered manifest (post-signing edit) or an unsigned manifest pass
through undetected; re-deriving the hash from the actually-compiled output
closes exactly that gap.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/manifest_compiler.py`,
  `ontology_integrity.py`, every connector's `ontology_<source>.ttl`
  regeneration path, `scripts/generate_connector_manifests.py`.
- **Backward Compatible**: Yes — `sync_leanix_ontology` production entry point
  is explicitly untouched.
- **Known weak point**: `is_wired()`'s anti-sprawl check
  (`manifest_compiler.py:180-191`) degrades to `False` on any exception from
  the federation registry lookup ("federation registry is best-effort here")
  — a transient failure in `registered_federated_iris()` would make a
  genuinely-wired connector look unwired, tripping `AntiSprawlError` for a
  spurious reason rather than a real sprawl violation.
