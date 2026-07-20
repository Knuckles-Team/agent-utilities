# Concept-ID coordination

Parallel sessions coordinate semantic concept IDs through the committed,
line-oriented `docs/concept_reservations.yaml` ledger.

Reserve an exact canonical ID:

```bash
agent-utilities concept reserve \
  --id AU-KG.ingest.entropy-dedup \
  --session build-session \
  --design-doc design-reference
```

The allocator validates the OKF-CIS grammar and closed domain vocabulary, then
checks source markers, `docs/concepts.yaml`, and live ledger claims while
holding the per-repository lock. Duplicate claims fail atomically. Session and
design values are persisted only as non-reversible references.

Other operations:

```bash
agent-utilities concept list --status reserved
agent-utilities concept release --id AU-KG.ingest.entropy-dedup
agent-utilities concept reconcile
agent-utilities concept resolve --id AU-KG.ingest.entropy-dedup
```

`reconcile` marks a claim `landed` once its exact marker is present in source,
and expires abandoned claims after their configured TTL.
