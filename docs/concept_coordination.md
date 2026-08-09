# Concept-ID coordination

There are two deliberately different paths:

* the existing `concept_allocator` is a **same-host compatibility path**. Its
  shared Git-common-dir lock protects linked worktrees of one checkout, but it
  is not a cross-host authority;
* `governance.concept_reservation.NativeConceptReservationAuthority` is the
  **cross-host path**. It uses the authoritative graph's existing
  `CreateNodeIfAbsent`/`CompareAndSetNodeFields` primitives and fails closed
  when those are unavailable or no authority-owned policy is configured. A
  local ledger, JSON file, process lock, or read-then-write graph sequence must
  never be presented as a global uniqueness guarantee.

The native contract and migration boundary are documented in
[`architecture/concept-reservation-authority.md`](architecture/concept-reservation-authority.md).

Parallel sessions coordinate semantic concept IDs through the committed,
line-oriented `docs/concept_reservations.yaml` ledger.

Reserve an exact canonical ID:

```bash
agent-utilities concept reserve \
  --id AU-KG.ingest.entropy-dedup \
  --session build-session \
  --design-doc design-reference
```

The compatibility allocator validates the OKF-CIS grammar and closed domain
vocabulary, then checks source markers, `docs/concepts.yaml`, and live ledger
claims while holding the per-repository lock. Duplicate claims fail atomically
within the same host/repository arbitration scope. Session and design values
are persisted only as non-reversible references. It must not be used when
separate clones or hosts need one global winner.

For a central reservation, construct a privacy-safe request and inject the
native authority into the lifecycle service:

```python
from agent_utilities.governance.concept_reservation import (
    ConceptReservationService,
    ConceptNamespacePolicy,
    NativeConceptReservationAuthority,
    reservation_request,
)

request = reservation_request(
    "AU-KG.ingest.example",
    tenant_id="tenant",
    repository="agent-utilities",
    lane="lane-1",
    owner="agent-1",
    request_key="stable-request-id",
    purpose="why this concept exists",
)
service = ConceptReservationService(
    NativeConceptReservationAuthority(
        engine,
        policies=(ConceptNamespacePolicy("AU-KG.ingest", policy_version="policy-1"),),
    )
)
record = service.reserve(request)
```

If graph-os does not expose the native create/CAS primitives or the authority
policy is unavailable, this raises `AuthorityUnavailable`; defer and retry
rather than calling the compatibility allocator.

Native request-key idempotency is scoped to the complete concept ID's
canonical node. Reusing a key for the same concept replays that node; the same
key for a different concept is a separate claim. No global request-key index is
claimed by this adapter.

Other operations:

```bash
agent-utilities concept list --status reserved
agent-utilities concept release --id AU-KG.ingest.entropy-dedup
agent-utilities concept reconcile
agent-utilities concept resolve --id AU-KG.ingest.entropy-dedup
```

`reconcile` marks a claim `landed` once its exact marker is present in source,
and expires abandoned claims after their configured TTL.
