# Cross-host concept reservation authority

<!-- CONCEPT:AU-OS.governance.cross-host-concept-reservation-authority -->

`agent_utilities.governance.concept_reservation` defines the authority boundary
for concept IDs used by Repository Manager and development agents. The
epistemic-graph engine already supplies the required durable primitives:

* `CreateNodeIfAbsent(node_id, properties)` atomically chooses the first writer;
* `CompareAndSetNodeFields(node_id, conditions, updates)` performs a fenced
  lifecycle/reclaim update under the engine write guard;
* `GetNodeProperties` and bounded `GetNodesByLabel` provide point/query reads.

The adapter uses one authoritative graph and one node identity per complete
concept ID: `concept-reservation:<concept_id>`. Cross-host clients must route to
that same graph authority; a local Git lock, JSON file, or fixture is never a
global fallback.

## Why the local allocator is not enough

The existing allocator's shared `git-common-dir` lock is useful when linked
worktrees share one checkout. Separate clones and hosts have different locks
and ledgers, so both can otherwise succeed. The local ledger remains an
auditable, merge-friendly projection only.

The generic native primitives are sufficient because the exact concept ID is
the graph node's identity. A failed create is followed by a point read: an
identical request key and immutable fingerprint on that same canonical node
replays the winner; changed input conflicts. The same request key on a
different concept names a different node, because this protocol does not add a
globally atomic secondary request-key index. Reclaim and lifecycle mutation
never delete the node. They CAS all immutable identity, state, owner, tenant,
and fence fields together. Therefore a concurrent reclaim has one winner
without a process-wide lock.

There is no read-then-write uniqueness check and no generic Cypher or
`BatchUpdate` fallback. If the native primitive is absent or the authoritative
graph is unavailable, the adapter raises `AuthorityUnavailable` and callers
defer/retry.

`FixtureConceptReservationAuthority` is useful for deterministic unit tests
only. It is thread-safe inside one Python process, advertises
`authoritative = False`, and provides no cross-process or cross-host guarantee.

## Authority-owned policy and durable record

The deployment supplies versioned `ConceptNamespacePolicy` values to the native
adapter. The adapter validates the complete OKF-CIS ID, selects the one matching
policy, and stores the policy namespace, numeric range, and `policy_version` in
the durable claim. Caller-provided range fields are normalized to the selected
policy; they are not authority policy. The constructor rejects any ambiguous
overlap instead of sorting version strings (so versions `2` and `10` cannot
silently select the wrong policy); disjoint ranges or semantic prefixes must be
declared explicitly.

Each record contains:

* concept ID/namespace and policy version/range;
* opaque tenant, repository, lane, owner, request, design, and provenance refs;
* immutable purpose digest and idempotency fingerprint;
* created/expiry times, lifecycle state, fence, visibility, and transition
  timestamps.

The state machine is:

```text
reserved -> materialized -> landed -> tombstoned
     |             |           |
     +-> released  +-> released +-> tombstoned
     +-> expired   +-> expired
```

`landed` means repository-visible. `tombstoned` means externally visible and
never reusable, even when source files disappear. A release or expiry is
reclaimable only while visibility is below repository. Any attempted release
or expiry after repository/external visibility is promoted to a tombstone;
visibility never decreases. Expiry is accepted only once `now >= expires_at`.

## Native primitive algorithm

For `reserve(request)`:

1. Bind the request to the authority-owned namespace/range policy.
2. Build the canonical node key and candidate record.
3. Call `CreateNodeIfAbsent`. A `true` result is the sole first claimant.
4. On `false`, read the node. Same request key/fingerprint is an idempotent
   replay; another active/landed/tombstoned request returns a conflict.
5. A released/expired pre-visibility node is reclaimed with one CAS conditioned
   on its current reservation ID, immutable fingerprint, state, and fence. A CAS
   loser rereads and classifies the new winner, up to a fixed retry bound; a
   perpetually contended authority fails closed.

For a lifecycle transition, the adapter reads the record and issues one CAS
conditioned on tenant, owner, reservation ID, immutable fingerprint, state, and
expected fence. The next record includes the incremented fence and all
lifecycle timestamps. A stale owner or fence cannot mutate the node. Exact
same-owner retries with the immediately previous fence return the already
committed record; other stale calls fail. Native label reads validate cursor
size/control characters, require strict cursor progress, and enforce page and
record bounds so a backend that ignores `after` cannot spin forever.

`reserve_next_numeric` probes an explicit bounded, deterministic candidate
range. Each candidate uses the same create-once path, so independent ranges do
not share a global process lock.

## Projection and recovery

After native reservation, Repository Manager calls
`ConceptReservationService.materialize`. The service first performs the fenced
native transition to `materialized`, then appends a privacy-safe record to the
caller's own concept fragment and regenerates the generated view. If the
process dies between those steps, retrying reads the already-materialized
claim and repeats only the idempotent local projection. The local fragment
never allocates or releases the ID.

`reconcile_projection` is read-only and bounded by `max_records`. It compares a
tenant-scoped native view with local fragments/view and source markers,
classifying missing projections, orphan records, state mismatches, and markers
without claims. It rejects repeated cursors or an exceeded bound rather than
spinning or materializing an unbounded response.

Existing `agent-utilities concept reserve` remains the limited same-host
compatibility entry point until Repository Manager migrates to this service.
Its file lock must not be described as globally safe.

## Evidence and deployment boundary

The epistemic-graph client and Rust engine already test durable create-once and
CAS exactly-one-winner behavior, including WAL/Raft replay of these native
methods. RMDD-16 adds 50 adapter instances contending through one shared native
port fixture, restart read/materialization tests, policy/fence/visibility/expiry
tests, and a bounded reconciliation test. That proves the adapter's use of the
native arbitration contract; it is not a separate-host network test. Production
rollout still requires exercising the configured authoritative graph across
hosts and after an engine restart, so this lane does not claim live cluster
evidence merely from the in-process fixture.
