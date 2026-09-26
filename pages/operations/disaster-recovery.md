# Backup, restore and cross-cell recovery

The production objective is an observed RPO no greater than 60 seconds and an observed
RTO no greater than 300 seconds. These are acceptance targets, not claims about an
unexecuted environment. A release is certified only when the 24–72 hour campaign signs
evidence that both targets were met under backup/restore and regional-recovery faults.

## Portable bundle contract

The minute-level online backup writes format-v3 bundles to the mounted archive. A bundle
is complete only when all of the following are present and validated:

- every authoritative `graph*.redb` shard, including Raft metadata/logs, mutation
  receipts/outbox/cursors and retained cross-shard prepare/decision records;
- `admin-mutations.redb`, the separate coordinator database containing parent receipts,
  child/outbox receipts, route fences and authenticated encrypted recovery plans for
  prepared transactions;
- `MANIFEST.json` with exact aggregate counts for graph state, coordinator receipts,
  encrypted plans and cross-shard decisions, plus SHA-256 for every portable graph
  shard and coordinator file. The manifest is synced and atomically renamed only
  after those files validate.

Backup snapshots graph shards first and the coordinator ledger last. Therefore a parent
is either terminal after its children, or remains prepared with the encrypted plan needed
to resume idempotently. The backup gate rejects a missing coordinator file, an unbound
receipt, plaintext/private recovery state, a prepared transaction without its plan, or
manifest count drift.

The archive PVC must be backed by a versioned object/RWX storage class replicated
outside the cell. The mounted hot window retains the newest two full bundles and
prunes older complete attempts under a cross-process archive lock; object versioning
and lifecycle policy own long-term retention. This bounds cell storage while
preserving the current and immediately prior recovery point. A large local filesystem
called “archive” does not satisfy disaster recovery.

An online attempt whose recovery boundary changes is never published. The backup job
removes that unpublished full copy immediately; the 24-hour incomplete-attempt sweep is
only a crash fallback. This prevents repeated retries from consuming the archive before
retention can run.

## Continuous restore validation

`graphos-restore-validation` runs daily against the latest complete bundle. It performs
an offline restore into a dedicated scratch claim, checks the bundle and restored tree,
starts the exact Epistemic Graph binary on loopback with verified context/RLS/signature
enforcement, waits for startup recovery to reconcile prepared parents and retained
cross-shard decisions, and runs a health RPC. Scratch data is deleted afterward.

The job verifies the manifest-bound portable-file digests, then emits only digests
and aggregate counts. It never emits archive locations,
engine locations, principals, graph content or source labels. A failed restore job pages
the recoverability SLO and blocks release promotion.

## Cross-cell cutover

1. Freeze release, ontology, index and placement changes in both cells.
2. Fence writes to the failed cell through the global control plane.
3. Select the newest bundle whose manifest and external object digest verify.
4. Restore into fresh retained claims in the recovery cell. Never overwrite a running
   or partially restored target.
5. Start the exact signed engine image with the same data-key authority. Startup must
   reconcile prepared transactions and retained cross-shard decisions before readiness.
6. Verify coordinator receipt totals, graph health, placement epoch, projection/index
   cursors, tenant policy version and checkpoint age using aggregate probes.
7. Shift a canary traffic slice, then all traffic, through the global control plane.
8. Keep the old cell fenced until the new checkpoint and external archive copy verify.
9. Record observed checkpoint age as RPO and fault-to-full-cutover time as RTO in signed
   operational evidence.

The executable regional-recovery scenario uses
`graphos-certification-fault`. Its runtime action command performs the real cell
fence/cutover; its probe command returns only invariant booleans and observed RPO. Both
commands are JSON argv references injected at runtime, never shell strings or committed
infrastructure locations.

## Restore acceptance invariants

- the bundle digest and external signature match the selected release;
- all admin parent/idempotency/child receipts remain internally bound;
- every prepared transaction recovery body remains authenticated ciphertext;
- retained cross-shard decisions and prepares survive byte-for-byte;
- no acknowledged graph write is lost and no partial domain state is visible;
- projection/index/reasoning cursors do not advance beyond authoritative state;
- tenant isolation, deletion propagation and stale-policy rejection remain active;
- the measured RPO/RTO fit the certified targets.
