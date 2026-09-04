# RF-021 architecture component registry — full reference

The mandatory pre-change workflow and its gate-pinned contract paragraph live in
`SKILL.md` under *Architecture component registry — mandatory pre-change workflow*.
This file carries the detail that section points at.

## The real operation map

Before adding or changing a capability, query the generated `ArchitectureComponent` /
`ArchitectureCapability` projection with the real Graph-OS operations `graph_query`
and `graph_search`, then use `graph_code(action=code_context)`
(`target=usage` or `impact`) to find live callers. These are the operation names the
runtime validation harness invokes; `code_context` is an action of `graph_code`, not
a second route. Verify the projected owner-repository source-manifest identity and
digest against its source. Match candidates by stable component/capability ID **and**
their behavioral, authority, and dependency signatures; a shared name, concept
label, or nearby implementation is not identity evidence. The Plans refactor
program's RF-021 architecture-component-registry contract governs this
cross-repository projection.

The conceptual phases have one real operation map: registry lookup → `graph_query`,
discovery → `graph_search`, and caller/impact evidence →
`graph_code(action=code_context)`. Regeneration and re-ingestion are a governed
RF-021 handoff after a read detects missing or stale data; they are not invented
Graph-OS mutation tools. Keep the read bounded, retain only typed/metadata evidence,
and stop when any required operation is unavailable or degraded. `graph_search` is
the existing flat-text result contract, not a structured discovery-row API: the
RF-021 checker accepts only one canonical `ArchitectureComponent` text record with
its source identity, authority signature, and contract digest, plus the exact
single-target connection trailer when Graph-OS appends it. A JSON-shaped
discovery fixture or a nonempty arbitrary string is not evidence.

## RF-019 source-acceptance reference

The immutable RF-019 source-acceptance reference (`641c780f42db1187ddf377d26b81ff553f5ffacb`,
tree `cb222da...`) may be folded into the CA/refactor reference plane as source
evidence only. Keep its qualification state `PROPOSED/NOT_RUN`; it does not
establish runtime acceptance, and the external `/etc` authority bundle remains
deferred until post-refactor deployment.

## The four rules

1. **Resolve before creating.** Find the existing authority and its callers. Extend,
   merge into, or replace it; otherwise obtain a reviewed, finite exception with a
   bounded scope, owner, and expiry or review trigger.
2. **Update the source-owned declaration.** Declare the stable ID, component
   contract version and status, owner, layer, boundary, interfaces, stores, policy
   and identity authorities, dependencies, consumers, entrypoints, tests, gates,
   acceptance evidence, concept tags, and `merged_from`/`replaces` history. Then
   generate the consolidated registry view; never hand-edit that projection.
3. **Prove consolidation in the same change.** Exercise a real caller through the
   declared entrypoint and delete the redundant implementation, facade, registry,
   declaration, or route that the new authority replaces.
4. **Fail closed.** A missing/stale projection or source-digest mismatch does not mean
   "no owner": stop and regenerate/re-ingest through the governed RF-021 path. Reject
   concept-only ownership, parallel facades or registries, missing consumers or tests,
   and stale source declarations. Concepts are many-to-many discovery tags, not
   component identities, and MUST NEVER trigger automatic merging.

The existing Plans `target_inventory` proposal remains the authority for reference-to-
target disposition, fresh target population, review/synthesis, cutover, and closure.
Link registry declarations to that lifecycle; do not create a second target inventory.

### Repository layout and worker lane ownership

RF-021 distinguishes a layer boundary/root seam from a cohesive implementation
component. A `layer_boundary_root_seam` is a navigation and coverage root, not a
monolithic implementation authority. Create an `implementation_component` only
under its declared `parent_layer` (and `parent_component_id`) after proving that no
current capability or behavioral, authority, or dependency signature matches. If a
signature matches, extend, merge, or replace the existing authority instead.
Every component records finite ownership for `owned_source_roots`,
`public_contract_roots`, `test_roots`, and isolated `generated_roots`; tests mirror
the component identity and generated artifacts stay out of handwritten roots.

For worker lanes, resolve those component-owned roots from the owner manifest before
editing. Shared roots or shared files are exceptional and limited: each exception
must use RF-021's finite metadata (`path`, `kind`, `owner_component_ids`,
`review_policy`, and `exception_id`) with explicit owner review. Do not create a
component for file count, team shape, a generic concept tag, or convenience, and
do not let a concept tag confer ownership. The Plans proposal-to-owner-manifest
cutover is the only authority transition; a missing, outdated, or disagreeing
manifest remains fail-closed until regeneration and re-ingestion are complete.
