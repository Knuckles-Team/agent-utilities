# Design Document: Fine-grained object permissioning adds column-redaction + mandatory marking propagation ON TOP of the existing row-level read path, never a second permission engine

CONCEPT:AU-KG.ontology.redact-object-materialize-restricted

> `agent_utilities/knowledge_graph/ontology/permissioning.py`,
> `agent_utilities/knowledge_graph/facade.py:302-333`
> (`restricted_view`, `apply_marking`).

## Decision — extend the existing company-brain read path with property-level redaction and marking-based mandatory control, reusing `PermissionsKernel`/`ActorContext`/`DataLevelPermissions` rather than a new engine

`permissioning.py:4-22` names the Foundry provenance and the gap: Palantir
"object-permissioning/overview" describes schema- and instance-level access
control where, beyond row-level visibility, property/column-level redaction
governs which FIELDS are visible, and marking-based mandatory controls
propagate along links/derivations "so a sensitive marking on one object
cannot be laundered through a derived object." This module "extends the
EXISTING company-brain read path (`secured_reads.py` + `company_brain_runtime.py`
+ `security/brain_context.py`)," adding three capabilities that path lacked:
`redact_object` (property-level masking, returns a filtered copy),
`Marking`/`propagate_markings` (mandatory control generalized "beyond the
4-level classification to arbitrary named markings" — a superset of
`secured_reads.inherit_inferred_acl`), and `restricted_view` (composes the
existing `permit()`/`filter_rows()` row gate WITH column redaction).

**The rejected alternative is a second, parallel permission engine** for
object-level concerns — plausible since this adds genuinely new capability
(property redaction, marking propagation) beyond what the row-level gate did.
Instead the module explicitly reuses `PermissionsKernel` semantics,
`ActorContext`, and `DataLevelPermissions`/`DataClassification` — "no new
permission engine is introduced." The single `enforce` entry point applies a
mandatory fail-closed policy: "every object requires a governed identifier,
verified tenant authority, and an explicit permitting ACL. Authorization or
marking-store failure never widens access" (`permissioning.py:27-30`) — an
error in checking access is treated the same as a denial, never silently
treated as permission. Mandatory markings are durably persisted as
`mandatory_marking` graph nodes; the in-process registry (`MARKING_REGISTRY`)
is explicitly only a CACHE of that durable authority, not the source of
truth — so a process restart cannot silently lose a marking that was already
applied.

The live facade exposes both halves (`facade.py:313-333`): `restricted_view`
delegates straight to `ontology.permissioning.restricted_view`, and
`apply_marking` to `ontology.permissioning.apply_marking` — the facade adds
no logic of its own, it's a thin, discoverable entrypoint onto the same
module.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/permissioning.py`,
  `knowledge_graph/facade.py` (`restricted_view`/`apply_marking`),
  `secured_reads.py`, `company_brain_runtime.py`, every reader that materializes
  an object set for an actor.
- **Backward Compatible**: Yes — additive; a caller not using
  `restricted_view`/`apply_marking` is unaffected by the new capability.
- **Known weak point**: `MARKING_REGISTRY` being an in-process cache means a
  freshly-started process has an EMPTY cache until markings are re-loaded from
  the durable `mandatory_marking` nodes — a window between process start and
  cache warm-up where propagation checks against the cache could miss a
  marking that exists durably but hasn't been read back into memory yet
  (mitigated only if callers always resolve through a path that checks
  durable storage, not the cache alone).
