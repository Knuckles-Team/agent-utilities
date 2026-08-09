#!/usr/bin/python
from __future__ import annotations

"""Admin ownership-claim capability — GOC-61 follow-on (BUG-033/034/039 remediation
context: ``plans/graph-os-completion-program/decisions/GOC-61-native-graph-sharing.md``
and ``decisions/GOC-61-unowned-node-disposition.md``, the latter the AUTHORIZED,
program-owner-ruled disposition this module implements the mechanism for).

Five ``except PermissionError: pass`` sites (``engine.py:723``/``:1178``,
``graph_compute.py:2281``, ``materialization.py:270``, ``enrichment/pipeline.py:236``)
have been silently producing **unowned** nodes for a long time. A separate, concurrent
GOC-61 worker is fixing those five sites and binding real actors on the three
private-conversation write paths (``chat_persistence.py``, ``conversation_ingestion.py``,
``messaging/enrichment.py``) — closing the SOURCE of new unowned nodes. **That work is
out of scope here and is not duplicated by this module**: this module remediates
already-written unowned nodes; it does not touch those five sites or three write paths.

**Disposition policy (``GOC-61-unowned-node-disposition.md``, authorized 2026-08-09,
REVISED same day — superseding an earlier "wipe conversational content" instruction):**
nothing is wiped by default; everything unowned is **claimed**, differing only in who
becomes the owner:

* **Operational/provenance/ontology** (``RuntimeSignal``, ``WorkItem``,
  ``IngestManifest``, ``Evidence``, ...) → a system/service identity.
* **Private conversational** (``Message``, ``Thread``, ``Memento``, ``ChatSummary``,
  ``InboundMessage``, ``EvictedBlock``) → the real human who wrote it. The program
  owner's own reasoning (quoted in the decision doc): *"the conversational content is
  the owner's own, written through their own sessions, and it lost its owner solely to
  a swallowed PermissionError — not to any ambiguity about whose it is."* Claiming these
  closes the BUG-033/034 visibility leak by itself — an owned private node is no longer
  "unowned", so ``tenant_sharing.filter_visible`` stops treating it as public. Attribution
  IS the fix; deletion was only ever a destructive way to reach the same end state.
* **``Concept`` is excluded from type-based sweeps entirely** (dual origin: some are
  general ontology concepts, operational; others are extracted from private chat text,
  conversational — type alone cannot tell them apart, per the decision doc). A
  ``node_types`` selection containing ``Concept`` is refused; an explicit, individually
  reviewed ``node_ids`` entry may still include one (a deliberate per-node human
  decision, not a blind type sweep).

**This module never resolves or hardcodes WHO the human/system owner is.** That is
precisely the decision GOC-61-unowned-node-disposition.md leaves to the authorized
caller, per its own invariant list and the coordinating instruction that amended this
module's brief: *"Resolve the human owner's identity properly... take the identity as a
parameter... The owner parameter must be explicit per invocation. Do not default it to
'the calling admin' — an admin running this on someone else's behalf must state whose
data it is. Silent self-assignment is exactly the takeover shape you were told to
prevent."* Accordingly ``owner_id`` is a **required** parameter of both
:func:`preview_claim` and :func:`apply_claim` — there is no implicit
"claims default to the caller" behavior anywhere in this module, for ANY node class.

Hard invariants, enforced here (not merely documented):

1. **Never reassign an already-owned node.** A node is claimable only while it is
   genuinely unowned (:data:`~.tenant_sharing.OWNER_KEY` absent or ``None``). Ownership
   is re-verified immediately before every mutating write, not only at enumeration time
   — a node that gained an owner between enumeration and write is skipped, not
   overwritten (BUG-034/E6-class TOCTOU defense). This is the one invariant the
   disposition decision itself calls out by name: *"The claim path never reassigns an
   already-owned node. That is an ownership takeover, a separate and more dangerous
   operation, and it is out of scope here."*
2. **``owner_id`` is always explicit, never a default-to-caller.** See above.
3. **``Concept`` cannot be swept by ``node_types``** — see above. Selecting it via
   ``node_types`` raises; it may only be reached through an explicit ``node_ids`` entry.
4. **Selectable, never blind.** There is no "claim everything" mode. Every call
   (preview or apply) requires an explicit ``node_types`` and/or ``node_ids``
   selection.
5. **Real counts, never an unconditional success string** (the BUG-049 lesson,
   generalized, and the disposition decision's own invariant: *"Both operations report
   the actual count, never an unconditional success string"*): every result reports
   exactly how many nodes were actually claimed and how many were skipped as
   already-owned.
6. **Idempotent.** Re-running the same claim after it has already landed finds zero
   remaining unowned candidates and reports ``claimed_total=0`` — never an error, never
   a fabricated re-claim.
7. **Admin-gated, fail-closed.** Every entry point requires
   :func:`tenant_sharing.is_privileged` (``kg:admin``). The MCP/REST surface
   additionally gates on the session's ``kg:admin`` scope before this module is even
   reached — this module's own check is defense-in-depth for any other caller
   (a script, a test, a future surface) that reaches it directly.

**Enumeration without the broken Cypher NULL predicate (BUG-035).** ``MATCH (n) WHERE
n._owner_id IS NULL`` was measured live returning 0 rows against a graph of 47,455
nodes — a silently wrong answer, not a real "nothing unowned" result. This module
never issues that predicate. Enumeration goes through the engine-side by-label
accessor (``IntelligenceGraphEngine.get_nodes_by_label`` / the backend's
``nodes_by_label`` / ``get_nodes_by_label`` — the accessor name differs by engine
shape, mirroring the exact resolution order BUG-049's ``graph_write
action=delete_node`` predicate-delete fix already established in
``mcp/tools/write_ingest_tools.py``), which returns each node's properties directly —
ownership is then a plain Python dict-membership test, never a server-side NULL
predicate. The one per-id point read used for the pre-write re-check and the explicit
``node_ids`` path is an id-equality ``MATCH``, never an ``IS NULL``/inequality
predicate either (BUG-035(a) additionally found ``= 'A'`` broken while ``IN [...]``
worked — id-equality via a map-pattern match, ``MATCH (n {id: $id})``, is the same
literal-map form ``tenant_sharing._node_properties``/``_set_scope`` already use
throughout this codebase, not the property-equality predicate form that bug flagged).
If no such accessor exists on the engine, this module fails loudly
(:class:`OwnershipClaimError`), never falls back to a Cypher scan.

**Execution preconditions this module does NOT enforce itself** (they gate the live
*execution* of the backlog-clearing pass, per the decision doc, not the existence of
this capability — this module is build/unit-test-only per its own worktree brief; it
never runs against the live engine): BUG-035 must be fixed and verified, and a real
by-type enumeration must be run and compared against the decision doc's provisional
type lists before anyone actually invokes ``apply`` against production. Those are
operational preconditions for a *future* invocation, tracked in the decision doc, not
something this library module can check for itself.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from ...security.brain_context import ActorContext, current_actor
from .tenant_sharing import (
    COMMONS_PRIVATE_NODE_TYPES,
    OWNER_KEY,
    SCOPE_COMMONS,
    SCOPE_KEY,
    SCOPE_ORG,
    SCOPE_PRIVATE,
    is_privileged,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AUDIT_NODE_LABEL",
    "AUDIT_MAX_EDGES",
    "CLAIMED_EDGE_TYPE",
    "DUAL_ORIGIN_NODE_TYPES",
    "PRIVATE_CONVERSATIONAL_NODE_TYPES",
    "ClaimCandidate",
    "ClaimPreview",
    "ClaimResult",
    "OwnershipClaimError",
    "VALID_SHARED_SCOPES",
    "apply_claim",
    "classify_node_type",
    "preview_claim",
    "record_claim_audit",
]

#: The only ``_shared_scope`` values an admin claim may stamp — the same three
#: values :mod:`tenant_sharing` defines; a claim never invents a fourth.
VALID_SHARED_SCOPES: frozenset[str] = frozenset(
    {SCOPE_PRIVATE, SCOPE_ORG, SCOPE_COMMONS}
)

#: Reported classification only (NOT a refusal list — see
#: ``GOC-61-unowned-node-disposition.md``: these ARE claimable, to an explicit
#: human owner_id the caller supplies). Reused, not redefined, from
#: :mod:`tenant_sharing` so the two modules can never drift on which types are
#: private-class.
PRIVATE_CONVERSATIONAL_NODE_TYPES: frozenset[str] = COMMONS_PRIVATE_NODE_TYPES

#: Node types excluded from a ``node_types``-SWEEP claim, dual-origin (some
#: rows operational/ontology, some extracted from private chat text — type
#: alone cannot tell them apart, per the disposition decision). Still
#: reachable through an explicit, individually-reviewed ``node_ids`` entry —
#: the decision doc's own words: "type is not sufficient to tell them apart",
#: not "no node of this type may ever be claimed".
DUAL_ORIGIN_NODE_TYPES: frozenset[str] = frozenset({"Concept"})

#: KG node label for the governance audit record this module writes on every
#: claim (preview AND apply — see :func:`record_claim_audit`), mirroring the
#: existing ``RiskVeto`` governance-event convention in
#: ``mcp/tools/governance_tools.py``'s ``submit_risk_veto`` (a durable KG node
#: per governance action, not a side-channel log).
AUDIT_NODE_LABEL = "OwnershipClaimAudit"

#: Edge type linking an ``OwnershipClaimAudit`` node to each node it claimed
#: (bounded by :data:`AUDIT_MAX_EDGES` — see :func:`record_claim_audit`).
CLAIMED_EDGE_TYPE = "CLAIMED_OWNERSHIP_OF"

#: Above this many claimed nodes in one call, stop creating individual
#: ``CLAIMED_OWNERSHIP_OF`` edges (an O(n) edge fan-out on a legitimately large
#: batch — e.g. thousands of ``RuntimeSignal`` rows — would itself become the
#: expensive part of the operation) and rely on the audit node's own
#: ``claimed_node_ids``/``claimed_total`` properties as the durable record
#: instead. The full id list is ALWAYS in the audit node's properties
#: regardless of this cap; only the edge fan-out is capped.
AUDIT_MAX_EDGES = 200


class OwnershipClaimError(PermissionError):
    """Raised for any unauthorized or invalid ownership-claim request.

    Subclasses :class:`PermissionError` so it composes with this codebase's
    existing "denial raises, is never silently swallowed at this layer"
    convention (unlike the BUG-033/039 write-path swallow this module is
    explicitly NOT part of — see the module docstring). Callers at the MCP/
    REST boundary surface this as a structured error, never a bare exception.
    """


def classify_node_type(node_type: str) -> str:
    """``"private_conversational"`` | ``"dual_origin_excluded"`` | ``"operational"``.

    Informational classification only — reported alongside every preview/result
    group (per the disposition ruling's own requirement that a human reviewer see
    "N nodes of type T → owner X", which is far more legible with the class
    attached: e.g. "Message: 11 nodes -> owner alice (private_conversational)").
    Never itself a refusal for ``private_conversational``; IS a refusal signal
    for ``dual_origin_excluded`` on the ``node_types``-sweep path — see
    :data:`DUAL_ORIGIN_NODE_TYPES`.
    """
    if node_type in DUAL_ORIGIN_NODE_TYPES:
        return "dual_origin_excluded"
    if node_type in PRIVATE_CONVERSATIONAL_NODE_TYPES:
        return "private_conversational"
    return "operational"


@dataclass(frozen=True)
class ClaimCandidate:
    """One node a claim call is considering, tagged with its enumeration-time
    ownership snapshot.

    ``currently_unowned`` is a SNAPSHOT taken at enumeration — it is never
    trusted as the final word on whether a node gets claimed. :func:`apply_claim`
    re-verifies immediately before every write (see its own docstring's TOCTOU
    note); this field only lets an already-owned-at-enumeration node be
    reported/skipped without a second point-read, since it was never going to
    be claimed regardless of what happens between enumeration and the write.
    """

    node_id: str
    node_type: str
    currently_unowned: bool


@dataclass(frozen=True)
class ClaimPreview:
    """The dry-run report :func:`preview_claim` returns. Never mutates anything.

    ``would_claim_by_type`` and ``owner_id`` together are the exact "N nodes of
    type T -> owner X" line the disposition ruling requires a human to be able
    to check before authorizing an ``apply`` call with the same parameters.
    ``already_owned_by_type`` reports the rest of the selection honestly (nodes
    that matched the selection but are already owned by someone else and would
    NOT be touched by the matching ``apply`` call) rather than silently
    omitting them.
    """

    owner_id: str
    requested_node_types: list[str]
    requested_node_ids: list[str]
    excluded_dual_origin_types: list[str]
    would_claim_total: int
    would_claim_by_type: dict[str, int]
    would_claim_classification_by_type: dict[str, str]
    already_owned_total: int
    already_owned_by_type: dict[str, int]
    sample_node_ids: list[str]


@dataclass(frozen=True)
class ClaimResult:
    """The real outcome of :func:`apply_claim` — always the ACTUAL count, per
    BUG-049's lesson that a destructive/mutating action must never report an
    unconditional or unverified success."""

    owner_id: str
    shared_scope: str
    claimed_total: int
    claimed_by_type: dict[str, int]
    claimed_classification_by_type: dict[str, str]
    claimed_node_ids: list[str]
    skipped_already_owned: int
    requested_node_types: list[str]
    requested_node_ids: list[str]


def _require_admin(actor: ActorContext | None) -> ActorContext:
    """Resolve and require ``kg:admin`` authority. Fails closed on anything else."""
    resolved = actor or current_actor()
    if not is_privileged(resolved):
        raise OwnershipClaimError(
            "Ownership claim requires verified kg:admin authority"
        )
    return resolved


def _require_owner_id(owner_id: str | None) -> str:
    """Require an explicit, non-empty ``owner_id`` — never a default-to-caller.

    Per the coordinating instruction amending this module's brief: *"The owner
    parameter must be explicit per invocation. Do not default it to 'the
    calling admin' — an admin running this on someone else's behalf must state
    whose data it is. Silent self-assignment is exactly the takeover shape you
    were told to prevent."* Applies uniformly to every node class, not only
    private-conversational ones — there is no code path in this module where a
    claim's new owner is anything other than what the caller explicitly typed.
    """
    resolved = str(owner_id or "").strip()
    if not resolved:
        raise OwnershipClaimError(
            "owner_id is required and must be explicit — this module never "
            "defaults an ownership claim to the calling actor's own id"
        )
    return resolved


def _normalize_types(node_types: list[str] | None) -> list[str]:
    seen: list[str] = []
    for raw in node_types or []:
        value = str(raw or "").strip()
        if value and value not in seen:
            seen.append(value)
    return seen


def _normalize_ids(node_ids: list[str] | None) -> list[str]:
    seen: list[str] = []
    for raw in node_ids or []:
        value = str(raw or "").strip()
        if value and value not in seen:
            seen.append(value)
    return seen


def _require_selection(node_types: list[str], node_ids: list[str]) -> None:
    if not node_types and not node_ids:
        raise OwnershipClaimError(
            "an ownership claim requires an explicit node_types and/or node_ids "
            "selection — there is no blind full-graph mode, by design (claiming "
            "everything blindly is exactly the mistake this capability exists to "
            "prevent)"
        )


def _is_unowned(props: dict[str, Any]) -> bool:
    """True when ``props`` carries no owner at all.

    Mirrors :func:`tenant_sharing.filter_visible`'s own test exactly
    (``OWNER_KEY not in props or owner is None``) — "unowned" means the key is
    absent or null, never merely falsy (an owner id that happens to be an
    empty string, if one ever existed, is not the same claim as "no owner was
    ever stamped").
    """
    return OWNER_KEY not in props or props.get(OWNER_KEY) is None


def _label_lookup(engine: Any):
    """Resolve the engine-side by-label accessor, or ``None`` if none exists.

    Same three-shape resolution order BUG-049's predicate-delete fix
    established in ``mcp/tools/write_ingest_tools.py`` (``IntelligenceGraphEngine``
    exposes it on ``.backend`` as ``nodes_by_label``; a ``GraphComputeEngine``
    exposes ``get_nodes_by_label`` directly) — duplicated here rather than
    imported from that tool module, since this is core library code and that
    module is an MCP tool registrar, not a shared library seam.
    """
    backend = getattr(engine, "backend", None)
    return (
        getattr(engine, "get_nodes_by_label", None)
        or getattr(backend, "nodes_by_label", None)
        or getattr(backend, "get_nodes_by_label", None)
    )


def _point_lookup(engine: Any, node_id: str) -> dict[str, Any] | None:
    """Fetch one node's current properties by id, or ``None`` if it doesn't exist.

    Used for the explicit ``node_ids`` path and — critically — the immediate
    pre-write re-check in :func:`apply_claim` that narrows the TOCTOU window
    between enumeration and mutation. Prefers the typed ``get_node_properties``
    point read; falls back to an id-EQUALITY ``MATCH`` (a literal map pattern,
    never an ``IS NULL``/property-equality predicate — BUG-035 does not apply
    to this form, per the module docstring).
    """
    backend = getattr(engine, "backend", None)
    reader = getattr(backend, "get_node_properties", None) or getattr(
        engine, "get_node_properties", None
    )
    if callable(reader):
        try:
            props = reader(node_id)
        except Exception:  # noqa: BLE001 — fall through to the Cypher point-read below
            props = None
        if props is not None:
            return dict(props) if isinstance(props, dict) else {}

    store = backend if backend is not None else engine
    execute = getattr(store, "execute", None)
    if not callable(execute):
        return None
    try:
        rows = (
            execute(
                "MATCH (n {id: $id}) RETURN properties(n) AS props LIMIT 1",
                {"id": node_id},
            )
            or []
        )
    except Exception:  # noqa: BLE001 — treat an unreadable point-lookup as "not found", never as owned or unowned
        return None
    if not rows:
        return None
    row0 = rows[0]
    props = row0.get("props") if isinstance(row0, dict) else None
    return dict(props) if isinstance(props, dict) else None


def _candidates_by_type(engine: Any, node_type: str) -> list[ClaimCandidate]:
    """EVERY node of ``node_type`` via the engine label index — owned or not.

    Ownership is not filtered here; each candidate carries its enumeration-time
    ``currently_unowned`` snapshot so callers can report already-owned matches
    honestly (per BUG-049's "never fabricate/omit a real count" lesson,
    generalized) instead of silently dropping them before the caller ever sees
    they existed.
    """
    lookup = _label_lookup(engine)
    if lookup is None:
        raise OwnershipClaimError(
            "no label-index accessor on this engine (tried "
            "engine.get_nodes_by_label, engine.backend.nodes_by_label, "
            "engine.backend.get_nodes_by_label); refusing to fall back to a "
            "Cypher scan — BUG-035's broken NULL predicate would make an "
            "IS-NULL-based scan silently return zero rows, and a plain scan "
            "would also bypass RLS row filtering and could under- or "
            "over-enumerate"
        )
    rows = lookup(node_type, 0) or []
    out: list[ClaimCandidate] = []
    for row in rows:
        if isinstance(row, (tuple, list)) and len(row) >= 2:
            node_id, props = row[0], row[1]
        else:
            continue
        if not isinstance(props, dict):
            continue
        out.append(
            ClaimCandidate(
                node_id=str(node_id),
                node_type=node_type,
                currently_unowned=_is_unowned(props),
            )
        )
    return out


def _candidates_by_ids(engine: Any, node_ids: list[str]) -> list[ClaimCandidate]:
    """Resolve explicit ``node_ids`` to existing nodes (owned or not — see
    :func:`_candidates_by_type`'s docstring for why ownership isn't filtered
    here), silently excluding ids that don't resolve to anything rather than
    fabricating a candidate for a node that was never written."""
    out: list[ClaimCandidate] = []
    for node_id in node_ids:
        props = _point_lookup(engine, node_id)
        if props is None:
            continue
        node_type = str(props.get("node_type") or "")
        out.append(
            ClaimCandidate(
                node_id=node_id,
                node_type=node_type,
                currently_unowned=_is_unowned(props),
            )
        )
    return out


def _split_dual_origin(node_types: list[str]) -> tuple[list[str], list[str]]:
    """``(allowed, excluded)`` — split ``node_types`` on :data:`DUAL_ORIGIN_NODE_TYPES`.

    NOT a private-content refusal (see the module docstring — private
    conversational types ARE claimable, to an explicit human owner). This is
    the narrower ``Concept``-only exclusion: type alone cannot separate its
    ontology rows from its chat-extracted rows, so a ``node_types`` sweep must
    never include it.
    """
    excluded = [t for t in node_types if t in DUAL_ORIGIN_NODE_TYPES]
    allowed = [t for t in node_types if t not in DUAL_ORIGIN_NODE_TYPES]
    return allowed, excluded


def preview_claim(
    engine: Any,
    *,
    owner_id: str,
    node_types: list[str] | None = None,
    node_ids: list[str] | None = None,
    actor: ActorContext | None = None,
) -> ClaimPreview:
    """Report what an :func:`apply_claim` with the same selection WOULD claim.

    Read-only — never mutates the graph. Groups the result by ``node_type``
    (with each group's classification and the resolved ``owner_id``) so the
    admin can see exactly "N nodes of type T -> owner X" before committing to
    anything — the disposition ruling's own required review line, and the
    safety mechanism the GOC-61 design calls for given the unowned population
    cannot currently be measured any other way (see the module docstring's
    BUG-035 note). ``owner_id`` is required here too (not merely for
    ``apply_claim``) so a preview is a genuine preview of what the identical
    ``apply`` call would do, never silently out of sync with it.
    """
    actor = _require_admin(actor)
    resolved_owner = _require_owner_id(owner_id)
    types = _normalize_types(node_types)
    ids = _normalize_ids(node_ids)
    _require_selection(types, ids)

    allowed_types, excluded_types = _split_dual_origin(types)

    candidates: list[ClaimCandidate] = []
    for node_type in allowed_types:
        candidates.extend(_candidates_by_type(engine, node_type))
    if ids:
        # Explicit ids MAY include a dual-origin (Concept) node — that is a
        # deliberate per-node human decision, not a blind type sweep; see
        # _split_dual_origin's docstring.
        candidates.extend(_candidates_by_ids(engine, ids))

    seen_ids: set[str] = set()
    deduped: list[ClaimCandidate] = []
    for cand in candidates:
        if cand.node_id in seen_ids:
            continue
        seen_ids.add(cand.node_id)
        deduped.append(cand)

    by_type: dict[str, int] = {}
    already_owned_by_type: dict[str, int] = {}
    class_by_type: dict[str, str] = {}
    for cand in deduped:
        class_by_type[cand.node_type] = classify_node_type(cand.node_type)
        if cand.currently_unowned:
            by_type[cand.node_type] = by_type.get(cand.node_type, 0) + 1
        else:
            already_owned_by_type[cand.node_type] = (
                already_owned_by_type.get(cand.node_type, 0) + 1
            )
    would_claim = [c for c in deduped if c.currently_unowned]

    result = ClaimPreview(
        owner_id=resolved_owner,
        requested_node_types=types,
        requested_node_ids=ids,
        excluded_dual_origin_types=excluded_types,
        would_claim_total=len(would_claim),
        would_claim_by_type=by_type,
        would_claim_classification_by_type=class_by_type,
        already_owned_total=len(deduped) - len(would_claim),
        already_owned_by_type=already_owned_by_type,
        sample_node_ids=[c.node_id for c in would_claim[:20]],
    )
    record_claim_audit(
        engine,
        actor=actor,
        dry_run=True,
        owner_id=resolved_owner,
        shared_scope="",
        requested_node_types=types,
        requested_node_ids=ids,
        claimed_node_ids=[],
        claimed_by_type={},
        skipped_already_owned=0,
        excluded_dual_origin_types=excluded_types,
    )
    return result


def apply_claim(
    engine: Any,
    *,
    owner_id: str,
    node_types: list[str] | None = None,
    node_ids: list[str] | None = None,
    shared_scope: str = SCOPE_ORG,
    actor: ActorContext | None = None,
) -> ClaimResult:
    """Actually claim ownership of every currently-unowned matching node.

    Only ever mutates a node that is verified unowned IMMEDIATELY before the
    write (never merely at enumeration time — see the module docstring's
    TOCTOU note). Refuses outright (raises :class:`OwnershipClaimError`,
    mutates nothing) if ``node_types`` includes a dual-origin type
    (:data:`DUAL_ORIGIN_NODE_TYPES` — currently just ``Concept``). Per
    ``GOC-61-unowned-node-disposition.md``, private-class conversational types
    ARE claimable here — to the caller's explicit ``owner_id``, never a
    default. Always records a governance audit (:func:`record_claim_audit`),
    including on a request that ends up claiming zero nodes.
    """
    actor = _require_admin(actor)
    resolved_owner = _require_owner_id(owner_id)
    types = _normalize_types(node_types)
    ids = _normalize_ids(node_ids)
    _require_selection(types, ids)

    if shared_scope not in VALID_SHARED_SCOPES:
        raise OwnershipClaimError(
            f"invalid shared_scope {shared_scope!r}; must be one of "
            f"{sorted(VALID_SHARED_SCOPES)}"
        )

    _allowed_types, excluded_types = _split_dual_origin(types)
    if excluded_types:
        raise OwnershipClaimError(
            f"refusing a node_types SWEEP of dual-origin type(s) "
            f"{excluded_types}: type alone cannot separate ontology rows from "
            "chat-extracted rows for these — claim individual, reviewed nodes "
            "via node_ids instead (see GOC-61-unowned-node-disposition.md and "
            "the module docstring)."
        )

    candidates: list[ClaimCandidate] = []
    for node_type in types:
        candidates.extend(_candidates_by_type(engine, node_type))
    if ids:
        candidates.extend(_candidates_by_ids(engine, ids))

    store = getattr(engine, "backend", None)
    if store is None:
        store = engine
    execute = getattr(store, "execute", None)
    if not callable(execute):
        raise OwnershipClaimError(
            "engine/backend exposes no execute() to apply the claim"
        )

    claimed: list[ClaimCandidate] = []
    skipped_already_owned = 0
    seen_ids: set[str] = set()
    for cand in candidates:
        if cand.node_id in seen_ids:
            continue  # a node reachable via both node_types and node_ids is claimed once
        seen_ids.add(cand.node_id)

        if not cand.currently_unowned:
            # Already known owned at enumeration -- report it honestly (never
            # silently omit a real, already-owned match, per BUG-049's lesson)
            # without spending a second round-trip: ownership does not
            # spontaneously revert to unowned, so there is no race to defend
            # against on THIS branch.
            skipped_already_owned += 1
            continue

        # The authoritative check for a node the enumeration snapshot saw as
        # unowned: re-verify IMMEDIATELY before writing, never trusting that
        # snapshot alone (BUG-034/E6 TOCTOU defense — see module docstring
        # invariant 1). This is what makes "never reassign an already-owned
        # node" a guarantee rather than a best-effort — the SAME guarantee for
        # every node class, private conversational content included.
        current = _point_lookup(engine, cand.node_id)
        if current is None:
            continue  # vanished between enumerate and apply — nothing to claim
        if not _is_unowned(current):
            skipped_already_owned += 1
            continue

        execute(
            f"MATCH (n {{id: $id}}) SET n.{OWNER_KEY} = $owner, n.{SCOPE_KEY} = $scope",
            {"id": cand.node_id, "owner": resolved_owner, "scope": shared_scope},
        )
        claimed.append(cand)

    by_type: dict[str, int] = {}
    class_by_type: dict[str, str] = {}
    for cand in claimed:
        by_type[cand.node_type] = by_type.get(cand.node_type, 0) + 1
        class_by_type[cand.node_type] = classify_node_type(cand.node_type)

    result = ClaimResult(
        owner_id=resolved_owner,
        shared_scope=shared_scope,
        claimed_total=len(claimed),
        claimed_by_type=by_type,
        claimed_classification_by_type=class_by_type,
        claimed_node_ids=[c.node_id for c in claimed],
        skipped_already_owned=skipped_already_owned,
        requested_node_types=types,
        requested_node_ids=ids,
    )
    record_claim_audit(
        engine,
        actor=actor,
        dry_run=False,
        owner_id=resolved_owner,
        shared_scope=shared_scope,
        requested_node_types=types,
        requested_node_ids=ids,
        claimed_node_ids=result.claimed_node_ids,
        claimed_by_type=by_type,
        skipped_already_owned=skipped_already_owned,
        excluded_dual_origin_types=[],
    )
    return result


def record_claim_audit(
    engine: Any,
    *,
    actor: ActorContext,
    dry_run: bool,
    owner_id: str,
    shared_scope: str,
    requested_node_types: list[str],
    requested_node_ids: list[str],
    claimed_node_ids: list[str],
    claimed_by_type: dict[str, int],
    skipped_already_owned: int,
    excluded_dual_origin_types: list[str],
) -> str | None:
    """Write one governance audit record for a claim call (preview or apply).

    Mirrors the existing ``RiskVeto`` governance-event convention
    (``mcp/tools/governance_tools.py``'s ``submit_risk_veto``): a durable KG
    node per governance action — who, when, what was requested, what actually
    happened — not a side-channel log file. Best-effort: an audit-write
    failure is logged, never raised into the caller (the claim/preview result
    the caller already has is the authoritative outcome; failing to *record*
    that outcome must not retroactively fail an already-decided operation).

    Returns the audit node id, or ``None`` if the audit write itself could not
    be attempted (no ``engine.add_node``) or failed.
    """
    add_node = getattr(engine, "add_node", None)
    if not callable(add_node):
        logger.warning("ownership claim audit skipped: engine exposes no add_node()")
        return None

    audit_id = f"ownership_claim_audit:{uuid.uuid4().hex}"
    props: dict[str, Any] = {
        "actor_id": actor.actor_id,
        "actor_tenant_id": actor.tenant_id,
        "dry_run": dry_run,
        "owner_id": owner_id,
        "shared_scope": shared_scope,
        "requested_node_types": list(requested_node_types),
        "requested_node_ids": list(requested_node_ids),
        "claimed_total": len(claimed_node_ids),
        "claimed_by_type": dict(claimed_by_type),
        "claimed_node_ids": list(claimed_node_ids),
        "skipped_already_owned": skipped_already_owned,
        "excluded_dual_origin_types": list(excluded_dual_origin_types),
    }
    try:
        add_node(audit_id, AUDIT_NODE_LABEL, props)
    except Exception:  # noqa: BLE001 — a provenance write must never fail an already-decided operation
        logger.warning(
            "ownership claim audit write failed for actor %r",
            actor.actor_id,
            exc_info=True,
        )
        return None

    if not dry_run and claimed_node_ids:
        add_edge = getattr(engine, "add_edge", None)
        if callable(add_edge):
            for node_id in claimed_node_ids[:AUDIT_MAX_EDGES]:
                try:
                    add_edge(audit_id, node_id, CLAIMED_EDGE_TYPE)
                except Exception:  # noqa: BLE001 — same "audit write never fails the operation" reasoning as above
                    logger.debug(
                        "ownership claim audit edge failed for %r -> %r",
                        audit_id,
                        node_id,
                        exc_info=True,
                    )
    return audit_id
