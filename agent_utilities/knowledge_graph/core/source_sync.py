"""Source-agnostic KG synchronization (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

One sync mechanism for every external source registered in the hydration
``CAPABILITY_REGISTRY`` (LeanIX, Camunda, ARIS, ServiceNow, …), so they share a
single entrypoint, scheduler dispatch, and operational model instead of each
re-hydrating ad hoc:

* **Watermark poll (delta)** — every durable source commits its typed cursor in
  the same ``ApplyChangeEnvelope`` transaction as its graph change; the next run
  fetches only what changed.
* **Reconcile** — watermark deltas never surface deletions, so a reconcile compares
  the live id set with the KG's ``domain=<source>`` nodes and tombstones the gone.
* **Webhook narrowing** — a source's webhook can drive a sync for a specific set of
  ``ids`` (near-real-time).

Sources opt into **delta** by registering a handler in :data:`_DELTA_HANDLERS`
(LeanIX is the first). Any other registered source still syncs through this one
entrypoint — it just falls back to a **full hydrate** via the capability registry
until it grows a delta handler. This keeps the surface uniform while being honest
about which sources are incremental today.

**AU-P1-5 ChangeEnvelope consolidation** (CONCEPT:AU-KG.ingest.envelope-atomic-
transaction, :mod:`~..ingestion.envelope_ingest`): every durable delta handler,
materialize extractor, and generic capability hydrate routes normalized graph
material through ``ChangeEnvelope`` / ``ingest_graph_slice`` — one native
``ApplyChangeEnvelope`` redb/Raft commit for graph material, policy, lineage,
typed content version, source cursor, and CDC/projection outbox — an atomic
validate -> identity -> write -> lineage+checkpoint -> CDC -> watermark unit
that is crash-resume safe at PER-RECORD granularity (a batch call is
all-or-visible; a crash mid-batch under the old ad hoc
``engine.ingest_external_batch(domain, entities, relationships)`` model could
silently skip the watermark advance for objects that DID get written). A
native capability/session/persistence failure is fail-closed; it never
downgrades to the historical Python write sequence. Every durable handler in
:data:`ENVELOPE_NATIVE_SOURCES` commits through that boundary. The sole
:data:`ORCHESTRATION_ONLY_SOURCES` entry, ``package_install``, emits no graph
material itself; a static architecture gate proves it only delegates to its
owned package-reload orchestrator.

**Ambient epistemics (W3.4, CONCEPT:AU-KG.ingest.ambient-connector-provenance /
AU-KG.temporal.ambient-connector-valid-time).** Connector-ingested rows carry
epistemic value BY DEFAULT: each record's own reported timestamp becomes its
bitemporal ``valid_from`` (never fabricated —
:mod:`~..ingestion.envelope_ingest`'s ``_stamp_ambient_valid_time``), and every
:func:`_ingest_entities_via_envelope` call records ONE PROV-O Activity for that
run plus one summary ``:Claim`` ("source X said N records as of T") —
:mod:`~..etl.lineage`'s ``record_connector_sync_activity`` /
``record_connector_sync_claim`` — never a claim per row. Flag-gated
(``KG_AMBIENT_EPISTEMIC``, default ON; per-source opt-out via
``KG_AMBIENT_EPISTEMIC_DISABLED_SOURCES``); flag off reproduces this module's
pre-W3.4 behavior byte-for-byte.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from ...security.identifiers import validate_identifier
from ..backends.sparql.source_partition import make_source_id

logger = logging.getLogger(__name__)

# Actions that route to source sync (used by the scheduler dispatch).
SYNC_ACTIONS = {"delta", "full", "reconcile"}

# CONCEPT:AU-ORCH.scheduling.hard-io-deadline — the fleet MCP tool-schema probe
# (:func:`_sync_fleet`) is synchronous, network-bound work invoked INLINE from an
# async task body (``connector_sync``/``capability_hydration`` in the ``connectors``
# lane, and the ``fleet-tool-schema-sync`` scheduled job in ``maint``) — it does not
# hit an ``await`` the surrounding cooperative-cancellation watchdog can act on.
# Proven live: both call sites were observed exceeding their lane's soft timeout
# (180s / 600s) with the worker still wedged afterward. Bound it at BOTH layers:
# a cooperative ``budget`` into ``probe_catalog`` (cancels straggling per-server
# probes cleanly — the correct outcome) and a slightly longer hard ``timeout`` on
# the outer sync wait (the backstop for whatever that cooperative cancellation
# itself cannot unblock, e.g. a child-process spawn that blocks before its first
# ``await``). Sized as a fraction of the tightest lane bound (``connectors``,
# 180s) so the whole call reliably returns well inside it, in either lane.
_FLEET_PROBE_BUDGET_FRACTION = 0.6
_FLEET_PROBE_GRACE_SEC = 15.0


def _read_envelope_watermark(
    engine: Any,
    connector: str,
    *,
    source_instance: str = "",
) -> str | None:
    """Read the native cursor for a ChangeEnvelope-emitting handler.

    An absent or temporarily unavailable native cursor produces a safe full pull
    rather than elevating a separately-written watermark to authority.
    """
    from ..ingestion.envelope_ingest import (
        NativeChangeEnvelopeUnavailable,
        read_change_cursor,
    )

    try:
        return read_change_cursor(engine, connector, source_instance=source_instance)
    except NativeChangeEnvelopeUnavailable:
        logger.warning(
            "%s native source cursor is unavailable; using a safe full pull",
            connector,
        )
        return None


@functools.lru_cache(maxsize=64)
def _load_connector_manifest(connector: str) -> Any | None:
    """Best-effort ``ConnectorManifest`` load for ``connector`` (CA-22/P11 preflight).

    ``sync_source`` already fails closed on a missing/invalid manifest via
    ``connector_manifest_gate.precheck_source`` BEFORE any handler runs
    (D17), so by the time a handler calls :func:`_apply_with_preflight` a
    valid manifest is known to exist on disk for every source except the
    internal-exempt ones (``INTERNAL_MANIFEST_EXEMPT_SOURCES``). This helper
    re-resolves and parses it (cached per connector name for the process
    lifetime -- manifests are static release artifacts, not runtime state)
    so the preflight chokepoint can read ``conflict_policy``/``backfeed``
    without threading a manifest object through all 31 handler signatures.
    Returns ``None`` on any resolution/parse failure or for an exempt
    source -- callers MUST treat ``None`` the same as "no policy declared"
    (fail-closed defaults apply via :func:`_manifest_conflict_policy`/
    :func:`_manifest_backfeed`, never a bypass).
    """
    try:
        import yaml

        from ..ontology.connector_manifest import ConnectorManifest
        from ..ontology.connector_manifest_gate import find_connector_manifest

        path = find_connector_manifest(connector)
        if path is None:
            return None
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ConnectorManifest.model_validate(raw)
    except Exception:  # noqa: BLE001 - preflight manifest read is best-effort
        logger.debug("preflight manifest load failed for %s", connector, exc_info=True)
        return None


def _manifest_conflict_policy(manifest: Any | None) -> Any:
    """The connector's declared :class:`~..ontology.sync_conflict.ConflictPolicySpec`.

    ``ConnectorManifest`` does not carry a ``conflict_policy`` field on
    ``main`` today -- that field is CA-32's exclusive territory
    (``file-ownership.yaml`` ``FO-CA-013``), not this lane's
    (``sync_conflict.py``'s own module docstring records the coordination
    note). Read defensively via ``getattr`` so this activates automatically,
    with zero further change here, the moment CA-32 lands the field --
    until then every manifest resolves to the safe, fail-closed default
    (``default_policy="manual_review"``, no fields declared, so
    :meth:`ConflictPolicySpec.declares` is ``False`` for everything and
    :func:`_apply_with_preflight` compares nothing -- additive, byte-
    identical to pre-CA-22 behavior for every connector today).
    """
    from ..ontology.sync_conflict import ConflictPolicySpec

    spec = getattr(manifest, "conflict_policy", None)
    return spec if isinstance(spec, ConflictPolicySpec) else ConflictPolicySpec()


def _manifest_backfeed(manifest: Any | None) -> Any:
    """The connector's declared :class:`~..ontology.sync_conflict.BackfeedCapabilitySpec`.

    Same defensive-``getattr`` rationale as :func:`_manifest_conflict_policy`.
    """
    from ..ontology.sync_conflict import BackfeedCapabilitySpec

    spec = getattr(manifest, "backfeed", None)
    return (
        spec if isinstance(spec, BackfeedCapabilitySpec) else BackfeedCapabilitySpec()
    )


def _current_graph_fields(
    engine: Any, node_id: str, fields: Iterator[str] | list[str]
) -> dict[str, Any]:
    """Best-effort read of ``node_id``'s CURRENT stored values for ``fields``.

    Used only for the fields a connector's :class:`ConflictPolicySpec`
    explicitly declares (see :func:`_apply_with_preflight`) -- never a
    blanket per-record read. Mirrors the established
    ``enrichment.writeback.core.resolve_external_id`` pattern (a direct
    ``backend.execute`` Cypher read). Fails OPEN on the read itself -- a
    backend that can't be queried this way (fixture/test doubles, a
    non-Cypher backend) returns ``{}``, which :func:`_apply_with_preflight`
    treats as "no prior graph value to conflict with" (the write proceeds);
    the WRITE side stays fail-closed once a real conflict is detected.
    """
    fields = list(fields)
    if not fields or not node_id:
        return {}
    backend = getattr(engine, "backend", engine)
    execute = getattr(backend, "execute", None)
    if execute is None:
        return {}
    try:
        rows = execute("MATCH (n {id: $id}) RETURN n AS node LIMIT 1", {"id": node_id})
    except Exception:  # noqa: BLE001 - conflict-detection read is best-effort
        logger.debug(
            "preflight current-state read failed for node %s", node_id, exc_info=True
        )
        return {}
    if not rows:
        return {}
    node = rows[0].get("node") if isinstance(rows[0], dict) else None
    if not isinstance(node, dict):
        return {}
    return {f: node[f] for f in fields if f in node}


def _preflight_declared_fields(record: dict[str, Any], policy_spec: Any) -> list[str]:
    """The record fields the connector's ``ConflictPolicySpec`` explicitly declares.

    An undeclared field is never compared against the graph, which is why every
    handler is a no-op through the preflight today.
    """
    return [f for f in record if f != "id" and policy_spec.declares(f)]


def _preflight_record_conflict(
    record: dict[str, Any],
    current: dict[str, Any],
    declared_fields: list[str],
    policy_spec: Any,
    ctx: SimpleNamespace,
) -> Any | None:
    """The first :class:`SyncConflict` among a record's declared fields, if any.

    Per-record scope, not per-field partial commit: the first conflicting field
    blocks the WHOLE envelope (never a silent partial overwrite; "Prohibited
    fallback" invariant). ``ctx`` carries ``connector``/``node_id``/
    ``source_instance`` for the resolver's provenance.
    """
    from ..ontology.sync_conflict import SyncConflict, resolve_field_conflict

    for field in declared_fields:
        if field not in current:
            continue  # no prior graph value -- nothing to conflict with
        graph_value = current[field]
        source_value = record.get(field)
        if graph_value == source_value:
            continue  # agreement -- not a conflict
        resolved = resolve_field_conflict(
            policy_spec.policy_for(field),
            source_value,
            graph_value,
            connector=ctx.connector,
            node_id=ctx.node_id,
            field_name=field,
            source_instance=ctx.source_instance,
        )
        if isinstance(resolved, SyncConflict):
            return resolved
    return None


def _preflight_block(
    index: int,
    env: Any,
    conflict: Any,
    connector: str,
    node_id: str,
    backfeed_spec: Any,
) -> dict[str, Any]:
    """The ``blocked`` entry for one envelope a declared-field conflict stopped."""
    from ..ontology.sync_conflict import evaluate_backfeed_preflight

    outcome = evaluate_backfeed_preflight(
        connector=connector,
        node_id=node_id,
        conflict=conflict,
        backfeed=backfeed_spec,
    )
    return {
        "index": index,
        "envelope": env,
        "conflict_or_rejection": outcome if outcome is not None else conflict,
    }


def _apply_with_preflight(
    engine: Any,
    connector: str,
    batch: list[Any],
    *,
    manifest: Any | None = None,
    source_instance: str = "",
) -> tuple[list[Any], list[dict[str, Any]]]:
    """The ONE chokepoint every ``_DELTA_HANDLERS`` write path calls before its
    existing commit call (CONCEPT:AU-KG.ingest.backfeed-preflight, DEC-CA-07/P11).

    Takes the ``ChangeEnvelope`` batch a handler already built and returns
    ``(allowed, blocked)`` -- ``allowed`` is the (possibly narrower) list a
    caller commits EXACTLY as it does today (``ingest_envelope``/
    ``ingest_envelopes``, unchanged idempotency-key derivation, per the
    lane's idempotency invariant); ``blocked`` is
    ``[{"index": int, "envelope": env, "conflict_or_rejection": SyncConflict |
    BackfeedProposal | PreflightRejection}, ...]`` for the caller to log/
    surface as ``failed``/``skipped`` counts. ``index`` is the position in
    the INPUT ``batch`` (not ``allowed``) -- callers needing a contiguous
    prefix (checkpoint-ordered batches) use it directly rather than relying
    on list membership, since :class:`~..ingestion.change_envelope.ChangeEnvelope`
    is a value-equality dataclass and two distinct envelopes can compare equal.

    Per-record scope, not per-field partial commit: ``ChangeEnvelope``
    commits atomically as one node upsert, so this never partially applies
    a record -- if ANY explicitly-declared field on a record resolves to a
    :class:`~..ontology.sync_conflict.SyncConflict` under
    ``manual_review``/``reject``, the WHOLE envelope is blocked (never a
    silent partial overwrite; "Prohibited fallback" invariant).

    Only fields the connector's ``ConflictPolicySpec`` explicitly declares
    (:meth:`~..ontology.sync_conflict.ConflictPolicySpec.declares`) are ever
    diffed against the current graph value -- an undeclared field is never
    compared, so every one of the 31 handlers is a no-op through this
    function today (no manifest declares ``conflict_policy`` yet; see
    :func:`_manifest_conflict_policy`), reachable and AST-verifiable, and
    activates the instant an operator declares a field policy. A record
    with no ``id``/no declared fields present skips the read entirely
    (:func:`_current_graph_fields` short-circuits on an empty field list).
    """
    manifest = manifest if manifest is not None else _load_connector_manifest(connector)
    policy_spec = _manifest_conflict_policy(manifest)
    backfeed_spec = _manifest_backfeed(manifest)

    allowed: list[Any] = []
    blocked: list[dict[str, Any]] = []
    for index, env in enumerate(batch):
        record = env.to_entity_dict() if hasattr(env, "to_entity_dict") else {}
        node_id = str(record.get("id") or getattr(env, "source_object_id", "") or "")
        declared_fields = _preflight_declared_fields(record, policy_spec)
        current = (
            _current_graph_fields(engine, node_id, declared_fields) if node_id else {}
        )
        record_conflict = _preflight_record_conflict(
            record,
            current,
            declared_fields,
            policy_spec,
            SimpleNamespace(
                connector=connector,
                node_id=node_id,
                source_instance=source_instance,
            ),
        )
        if record_conflict is None:
            allowed.append(env)
            continue
        blocked.append(
            _preflight_block(
                index, env, record_conflict, connector, node_id, backfeed_spec
            )
        )
    return allowed, blocked


def _apply_with_preflight_one(
    engine: Any,
    connector: str,
    envelope: Any,
    *,
    manifest: Any | None = None,
    source_instance: str = "",
) -> tuple[Any | None, dict[str, Any] | None]:
    """Single-envelope convenience wrapper over :func:`_apply_with_preflight`.

    Returns ``(envelope, None)`` when clear to commit exactly as today, or
    ``(None, block_info)`` when blocked -- ``block_info`` is the same shape
    as one entry of :func:`_apply_with_preflight`'s ``blocked`` list.
    """
    allowed, blocked = _apply_with_preflight(
        engine,
        connector,
        [envelope],
        manifest=manifest,
        source_instance=source_instance,
    )
    if allowed:
        return allowed[0], None
    return None, blocked[0] if blocked else None


def _reconcile_allowed_empty_sources() -> set[str]:
    """Source keys allowed to tombstone on an authoritatively-empty snapshot (CONCEPT:AU-P0-4).

    ``SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE`` is a comma-separated allowlist of
    ``sync_source`` source keys (e.g. ``"leanix,twenty"``) an operator has
    explicitly confirmed CAN legitimately report a fully-empty authoritative
    snapshot (a real "everything was deleted upstream" event). Empty by
    default: no source tombstones on an empty live-id set unless named here,
    so a transient upstream/client hiccup that happens to yield zero ids can
    never wipe previously-known data.
    """
    from ...core.config import setting

    raw = setting("SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE", default="") or ""
    return {s.strip().lower() for s in str(raw).split(",") if s.strip()}


def _reconcile(
    engine: Any,
    source: str,
    live_ids: set[str],
    *,
    source_instance: str = "",
    fetch_ok: bool = True,
) -> dict[str, Any]:
    """Commit an authoritative snapshot decision through ApplyChangeEnvelope.

    CONCEPT:AU-P0-4 fail-closed reconcile — an EMPTY ``live_ids`` is
    ambiguous: it can mean the upstream authoritatively has zero live objects
    (a genuine "everything was deleted" event that SHOULD tombstone every
    previously-known node for this source), or that the live-id fetch itself
    failed/was skipped (a transient error that must NEVER be allowed to wipe
    data). Callers report which one happened via ``fetch_ok`` — ``False``
    means the fetch errored/was skipped, and reconcile always no-ops in that
    case regardless of policy. A *successful* empty fetch (``fetch_ok=True``)
    only tombstones when ``source`` is named in
    :func:`_reconcile_allowed_empty_sources` (``SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE``)
    — an explicit per-source opt-in. Default is conservative: skip. The engine
    enforces this policy server-side inside ``ChangeEnvelope.snapshot_complete``
    before it commits any tombstone. ``source_instance`` identifies the configured
    instance whose rows the marker is allowed to inspect; leaving it empty preserves
    the single-instance connector behavior.
    """
    from ..ingestion.change_envelope import ChangeEnvelope
    from ..ingestion.envelope_ingest import ingest_envelope

    snapshot_digest = hashlib.sha256(
        json.dumps(sorted(str(value) for value in live_ids)).encode("utf-8")
    ).hexdigest()
    marker = ChangeEnvelope.snapshot_complete(
        connector=source,
        source_instance=source_instance,
        live_ids=live_ids,
        fetch_ok=fetch_ok,
        source_version=snapshot_digest,
    )
    applied = ingest_envelope(engine, marker)
    if applied.get("status") not in {"success", "skipped"}:
        return {
            "status": "error",
            "live": len(live_ids),
            "tombstoned": 0,
            "reason": applied.get("error") or applied.get("status"),
        }
    writes = applied.get("write_result") or {}
    return {
        "status": "completed",
        "live": len(live_ids),
        "tombstoned": int(writes.get("tombstoned", 0) or 0),
    }


def _ingest_graph_slice_via_envelope(
    engine: Any,
    connector: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    source_instance: str = "",
    checkpoint: str | None = None,
    version_field: str = "updatedAt",
) -> dict[str, Any]:
    """Map source DTO kinds to canonical graph properties and commit atomically."""
    from ..ingestion.envelope_ingest import ingest_graph_slice

    canonical_entities = []
    for item in entities:
        row = dict(item)
        row["node_type"] = row.pop("type")
        canonical_entities.append(row)
    canonical_relationships = []
    for item in relationships or []:
        row = dict(item)
        row["relationship"] = row.pop("type")
        canonical_relationships.append(row)

    return ingest_graph_slice(
        engine,
        connector,
        canonical_entities,
        canonical_relationships,
        source_instance=source_instance,
        checkpoint=checkpoint,
        version_field=version_field,
    )


# ── Fleet capability elevation (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical) ────────────────────────────
#
# The ~62 fleet MCP servers' tools were AST-ingested as generic ``Code`` symbols
# and never elevated to capability nodes, so the KG lacked the fleet capability
# vocabulary: a query naming "portainer"/"github" matched no ``Tool`` node, so
# neither the ontology classification gate nor the dispatcher's specialist
# routing (``config._fetch_tools`` → ``MATCH (t:Tool)``) could act on it. This
# handler enumerates the **served multiplexer catalog** (the source of truth that
# already lists every fleet tool with name+description+owning server) and writes
# each tool as a ``Tool`` capability node linked to its ``MCPServer`` — fixing the
# classification gate AND the "no fleet specialist registered" hole with one pass.

# Suffix tokens stripped to recover a product/brand synonym from a server name
# (``portainer-agent`` → ``portainer``), mirroring how ``config.py``'s
# ``_synthesize_partition_agents`` derives its ``server_tag``.
_CAPABILITY_GENERIC_TOKENS = frozenset(
    {
        "mcp",
        "agent",
        "api",
        "server",
        "service",
        "manager",
        "tool",
        "tools",
        "client",
        "connector",
        "package",
    }
)
_CAPABILITY_NAME_SUFFIXES = (
    "-mcp",
    "_mcp",
    "-agent",
    "_agent",
    "-api",
    "_api",
    "-server",
    "_server",
    "-manager",
    "_manager",
    "-service",
    "_service",
)


def derive_capability_synonyms(server_name: str) -> list[str]:
    """Matchable terms for a fleet server, for the ontology lexical gate.

    Returns the full server name, its de-suffixed product name, and each
    non-generic token — so a chat turn naming the product ("portainer") matches
    the capability node even though the server is registered as
    ``portainer-agent``. Deterministic and embedding-free.
    """
    import re

    base = (server_name or "").lower().strip()
    if not base:
        return []
    product = base
    for suf in _CAPABILITY_NAME_SUFFIXES:
        if product.endswith(suf):
            product = product[: -len(suf)]
            break
    syns = {base}
    if product:
        syns.add(product)
    for tok in re.split(r"[-_\s]+", base):
        if len(tok) > 1 and tok not in _CAPABILITY_GENERIC_TOKENS:
            syns.add(tok)
    return sorted(syns)


def _capability_product(server_name: str) -> str:
    """The de-suffixed product tag (``portainer-agent`` → ``portainer``)."""
    syns = derive_capability_synonyms(server_name)
    base = (server_name or "").lower().strip()
    # prefer the de-suffixed product (the shortest synonym that is a prefix of base)
    for s in syns:
        if s != base and base.startswith(s):
            return s
    return base


# Verified labels of the three call sites that pass a bare node id here (the
# fleet-catalog write loop below builds ``entities`` with ``"type":
# "MCPServer"/"Tool"/"Skill"`` respectively, and ``_write_fleet_slice`` ->
# ``_ingest_graph_slice_via_envelope`` writes each row's ``type`` straight
# through as the node's Cypher label). An unlabeled ``MATCH (n)`` clones every
# node's property blob in the whole graph on every call (no id index exists
# in the Cypher engine, only a label index); trying these first turns the hot
# "once per tool during fleet registration" path into an indexed lookup.
_FLEET_DISABLED_LOOKUP_LABELS: tuple[str, ...] = ("MCPServer", "Tool", "Skill")


def _disabled_from_cache(engine: Any, node_id: str) -> bool | None:
    """The in-memory graph's ``disabled`` flag, or ``None`` when it has no such node."""
    gc = getattr(engine, "graph_compute", None)
    graph = getattr(gc, "graph", None)
    if graph is not None and node_id in graph:
        return bool(graph.nodes[node_id].get("disabled", False))
    return None


def _disabled_from_query(engine: Any, node_id: str, label: str | None) -> bool | None:
    """The ``disabled`` flag from a Cypher lookup, or ``None`` when nothing matched.

    ``label=None`` is the unlabeled correctness fallback for an id outside the
    verified fleet label set.
    """
    if label is None:
        query = "MATCH (n) WHERE n.id = $id RETURN n.id AS id, n.disabled AS disabled"
    else:
        safe_label = validate_identifier(label, kind="label")
        query = (
            f"MATCH (n:{safe_label}) WHERE n.id = $id "
            "RETURN n.id AS id, n.disabled AS disabled"
        )
    rows = engine.query_cypher(query, {"id": node_id})
    if rows and isinstance(rows, list) and len(rows) > 0:
        return bool(rows[0].get("disabled", False))
    return None


def _existing_disabled(engine: Any, node_id: str) -> bool:
    """Best-effort read of a node's ``disabled`` flag so a re-sync preserves an
    operator's manual disable (mirrors ``kg_server.get_existing_disabled`` without
    creating a knowledge_graph → mcp import inversion).

    Fail-closed: this flag feeds an enable/disable decision written straight
    back into the node on re-sync, so a lookup that could not complete (an
    exception from the in-memory cache or ``query_cypher``) returns ``True``
    (treat as disabled) instead of silently defaulting to "not disabled" —
    the prior contract here conflated "confirmed not disabled" with "the
    check itself failed", which would silently re-enable an operator's
    manual disable on any transient engine hiccup during a re-sync. A
    genuinely absent node (every query executed successfully and found
    nothing — a brand-new node with no prior state) still returns ``False``;
    that is not a failure.
    """
    try:
        cached = _disabled_from_cache(engine, node_id)
        if cached is not None:
            return cached
        for candidate_label in _FLEET_DISABLED_LOOKUP_LABELS:
            labelled = _disabled_from_query(engine, node_id, candidate_label)
            if labelled is not None:
                return labelled
        # Correctness fallback for a node outside the verified fleet label
        # set above — the original (unoptimized) cost, only paid for ids
        # this loop doesn't already know the label of.
        unlabelled = _disabled_from_query(engine, node_id, None)
        if unlabelled is not None:
            return unlabelled
    except Exception as exc:  # noqa: BLE001 — surfaced as a fail-closed True below
        logger.error(
            "_existing_disabled(%s) lookup failed — failing closed "
            "(treating as disabled): %s",
            node_id,
            type(exc).__name__,
        )
        return True
    return False


def _derive_tool_mode(input_schema: dict | None) -> str:
    """Classify a served tool as ``condensed`` or ``verbose`` (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical).

    A probed catalog may expose a *condensed* action-routed tool (one tool with
    ``action`` + ``params_json``) and *verbose* 1:1 tools (one typed tool per
    operation). Both are ingested as distinct ``Tool`` nodes; tagging the
    variant lets selection/analytics prefer the right altitude (condensed for broad, verbose
    for a specific operation) instead of guessing from the name.
    """
    props = (input_schema or {}).get("properties")
    if isinstance(props, dict) and "action" in props and "params_json" in props:
        return "condensed"
    return "verbose"


def _entity_rows(
    records: list[Any], build: Callable[[Any], Any]
) -> list[dict[str, Any]]:
    """Every record ``build`` could turn into an entity (unidentifiable → dropped).

    The shared record→entity fan-out for the connector handlers: each source owns
    its own per-record builder, and this owns the "drop what has no identity" rule.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        entity = build(record)
        if entity is not None:
            rows.append(entity)
    return rows


def _privacy_safe(text: str) -> str:
    """Redact fleet-supplied prose before it becomes graph state.

    A child's tool/skill descriptions are EXTERNAL material: they routinely
    embed absolute paths and other host detail. The native
    ``ApplyChangeEnvelope`` commit rejects (does not redact) such inline text,
    so a single offending description used to fail the ENTIRE fleet slice —
    every tool and skill from every reachable server — with the unactionable
    message "persistence privacy policy rejected inline text". Redacting here
    mirrors what :func:`~..ingestion.skill_workflow_ingest.ingest_runnable_skill`
    already does for skill bodies, so one noisy description degrades to a
    redacted description instead of dropping the whole sync.
    """
    from ...security.persistence_privacy import PersistencePrivacyGuard

    safe, _privacy = PersistencePrivacyGuard().sanitize_text(str(text or ""))
    return safe


@contextmanager
def _fresh_write_authority() -> Iterator[None]:
    """Re-mint this process's verified write authority for the write phase.

    CONCEPT:AU-OS.identity.authenticated-identity-enforcement. The fleet
    probe is minutes-long network work across the whole fleet, and the
    bearer JWT behind a :class:`~.session.GraphSession` has a Keycloak
    access-token lifetime measured in minutes. A sync that bound its session
    BEFORE the probe therefore routinely reaches the write phase holding an
    already-EXPIRED authority — measured live 2026-08-26: the probe
    succeeded for all 66 servers, then every single write failed
    (11,032 KG rows rejected with ``SessionExpiredError`` and the relational
    catalog write skipped with "Verified graph authority has expired").

    ``suspend_session()`` first, for exactly the reason
    ``gateway/registry_api._catalog_service_session`` documents: a bare
    :func:`~...security.request_identity.system_write_session` call PREFERS
    an already-bound ambient session, so without suspending it would hand
    back the very expired session this is replacing. Best-effort — if the
    authority cannot be re-minted, the block still runs under whatever was
    already bound, exactly as before.
    """
    from ...security.brain_context import use_actor
    from ...security.request_identity import system_write_session
    from .session import suspend_session, use_session

    try:
        with suspend_session():
            session = system_write_session()
    except Exception as exc:  # noqa: BLE001 — re-minting is best-effort
        logger.warning(
            "could not re-mint write authority before the fleet write (%s: %s); "
            "continuing with the ambient session",
            type(exc).__name__,
            exc,
        )
        yield
        return
    with use_actor(session.actor), use_session(session):
        yield


def _write_fleet_relational(
    engine: Any,
    catalog: dict[str, dict],
    *,
    configs: dict[str, dict] | None = None,
    discovery_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mirror the probed ``catalog`` into the relational fleet-catalog tables.

    CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables. Runs FIRST, from the
    SAME probed ``catalog`` the KG entities below are built from, so the
    relational rows and the KG nodes can never observe a different fleet
    state. This is the cheap, synchronous half the frontend should read —
    best-effort and independently wrapped so a failure here (e.g. the engine
    SQL surface is unavailable) is reported but never blocks the KG write
    that follows, and — just as important — the reverse: the KG write's own
    ACL gate (``IsolationLayer::check_access`` on
    ``tenant__homelab____commons__``) is a SEPARATE, currently-broken gate
    this write does not share (see ``fleet_catalog_tables`` module docstring),
    so this table gets populated even while that Cypher write is denied.
    """
    from .fleet_catalog_tables import write_fleet_catalog

    try:
        return write_fleet_catalog(
            engine,
            catalog,
            configs=configs,
            discovery_bindings=discovery_bindings,
        )
    except Exception as exc:  # noqa: BLE001 — relational write is best-effort
        logger.error(
            "fleet catalog relational write failed (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return {"status": "error", "reason": str(exc)}


def _fleet_tool_entity(
    engine: Any,
    entry: Any,
    server_name: str,
    product: str,
    synonyms: list[str],
) -> dict[str, Any] | None:
    """One probed tool as a ``Tool`` capability node, or ``None`` when unnamed."""
    if not isinstance(entry, dict):
        return None
    tool_name = entry.get("name")
    if not tool_name:
        return None
    tool_node_id = f"tool_{server_name}_{tool_name}"
    return {
        "id": tool_node_id,
        "type": "Tool",
        "name": tool_name,
        "description": _privacy_safe(entry.get("description", "")),
        "mcp_server": server_name,
        "tags": [product] if product else [],
        # ``ToolShape`` (governance.shapes.ttl) requires minCount 1
        # ``capabilityCategory``, and it was never written — so the
        # SHACL gate rejected the WHOLE fleet slice, silently: the
        # rejection surfaced only as the class name "ValueError".
        # The category is the de-suffixed product this server
        # provides (``servicenow-mcp`` → ``servicenow``), falling
        # back to the server name so it is never empty.
        "capabilityCategory": product or server_name,
        "relevance_score": 50,
        "requires_approval": False,
        "synonyms": synonyms,
        "kind": "mcp_tool",
        "tool_mode": _derive_tool_mode(entry.get("inputSchema")),
        "disabled": _existing_disabled(engine, tool_node_id),
    }


def _fleet_skill_entity(
    engine: Any,
    entry: Any,
    server_name: str,
    product: str,
    synonyms: list[str],
) -> dict[str, Any] | None:
    """One probed Skills-over-MCP resource as a ``Skill`` node, or ``None``."""
    from ..ingestion.skill_workflow_ingest import skill_reference

    if not isinstance(entry, dict):
        return None
    skill_name = entry.get("name")
    if not skill_name:
        return None
    skill_node_id = f"skill_{server_name}_{skill_name}"
    skill_props: dict[str, Any] = {
        "id": skill_node_id,
        "type": "Skill",
        "name": skill_name,
        "description": _privacy_safe(entry.get("description", "")),
        "mcp_server": server_name,
        "tags": [product] if product else [],
        "relevance_score": 0.5,
        "requires_approval": False,
        "synonyms": synonyms,
        "kind": "mcp_skill",
        "disabled": _existing_disabled(engine, skill_node_id),
        # CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — WHY this
        # fleet skill is (or is not) runnable, recorded on the node
        # so the execution path can name the unmet precondition
        # instead of failing with a generic "not found or runnable".
        "runnable_blocked_by": _privacy_safe(entry.get("harvest_error", "")),
    }
    # ``skill://<name>`` collides with the persistence-privacy policy's
    # posix-path heuristics whenever a skill name starts with a
    # filesystem-root token (``opt``ions-…, ``workspace``-manager,
    # ``tmp``…, ``var``…). The native ApplyChangeEnvelope commit REJECTS
    # such text, and a rejection fails the WHOLE slice — so six oddly
    # named skills silently cost every tool and skill from every
    # reachable server. Writing a redacted ref instead would be worse:
    # it is no longer a ``skill://`` reference at all, which is the
    # contract every ranking consumer checks. So the ref is omitted and
    # the reason logged; the skill's canonical runnable identity comes
    # from the promotion path, which does not use this envelope.
    source_ref = skill_reference(skill_name)
    if _privacy_safe(source_ref) == source_ref:
        skill_props["source_ref"] = source_ref
    else:
        logger.warning(
            "Fleet skill %s omits its source_ref: %r trips the "
            "persistence privacy policy's local-path heuristic",
            skill_name,
            source_ref,
        )
    return skill_props


def _fleet_server_slice(
    engine: Any, server_name: str, info: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The ``MCPServer`` node plus every ``Tool``/``Skill`` node one server serves."""
    tools = info.get("tools") or []
    skills = info.get("skills") or []
    if not tools and not skills:
        return [], []

    synonyms = derive_capability_synonyms(server_name)
    product = _capability_product(server_name)
    server_node_id = f"mcp_server_{server_name}"
    entities: list[dict[str, Any]] = [
        {
            "id": server_node_id,
            "type": "MCPServer",
            "name": server_name,
            "synonyms": synonyms,
            "disabled": _existing_disabled(engine, server_node_id),
        }
    ]
    relationships: list[dict[str, Any]] = []

    for entry in tools:
        entity = _fleet_tool_entity(engine, entry, server_name, product, synonyms)
        if entity is None:
            continue
        entities.append(entity)
        relationships.append(
            {"source": server_node_id, "target": entity["id"], "type": "SERVES"}
        )

    for entry in skills:
        entity = _fleet_skill_entity(engine, entry, server_name, product, synonyms)
        if entity is None:
            continue
        entities.append(entity)
        relationships.append(
            {"source": server_node_id, "target": entity["id"], "type": "SERVES"}
        )

    return entities, relationships


def _fleet_catalog_slice(
    engine: Any, catalog: dict[str, dict] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Project a probed catalog into ``(entities, relationships, unreachable)``.

    A server that reported an ``error`` is recorded as unreachable and contributes
    no nodes; coverage is "the currently registered + reachable fleet".
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    unreachable: dict[str, str] = {}

    for server_name, info in (catalog or {}).items():
        if not isinstance(info, dict):
            continue
        err = info.get("error")
        if err:
            unreachable[server_name] = str(err)
            continue
        server_entities, server_relationships = _fleet_server_slice(
            engine, server_name, info
        )
        entities.extend(server_entities)
        relationships.extend(server_relationships)

    return entities, relationships, unreachable


def _fleet_type_counts(entities: list[dict[str, Any]]) -> dict[str, int]:
    """How many ``MCPServer``/``Tool``/``Skill`` nodes the slice carries."""
    counts = {"MCPServer": 0, "Tool": 0, "Skill": 0}
    for item in entities:
        if item["type"] in counts:
            counts[item["type"]] += 1
    return counts


def _write_fleet_nodes(
    engine: Any,
    catalog: dict[str, dict],
    *,
    configs: dict[str, dict] | None = None,
    discovery_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a probed multiplexer catalog into the KG as capability nodes.

    ``catalog`` is the ``{server: {"tools": [{name, description, ...}],
    "skills": [{name, uri, description}], "error": str|None}}`` map returned by
    :meth:`MCPMultiplexer.probe_catalog` (``skills`` is the Skills-over-MCP
    subset of a probed server's Resources, CONCEPT:AU-ECO.mcp.skills-over-mcp-provider — absent/empty for
    a fastmcp-3 or pre-skills server). For each reachable server every tool
    becomes a ``Tool`` node and every ``skill://{name}/SKILL.md`` resource
    becomes a ``Skill`` node, both carrying the schema the dispatcher/ranker
    reads (``name``, ``description``, ``mcp_server``, ``tags``,
    ``relevance_score``, ``requires_approval``) plus ``synonyms`` for the
    lexical gate, linked to their (defensively upserted) ``MCPServer`` via
    ``SERVES``. A fleet ``Skill`` node is id-namespaced per-server
    (``skill_{server}_{name}``), distinct from a locally-executed skill's
    canonical ``skill:<slug>`` identity (:func:`~..ingestion.skill_workflow_ingest.skill_reference`)
    so a thin fleet probe write can never clobber a richer in-loop skill's
    ``body``/``instruction`` properties — ``kind='mcp_skill'`` plus
    ``source_ref='skill://<slug>'`` still lets ranking recognize both as the
    same capability *kind*. Idempotent: stable node ids + the write-layer
    content-hash delta (over the whole probed slice) skip unchanged
    tools/skills on re-sync. Factored out of :func:`_sync_fleet` so it is
    testable without spawning any servers.
    """
    # Relational write FIRST (see CONCEPT:AU-KG.ingest.fleet-catalog-relational-tables
    # docstring on :func:`_write_fleet_relational`) — cheap, synchronous, and
    # independent of the KG write below succeeding, failing, or being rejected
    # by the engine's Cypher ACL gate.
    relational = _write_fleet_relational(
        engine,
        catalog,
        configs=configs,
        discovery_bindings=discovery_bindings,
    )

    entities, relationships, unreachable = _fleet_catalog_slice(engine, catalog)

    # CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — promotion runs BEFORE the
    # catalog slice write, and deliberately so. The catalog write is ONE native
    # ChangeEnvelope over every server's every tool and skill: it is atomic, so
    # a single rejected row (observed live: one tool description whose prose
    # about credential handling tripped the engine's persistence-privacy policy)
    # discards the whole slice. Making a fleet skill RUNNABLE must not be
    # hostage to that all-or-nothing write — promotion writes per skill through
    # ``ingest_runnable_skill`` and fails closed per skill, so one noisy
    # description can no longer cost the entire fleet its runnable capability.
    harvest = _promote_fleet_skills(engine, catalog)
    # CONCEPT:AU-ECO.mcp.cross-process-prompt-harvest — promotes prompts
    # harvested from fleet MCP children into the Prompt corpus, same rationale
    # as the skill promotion above applied to prompt bodies: runs BEFORE the
    # catalog slice write and fails closed per prompt via ``ingest_prompt_node``,
    # so one malformed fleet prompt can never cost the rest of the corpus.
    harvest.update(_promote_fleet_prompts(engine, catalog))

    rejected: list[str] = []
    materialization_pending: list[str] = []
    if entities:
        rejected, materialization_pending = _write_fleet_slice(
            engine, entities, relationships
        )

    written = _fleet_type_counts(entities)
    return {
        # Genuinely rejected — the engine judged the row's content unacceptable
        # (privacy policy, validation); cached so future syncs skip re-deriving it.
        "catalog_rows_rejected": len(rejected),
        "catalog_rejected_ids": rejected,
        # Retried against a still-materializing engine and gave up within the
        # bounded budget (envelope_ingest.ingest_envelope) — a THIS-SYNC-ONLY
        # omission, never cached, distinct from a genuine rejection above.
        "catalog_rows_materialization_pending": len(materialization_pending),
        "catalog_materialization_pending_ids": materialization_pending,
        "servers_written": written["MCPServer"],
        "tools_written": written["Tool"],
        "skills_written": written["Skill"],
        "unreachable": unreachable,
        "relational": relational,
        **harvest,
    }


_REJECTED_ROW_CACHE_FILE = "fleet_sync_rejected_rows.json"


def _row_content_hash(row: dict[str, Any]) -> str:
    """Stable content fingerprint for one catalog row (order-independent)."""
    payload = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rejected_row_cache_path():
    from ...core.paths import cache_dir

    return cache_dir() / _REJECTED_ROW_CACHE_FILE


def _load_rejected_row_cache() -> dict[str, str]:
    """Return ``{row_id: content_hash}`` for rows known-bad from a prior sync.

    CONCEPT:AU-KG.ingest.fleet-sync-rejected-row-cache — best-effort local
    cache, never authoritative: a missing/corrupt/unreadable file degrades to
    "nothing known bad yet", never an error.
    """
    try:
        path = _rejected_row_cache_path()
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    except Exception as e:  # noqa: BLE001 — cache is a pure optimization; any read failure just means every row is re-attempted this sync, same as before this cache existed
        logger.debug("fleet_sync: rejected-row cache read failed: %s", e)
        return {}


def _save_rejected_row_cache(cache: dict[str, str]) -> None:
    try:
        path = _rejected_row_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — cache is a pure optimization; a write failure only costs a repeat bisection next sync, never a correctness issue
        logger.debug("fleet_sync: rejected-row cache write failed: %s", e)


def _write_fleet_slice(
    engine: Any, entities: list[dict[str, Any]], relationships: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Write the fleet catalog slice, isolating rows the engine rejects.

    The native ``ApplyChangeEnvelope`` commit is ATOMIC, so one unacceptable row
    discards the entire slice — observed live: a single ServiceNow tool whose
    description explains its credential handling tripped the engine's
    persistence-privacy policy and cost every tool and skill of every reachable
    fleet server. The engine reports rejection for the slice, not the row, and
    that policy lives engine-side (it is NOT reproducible from the Python
    ``PersistencePrivacyGuard``, which leaves that description unchanged), so
    the offender can only be identified by bisection.

    D-SH-2 (``reports/deferred/lane-skill-harvest.md``): bisection alone repeats
    the SAME O(k log n) discovery cost on every sync, because the offending
    row is re-derived from scratch each time. A row previously confirmed
    rejected (by its id AND an exact content hash — so an edited row is never
    stuck on a stale verdict) is pre-excluded from THIS attempt entirely: the
    common steady-state case (no NEW offenders since last sync) then commits
    the whole remaining slice in one shot, no bisection at all. Bisection
    still runs (only) over rows not already known-bad, so it discovers a truly
    NEW offender at the same O(k log n) cost this always had — just no longer
    PAID every sync for offenders already known about.

    On a GENUINE rejection this halves the slice and retries, isolating each
    offending row, dropping ONLY those. Cost is O(k log n) commits for k
    *newly* discovered offenders. A rejected row is logged at error so it is
    never a silent omission.

    A row the engine could not commit only because it was still
    mid-materialization (``ingest_envelope``'s own bounded resume, see
    ``envelope_ingest.PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED_MARKER``, gave
    up within its budget) is a DIFFERENT outcome from a genuine content
    rejection and is handled without bisecting further: splitting would only
    re-pay the same bounded wait against the same still-materializing engine
    state for every half, turning one shared, transient condition into
    O(k log n) repeats of it. Instead the WHOLE batch still pending at that
    point is logged once (at warning, not error — it was never judged
    unacceptable) and reported back separately; it is never written to the
    known-bad cache, so the next sync re-attempts it with a clean slate.

    Returns ``(rejected_ids, materialization_pending_ids)``: ``rejected_ids``
    are rows genuinely rejected (known-bad rows from the cache are included,
    so callers see the full current exclusion set); ``materialization_pending_ids``
    are rows retried and given up on for THIS sync only.
    """
    from ..ingestion.envelope_ingest import (
        PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED_MARKER,
    )

    cache = _load_rejected_row_cache()
    known_bad: list[dict[str, Any]] = []
    to_attempt: list[dict[str, Any]] = []
    for row in entities:
        row_id = str(row.get("id"))
        if cache.get(row_id) == _row_content_hash(row):
            known_bad.append(row)
        else:
            to_attempt.append(row)

    def _attempt(rows: list[dict[str, Any]]) -> tuple[bool, bool]:
        """Returns ``(succeeded, gave_up_on_materialization)``."""
        by_id = {row["id"] for row in rows}
        edges = [
            edge
            for edge in relationships
            if edge["source"] in by_id and edge["target"] in by_id
        ]
        try:
            _ingest_graph_slice_via_envelope(
                engine, "fleet", rows, edges, source_instance="catalog"
            )
        except Exception as exc:  # noqa: BLE001 — retried by bisection below;
            # the final per-row failure is logged with its reason by the caller.
            gave_up_on_materialization = (
                PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED_MARKER in str(exc)
            )
            logger.debug(
                "fleet catalog slice of %d row(s) rejected (%s: %s)",
                len(rows),
                type(exc).__name__,
                exc,
            )
            return False, gave_up_on_materialization
        return True, False

    newly_rejected: list[str] = []
    materialization_pending: list[str] = []

    def _bisect(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        succeeded, gave_up_on_materialization = _attempt(rows)
        if succeeded:
            return
        if gave_up_on_materialization:
            for row in rows:
                row_id = str(row.get("id"))
                logger.warning(
                    "fleet catalog row %s retried against a still-materializing "
                    "engine and gave up within its bounded budget; it is "
                    "omitted from THIS sync only (not cached as rejected) and "
                    "will be re-attempted next sync",
                    row_id,
                )
                materialization_pending.append(row_id)
            return
        if len(rows) == 1:
            row_id = str(rows[0].get("id"))
            logger.error(
                "fleet catalog row %s was REJECTED by the engine and is omitted "
                "from the catalog; every other row was still written",
                row_id,
            )
            newly_rejected.append(row_id)
            cache[row_id] = _row_content_hash(rows[0])
            return
        middle = len(rows) // 2
        _bisect(rows[:middle])
        _bisect(rows[middle:])

    if known_bad:
        logger.debug(
            "fleet catalog: pre-excluding %d row(s) already known-bad from a "
            "prior sync (unchanged content) — attempting %d row(s)",
            len(known_bad),
            len(to_attempt),
        )
    _bisect(to_attempt)

    if newly_rejected:
        _save_rejected_row_cache(cache)

    known_bad_ids = [str(row.get("id")) for row in known_bad]
    return known_bad_ids + newly_rejected, materialization_pending


def _promote_fleet_skills(engine: Any, catalog: dict[str, dict]) -> dict[str, Any]:
    """Promote harvested fleet skills, reporting the outcome under its own keys."""
    from ..ingestion.fleet_skill_harvest import promote_harvested_skills

    report = promote_harvested_skills(engine, catalog)
    return {
        "skills_promoted": report["promoted"],
        "skills_promoted_names": report["promoted_skills"],
        "skills_blocked": report["blocked"],
        "skills_blocked_detail": report["blocked_detail"],
        "skills_promote_errors": report["errors"],
        "skills_promote_error_detail": report["error_detail"],
    }


def _promote_fleet_prompts(engine: Any, catalog: dict[str, dict]) -> dict[str, Any]:
    """Promote harvested fleet prompts, reporting the outcome under its own keys."""
    from ..ingestion.fleet_prompt_harvest import promote_harvested_prompts

    report = promote_harvested_prompts(engine, catalog)
    return {
        "prompts_promoted": report["promoted"],
        "prompts_promoted_ids": report["promoted_prompts"],
        "prompts_blocked": report["blocked"],
        "prompts_blocked_detail": report["blocked_detail"],
        "prompts_promote_errors": report["errors"],
        "prompts_promote_error_detail": report["error_detail"],
    }


def _fleet_config_candidates() -> list[Any]:
    """The ``mcp_config.json`` candidates, in connector-convention order.

    ``MCP_CONFIG`` → ``WORKSPACE_PATH/mcp_config.json`` → the multiplexer's own
    default search, so resolution stays deployment-agnostic (genesis sets the env).
    """
    from pathlib import Path

    from ...core.config import setting

    candidates: list[Path] = []
    configured = (setting("MCP_CONFIG", default="") or "").strip()
    if configured:
        candidates.append(Path(configured))
    ws = (setting("WORKSPACE_PATH", default="") or "").strip()
    if ws:
        candidates.append(Path(ws) / "mcp_config.json")
    try:
        from ...mcp.multiplexer import _resolve_config_path

        rp = _resolve_config_path(None)
        if rp is not None:
            candidates.append(rp)
    except Exception:  # noqa: BLE001 — multiplexer default search is a fallback
        pass
    return candidates


def _fleet_config_has_servers(path: Any) -> bool:
    """Whether a candidate file actually parses to ≥1 ``mcpServers`` entry.

    An empty/placeholder file (e.g. a 0-byte ``~/.gemini/antigravity/
    mcp_config.json``) is skipped rather than silently yielding a 0-server probe.
    """
    import json as _json

    try:
        if path.exists() and path.stat().st_size > 0:
            data = _json.loads(path.read_text(encoding="utf-8"))
            return bool(data.get("mcpServers"))
    except Exception:  # noqa: BLE001 — skip unreadable/invalid candidates
        return False
    return False


def _resolve_fleet_config():
    """Resolve the fleet ``mcp_config.json`` — the one the multiplexer serves.

    Returns the first candidate that actually parses to ≥1 ``mcpServers`` entry,
    so an empty/placeholder file (e.g. a 0-byte ``~/.gemini/antigravity/
    mcp_config.json``) is skipped rather than silently yielding a 0-server probe.
    Order follows the connector convention (``MCP_CONFIG`` →
    ``WORKSPACE_PATH/mcp_config.json``) before the multiplexer's own default
    search, so it stays deployment-agnostic (genesis sets the env).
    """
    for path in _fleet_config_candidates():
        if _fleet_config_has_servers(path):
            return path
    return None


def _declared_fleet_services() -> list[dict[str, Any]] | None:
    """The declared fleet universe from ``deploy/mcp-fleet.registry.yml``.

    ``None`` when the registry is absent or unparsable — this reconcile is purely
    informational and a broken/missing file must never block a sync.
    """
    try:
        import yaml

        from ...orchestration.fleet_reconciler import resolve_registry_path

        path = resolve_registry_path()
        if path is None:
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [
            s
            for s in (data.get("services") or [])
            if isinstance(s, dict) and s.get("name")
        ]
    except Exception:  # noqa: BLE001 — registry reconcile is informational only
        return None


def _reconcile_declared_fleet(catalog: dict[str, dict] | None) -> dict[str, Any] | None:
    """Best-effort reconcile the probed catalog against the DECLARED fleet universe.

    ``deploy/mcp-fleet.registry.yml`` (the ~62-server desired-state manifest) was,
    until now, read only by the k8s reconciler — ingestion had no notion of "the
    declared universe" to compare probe coverage against. This makes that
    coverage visible: a registry entry is *covered* when either its ``name`` or
    its ``package`` (they diverge for ~half the fleet, e.g. ``github-mcp`` /
    ``github-agent``) appears as a probed ``mcp_config.json`` server key —
    mirroring the exact name-or-package membership check
    :func:`_sync_fleet_connectors` already uses for this same registry/config
    naming mismatch. Purely additive/informational: never raises, never affects
    which nodes get written, and returns ``None`` (added onto nothing) when the
    registry is absent or unparsable so a broken/missing file never blocks a sync.
    """
    services = _declared_fleet_services()
    if services is None:
        return None

    probed = set(catalog or {})
    uncovered = sorted(
        str(svc["name"])
        for svc in services
        if str(svc["name"]) not in probed
        and str(svc.get("package") or "") not in probed
    )
    return {
        "declared_total": len(services),
        "declared_uncovered": uncovered,
    }


def _fleet_probe_budget() -> float:
    """The cooperative probe budget, sized from the tightest lane's soft timeout."""
    from .task_lanes import lane_soft_timeout

    return max(10.0, lane_soft_timeout("connectors") * _FLEET_PROBE_BUDGET_FRACTION)


def _fleet_mux_metadata(mux: Any, catalog: dict[str, Any] | None) -> tuple[Any, Any]:
    """Best-effort ``(transport configs, discovery bindings)`` for a probed fleet."""
    try:
        # Best-effort transport/url metadata for the relational ``mcp_servers``
        # rows (see ``fleet_catalog_tables.write_fleet_catalog``); never fatal to
        # the sync if the config can't be re-read.
        configs = mux.load_catalog()
    except Exception:  # noqa: BLE001 — server-row transport/url is best-effort
        configs = None

    try:
        # Broker authority is process-owned multiplexer state, never a field in
        # the caller-visible catalog.  The identity-bound lookup also rejects
        # copied/spoofed catalog dictionaries.  Minting a tenant-local binding
        # reads the ambient session, which the probe may have outlived -- see
        # :func:`_fresh_write_authority`.
        with _fresh_write_authority():
            mux._bind_local_discovery_bindings(catalog or {})
            discovery_bindings = mux._take_discovery_bindings(catalog or {})
    except Exception:  # noqa: BLE001 - private binding metadata is optional
        discovery_bindings = None
    return configs, discovery_bindings


def _probe_fleet_catalog() -> SimpleNamespace:
    """Build the multiplexer from ``mcp_config.json`` and probe the served catalog.

    Returns ``SimpleNamespace(skip, catalog, configs, discovery_bindings)`` — a
    non-``None`` ``skip`` is a ready-made handler result (the multiplexer is
    optional at import, and a probe failure is never fatal to the caller).
    """
    try:
        from ...mcp.multiplexer import MCPMultiplexer
        from ...protocols.source_connectors.connectors.mcp_package import _run_async
    except Exception as exc:  # noqa: BLE001 — multiplexer optional at import
        return SimpleNamespace(
            skip={
                "status": "skipped",
                "source": "fleet",
                "reason": f"multiplexer unavailable: {exc}",
            }
        )

    config_path = _resolve_fleet_config()
    if config_path is None:
        return SimpleNamespace(
            skip={
                "status": "skipped",
                "source": "fleet",
                "reason": "no mcp_config.json with servers found",
            }
        )

    probe_budget = _fleet_probe_budget()
    try:
        mux = MCPMultiplexer(config_path)
        catalog = _run_async(
            mux.probe_catalog(budget=probe_budget),
            timeout=probe_budget + _FLEET_PROBE_GRACE_SEC,
        )
    except Exception as exc:  # noqa: BLE001 — probe is best-effort
        return SimpleNamespace(
            skip={"status": "error", "source": "fleet", "reason": str(exc)}
        )

    configs, discovery_bindings = _fleet_mux_metadata(mux, catalog)
    return SimpleNamespace(
        skip=None,
        catalog=catalog,
        configs=configs,
        discovery_bindings=discovery_bindings,
    )


def _sync_fleet(
    engine: Any, *, mode: str = "full", ids: list[str] | None = None, client: Any = None
) -> dict[str, Any]:
    """Elevate fleet MCP-server tools to KG capability nodes (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical).

    Probes the served multiplexer catalog (each fleet server's real tools, via a
    bounded connect→list_tools→release sweep) and writes them as ``Tool``
    capability nodes. ``client`` may inject a pre-probed catalog dict (tests /
    callers that already hold one); otherwise the multiplexer is built from the
    fleet ``mcp_config.json`` and probed. Unreachable servers are recorded, never
    fatal — coverage is "the currently registered + reachable fleet". When
    ``deploy/mcp-fleet.registry.yml`` resolves, the result also carries
    ``declared_total``/``declared_uncovered`` (:func:`_reconcile_declared_fleet`)
    so the declared universe is visible alongside what was actually probed.
    """
    # CA-22/P11: document-shaped delegated pipeline (commits Tool capability nodes
    # via its own write path below, not a ChangeEnvelope built here) -- AST-
    # reachability call to the preflight chokepoint; batch=[] checks nothing
    # per-record until a manifest declares conflict_policy for "fleet".
    _apply_with_preflight(engine, "fleet", [])

    probe = (
        SimpleNamespace(
            skip=None, catalog=client, configs=None, discovery_bindings=None
        )
        if isinstance(client, dict)
        else _probe_fleet_catalog()
    )
    if probe.skip is not None:
        return probe.skip
    catalog = probe.catalog

    with _fresh_write_authority():
        counts = _write_fleet_nodes(
            engine,
            catalog,
            configs=probe.configs,
            discovery_bindings=probe.discovery_bindings,
        )
    return {
        "status": "ok",
        "source": "fleet",
        "mode": mode,
        "delta_capable": True,
        "servers_seen": len(catalog or {}),
        **counts,
        **(_reconcile_declared_fleet(catalog) or {}),
    }


# ── LeanIX delta handler (the first delta-capable source) ────────────────────


def _leanix_reconcile(engine: Any, client: Any) -> dict[str, Any]:
    """Reconcile the KG's LeanIX slice against the live fact-sheet id set.

    CONCEPT:AU-P0-4: track whether the live-id fetch actually succeeded — an
    exception here must NOT be silently indistinguishable from a legitimate
    authoritatively-empty snapshot (both used to collapse to ``live = set()``,
    and an empty set used to always tombstone).
    """
    live: set[str] = set()
    fetch_ok = False
    getter = getattr(client, "fact_sheet_ids", None)
    if callable(getter):
        try:
            live = getter() or set()
            fetch_ok = True
        except Exception:  # noqa: BLE001
            live = set()
            fetch_ok = False
    return _reconcile(engine, "leanix", live, fetch_ok=fetch_ok)


def _leanix_batch_rows(
    batch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The extractor batch as checkpoint-ordered entity rows plus edge rows."""
    entities = [{"id": n.id, "type": n.type, **n.props} for n in batch.nodes]
    relationships = [
        {"source": e.source, "target": e.target, "type": e.rel_type, **e.props}
        for e in batch.edges
    ]
    ordered = sorted(
        entities, key=lambda item: _checkpoint_order(item.get("updatedAt"))
    )
    return ordered, relationships


def _leanix_apply_entities(
    engine: Any, records: list[dict[str, Any]], since: str | None
) -> SimpleNamespace:
    """Apply each fact-sheet envelope in order, stopping at the first failure.

    Returns ``SimpleNamespace(applied, failed, watermark)``.
    """
    from ..ingestion.change_envelope import ChangeEnvelope
    from ..ingestion.envelope_ingest import ingest_envelope

    state = SimpleNamespace(applied=0, failed=0, watermark=since)
    for record in records:
        env: ChangeEnvelope | None = ChangeEnvelope.from_connector_record(
            record,
            connector="leanix",
            id_field="id",
            version_field="updatedAt",
            checkpoint=record.get("updatedAt"),
        )
        env, blocked = _apply_with_preflight_one(engine, "leanix", env)
        if env is None:
            state.failed += 1
            logger.warning("leanix envelope blocked by backfeed preflight: %s", blocked)
            break
        result = ingest_envelope(engine, env)
        if result.get("status") not in {"success", "skipped"}:
            state.failed += 1
            logger.warning(
                "leanix envelope %s failed: %s",
                env.idempotency_key,
                result.get("error"),
            )
            # A later record may carry a newer cursor. Stop here so a retry can
            # still observe this failed record instead of skipping past it.
            break
        state.applied += 1
        checkpoint = record.get("updatedAt")
        if checkpoint:
            state.watermark = str(checkpoint)
    return state


def _leanix_relation_record(
    relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    """The deterministic relation-projection record for one delta's edges."""
    canonical_relationships = sorted(
        relationships,
        key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), default=str
        ),
    )
    relation_version = hashlib.sha256(
        json.dumps(
            canonical_relationships,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "id": make_source_id("leanix", "relationship-projection"),
        "type": "SourceRelationshipProjection",
        "updatedAt": relation_version,
        "relationship_count": len(canonical_relationships),
        "_links": canonical_relationships,
    }


def _leanix_apply_relations(
    engine: Any, relationships: list[dict[str, Any]], watermark: str | None
) -> tuple[int, int]:
    """Apply the relation projection, as ``(relations_hydrated, failures)``."""
    from ..ingestion.change_envelope import ChangeEnvelope
    from ..ingestion.envelope_ingest import ingest_envelope

    relation_record = _leanix_relation_record(relationships)
    relation_env: ChangeEnvelope | None = ChangeEnvelope.from_connector_record(
        relation_record,
        connector="leanix",
        id_field="id",
        version_field="updatedAt",
        checkpoint=watermark,
    )
    relation_env, blocked = _apply_with_preflight_one(engine, "leanix", relation_env)
    if relation_env is None:
        logger.warning(
            "leanix relation envelope blocked by backfeed preflight: %s",
            blocked,
        )
        return 0, 1
    relation_result = ingest_envelope(engine, relation_env)
    if relation_result.get("status") not in {"success", "skipped"}:
        logger.warning(
            "leanix relation envelope %s failed: %s",
            relation_env.idempotency_key,
            relation_result.get("error"),
        )
        return 0, 1
    return int(relation_record["relationship_count"]), 0


def _sync_leanix(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): object

    writes route through :func:`~..ingestion.envelope_ingest.ingest_envelope`.
    Each fact sheet is one native ``ApplyChangeEnvelope`` transaction. The
    relation projection is a separate, deterministic ChangeEnvelope applied
    after the object envelopes, so cross-delta endpoints may already exist while
    graph rows, edge rows, policy, lineage, content version, cursor, and outbox
    still share one authoritative engine transaction. There is no direct batch
    write or post-commit watermark write on this path. Migrated first as the
    flagship delta connector (referenced by name in
    ``ingestion.change_envelope``'s own module docstring).
    """
    if client is None:
        from ...ecosystem.ea_clients import get_leanix_client

        client = get_leanix_client()
    if client is None:
        return {"status": "skipped", "reason": "no LeanIX client configured"}

    if mode == "reconcile":
        return _leanix_reconcile(engine, client)

    since = None if mode == "full" else _read_envelope_watermark(engine, "leanix")

    from ..enrichment.extractors.leanix import extract as leanix_extract

    batch = leanix_extract(SimpleNamespace(client=client, since=since, ids=ids))
    ordered_entities, relationships = _leanix_batch_rows(batch)

    applied = _leanix_apply_entities(engine, ordered_entities, since)
    failed = applied.failed
    relations_hydrated = 0
    if relationships and not failed:
        # A relation can connect nodes from different delta pages. Applying the
        # projection only after object envelopes preserves that shape without
        # choosing an arbitrary object as the relation owner. Sorting makes an
        # upstream ordering-only change idempotent.
        relations_hydrated, relation_failures = _leanix_apply_relations(
            engine, relationships, applied.watermark
        )
        failed += relation_failures

    # This is only a reported summary; the native cursor is committed inside
    # each successful ChangeEnvelope transaction.

    return {
        "status": "partial" if failed else "ok",
        "source": "leanix",
        "mode": mode,
        "delta_capable": True,
        "nodes_hydrated": applied.applied,
        "relations_hydrated": relations_hydrated,
        "failed": failed,
        "since": since,
        "watermark": applied.watermark or since,
    }


def _archivebox_params(since: str | None, ids: list[str] | None) -> dict[str, Any]:
    """The ``archivebox`` preset params — delta watermark and/or explicit ids."""
    params: dict[str, Any] = {}
    if since:
        params["created_at__gte"] = since
    if ids:
        params["id"] = ",".join(ids)
    return params


def _archivebox_docs(since: str | None, ids: list[str] | None) -> list[Any]:
    """Enumerate ArchiveBox snapshots through the ``archivebox`` mcp_tool preset."""
    from ...protocols.source_connectors.registry import build_connector

    config: dict[str, Any] = {"preset": "archivebox"}
    params = _archivebox_params(since, ids)
    if params:
        config["params"] = params
    conn = build_connector("mcp_tool", config)
    if hasattr(conn, "poll_all"):
        return list(conn.poll_all())  # type: ignore[attr-defined]
    return list(conn.load())  # type: ignore[attr-defined]


def _archivebox_page_text(doc: Any, url: str) -> tuple[Any, str]:
    """The resolved page and its body text, falling back to the connector's own."""
    from ..ingestion.web_fetch import resolve_web_fetch

    page = resolve_web_fetch(url)
    text = page.markdown if page is not None else (getattr(doc, "text", "") or "")
    return page, text


def _archivebox_ingest_page(processor: Any, doc: Any) -> bool:
    """Ingest one archived URL through the unified DOCUMENT path.

    The body is retrieved robustly via ``web_fetch.resolve_web_fetch``
    (ArchiveBox-preferred when configured); ``False`` means nothing was ingested.
    """
    url = str((getattr(doc, "metadata", None) or {}).get("url") or "")
    if not url.startswith(("http://", "https://")):
        return False
    page, text = _archivebox_page_text(doc, url)
    if not text.strip():
        return False
    stable = hashlib.sha256(
        str(getattr(doc, "id", "") or url).encode("utf-8")
    ).hexdigest()[:32]
    processor.process(
        text,
        document_id=f"archivebox:{stable}",
        title=getattr(doc, "title", "") or url,
        doc_type="archived_web_page",
        source=url,
        metadata={
            "source_system": "archivebox",
            "fetch_backend": page.backend if page is not None else "connector",
            "updated_at": getattr(doc, "updated_at", None),
        },
        external_access=getattr(doc, "external_access", None),
        connector="archivebox",
        checkpoint=getattr(doc, "updated_at", None),
    )
    return True


def _archivebox_checkpoint(engine: Any, docs: list[Any], since: str | None) -> Any:
    """Commit a review checkpoint when this drain saw newer snapshots."""
    seen = [d.updated_at for d in docs if d.updated_at]
    new_watermark = max(seen) if seen else None
    if new_watermark and (since is None or str(new_watermark) > str(since)):
        _ingest_graph_slice_via_envelope(
            engine,
            "archivebox",
            [
                {
                    "id": "archivebox:review-checkpoint",
                    "type": "SourceReviewCheckpoint",
                    "updatedAt": new_watermark,
                    "items_seen": len(docs),
                }
            ],
            checkpoint=new_watermark,
        )
    return new_watermark


def _sync_archivebox(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest preserved ArchiveBox snapshots into the KG (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

    Enumerates snapshots via the ``archivebox`` mcp_tool source preset (delta =
    ``created_at__gte`` watermark; "pull all" = ``mode='full'``; ``ids`` selects
    specific snapshots), then ingests each archived URL through the unified
    ``DOCUMENT`` path — so the body is retrieved robustly via
    ``web_fetch.resolve_web_fetch`` (ArchiveBox-preferred when configured) and a
    research-roundup snapshot also auto-acquires the papers it cites (Phase 2).
    """
    # CA-22/P11: document-shaped delegated pipeline -- no single ChangeEnvelope
    # record is built here (the commit happens inside the delegated module
    # below). This call establishes AST-verifiable reachability to the
    # backfeed preflight chokepoint at this handler's entry point (the third
    # of the lane's three write-path shapes); it checks nothing per-record
    # (batch=[]) since no manifest declares a field-level conflict_policy for
    # "archivebox" today -- additive, matches every other undeclared connector.
    _apply_with_preflight(engine, "archivebox", [])

    from ...core.config import setting

    if not (setting("ARCHIVEBOX_URL", default="") or "").strip():
        return {"status": "skipped", "reason": "ARCHIVEBOX_URL not configured"}

    since = None if mode == "full" else _read_envelope_watermark(engine, "archivebox")

    docs = _archivebox_docs(since, ids)
    processor = _confluence_processor(engine)
    ingested = sum(
        1 for doc in _ordered_documents(docs) if _archivebox_ingest_page(processor, doc)
    )
    new_watermark = _archivebox_checkpoint(engine, docs, since)

    return {
        "status": "ok",
        "source": "archivebox",
        "mode": mode,
        "delta_capable": True,
        "snapshots_seen": len(docs),
        "documents_ingested": ingested,
        "since": since,
        "watermark": new_watermark or since,
    }


# PACKAGE_PRESETS packages whose upstream is ALREADY ingested by a dedicated
# _DELTA_HANDLERS source — excluded from the fleet sweep to avoid double-ingestion
# (CONCEPT:AU-KG.compute.gitlab-api-gitlab-atlassian). gitlab-api→gitlab, atlassian-agent→jira/confluence,
# plane-agent→plane, scholarx→the research feed. Keep this in sync with
# _DELTA_HANDLERS: a package gains a dedicated handler ⇒ add it here.
_FLEET_DEDICATED_PACKAGES: frozenset[str] = frozenset(
    {"gitlab-api", "atlassian-agent", "plane-agent", "scholarx"}
)


_FLEET_CONNECTOR_UNCONFIGURED = (
    "not configured",
    "no client",
    "missing",
    "credential",
)


def _fleet_connector_ingest(
    state: SimpleNamespace, package: str, docs: list[Any], doc_type: str
) -> int:
    """Ingest one package's drained documents through the unified DOCUMENT path."""
    ingested = 0
    for doc in docs:
        text = getattr(doc, "text", "") or ""
        if not text.strip():
            continue
        _lazy_document_processor(state).process(
            text,
            document_id=f"fleet:{package}:{getattr(doc, 'id', '')}",
            title=getattr(doc, "title", "") or str(getattr(doc, "id", "")),
            doc_type=doc_type,
            source=getattr(doc, "source_uri", "") or "",
            metadata={
                "source_system": make_source_id("fleet", package),
                "package": package,
                "updated_at": getattr(doc, "updated_at", None),
            },
            external_access=getattr(doc, "external_access", None),
            connector="fleet_connectors",
            source_instance=package,
            checkpoint=getattr(doc, "updated_at", None),
        )
        ingested += 1
    return ingested


def _fleet_connector_checkpoint(
    engine: Any, package: str, docs: list[Any], since: str | None
) -> str | None:
    """Commit a ``SourceReviewCheckpoint`` when this drain saw newer records."""
    watermark = _max_updated(docs)
    if watermark and (since is None or str(watermark) > str(since)):
        _ingest_graph_slice_via_envelope(
            engine,
            "fleet_connectors",
            [
                {
                    "id": f"fleet:{package}:review-checkpoint",
                    "type": "SourceReviewCheckpoint",
                    "updatedAt": watermark,
                    "items_seen": len(docs),
                }
            ],
            source_instance=package,
            checkpoint=watermark,
        )
    return watermark


def _fleet_connector_sync(
    state: SimpleNamespace, package: str, preset: dict[str, Any], mode: str
) -> dict[str, Any]:
    """Drain + ingest one ``agents/*`` package, returning its ``synced`` summary."""
    from ...protocols.source_connectors.registry import build_connector

    since = (
        None
        if mode == "full"
        else _read_envelope_watermark(
            state.engine,
            "fleet_connectors",
            source_instance=package,
        )
    )
    conn = build_connector("mcp", {"package": package})
    drained, _ = _drain_incremental(conn, since)
    docs = _ordered_documents(drained)
    ingested = _fleet_connector_ingest(
        state, package, docs, str(preset.get("doc_type") or "document")
    )
    return {
        "records_seen": len(docs),
        "documents_ingested": ingested,
        "watermark": _fleet_connector_checkpoint(state.engine, package, docs, since),
    }


def _classify_fleet_connector_exception(
    package: str, exc: Exception
) -> tuple[str, str]:
    """Bucket one package's failure — an unconfigured upstream is a skip."""
    msg = str(exc)
    if any(token in msg.lower() for token in _FLEET_CONNECTOR_UNCONFIGURED):
        return "skipped", f"unconfigured: {msg[:120]}"
    logger.warning("fleet_connectors: %s failed: %s", package, exc)
    return "errors", msg[:200]


def _sync_fleet_connectors(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Drain EVERY configured ``agent-packages/agents/*`` connector in one pass (CONCEPT:AU-KG.compute.gitlab-api-gitlab-atlassian).

    The fleet ships ~50 sibling packages, each a FastMCP server with a declared
    document-yielding tool (the :data:`package_manifest.PACKAGE_PRESETS` catalog —
    scholarx/github-agent/gitlab-api/servicenow-api/mattermost/nextcloud/microsoft/
    atlassian/plane/erpnext/mealie/langfuse/…). Rather than a per-package handler,
    this ONE declarative handler iterates the preset catalog, reaches each package
    through the generic ``mcp`` connector (:class:`MCPPackageConnector`, ECO-4.29),
    and ingests every yielded record through the unified ``DOCUMENT`` path — so the
    "every agents/* connector" leg of a FULL ingest is a single registered source
    (``fleet_connectors``) that the ``source="all"`` sweep fans out as its own laned
    ``connector_sync`` task.

    A package is only attempted when its MCP server is registered in the workspace
    ``mcp_config.json`` (the same source the multiplexer/connector use), so
    unconfigured packages are reported *skipped* — never errored — and one bad
    package never aborts the rest. Delta = a per-package ISO ``updated_at`` watermark
    (``fleet:<package>``); the write-layer content-hash is the second guard, so a
    re-run is a no-op for unchanged records. ``mode='full'`` drains from scratch.
    """
    # CA-22/P11: document-shaped delegated pipeline, fanning out across ~50
    # sibling packages via the unified DOCUMENT path (each commits its own
    # ApplyChangeEnvelope, not a ChangeEnvelope built here) -- AST-reachability
    # call to the preflight chokepoint; batch=[] checks nothing per-record
    # until a manifest declares conflict_policy for "fleet_connectors".
    _apply_with_preflight(engine, "fleet_connectors", [])

    from ...protocols.source_connectors.connectors.mcp_package import _load_mcp_config
    from ...protocols.source_connectors.connectors.package_manifest import (
        PACKAGE_PRESETS,
        get_preset,
    )

    servers = _load_mcp_config() or {}
    state = SimpleNamespace(engine=engine, proc=None)
    buckets: dict[str, dict[str, Any]] = {"synced": {}, "skipped": {}, "errors": {}}

    for package in sorted(PACKAGE_PRESETS):
        # Skip packages already covered by a DEDICATED delta handler — otherwise
        # source="all" enqueues BOTH this fleet leg AND the dedicated source for the
        # same upstream, writing under non-matching doc-id namespaces so the
        # content-hash delta can't dedup → duplicate Document/Chunk nodes every run
        # (CONCEPT:AU-KG.compute.gitlab-api-gitlab-atlassian). The dedicated handler owns these upstreams.
        if package in _FLEET_DEDICATED_PACKAGES:
            buckets["skipped"][package] = "covered by a dedicated delta handler"
            continue
        preset = get_preset(package)
        server = str(preset.get("server") or f"{package}-mcp")
        # Configured = the package's MCP server is registered with the multiplexer.
        if server not in servers and package not in servers:
            buckets["skipped"][package] = f"{server} not in mcp_config"
            continue
        try:
            buckets["synced"][package] = _fleet_connector_sync(
                state, package, preset, mode
            )
        except Exception as exc:  # noqa: BLE001 — isolate one bad package
            bucket, value = _classify_fleet_connector_exception(package, exc)
            buckets[bucket][package] = value

    return {
        "status": "partial" if buckets["errors"] else "ok",
        "source": "fleet_connectors",
        "mode": mode,
        "delta_capable": True,
        "synced": buckets["synced"],
        "skipped": buckets["skipped"],
        "errors": buckets["errors"],
        "counts": {
            "synced": len(buckets["synced"]),
            "skipped": len(buckets["skipped"]),
            "errors": len(buckets["errors"]),
        },
    }


# ── L27: live sync_source call sites for 6 mandatory-manifest ops connectors ──
#
# ``connector_manifest_gate.MANDATORY_NAMED_CONNECTOR_SOURCES`` names 12 connectors
# whose ``connector_manifest.yml`` is unconditionally required (AU-P1-6). 7 already
# had a live ``sync_source`` call site; 6 did not (its own docstring said so
# explicitly): ``microsoft-agent``, ``container-manager-mcp``, ``documentdb-mcp``,
# ``repository-manager``, ``systems-manager``, ``vector-mcp`` — the compile-before-
# sync gate was correct but reached no runtime path. This closes that ledger item
# (L27): each gets a real, dispatchable ``_DELTA_HANDLERS`` entry, envelope-native
# from day one (CONCEPT:AU-KG.ingest.envelope-atomic-transaction, AU-P1-5) — the
# gate now actually runs whenever ``sync_source(<connector>)`` is called.
#
# These are action/ops MCP servers, not document-corpus sources.  Their listing
# tools and field maps are owned by the connector packages as data-only preset
# providers and copied into the signed manifest.  Runtime deliberately has no
# central fallback table: a missing provider or any preset drift is a gate error,
# never permission to invoke a guessed tool.

_OPS_PRIVATE_METADATA_KEYS = frozenset(
    {
        "base_url",
        "endpoint",
        "file_path",
        "overlay_source",
        "path",
        "source_url",
        "url",
    }
)


def _is_filesystem_path(value: str) -> bool:
    """Whether a string looks like a local filesystem path or ``file://`` URL."""
    return (
        os.path.isabs(value)
        or value.startswith("file://")
        or (len(value) > 2 and value[1] == ":" and value[2] in "\\/")
    )


def _filter_ops_value(value: Any) -> Any:
    """Drop private keys and replace path-shaped strings.

    Recurses through :func:`_safe_ops_value` so every nested level is filtered
    AND sanitized, exactly as the original single recursive function did.
    """
    if isinstance(value, dict):
        return {
            key: _safe_ops_value(item)
            for key, item in value.items()
            if str(key).lower() not in _OPS_PRIVATE_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_safe_ops_value(item) for item in value]
    if isinstance(value, str) and _is_filesystem_path(value):
        return "configured-resource"
    return value


def _safe_ops_value(value: Any) -> Any:
    """Remove private fields and sanitize an ops record before persistence."""
    from ...security.persistence_privacy import sanitize_for_persistence

    clean, _ = sanitize_for_persistence(_filter_ops_value(value))
    return clean


def _safe_ops_id(value: Any, *, source: str) -> str:
    from ...security.persistence_privacy import persistence_reference

    return persistence_reference("connector_object", value, namespace=source)


def _ops_connector_config(package: str) -> tuple[dict[str, Any], Any]:
    """Resolve one mandatory connector's signed, provider-owned sync preset."""
    import yaml

    from ...protocols.source_connectors.connectors.mcp_tool import (
        provider_tool_presets,
        provider_tool_schema_fingerprints,
    )
    from ..ontology.connector_manifest import ConnectorManifest
    from ..ontology.connector_manifest_gate import (
        check_manifest_bytes,
        find_connector_manifest,
    )

    manifest_path = find_connector_manifest(package)
    if manifest_path is None:
        raise RuntimeError("mandatory connector manifest is unavailable")
    if check_manifest_bytes(
        manifest_path, require_signature=True, require_provider=True
    ):
        raise RuntimeError("mandatory connector manifest contract did not verify")
    manifest = ConnectorManifest.model_validate(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )
    provider_presets = provider_tool_presets(manifest.connector)
    if provider_presets is None:
        raise RuntimeError("mandatory connector preset provider is unavailable")
    if not manifest.sync:
        raise RuntimeError("mandatory connector manifest has no sync preset")

    schema_pins = provider_tool_schema_fingerprints(manifest.connector)
    if not schema_pins:
        raise RuntimeError(
            "mandatory connector tool-schema certification is unavailable"
        )

    sync = manifest.sync[0]
    provider_preset = provider_presets.get(sync.preset)
    if provider_preset is None:
        raise RuntimeError("mandatory connector provider is missing its signed preset")
    if json.dumps(provider_preset, sort_keys=True, separators=(",", ":")) != json.dumps(
        sync.raw, sort_keys=True, separators=(",", ":")
    ):
        raise RuntimeError(
            "mandatory connector provider differs from its signed preset"
        )
    schema_pin = schema_pins.get(sync.tool)
    if not schema_pin or schema_pin != sync.tool_schema_sha256:
        raise RuntimeError("mandatory connector live tool-schema pin is invalid")
    return {
        **provider_preset,
        "strict_schema": True,
        "verify_live_schema": True,
        # Release-certified structural pin. Presentation text and runtime
        # defaults may change; arguments and constraints remain fail-closed.
        "tool_schema_sha256": schema_pin,
    }, sync


def _ops_drain(conn_config: dict[str, Any], mode: str, since: str | None) -> list[Any]:
    """Drain one ops connector's signed listing tool (full snapshot on reconcile)."""
    from ...protocols.source_connectors.registry import build_connector

    conn = build_connector("mcp_tool", conn_config)
    if mode == "reconcile":
        return list(conn.load())  # type: ignore[attr-defined]
    drained, _ = _drain_incremental(conn, since)
    return _ordered_documents(drained)


def _ops_live_ids(docs: list[Any], package: str) -> set[str]:
    """The privacy-safe live id set a reconcile pass compares the KG against."""
    raw_live = {str(getattr(d, "id", "")) for d in docs if getattr(d, "id", None)}
    return {_safe_ops_id(value, source=package) for value in raw_live}


def _ops_record(doc: Any, package: str, sync: Any) -> dict[str, Any]:
    """One drained ops document as a privacy-safe, id-stamped connector record."""
    raw = _record_of(doc)
    record: dict[str, Any] = (
        _safe_ops_value(dict(raw))
        if isinstance(raw, dict)
        else {
            "id": getattr(doc, "id", ""),
            "name": getattr(doc, "title", ""),
            "text": getattr(doc, "text", ""),
        }
    )
    safe_id = _safe_ops_id(getattr(doc, "id", ""), source=package)
    record["id"] = safe_id
    if sync.id_field and "." not in sync.id_field:
        record[sync.id_field] = safe_id
    record.setdefault("name", _safe_ops_value(getattr(doc, "title", "")))
    record.setdefault("text", _safe_ops_value(getattr(doc, "text", "")))
    return record


def _ops_apply_envelopes(
    engine: Any, docs: list[Any], package: str, sync: Any, since: str | None
) -> SimpleNamespace:
    """Ingest each ops record, stopping at the first failure.

    Returns ``SimpleNamespace(processed, failed, watermark)``.
    """
    from ..ingestion.change_envelope import ChangeEnvelope
    from ..ingestion.envelope_ingest import ingest_envelope

    state = SimpleNamespace(processed=0, failed=0, watermark=since)
    for doc in docs:
        updated_at = getattr(doc, "updated_at", None)
        env: ChangeEnvelope | None = ChangeEnvelope.from_connector_record(
            _ops_record(doc, package, sync),
            connector=package,
            id_field="id",
            version_field=sync.updated_field or "id",
            checkpoint=updated_at,
            source_acl=getattr(doc, "external_access", None),
        )
        env, blocked = _apply_with_preflight_one(engine, package, env)
        if env is None:
            state.failed += 1
            logger.warning(
                "%s envelope blocked by backfeed preflight: %s", package, blocked
            )
            break
        result = ingest_envelope(engine, env)
        if result.get("status") not in {"success", "skipped"}:
            state.failed += 1
            logger.warning(
                "%s envelope %s failed closed",
                package,
                env.idempotency_key,
            )
            break
        state.processed += 1
        if updated_at and (
            state.watermark is None or str(updated_at) > str(state.watermark)
        ):
            state.watermark = updated_at
    return state


def _sync_ops_mcp_connector(
    engine: Any,
    *,
    mode: str,
    ids: list[str] | None,
    client: Any,
    package: str,
) -> dict[str, Any]:
    """L27 minimal snapshot pull for one action/ops MCP connector (CONCEPT:AU-KG.ontology.connector-manifest-gate).

    Shared by the thin ``_sync_<package>`` wrappers below. Reaches the connector
    through the generic strict ``mcp_tool`` adapter using its installed,
    provider-owned preset, drains the signed listing tool, and routes each record through
    :func:`~..ingestion.envelope_ingest.ingest_envelope` — envelope-native from
    day one and subject to the same fail-closed native capability boundary.
    """
    try:
        conn_config, sync = _ops_connector_config(package)
    except Exception as exc:  # noqa: BLE001 - mandatory contract lookup is fail-closed
        return {
            "status": "error",
            "source": package,
            "reason": f"mandatory connector contract unavailable ({type(exc).__name__})",
        }

    since = None if mode == "full" else _read_envelope_watermark(engine, package)

    if client is not None:
        conn_config["client"] = client

    try:
        docs = _ops_drain(conn_config, mode, since)
    except Exception as exc:  # noqa: BLE001 - mandatory execution is fail-closed
        return {
            "status": "error",
            "source": package,
            "reason": f"mandatory connector execution failed ({type(exc).__name__})",
        }

    if mode == "reconcile":
        live = _ops_live_ids(docs, package)
        return _reconcile(engine, package, live) | {"source": package}

    state = _ops_apply_envelopes(engine, docs, package, sync, since)
    return {
        "status": "partial" if state.failed else "ok",
        "source": package,
        "mode": mode,
        "delta_capable": bool(sync.updated_field),
        "records_seen": len(docs),
        "ingested": state.processed,
        "failed": state.failed,
        "since": since,
        "watermark": state.watermark,
    }


def _sync_microsoft_agent(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """L27 live call site for ``microsoft-agent`` (Graph API messages/Teams/SharePoint)."""
    return _sync_ops_mcp_connector(
        engine, mode=mode, ids=ids, client=client, package="microsoft-agent"
    )


def _sync_container_manager_mcp(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """L27 live call site for ``container-manager-mcp`` (docker/podman/k8s fleet)."""
    return _sync_ops_mcp_connector(
        engine, mode=mode, ids=ids, client=client, package="container-manager-mcp"
    )


def _sync_documentdb_mcp(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """L27 live call site for ``documentdb-mcp``."""
    return _sync_ops_mcp_connector(
        engine, mode=mode, ids=ids, client=client, package="documentdb-mcp"
    )


def _sync_repository_manager(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """L27 live call site for ``repository-manager`` (git repo/worktree fleet)."""
    return _sync_ops_mcp_connector(
        engine, mode=mode, ids=ids, client=client, package="repository-manager"
    )


def _sync_systems_manager(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """L27 live call site for ``systems-manager`` (host/systems inventory)."""
    return _sync_ops_mcp_connector(
        engine, mode=mode, ids=ids, client=client, package="systems-manager"
    )


def _sync_vector_mcp(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """L27 live call site for ``vector-mcp`` (vector-store collections)."""
    return _sync_ops_mcp_connector(
        engine, mode=mode, ids=ids, client=client, package="vector-mcp"
    )


def _as_epoch(value: Any) -> int | None:
    """Best-effort parse of a watermark value to int unix-seconds."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _freshrss_configured() -> bool:
    """FreshRSS is configured when ``FRESHRSS_URL`` is set OR ``freshrss-mcp`` is.

    The connector reaches FreshRSS through that server (which holds the GReader
    credentials), so graph-os itself needs no direct FreshRSS env.
    """
    from ...core.config import setting

    if (setting("FRESHRSS_URL", default="") or "").strip():
        return True
    try:
        from ...protocols.source_connectors.connectors.mcp_tool import (
            _load_mcp_config,
        )

        servers = _load_mcp_config() or {}
        return "freshrss-mcp" in servers or "freshrss" in servers
    except Exception:  # noqa: BLE001 — best-effort discovery
        return False


def _freshrss_docs(since: str | None) -> list[Any]:
    """Drain the ``freshrss`` mcp_tool preset over the Google-Reader API.

    Bound each run (the */20min sweep drains incrementally) so a cold first run —
    thousands of backlog articles before any watermark — can't run unbounded. Each
    cursor batch is ~100 items; default 3 pages ≈ 300 items/run. Override with
    ``FRESHRSS_MAX_BATCHES``.
    """
    from ...core.config import setting
    from ...protocols.source_connectors.registry import build_connector

    params: dict[str, Any] = {}
    if since and (since_epoch := _as_epoch(since)) is not None:
        params["newer_than"] = since_epoch  # GReader ``ot`` — unix seconds
    config: dict[str, Any] = {"preset": "freshrss"}
    if params:
        config["params"] = params
    conn = build_connector("mcp_tool", config)
    try:
        max_batches = int(setting("FRESHRSS_MAX_BATCHES", default="3") or 3)
    except (TypeError, ValueError):
        max_batches = 3
    if hasattr(conn, "poll_all"):
        return list(conn.poll_all(max_batches=max_batches))  # type: ignore[attr-defined]
    return list(conn.load())  # type: ignore[attr-defined]


def _freshrss_checkpoint(
    engine: Any, docs: list[Any], since: str | None, failed: int
) -> str | None:
    """Commit the FreshRSS review checkpoint (a unix-seconds GReader watermark)."""
    seen = [e for d in docs if (e := _as_epoch(d.updated_at)) is not None]
    new_watermark = str(max(seen)) if seen else None
    since_epoch = _as_epoch(since) if since else None
    if (
        not failed
        and new_watermark
        and (since_epoch is None or int(new_watermark) > since_epoch)
    ):
        _ingest_graph_slice_via_envelope(
            engine,
            "freshrss",
            [
                {
                    "id": "freshrss:review-checkpoint",
                    "type": "SourceReviewCheckpoint",
                    "updatedAt": new_watermark,
                    "items_seen": len(docs),
                }
            ],
            checkpoint=new_watermark,
        )
    return new_watermark


def _sync_freshrss(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Relevance-gated ingestion of curated FreshRSS items.

    Enumerates items via the ``freshrss`` mcp_tool preset over the Google-Reader API
    (delta = ``newer_than`` → GReader ``ot`` **unix-seconds** watermark on
    ``published``; ``mode='full'`` drains all). Unlike a mirror connector this source
    is INTENTIONALLY GATED: each item passes the world-model relevance gate
    (:class:`WorldModelPipelineRunner`, CONCEPT:AU-KG.ingest.news-finance-tech-sibling) — only items relevant to the
    existing KG (taxonomy score OR concept-novelty) or agent-force flagged are fully
    ingested as ``news_article`` Documents; the rest get a marginal footprint or are
    skipped. Research/arXiv-feed items route to the research path (CONCEPT:AU-KG.ingest.worldmodel-gated-ingestion),
    unifying RSS intake. ``skipped_unchanged`` plus the watermark prove the delta on a
    re-run; the write-layer content-hash delta (KG_WRITE_DELTA) is the second guard.
    """
    # CA-22/P11: document-shaped delegated pipeline (commits via the world-model
    # gate's own ApplyChangeEnvelope path) -- AST-reachability call to the
    # preflight chokepoint; batch=[] checks nothing per-record until a manifest
    # declares conflict_policy for "freshrss".
    _apply_with_preflight(engine, "freshrss", [])

    from ...core.config import setting

    if not _freshrss_configured():
        return {
            "status": "skipped",
            "reason": "FreshRSS not configured (set FRESHRSS_URL or add the "
            "freshrss-mcp server to mcp_config)",
        }
    # Registry material is part of this external source's durable projection.
    # A native commit failure is not a reason to continue and advance its cursor.
    from ...automation.feed_sources import upsert_feed_source

    upsert_feed_source(
        engine,
        key="freshrss",
        source_system="freshrss",
        feed_url=(setting("FRESHRSS_URL", default="") or ""),
        kind="FeedSource",
        name="FreshRSS",
    )

    since = None if mode == "full" else _read_envelope_watermark(engine, "freshrss")

    docs = _freshrss_docs(since)

    from ...automation.worldmodel_pipeline import (
        WorldModelConfig,
        WorldModelPipelineRunner,
    )
    from ...base_utilities import to_boolean

    wm_config = WorldModelConfig(
        use_novelty=to_boolean(setting("FRESHRSS_USE_NOVELTY", default="False"))
    )
    report = WorldModelPipelineRunner(
        engine=engine, config=wm_config, connector="freshrss"
    ).run_gated_ingest(docs)

    new_watermark = _freshrss_checkpoint(engine, docs, since, report.failed)

    return {
        "status": "partial" if report.failed else "ok",
        "source": "freshrss",
        "mode": mode,
        "delta_capable": True,
        "items_seen": len(docs),
        "ingested": report.ingested,
        "relevant": report.relevant,
        "marginal": report.marginal,
        "research": report.research,
        "skipped_unchanged": report.skipped,
        "failed": report.failed,
        "since": since,
        "watermark": new_watermark or since,
    }


def _feed_checkpoint(
    engine: Any, source: str, docs: list[Any], since: str | None, failed: int
) -> Any:
    """Commit a feed connector's review checkpoint when this run saw newer items.

    Shared by the ``rss``/``arxiv``/``freshrss`` world-model feed handlers. A run
    with any failure never advances the watermark.
    """
    iso_dates = [d.updated_at for d in docs if getattr(d, "updated_at", None)]
    new_watermark = max(iso_dates) if iso_dates else None
    if not failed and new_watermark and (since is None or new_watermark > since):
        _ingest_graph_slice_via_envelope(
            engine,
            source,
            [
                {
                    "id": f"{source}:review-checkpoint",
                    "type": "SourceReviewCheckpoint",
                    "updatedAt": new_watermark,
                    "items_seen": len(docs),
                }
            ],
            checkpoint=new_watermark,
        )
    return new_watermark


def _rss_native_urls(engine: Any) -> list[str]:
    """Native feed URLs: the ``KG_RSS_FEEDS`` seed UNION the ``:FeedSource`` registry.

    So ``graph_feeds add`` → the next sweep ingests it.
    """
    from ...automation.feed_sources import list_feed_sources
    from ...core.config import config as _cfg

    seed = (getattr(_cfg, "kg_rss_feeds", "") or "").split(",")
    native_url_set = {u.strip() for u in seed if u.strip()}
    for node in list_feed_sources(engine):
        if (
            node.get("source_system") == "rss"
            and node.get("enabled", True)
            and node.get("feed_url")
        ):
            native_url_set.add(str(node["feed_url"]))
    return sorted(native_url_set)


def _scholarx_available() -> bool:
    """ScholarX is reachable via the local package OR the fleet ``scholarx-mcp``.

    Per CONCEPT:AU-KG.ingest.research-connector-presets either is sufficient to
    enable the research feed (``scholarx_feed_documents`` itself prefers the
    package and falls back to MCP).
    """
    from ...automation.feed_sources import _scholarx_mcp_configured

    try:
        import scholarx  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return _scholarx_mcp_configured()


def _rss_native_docs(native_urls: list[str], since: str | None) -> list[Any]:
    """Drain the native feed URLs through the zero-infra ``rss`` connector."""
    from ...protocols.source_connectors.base import ConnectorCheckpoint
    from ...protocols.source_connectors.registry import build_connector

    conn = build_connector("rss", {"feed_urls": native_urls})
    cp = ConnectorCheckpoint(watermark=since) if since else None
    if hasattr(conn, "poll_all"):
        return list(conn.poll_all(cp))  # type: ignore[attr-defined]
    return list(conn.load())  # type: ignore[attr-defined]


def _sync_rss(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Native RSS/Atom feeds + ScholarX arXiv through the ONE world-model gate (KG-2.121).

    The unified feed handler: native feed URLs (``KG_RSS_FEEDS``) are drained by the
    zero-infra ``rss`` connector and ScholarX arXiv items by the scholarx feed bridge;
    both emit the same ``SourceDocument`` shape and flow through
    :meth:`WorldModelPipelineRunner.run_gated_ingest` — research/arXiv items take the
    prioritized ``research_paper_fetch`` path, news items the relevance+novelty gate.
    Each configured feed is materialized as a first-class ``:FeedSource`` node on this
    live path. Delta = an ISO publish-date watermark; node-existence (``_is_known``)
    is the cross-run dedup.
    """
    # CA-22/P11: document-shaped delegated pipeline (commits via the world-model
    # gate's own ApplyChangeEnvelope path) -- AST-reachability call to the
    # preflight chokepoint; batch=[] checks nothing per-record until a manifest
    # declares conflict_policy for "rss".
    _apply_with_preflight(engine, "rss", [])
    from ...automation.feed_sources import (
        register_feed_nodes,
        scholarx_feed_documents,
    )
    from ...automation.worldmodel_pipeline import WorldModelPipelineRunner

    native_urls = _rss_native_urls(engine)
    scholarx_ok = _scholarx_available()
    if not native_urls and not scholarx_ok:
        return {
            "status": "skipped",
            "reason": "no native RSS feeds (set KG_RSS_FEEDS) and scholarx not "
            "configured (install scholarx or add scholarx-mcp to mcp_config)",
        }

    since = None if mode == "full" else _read_envelope_watermark(engine, "rss")

    # Materialize the feed registry on the live sweep path (Wire-First, KG-2.122).
    register_feed_nodes(
        engine,
        native_urls=native_urls,
        scholarx_categories=(["arxiv"] if scholarx_ok else []),
    )

    docs: list[Any] = []
    if native_urls:
        docs.extend(_rss_native_docs(native_urls, since))
    if scholarx_ok:
        docs.extend(scholarx_feed_documents())

    report = WorldModelPipelineRunner(engine=engine, connector="rss").run_gated_ingest(
        docs
    )

    new_watermark = _feed_checkpoint(engine, "rss", docs, since, report.failed)

    return {
        "status": "partial" if report.failed else "ok",
        "source": "rss",
        "mode": mode,
        "delta_capable": True,
        "items_seen": len(docs),
        "ingested": report.ingested,
        "relevant": report.relevant,
        "marginal": report.marginal,
        "research": report.research,
        "skipped_unchanged": report.skipped,
        "failed": report.failed,
        "since": since,
        "watermark": new_watermark or since,
    }


def _arxiv_categories() -> list[str]:
    """The opt-in ``KG_ARXIV_CATEGORIES`` listing scope.

    An unscoped arXiv query is not a valid listing (``ArxivConnector`` raises
    without categories), so an empty result means "skip", never "firehose".
    """
    from ...core.config import config as _cfg

    return [
        c.strip()
        for c in (getattr(_cfg, "kg_arxiv_categories", "") or "").split(",")
        if c.strip()
    ]


def _arxiv_docs(categories: list[str], since: str | None) -> list[Any]:
    """Drain the native ``arxiv`` connector for the configured categories."""
    from ...core.config import config as _cfg
    from ...protocols.source_connectors.base import ConnectorCheckpoint
    from ...protocols.source_connectors.registry import build_connector

    try:
        max_results = int(getattr(_cfg, "kg_arxiv_max_results", 50) or 50)
    except (TypeError, ValueError):
        max_results = 50
    conn = build_connector(
        "arxiv", {"categories": categories, "max_results": max_results}
    )
    cp = ConnectorCheckpoint(watermark=since) if since else None
    if hasattr(conn, "poll_all"):
        return list(conn.poll_all(cp))  # type: ignore[attr-defined]
    return list(conn.load())  # type: ignore[attr-defined]


def _sync_arxiv(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Native arXiv category listings through the ONE world-model research gate.

    Zero-infra sibling to ``rss``/``freshrss``/ScholarX (CONCEPT:AU-KG.ingest.arxiv-feed-connector, KG-7.3):
    drains the native ``arxiv`` connector (``export.arxiv.org``, no MCP server or
    account needed) and routes every item through
    :meth:`WorldModelPipelineRunner.run_gated_ingest`, which recognizes it as
    research (``origin.streamId="arxiv:api"``) and defers to the SAME
    ``grade_and_enqueue_paper`` budget gate as FreshRSS-arXiv and ScholarX — this
    connector only widens the funnel's mouth, never bypasses its throat.

    **Opt-in only** (``KG_ARXIV_CATEGORIES``): an unscoped arXiv query is not a
    valid listing (``ArxivConnector`` raises without categories), so this handler
    skips cleanly rather than defaulting to a firehose-shaped category set.
    """
    # CA-22/P11: document-shaped delegated pipeline (commits via the world-model
    # gate's own ApplyChangeEnvelope path) -- AST-reachability call to the
    # preflight chokepoint; batch=[] checks nothing per-record until a manifest
    # declares conflict_policy for "arxiv".
    _apply_with_preflight(engine, "arxiv", [])

    from ...automation.feed_sources import upsert_feed_source
    from ...automation.worldmodel_pipeline import WorldModelPipelineRunner

    categories = _arxiv_categories()
    if not categories:
        return {
            "status": "skipped",
            "reason": "arXiv not configured (set KG_ARXIV_CATEGORIES, e.g. 'cs.AI,cs.LG')",
        }

    for category in categories:
        upsert_feed_source(
            engine,
            key=category,
            source_system="arxiv",
            feed_url=f"https://export.arxiv.org/api/query?search_query=cat:{category}",
            kind="RssFeed",
            name=f"arXiv {category}",
        )

    since = None if mode == "full" else _read_envelope_watermark(engine, "arxiv")
    docs = _arxiv_docs(categories, since)

    report = WorldModelPipelineRunner(
        engine=engine, connector="arxiv"
    ).run_gated_ingest(docs)

    new_watermark = _feed_checkpoint(engine, "arxiv", docs, since, report.failed)

    return {
        "status": "partial" if report.failed else "ok",
        "source": "arxiv",
        "mode": mode,
        "delta_capable": True,
        "categories": categories,
        "items_seen": len(docs),
        "ingested": report.ingested,
        "relevant": report.relevant,
        "marginal": report.marginal,
        "research": report.research,
        "skipped_unchanged": report.skipped,
        "failed": report.failed,
        "since": since,
        "watermark": new_watermark or since,
    }


def _gitlab_repository_checkpoint(entities: list[dict[str, Any]]) -> str | None:
    """The ``updatedAt`` of the Repository row in one indexed project slice."""
    return next(
        (
            str(item.get("updatedAt"))
            for item in entities
            if item.get("type") == "Repository" and item.get("updatedAt")
        ),
        None,
    )


def _gitlab_totals(results: list[dict[str, Any]]) -> dict[str, int]:
    """Index totals across every GitLab instance summary."""
    return {
        "failed": sum(len(r.get("errors") or []) for r in results),
        "projects_indexed": sum(r["projects_indexed"] for r in results),
        "symbols": sum(r["symbols"] for r in results),
        "calls_resolved": sum(r["calls_resolved"] for r in results),
    }


def _gitlab_instance_summary(
    engine: Any,
    inst: Any,
    client: Any,
    mode: str,
    project_ids: set[str] | None,
    index_fn: Callable[..., Any],
) -> dict[str, Any]:
    """Index one configured GitLab instance and return its summary row."""
    from .gitlab_indexer import GitLabRestSource, GitLabSource, index_instance

    name = inst.name if inst is not None else "gitlab"
    since = (
        None
        if mode == "full"
        else _read_envelope_watermark(
            engine,
            "gitlab",
            source_instance=name,
        )
    )
    # `inst is None` only occurs on the injected-client override path (in
    # :func:`_sync_gitlab`), so a real instance always pairs with the REST source.
    if client is not None:
        source: GitLabSource = client
    else:
        assert inst is not None
        source = GitLabRestSource(inst)

    def _commit_project_slice(
        _domain: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        instance_name: str = name,
    ) -> dict[str, Any]:
        return _ingest_graph_slice_via_envelope(
            engine,
            "gitlab",
            entities,
            relationships,
            source_instance=instance_name,
            checkpoint=_gitlab_repository_checkpoint(entities),
        )

    summary = index_instance(
        instance=name,
        source=source,
        index_fn=index_fn,
        ingest=_commit_project_slice,
        project_ids=project_ids,
        since=since,
    )
    return summary.as_dict()


def _sync_gitlab(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Index whole GitLab instances as a resolved code graph (CONCEPT:AU-KG.backend.declared-columns-so-schema).

    For every configured instance (``GITLAB_INSTANCES`` JSON, else single
    ``GITLAB_URL``/``GITLAB_TOKEN``) this enumerates projects → default-branch code
    files and ships each project to the engine's ``index_repository`` resolver
    (CONCEPT:EG-KG.compute.turn-each-project), writing ``:Code`` symbols + resolved ``calls``/``depends_on``
    + ``Repository``/``File`` structure under ``source_system = gitlab:<instance>``.
    ``mode='full'`` re-indexes all; delta uses a per-instance ``last_activity_at``
    watermark; ``ids`` narrows to specific projects (webhook delta).
    """
    # CA-22/P11: document-shaped delegated pipeline (commits via gitlab_indexer's
    # own index_repository/ApplyChangeEnvelope path, not a ChangeEnvelope built
    # here) -- AST-reachability call to the preflight chokepoint; batch=[] checks
    # nothing per-record until a manifest declares conflict_policy for "gitlab".
    _apply_with_preflight(engine, "gitlab", [])

    from .gitlab_indexer import instances_from_config

    graph_compute = getattr(engine, "graph_compute", None)
    if graph_compute is None or not callable(
        getattr(graph_compute, "index_repository", None)
    ):
        raise RuntimeError("current engine is missing mandatory IndexRepository")

    # An injected `client` is an explicit single-source override (tests / a caller
    # supplying its own GitLabSource): use one sentinel instance, ignore config.
    instances = [None] if client is not None else instances_from_config()  # type: ignore[list-item]
    if not instances:
        return {"status": "skipped", "reason": "no GitLab instance configured"}

    project_ids = {str(i) for i in ids} if ids else None

    results: list[dict[str, Any]] = [
        _gitlab_instance_summary(
            engine, inst, client, mode, project_ids, graph_compute.index_repository
        )
        for inst in instances
    ]

    totals = _gitlab_totals(results)
    return {
        "status": "partial" if totals["failed"] else "ok",
        "source": "gitlab",
        "mode": mode,
        "delta_capable": True,
        "instances": results,
        "projects_indexed": totals["projects_indexed"],
        "symbols": totals["symbols"],
        "calls_resolved": totals["calls_resolved"],
        "failed": totals["failed"],
    }


# ── Atlassian + Plane issue trackers / wiki (KG-2.123/2.124/2.125) ────────────
#
# Three first-class delta connectors that reach Jira / Confluence / Plane through
# their fleet MCP servers (``atlassian-mcp`` / ``plane-mcp``) via the declarative
# mcp_tool presets — never a direct vendor client. Each is **multi-instance** (the
# GitLab pattern): a second Atlassian site or Plane workspace is a second ``*-mcp``
# server entry + a typed ``*_instances`` config row, so the same logic ingests both.
#
# CONCEPT:AU-KG.compute.confluence-first-class-delta — Confluence first-class delta connector
# CONCEPT:AU-KG.compute.jira-first-class-delta — Jira first-class delta connector
# CONCEPT:AU-KG.compute.plane-first-class-delta — Plane first-class delta connector


def _resolve_tracker_instances(
    field: str,
    *,
    default_name: str,
    default_server: str,
    scope_key: str,
    scope_setting: str,
) -> list[dict[str, Any]]:
    """Configured ``*_instances`` rows, or one synthetic instance from the single-host
    settings (mirrors ``gitlab_indexer.instances_from_config``)."""
    from ...core.config import config as cfg
    from ...core.config import setting

    rows = [r for r in (getattr(cfg, field, None) or []) if isinstance(r, dict)]
    if rows:
        return rows
    scope = [
        s.strip()
        for s in (setting(scope_setting, default="") or "").split(",")
        if s.strip()
    ]
    return [{"name": default_name, "server": default_server, scope_key: scope}]


def _build_preset_conn(preset: str, server: str, params: dict[str, Any]) -> Any:
    """Build the mcp_tool connector for one tracker instance (preset + per-instance
    server override + per-run params)."""
    from ...protocols.source_connectors.registry import build_connector

    config: dict[str, Any] = {"preset": preset, "server": server}
    if params:
        config["params"] = params
    return build_connector("mcp_tool", config)


def _drain_incremental(
    conn: Any, since: str | None, *, max_batches: int = 25
) -> tuple[list[Any], bool]:
    """Drain a connector incrementally via ``poll()`` — binds the ``since`` watermark
    (client-side ``updated_field`` filter), resumes the cursor across batches, and is
    bounded so a cold first run can't run unbounded. Returns ``(documents, fetch_ok)``;
    a drain that reaches the batch bound with more pages pending is explicitly
    incomplete so callers must not treat its IDs as an authoritative snapshot."""
    from ...protocols.source_connectors.base import ConnectorCheckpoint

    docs: list[Any] = []
    cp = ConnectorCheckpoint(watermark=since) if since else None
    for _ in range(max(1, max_batches)):
        batch = conn.poll(cp)
        docs.extend(batch.documents)
        cp = batch.checkpoint
        if not getattr(cp, "has_more", False):
            return docs, True
    return docs, False


def _record_of(doc: Any) -> dict[str, Any]:
    """The raw source record the connector preserved in ``metadata.record``."""
    rec = (getattr(doc, "metadata", None) or {}).get("record")
    return rec if isinstance(rec, dict) else {}


def _max_updated(docs: list[Any]) -> str | None:
    seen = [u for d in docs if (u := getattr(d, "updated_at", None))]
    return max(seen, key=str) if seen else None


def _checkpoint_order(value: Any) -> tuple[int, int, float | str]:
    """Order source records oldest-first before any cursor-bearing commit.

    Versionless dependency/container rows sort first. Numeric and ISO timestamps
    retain their source order semantics; provider-opaque values get a stable
    lexical order only as a deterministic last resort.
    """
    if value in (None, ""):
        return (0, 0, 0.0)
    raw = str(value)
    try:
        return (1, 0, float(raw))
    except (TypeError, ValueError):  # noqa: BLE001 — ordinary format-detection fallthrough: not a float, try ISO-8601 next
        pass
    try:
        from datetime import UTC, datetime

        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return (1, 1, parsed.timestamp())
    except (TypeError, ValueError, OverflowError):
        return (1, 2, raw)


def _ordered_documents(docs: list[Any]) -> list[Any]:
    """Stable oldest-first order for per-document transactional cursors."""
    return sorted(
        docs, key=lambda doc: _checkpoint_order(getattr(doc, "updated_at", None))
    )


def _jira_jql_date(value: Any) -> str | None:
    """Render an ISO8601 watermark as a Jira JQL datetime (``yyyy-MM-dd HH:mm``)."""
    import re

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})", str(value))
    return (
        f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}"
        if m
        else None
    )


def _jira_scope_clauses(inst: dict[str, Any], ids: list[str] | None) -> list[str]:
    """The JQL clauses one instance's project scope and id narrowing imply."""
    clauses: list[str] = []
    keys = [str(k) for k in (inst.get("project_keys") or []) if k]
    if keys:
        clauses.append(f"project in ({','.join(keys)})")
    if ids:
        clauses.append(f"key in ({','.join(str(i) for i in ids)})")
    return clauses


def _jira_jql_clauses(
    inst: dict[str, Any], since: str | None, ids: list[str] | None
) -> list[str]:
    """The JQL clauses one instance's scope, id narrowing, and delta imply."""
    clauses = _jira_scope_clauses(inst, ids)
    if since and (d := _jira_jql_date(since)):
        clauses.append(f'updated >= "{d}"')
    if extra := str(inst.get("jql") or "").strip():
        clauses.append(f"({extra})")
    return clauses


def _jira_jql(inst: dict[str, Any], since: str | None, ids: list[str] | None) -> str:
    where = " AND ".join(_jira_jql_clauses(inst, since, ids))
    return (
        f"{where} ORDER BY updated DESC"
        if where
        # Jira Cloud /search/jql (search-and-reconcile) rejects an UNBOUNDED query
        # (400); a wide created-bound keeps "all issues" valid (CONCEPT:AU-KG.compute.jira-first-class-delta).
        else 'created >= "1970-01-01" ORDER BY updated DESC'
    )


def _jira_assignee_rows(
    fields: dict[str, Any], instance: str, node_id: str, src: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The issue's assignee as a :Person plus the issue's ``has_role`` edge."""
    assignee = fields.get("assignee")
    if not isinstance(assignee, dict):
        return [], []
    uid = assignee.get("accountId") or assignee.get("name")
    if not uid:
        return [], []
    user_node = f"jira:{instance}:user:{uid}"
    return (
        [
            {
                "id": user_node,
                "type": "person",
                "name": assignee.get("displayName") or f"User {uid}",
                "domain": "jira",
                "source_system": src,
            }
        ],
        [
            {
                "source": node_id,
                "target": user_node,
                "type": "has_role",
                "domain": "jira",
            }
        ],
    )


def _jira_epic_rows(
    fields: dict[str, Any], instance: str, node_id: str, src: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The issue's epic as a :Goal plus the issue's ``part_of`` edge."""
    parent = fields.get("parent")
    epic = (
        parent.get("key")
        if isinstance(parent, dict)
        else fields.get("customfield_10014")
    )
    if not epic:
        return [], []
    epic_node = f"jira:{instance}:epic:{epic}"
    return (
        [
            {
                "id": epic_node,
                "type": "goal",
                "name": f"Epic {epic}",
                "domain": "jira",
                "source_system": src,
            }
        ],
        [
            {
                "source": node_id,
                "target": epic_node,
                "type": "part_of",
                "domain": "jira",
            }
        ],
    )


def _jira_issue_rows(doc: Any, instance: str, src: str) -> list[dict[str, Any]]:
    """One Jira record as its person/epic nodes plus its own :Issue node."""
    key = getattr(doc, "id", None)
    if not key:
        return []
    fields = _record_of(doc).get("fields") or {}
    node_id = f"jira:{instance}:issue:{key}"
    user_entities, user_links = _jira_assignee_rows(fields, instance, node_id, src)
    epic_entities, epic_links = _jira_epic_rows(fields, instance, node_id, src)
    return [
        *user_entities,
        *epic_entities,
        {
            "id": node_id,
            "type": "issue",
            "name": fields.get("summary") or f"Issue {key}",
            "status": (fields.get("status") or {}).get("name", ""),
            "priority": (fields.get("priority") or {}).get("name", ""),
            "issueKey": str(key),
            "domain": "jira",
            "source_system": src,
            "externalToolId": str(key),
            "updatedAt": fields.get("updated"),
            "_links": [*user_links, *epic_links],
        },
    ]


def _jira_entities(docs: list[Any], instance: str) -> list[dict[str, Any]]:
    """Map drained Jira records → issue/person/epic entities (the mapping inherited
    from the removed ``_hydrate_jira``).

    AU-P1-5 (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): the assignee's
    ``has_role`` edge and the epic's ``part_of`` edge are both self-sourced from the
    issue's own record (assignee/epic are derived from the SAME issue document, never
    fetched separately) and carried on the ISSUE's own ``_links`` — the issue is also
    the only one of the three entity types with a real per-record ``updatedAt``.
    """
    src = f"jira:{instance}"
    entities: list[dict[str, Any]] = []
    for doc in docs:
        entities.extend(_jira_issue_rows(doc, instance, src))
    return entities


def _sync_jira(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Jira issues as typed issue/person/epic entities (CONCEPT:AU-KG.compute.jira-first-class-delta).

    Per configured instance, drains the ``jira`` mcp_tool preset over its
    ``atlassian-mcp`` server with a JQL ``updated >= <watermark>`` server-side delta
    (the write-layer content-hash is the second guard); rebuilds the issue graph from
    each record and emits one ChangeEnvelope per typed object. ``ids`` narrows to
    specific keys (webhook). Replaces the removed single-shot ``_hydrate_jira``.

    Deployment note (CONCEPT:AU-KG.compute.jira-first-class-delta): wire ``atlassian-mcp`` in the source
    ``mcp_config`` over a runtime-injected streamable-http ``url``, mirroring
    freshrss-mcp / plane-mcp — never a local ``command`` venv binary, which would
    incorrectly spawn a stdio server on the host.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    issue/person/epic is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`,
    ``source_instance=<jira instance name>`` so the per-envelope watermark key
    (``jira:<name>``) matches this handler's own existing ``wm_key`` format exactly.
    """
    instances = _resolve_tracker_instances(
        "jira_instances",
        default_name="jira",
        default_server="atlassian-mcp",
        scope_key="project_keys",
        scope_setting="JIRA_PROJECT_KEYS",
    )
    results: list[dict[str, Any]] = []
    total_e = total_failed = 0
    for inst in instances:
        name = str(inst.get("name") or "jira")
        server = str(inst.get("server") or "atlassian-mcp")
        since = (
            None
            if mode == "full"
            else _read_envelope_watermark(
                engine,
                "jira",
                source_instance=name,
            )
        )
        if mode == "reconcile":
            conn = _build_preset_conn(
                "jira", server, {"jql": _jira_jql(inst, None, None)}
            )
            drained, fetch_ok = _drain_incremental(conn, None)
            live = {str(getattr(d, "id", "")) for d in drained}
            results.append(
                _reconcile(
                    engine,
                    "jira",
                    live,
                    source_instance=name,
                    fetch_ok=fetch_ok,
                )
                | {"instance": name}
            )
            continue
        conn = _build_preset_conn("jira", server, {"jql": _jira_jql(inst, since, ids)})
        docs, _ = _drain_incremental(conn, since)
        entities = _jira_entities(docs, name)
        ok, failed = _ingest_entities_via_envelope(
            engine, "jira", entities, source_instance=name
        )
        total_e += ok
        total_failed += failed
        results.append({"instance": name, "issues": len(docs), "since": since})
    return {
        "status": "ok",
        "source": "jira",
        "mode": mode,
        "delta_capable": True,
        "instances": results,
        "nodes_hydrated": total_e,
        "failed": total_failed,
    }


def _plane_entities(
    docs: list[Any], instance: str, project_id: str
) -> list[dict[str, Any]]:
    """Map drained Plane work items → issue + project entities (inherited from the
    removed ``_hydrate_plane``).

    AU-P1-5 (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): an issue's ``part_of``
    edge to its project is self-sourced (both are resolved from the SAME per-project
    drain) and carried on the ISSUE's own ``_links`` — the issue also carries the real
    per-record ``updated_at``, unlike the versionless synthetic project node.
    """
    entities: list[dict[str, Any]] = []
    src = f"plane:{instance}"
    proj_node = f"plane:{instance}:proj:{project_id}"
    proj_emitted = False
    for doc in docs:
        iid = getattr(doc, "id", None)
        if not iid:
            continue
        rec = _record_of(doc)
        state = rec.get("state")
        state_name = state.get("name", "") if isinstance(state, dict) else (state or "")
        node_id = f"plane:{instance}:issue:{iid}"
        if not proj_emitted:
            entities.append(
                {
                    "id": proj_node,
                    "type": "software_project",
                    "name": f"Plane Project {project_id}",
                    "domain": "plane",
                    "source_system": src,
                }
            )
            proj_emitted = True
        entities.append(
            {
                "id": node_id,
                "type": "issue",
                "name": rec.get("name") or f"Plane Issue {iid}",
                "state": state_name,
                "priority": rec.get("priority") or "",
                "domain": "plane",
                "source_system": src,
                "externalToolId": str(iid),
                "updatedAt": rec.get("updated_at"),
                "_links": [
                    {
                        "source": node_id,
                        "target": proj_node,
                        "type": "part_of",
                        "domain": "plane",
                    }
                ],
            }
        )
    return entities


def _plane_project_slice(
    engine: Any, name: str, server: str, pid: str, mode: str, ids: list[str] | None
) -> tuple[int, int, int]:
    """Drain + ingest one Plane project, as ``(issues_written, failed, records_seen)``.

    ``source_instance=<instance>:<project_id>`` so the per-envelope watermark key
    (``plane:<instance>:<project_id>``) matches this handler's own key format.
    """
    source_instance = f"{name}:{pid}"
    since = (
        None
        if mode == "full"
        else _read_envelope_watermark(
            engine,
            "plane",
            source_instance=source_instance,
        )
    )
    params: dict[str, Any] = {"project_id": pid}
    if ids:
        params["filters"] = {"id": ids}
    conn = _build_preset_conn("plane", server, params)
    docs, _ = _drain_incremental(conn, since)
    entities = _plane_entities(docs, name, pid)
    _ok, failed = _ingest_entities_via_envelope(
        engine,
        "plane",
        entities,
        source_instance=source_instance,
    )
    issues = sum(1 for e in entities if e["type"] == "issue")
    return issues, failed, len(docs)


def _plane_instance_result(
    engine: Any, inst: dict[str, Any], mode: str, ids: list[str] | None
) -> tuple[dict[str, Any], int, int]:
    """One Plane instance's configured projects, as ``(result row, written, failed)``."""
    name = str(inst.get("name") or "plane")
    server = str(inst.get("server") or "plane-mcp")
    projects = [str(p) for p in (inst.get("projects") or []) if p]
    if not projects:
        return (
            {
                "instance": name,
                "status": "skipped",
                "reason": "no projects configured",
            },
            0,
            0,
        )
    inst_ok = inst_failed = inst_issues = 0
    for pid in projects:
        issues, failed, seen = _plane_project_slice(
            engine, name, server, pid, mode, ids
        )
        inst_ok += issues
        inst_failed += failed
        inst_issues += seen
    return {"instance": name, "issues": inst_issues}, inst_ok, inst_failed


def _sync_plane(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Plane work items as typed issue/project entities (CONCEPT:AU-KG.compute.plane-first-class-delta).

    Per configured instance × project, drains the ``plane`` mcp_tool preset over its
    ``plane-mcp`` server (a SECOND Plane workspace is a second instance row pointing at
    a second server). Delta = the ``updated_at`` watermark + content-hash. Replaces the
    removed ``_hydrate_plane``.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    issue/project is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`,
    ``source_instance=<instance>:<project_id>`` so the per-envelope watermark key
    (``plane:<instance>:<project_id>``) matches this handler's own existing ``wm_key``
    format exactly.
    """
    instances = _resolve_tracker_instances(
        "plane_instances",
        default_name="plane",
        default_server="plane-mcp",
        scope_key="projects",
        scope_setting="PLANE_PROJECT_IDS",
    )
    results: list[dict[str, Any]] = []
    total_e = total_failed = 0
    for inst in instances:
        row, written, failed = _plane_instance_result(engine, inst, mode, ids)
        results.append(row)
        total_e += written
        total_failed += failed
    return {
        "status": "ok",
        "source": "plane",
        "mode": mode,
        "delta_capable": True,
        "instances": results,
        "nodes_hydrated": total_e,
        "failed": total_failed,
    }


def _confluence_processor(engine: Any) -> Any:
    from ..ontology.document_processing import ChunkingConfig, DocumentProcessor

    return DocumentProcessor(
        getattr(engine, "backend", None),
        engine=engine,
        chunking=ChunkingConfig(),
        contextual=True,
    )


def _lazy_document_processor(state: SimpleNamespace) -> Any:
    """The lazily-built ``DocumentProcessor``, created on the first real page."""
    if state.proc is None:
        state.proc = _confluence_processor(state.engine)
    return state.proc


def _confluence_page_params(space: str | None, ids: list[str] | None) -> dict[str, Any]:
    """The ``confluence`` preset params for one space (``ids`` narrows to pages)."""
    params: dict[str, Any] = {}
    if space:
        params["space_id"] = [space]
    if ids:
        params["id_"] = ids
    return params


def _confluence_ingest_page(
    proc: Any, doc: Any, inst_name: str, source_instance: str
) -> bool:
    """Ingest one Confluence page; ``False`` when that one page failed."""
    rec = _record_of(doc)
    try:
        proc.process(
            getattr(doc, "text", "") or "",
            document_id=f"confluence:{inst_name}:{getattr(doc, 'id', '')}",
            title=getattr(doc, "title", "") or str(getattr(doc, "id", "")),
            doc_type="wiki",
            source=getattr(doc, "source_uri", ""),
            metadata={
                "source_system": make_source_id("confluence", inst_name),
                "space_id": rec.get("spaceId"),
                "version": (rec.get("version") or {}).get("number"),
                "confluence_page_id": str(getattr(doc, "id", "")),
                "updated_at": getattr(doc, "updated_at", None),
            },
            external_access=getattr(doc, "external_access", None),
            connector="confluence",
            source_instance=source_instance,
            checkpoint=getattr(doc, "updated_at", None),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — one bad page must not abort
        logger.warning(
            "[KG-2.123] confluence page ingest failed for %s: %s",
            getattr(doc, "id", "?"),
            exc,
        )
        return False


def _confluence_space_pages(
    state: SimpleNamespace,
    inst_name: str,
    server: str,
    space: str | None,
    mode: str,
    ids: list[str] | None,
) -> tuple[int, bool]:
    """Drain + ingest one space, as ``(pages_ingested, partition_failed)``.

    The first failed page aborts this partition so the watermark is never advanced
    past a page that did not land.
    """
    source_instance = f"{inst_name}:{space or 'all'}"
    since = (
        None
        if mode == "full"
        else _read_envelope_watermark(
            state.engine,
            "confluence",
            source_instance=source_instance,
        )
    )
    conn = _build_preset_conn("confluence", server, _confluence_page_params(space, ids))
    drained, _ = _drain_incremental(conn, since)
    docs = _ordered_documents(drained)
    pages = 0
    for doc in docs:
        if not _confluence_ingest_page(
            _lazy_document_processor(state), doc, inst_name, source_instance
        ):
            state.failed += 1
            return pages, True
        pages += 1
    return pages, False


def _confluence_spaces(inst: dict[str, Any]) -> list[str | None]:
    """The configured space ids for one instance, or ``[None]`` for site-wide."""
    return [str(s) for s in (inst.get("spaces") or []) if s] or [None]


def _confluence_instance_pages(
    state: SimpleNamespace, inst: dict[str, Any], mode: str, ids: list[str] | None
) -> dict[str, Any]:
    """Every configured space of one Confluence instance, as a result row."""
    name = str(inst.get("name") or "confluence")
    server = str(inst.get("server") or "atlassian-mcp")
    spaces = _confluence_spaces(inst)
    pages = 0
    instance_failed = False
    for space in spaces:
        space_pages, partition_failed = _confluence_space_pages(
            state, name, server, space, mode, ids
        )
        pages += space_pages
        if partition_failed:
            instance_failed = True
            break
    state.total += pages
    return {
        "instance": name,
        "pages": pages,
        "status": "partial" if instance_failed else "ok",
    }


def _sync_confluence(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Full-mirror Confluence pages as ``:ConfluencePage`` Documents (CONCEPT:AU-KG.compute.confluence-first-class-delta).

    Per configured instance × space, drains the ``confluence`` mcp_tool preset
    (Cloud-v2 ``get_pages``, recency-sorted, body inline) over its ``atlassian-mcp``
    server and ingests each page through the KG-2.48 ``DocumentProcessor`` (chunk +
    embed) so the wiki is fully searchable. Delta = the ``version.createdAt`` since
    filter + the write-layer content-hash. ``ids`` narrows to specific pages (webhook).
    NOT relevance-gated — internal wiki is curated knowledge.

    Deployment note (CONCEPT:AU-KG.compute.confluence-first-class-delta): the ``atlassian-mcp`` server is reached over
    streamable-http at its runtime-injected fleet URL — wire it in the source
    ``mcp_config`` with ``transport``/``url`` (mirroring freshrss-mcp / plane-mcp),
    never a local ``command`` venv binary. Confluence Cloud v2 paths are
    bare (``/spaces``, ``/pages``), so the **service** must set
    ``ATLASSIAN_CONFLUENCE_CLOUD_URL=https://<site>.atlassian.net/wiki/api/v2`` — the
    per-suite override in ``atlassian_agent.auth.get_confluence_cloud_client``;
    otherwise the client falls back to the Jira base URL and every call 404s.
    """
    # CA-22/P11: document-shaped delegated pipeline (commits via DocumentProcessor's
    # own ApplyChangeEnvelope path, not a ChangeEnvelope built here) -- AST-
    # reachability call to the preflight chokepoint; batch=[] checks nothing
    # per-record until a manifest declares conflict_policy for "confluence".
    _apply_with_preflight(engine, "confluence", [])

    instances = _resolve_tracker_instances(
        "confluence_instances",
        default_name="confluence",
        default_server="atlassian-mcp",
        scope_key="spaces",
        scope_setting="CONFLUENCE_SPACE_IDS",
    )
    state = SimpleNamespace(engine=engine, proc=None, total=0, failed=0)
    results = [_confluence_instance_pages(state, inst, mode, ids) for inst in instances]
    return {
        "status": "partial" if state.failed else "ok",
        "source": "confluence",
        "mode": mode,
        "delta_capable": True,
        "instances": results,
        "pages_ingested": state.total,
        "failed": state.failed,
    }


# ── Ops / platform connectors as typed OWL entities (CONCEPT:AU-KG.compute.dockerhub-repositories–2.161) ──────
#
# Seven first-class delta connectors that reach their upstream ONLY through a fleet
# ``*-mcp`` server (like jira/confluence/plane) and rebuild **typed** entities mapped to
# OWL classes — not generic Documents. Each is MCP-configured: its "configured" signal is
# *"the server is registered in mcp_config.json"*. Delta = a per-source ISO ``updated_at``
# watermark + the write-layer content-hash; the server it reaches is in ``_MCP_TRACKER_SERVERS``.
#
# CONCEPT:AU-KG.compute.dockerhub-repositories — DockerHub repositories → :Repository / :ContainerImage
# CONCEPT:AU-KG.compute.langfuse-traces-observations — Langfuse traces/observations → :Trace / :Observation / :Generation
# CONCEPT:AU-KG.compute.technitium-dns-zones-records — Technitium DNS zones+records → :DnsZone / :DnsRecord
# CONCEPT:AU-KG.compute.tunnel-manager-hosts — tunnel-manager hosts → :Host / :Tunnel
# CONCEPT:AU-KG.compute.uptime-kuma-monitors — Uptime Kuma monitors → :Monitor / :HeartbeatStat
# CONCEPT:AU-KG.compute.home-assistant-states — Home Assistant states → :Device / :Entity
# CONCEPT:AU-KG.compute.twenty-crm-people-companies — Twenty CRM people/companies/opportunities → :Person / :Company / :Opportunity


def _configured_server(server_candidates: tuple[str, ...]) -> str | None:
    """The first candidate ``*-mcp`` server actually registered in ``mcp_config.json``
    (or its ``<name>-mcp`` alias), or ``None`` when none is — so a handler reaches the
    upstream through the server the operator really configured (the catalog name and the
    local config key can differ, e.g. ``uptime-mcp`` vs ``uptime-kuma-mcp``)."""
    try:
        from ...protocols.source_connectors.connectors.mcp_package import (
            _load_mcp_config,
        )

        servers = _load_mcp_config() or {}
    except Exception:  # noqa: BLE001 — no readable config → not configured here
        return None
    for cand in server_candidates:
        if cand in servers:
            return cand
        if f"{cand}-mcp" in servers:
            return f"{cand}-mcp"
    return None


def _server_configured(server_candidates: tuple[str, ...]) -> bool:
    """True when any candidate ``*-mcp`` server (or its de-suffixed alias) is in
    ``mcp_config.json`` — the connector reaches the upstream only through that server."""
    return _configured_server(server_candidates) is not None


def _drain_preset(
    preset: str, *, server: str = "", params: dict[str, Any] | None = None
) -> list[Any]:
    """Build the ``mcp_tool`` connector for a preset and drain ONE full sweep.

    Used by the typed handlers below: the preset does the list/page/cursor drain, the
    handler maps each ``metadata.record`` to a typed entity. Bounded by the connector's
    own ``max_pages`` so a cold run can't loop unbounded.
    """
    from ...protocols.source_connectors.registry import build_connector

    config: dict[str, Any] = {"preset": preset}
    if server:
        config["server"] = server
    if params:
        config["params"] = params
    conn = build_connector("mcp_tool", config)
    if hasattr(conn, "poll_all"):
        return list(conn.poll_all())  # type: ignore[attr-defined]
    return list(conn.load())  # type: ignore[attr-defined]


def _envelope_batch(
    entities: list[dict[str, Any]],
    connector: str,
    source_instance: str,
    version_field: str,
    activity_id: str | None,
) -> list[Any]:
    """The checkpoint-ordered ``ChangeEnvelope`` page for one connector call.

    When ambient provenance is active, every entity's own ``_links`` gains a
    ``derived_from`` edge to this run's Activity, committed atomically with that
    entity's own envelope (no extra engine round-trip).
    """
    from ..ingestion.change_envelope import ChangeEnvelope

    batch: list[ChangeEnvelope] = []
    for record in sorted(
        entities, key=lambda item: _checkpoint_order(item.get(version_field))
    ):
        if activity_id:
            record = {
                **record,
                "_links": [
                    *(record.get("_links") or []),
                    {"target": activity_id, "type": "derived_from"},
                ],
            }
        batch.append(
            ChangeEnvelope.from_connector_record(
                record,
                connector=connector,
                source_instance=source_instance,
                id_field="id",
                version_field=version_field,
                checkpoint=record.get(version_field),
            )
        )
    return batch


def _truncate_at_preflight_block(
    engine: Any, connector: str, batch: list[Any], source_instance: str
) -> tuple[list[Any], int]:
    """Cut the page to the CONTIGUOUS prefix before the first blocked envelope.

    CA-22/P11 preflight chokepoint: a preflight block must stop the watermark
    advance exactly like a backend rejection does (a later envelope's newer
    checkpoint must never be committed past an unapplied earlier one). Blocked
    envelopes — and everything after them this pass — are retried on the next sync
    run, same as any other break-on-first-failure outcome. Returns
    ``(truncated batch, blocked count)``.
    """
    _, blocked = _apply_with_preflight(
        engine, connector, batch, source_instance=source_instance
    )
    if not blocked:
        return batch, 0
    first_blocked_idx = min(entry["index"] for entry in blocked)
    for entry in blocked:
        logger.warning(
            "%s envelope %s blocked by backfeed preflight: %s",
            connector,
            getattr(entry["envelope"], "idempotency_key", "?"),
            entry["conflict_or_rejection"],
        )
    return batch[:first_blocked_idx], len(blocked)


def _commit_envelope_batch(
    engine: Any, connector: str, batch: list[Any]
) -> tuple[int, int]:
    """Commit the page, counting only the last CONTIGUOUS run of successes.

    A later envelope can carry a newer cursor, so a gap must never be crossed.
    Returns ``(succeeded, failed)``.
    """
    from ..ingestion.envelope_ingest import ingest_envelopes

    ok = 0
    for env, result in zip(batch, ingest_envelopes(engine, batch), strict=True):
        if result.get("status") not in {"success", "skipped"}:
            logger.warning(
                "%s envelope %s failed: %s",
                connector,
                env.idempotency_key,
                result.get("error"),
            )
            return ok, 1
        ok += 1
    return ok, 0


def _close_ambient_provenance(
    engine: Any,
    connector: str,
    source_instance: str,
    activity_id: str,
    ok: int,
    failed: int,
) -> None:
    """Finalize this run's PROV-O Activity and persist its ONE summary ``:Claim``."""
    from ..etl.lineage import (
        record_connector_sync_activity,
        record_connector_sync_claim,
    )

    record_connector_sync_activity(
        engine,
        connector=connector,
        source_instance=source_instance,
        status="ok" if not failed else "partial",
        record_count=ok,
        failed_count=failed,
        activity_id=activity_id,
    )
    record_connector_sync_claim(
        engine,
        connector=connector,
        source_instance=source_instance,
        record_count=ok,
        activity_id=activity_id,
    )


def _ingest_entities_via_envelope(
    engine: Any,
    connector: str,
    entities: list[dict[str, Any]],
    *,
    source_instance: str = "",
    version_field: str = "updatedAt",
) -> tuple[int, int]:
    """AU-P1-5 (CONCEPT:AU-KG.ingest.envelope-atomic-transaction) shared migration tail for the
    "typed OWL entity" handlers below — replaces the old ``_ingest_typed`` single
    ``engine.ingest_external_batch(source, entities, rels)`` call.

    Each handler already builds its whole ``entities`` list from records fetched
    in ONE handler invocation (self-sourced: an entity's cross-references —
    "this image's repo", "this issue's assignee", "this transaction's account"
    — are always derived from the SAME drain call, never a separate delta
    batch), so a relationship is attached to whichever entity in this list is
    the one that actually carries a real per-record ``version_field`` (that
    entity's own ``_links`` key, built by the caller BEFORE this is invoked) —
    an unchanged entity's envelope is safely idempotent-skipped, while a
    new/changed entity's envelope still (re-)asserts every edge attached to
    it. A relationship attached to a VERSIONLESS entity (e.g. a static
    namespace/registry/project container whose own fields never change) would
    silently stop being re-asserted after that entity's first successful
    write, since its idempotency key never changes again — callers MUST NOT
    attach ``_links`` to a versionless entity for this reason.

    Returns ``(succeeded, failed)`` entity counts. The watermark advance
    happens per-envelope, atomically, inside :func:`~..ingestion.envelope_ingest.ingest_envelope`
    itself — there is no separate ``_write_watermark`` call left to make here.

    **Ambient provenance (W3.4, CONCEPT:AU-KG.ingest.ambient-connector-provenance).**
    When ambient epistemics is enabled for ``connector``
    (:func:`~..ingestion.envelope_ingest._ambient_epistemic_enabled`) and this
    call has entities to ingest, ONE PROV-O Activity node is recorded for this
    whole call (:func:`~..etl.lineage.record_connector_sync_activity`) — never
    per row — and every entity's own ``_links`` gains a ``"derived_from"`` edge
    to it, committed atomically with that entity's own envelope (no extra
    engine round-trip). After the loop, the Activity is updated with final
    counts/status and ONE summary ``:Claim`` is persisted
    (:func:`~..etl.lineage.record_connector_sync_claim`) — "``connector``
    reported N record(s) as of T". Both are best-effort: a provenance-write
    failure (or ambient epistemics disabled) never affects the entities
    themselves or this function's return value.
    """
    from ..ingestion.envelope_ingest import _ambient_epistemic_enabled

    activity_id: str | None = None
    if entities and _ambient_epistemic_enabled(connector):
        from ..etl.lineage import record_connector_sync_activity

        activity_id = record_connector_sync_activity(
            engine,
            connector=connector,
            source_instance=source_instance,
            status="running",
        )

    batch = _envelope_batch(
        entities, connector, source_instance, version_field, activity_id
    )
    batch, failed = _truncate_at_preflight_block(
        engine, connector, batch, source_instance
    )
    ok, commit_failed = _commit_envelope_batch(engine, connector, batch)
    failed += commit_failed

    if activity_id:
        _close_ambient_provenance(
            engine, connector, source_instance, activity_id, ok, failed
        )
    return ok, failed


def _dockerhub_namespaces(ids: list[str] | None) -> list[str]:
    """The configured namespaces: ``DOCKERHUB_NAMESPACES`` CSV, else ``ids``."""
    from ...core.config import setting

    return [
        n.strip()
        for n in (
            setting("DOCKERHUB_NAMESPACES", default="")
            or setting("DOCKERHUB_NAMESPACE", default="")
        ).split(",")
        if n.strip()
    ] or [str(i) for i in (ids or [])]


def _dockerhub_image_entity(doc: Any, ns: str, repo_node: str) -> dict[str, Any] | None:
    """One DockerHub repo record as a :ContainerImage the namespace ``contains``."""
    from ..etl.transforms import coalesce

    rec = _record_of(doc)
    name = coalesce(rec, "name") or getattr(doc, "id", None)
    if not name:
        return None
    img_id = f"dockerhub:{ns}/{name}"
    return {
        "id": img_id,
        "type": "container_image",
        "name": f"{ns}/{name}",
        "description": rec.get("description") or "",
        "pull_count": rec.get("pull_count"),
        "star_count": rec.get("star_count"),
        "is_private": rec.get("is_private"),
        "domain": "dockerhub",
        "source_system": make_source_id("dockerhub", ns),
        "externalToolId": f"{ns}/{name}",
        "updatedAt": rec.get("last_updated"),
        "_links": [
            {
                "source": repo_node,
                "target": img_id,
                "type": "contains",
                "domain": "dockerhub",
            }
        ],
    }


def _dockerhub_namespace_slice(
    engine: Any, ns: str, mode: str
) -> tuple[int, int, str | None]:
    """Drain + ingest one namespace, as ``(images_seen, failed, since)``."""
    from ..etl.transforms import stable_id

    since = (
        None
        if mode == "full"
        else _read_envelope_watermark(
            engine,
            "dockerhub",
            source_instance=ns,
        )
    )
    docs = _drain_preset("dockerhub-repos", params={"namespace": ns})
    repo_node = stable_id(ns, prefix="dockerhub")
    entities: list[dict[str, Any]] = [
        {
            "id": repo_node,
            "type": "repository",
            "name": ns,
            "domain": "dockerhub",
            "source_system": make_source_id("dockerhub", ns),
        },
        *_entity_rows(docs, lambda doc: _dockerhub_image_entity(doc, ns, repo_node)),
    ]
    _ok, failed = _ingest_entities_via_envelope(
        engine, "dockerhub", entities, source_instance=ns
    )
    return len(docs), failed, since


def _sync_dockerhub(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest DockerHub repositories as :Repository + :ContainerImage (CONCEPT:AU-KG.compute.dockerhub-repositories).

    Per configured namespace (``DOCKERHUB_NAMESPACES`` CSV, else ``DOCKERHUB_NAMESPACE``,
    else ``ids`` as namespaces) drains the ``dockerhub-repos`` preset over ``dockerhub-mcp``
    and rebuilds each repo as a :ContainerImage (image coordinates + pull/star counts) that
    ``contains`` the namespace's :Repository. Delta = the ``last_updated`` watermark.

    Uses the shared transform primitives (CONCEPT:AU-KG.etl.transform-primitives) —
    :func:`~..etl.transforms.coalesce` for the image-name fallback and
    :func:`~..etl.transforms.stable_id` for the ``dockerhub:<ns>[/<name>]`` node ids
    — as the first migrated handler proving the pattern.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    image is one ``ChangeEnvelope`` routed through
    :func:`~..ingestion.envelope_ingest.ingest_envelope` via the shared
    :func:`_ingest_entities_via_envelope` tail. The namespace's :Repository is
    versionless (its own fields never change), so the ``contains`` edge is
    carried on the IMAGE's ``_links`` (self-sourced — repo and image are always
    resolved together from the SAME per-namespace drain) rather than the
    repo's: attaching it to the repo would silently stop re-asserting new
    images once the repo's own idempotency-key-fixed envelope stops
    re-applying after its first successful write.
    """
    if not _server_configured(("dockerhub-mcp", "dockerhub-api")):
        return {"status": "skipped", "reason": "dockerhub-mcp not in mcp_config"}

    namespaces = _dockerhub_namespaces(ids)
    if not namespaces:
        return {"status": "skipped", "reason": "no DockerHub namespace configured"}

    total = 0
    total_failed = 0
    results: list[dict[str, Any]] = []
    for ns in namespaces:
        images, failed, since = _dockerhub_namespace_slice(engine, ns, mode)
        total_failed += failed
        # NOTE: the per-envelope watermark advance already happened atomically
        # inside ingest_envelope (monotonic-guarded per entity) — `since` above
        # is only this run's read of the last-advanced watermark, not a second
        # write (AU-P1-5).
        total += images
        results.append({"namespace": ns, "images": images, "since": since})
    return {
        "status": "ok",
        "source": "dockerhub",
        "mode": mode,
        "delta_capable": True,
        "namespaces": results,
        "images_ingested": total,
        "failed": total_failed,
    }


def _langfuse_trace_entity(doc: Any) -> dict[str, Any] | None:
    """One Langfuse trace record as a :Trace entity."""
    tid = getattr(doc, "id", None)
    if not tid:
        return None
    rec = _record_of(doc)
    return {
        "id": f"langfuse:trace:{tid}",
        "type": "trace",
        "name": rec.get("name") or f"Trace {tid}",
        "user_id": rec.get("userId"),
        "session_id": rec.get("sessionId"),
        "domain": "langfuse",
        "source_system": "langfuse",
        "externalToolId": str(tid),
        "updatedAt": rec.get("timestamp"),
    }


def _langfuse_observation_entity(doc: Any) -> dict[str, Any] | None:
    """One Langfuse observation as an :Observation (LLM calls → :Generation)."""
    oid = getattr(doc, "id", None)
    if not oid:
        return None
    rec = _record_of(doc)
    is_gen = str(rec.get("type") or "").upper() == "GENERATION"
    node_id = f"langfuse:obs:{oid}"
    entity: dict[str, Any] = {
        "id": node_id,
        "type": "generation" if is_gen else "observation",
        "name": rec.get("name") or f"Observation {oid}",
        "model": rec.get("model"),
        "domain": "langfuse",
        "source_system": "langfuse",
        "externalToolId": str(oid),
        "updatedAt": rec.get("startTime"),
    }
    if tid := rec.get("traceId"):
        entity["_links"] = [
            {
                "source": node_id,
                "target": f"langfuse:trace:{tid}",
                "type": "part_of",
                "domain": "langfuse",
            }
        ]
    return entity


def _sync_langfuse(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Langfuse traces + observations as :Trace / :Observation / :Generation
    (CONCEPT:AU-KG.compute.langfuse-traces-observations).

    Drains the ``langfuse-traces`` and ``langfuse-observations`` presets over
    ``langfuse-mcp``; each trace is a :Trace, each observation a :Observation (LLM-call
    observations — ``type == 'GENERATION'`` — are :Generation), linked ``part_of`` their
    trace via ``traceId``. Delta = the ``timestamp`` / ``startTime`` watermark.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    trace/observation is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`.
    An observation's ``part_of`` edge to its trace is carried on the OBSERVATION's own
    ``_links`` — self-sourced (both are drained together in this one call) and the
    observation is itself the entity that changes version-to-version.
    """
    if not _server_configured(("langfuse-mcp", "langfuse-agent")):
        return {"status": "skipped", "reason": "langfuse-mcp not in mcp_config"}
    since = None if mode == "full" else _read_envelope_watermark(engine, "langfuse")
    src = "langfuse"

    trace_docs = _drain_preset("langfuse-traces")
    obs_docs = _drain_preset("langfuse-observations")
    entities: list[dict[str, Any]] = [
        *_entity_rows(trace_docs, _langfuse_trace_entity),
        *_entity_rows(obs_docs, _langfuse_observation_entity),
    ]
    ok, failed = _ingest_entities_via_envelope(engine, src, entities)
    return {
        "status": "ok",
        "source": "langfuse",
        "mode": mode,
        "delta_capable": True,
        "traces": len(trace_docs),
        "observations": len(obs_docs),
        "nodes_hydrated": ok,
        "failed": failed,
        "since": since,
    }


def _technitium_record_value(rec: dict[str, Any]) -> str:
    """A DNS record's rendered value, out of its ``rData`` envelope."""
    rdata = rec.get("rData")
    if not isinstance(rdata, dict):
        return ""
    return str(rdata.get("ipAddress") or rdata.get("value") or rdata.get("text") or "")


def _technitium_record_entity(
    rec: Any, zname: str, zone_node: str
) -> dict[str, Any] | None:
    """One Technitium record as a :DnsRecord ``part_of`` its :DnsZone."""
    if not isinstance(rec, dict):
        return None
    rname = rec.get("name")
    rtype = rec.get("type")
    value = _technitium_record_value(rec)
    rec_node = f"technitium:rec:{zname}:{rname}:{rtype}:{value}"
    return {
        "id": rec_node,
        "type": "dns_record",
        "name": f"{rname} {rtype}".strip(),
        "record_type": rtype,
        "ttl": rec.get("ttl"),
        "value": value,
        "disabled": rec.get("disabled"),
        "domain": "technitium",
        "source_system": "technitium",
        "_links": [
            {
                "source": rec_node,
                "target": zone_node,
                "type": "part_of",
                "domain": "technitium",
            }
        ],
    }


def _technitium_zone_records(
    call: Callable[..., Any], zname: str, zone_node: str
) -> list[dict[str, Any]]:
    """Every :DnsRecord in one zone; one bad zone never aborts the rest."""
    from ...protocols.source_connectors.connectors.rest import _dig

    try:
        rec_res = call(
            "get_records", {"domain": zname, "zone": zname, "list_zone": True}
        )
    except Exception as exc:  # noqa: BLE001 — one bad zone never aborts the rest
        logger.warning(
            "[KG-2.157] technitium records fetch failed for %s: %s", zname, exc
        )
        return []
    records = (
        (_dig(rec_res, "response.records") or []) if isinstance(rec_res, dict) else []
    )
    return _entity_rows(
        records, lambda rec: _technitium_record_entity(rec, zname, zone_node)
    )


def _technitium_zone_slice(
    call: Callable[..., Any], zone: Any
) -> tuple[list[dict[str, Any]], int]:
    """One zone plus its records, as ``(entities, record_count)``."""
    if not isinstance(zone, dict):
        return [], 0
    zname = zone.get("name")
    if not zname:
        return [], 0
    zone_node = f"technitium:zone:{zname}"
    entities: list[dict[str, Any]] = [
        {
            "id": zone_node,
            "type": "dns_zone",
            "name": zname,
            "zone_type": zone.get("type"),
            "disabled": zone.get("disabled"),
            "domain": "technitium",
            "source_system": "technitium",
            "externalToolId": zname,
        }
    ]
    records = _technitium_zone_records(call, zname, zone_node)
    entities.extend(records)
    return entities, len(records)


def _sync_technitium(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Technitium DNS zones + records as :DnsZone / :DnsRecord (CONCEPT:AU-KG.compute.technitium-dns-zones-records).

    Lists zones via ``technitium_dns_zones`` (action=list_zones), then per zone lists its
    records (action=get_records, list_zone=true). Each zone → a :DnsZone; each record →
    a :DnsRecord ``part_of`` its zone. Dict-shaped Technitium envelope (``response.zones`` /
    ``response.records``) → calls the tool directly via ``call_tool_once``. Full snapshot
    each run (DNS is small); the write-layer content-hash makes a re-run a no-op.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each zone
    and record is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`. Neither
    Technitium object carries a natural version marker (no per-record ``updatedAt`` — same
    as the L27 ops connectors with no ``updated_field``), so a record's ``part_of`` edge is
    carried on the RECORD's own ``_links`` (self-sourced — zone and record are always
    drained together in this one call).
    """
    server = _configured_server(("technitium-dns-mcp", "technitium-dns"))
    if server is None:
        return {"status": "skipped", "reason": "technitium-dns-mcp not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once
    from ...protocols.source_connectors.connectors.rest import _dig

    def _call(action: str, params: dict[str, Any]) -> Any:
        return _run_async(
            call_tool_once(
                server=server,
                tool="technitium_dns_zones",
                action=action,
                params=params,
            )
        )

    zones_res = _call("list_zones", {})
    zones = (
        (_dig(zones_res, "response.zones") or []) if isinstance(zones_res, dict) else []
    )
    entities: list[dict[str, Any]] = []
    records_total = 0
    for zone in zones:
        zone_entities, zone_records = _technitium_zone_slice(_call, zone)
        entities.extend(zone_entities)
        records_total += zone_records
    ok, failed = _ingest_entities_via_envelope(engine, "technitium", entities)
    return {
        "status": "ok",
        "source": "technitium",
        "mode": mode,
        "delta_capable": False,
        "zones": len(zones),
        "records": records_total,
        "nodes_hydrated": ok,
        "failed": failed,
    }


def _tunnel_host_rows(alias: Any, cfg: Any) -> list[dict[str, Any]]:
    """One tunnel-manager alias as its optional :Tunnel plus its :Host node."""
    if not isinstance(cfg, dict):
        return []
    extra = ec if isinstance((ec := cfg.get("extra_config")), dict) else {}
    host_node = f"tunnel:host:{alias}"
    host_entity: dict[str, Any] = {
        "id": host_node,
        "type": "host",
        "name": str(alias),
        "hostname": cfg.get("hostname"),
        "ssh_user": cfg.get("user"),
        "ssh_port": cfg.get("port"),
        "group": extra.get("group") or extra.get("ansible_group"),
        "ip_address": extra.get("ansible_host") or cfg.get("hostname"),
        "domain": "tunnel_manager",
        "source_system": "tunnel_manager",
        "externalToolId": str(alias),
    }
    proxy = cfg.get("proxy_command")
    if not proxy:
        return [host_entity]
    tun_node = f"tunnel:link:{alias}"
    host_entity["_links"] = [
        {
            "source": host_node,
            "target": tun_node,
            "type": "connects_via",
            "domain": "tunnel_manager",
        }
    ]
    return [
        {
            "id": tun_node,
            "type": "tunnel",
            "name": f"tunnel:{alias}",
            "proxy_command": str(proxy),
            "domain": "tunnel_manager",
            "source_system": "tunnel_manager",
        },
        host_entity,
    ]


def _sync_tunnel_manager(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest tunnel-manager host inventory as :Host / :Tunnel (CONCEPT:AU-KG.compute.tunnel-manager-hosts).

    Calls ``tm_hosts`` (action=list) — a dict ``{"hosts": {alias: HostConfig}}`` (args-style,
    not a record list) — so it goes through ``call_tool_once`` directly. Each alias → a
    :Host (hostname/user/port + any ``extra_config`` inventory keys); a configured
    ``proxy_command`` (a jump/tunnel) → a :Tunnel the host ``connects_via``.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each host
    (+ its optional tunnel) is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`.
    The ``connects_via`` edge is self-sourced from the host's own record and carried on
    the HOST's own ``_links``.
    """
    server = _configured_server(("tunnel-manager-mcp", "tunnel-manager"))
    if server is None:
        return {"status": "skipped", "reason": "tunnel-manager-mcp not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once

    res = _run_async(
        call_tool_once(
            server=server,
            tool="tm_hosts",
            params={"action": "list"},
            params_style="args",
            action="",
        )
    )
    hosts = (res.get("hosts") if isinstance(res, dict) else None) or {}
    entities: list[dict[str, Any]] = []
    for alias, cfg in hosts.items():
        entities.extend(_tunnel_host_rows(alias, cfg))
    ok, failed = _ingest_entities_via_envelope(engine, "tunnel_manager", entities)
    return {
        "status": "ok",
        "source": "tunnel_manager",
        "mode": mode,
        "delta_capable": False,
        "hosts": sum(1 for e in entities if e["type"] == "host"),
        "tunnels": sum(1 for e in entities if e["type"] == "tunnel"),
        "nodes_hydrated": ok,
        "failed": failed,
    }


def _uptime_monitor_list(monitors: Any) -> list[dict[str, Any]]:
    """Normalize ``get_monitors`` to a list of monitor dicts.

    It may return a bare list OR a dict keyed by id, depending on the
    ``uptime_kuma_api`` version.
    """
    if isinstance(monitors, dict):
        return [m for m in monitors.values() if isinstance(m, dict)]
    if isinstance(monitors, list):
        return [m for m in monitors if isinstance(m, dict)]
    return []


def _uptime_heartbeat_map(server: str) -> dict[Any, Any]:
    """The latest heartbeats keyed by monitor id (best-effort enrichment)."""
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once

    try:
        heartbeats = _run_async(
            call_tool_once(
                server=server,
                tool="uptime_kuma_status",
                params={"action": "get_heartbeats"},
                params_style="json",
                action="",
            )
        )
    except Exception:  # noqa: BLE001 — heartbeats are best-effort enrichment
        heartbeats = {}
    return heartbeats if isinstance(heartbeats, dict) else {}


def _uptime_monitor_entity(
    mon: dict[str, Any], mid: Any, mon_node: str
) -> dict[str, Any]:
    """One Uptime Kuma monitor as a :Monitor entity."""
    return {
        "id": mon_node,
        "type": "uptime_monitor",
        "name": mon.get("name") or f"Monitor {mid}",
        "url": mon.get("url"),
        "monitor_type": mon.get("type"),
        "active": mon.get("active"),
        "domain": "uptime_kuma",
        "source_system": "uptime_kuma",
        "externalToolId": str(mid),
    }


def _uptime_heartbeat_entity(
    hb_map: dict[Any, Any], mid: Any, mon_node: str
) -> dict[str, Any] | None:
    """One monitor's latest heartbeat as a :HeartbeatStat ``part_of`` it."""
    beats = hb_map.get(str(mid)) or hb_map.get(mid) or []
    last = beats[-1] if isinstance(beats, list) and beats else None
    if not isinstance(last, dict):
        return None
    hb_node = f"uptime:hb:{mid}"
    return {
        "id": hb_node,
        "type": "heartbeat_stat",
        "name": f"heartbeat:{mid}",
        "up": last.get("status") == 1,
        "ping": last.get("ping"),
        "msg": last.get("msg"),
        "domain": "uptime_kuma",
        "source_system": "uptime_kuma",
        "updatedAt": last.get("time"),
        "_links": [
            {
                "source": hb_node,
                "target": mon_node,
                "type": "part_of",
                "domain": "uptime_kuma",
            }
        ],
    }


def _uptime_entities(
    mon_list: list[dict[str, Any]], hb_map: dict[Any, Any]
) -> list[dict[str, Any]]:
    """Every monitor plus its latest heartbeat, in monitor order."""
    entities: list[dict[str, Any]] = []
    for mon in mon_list:
        mid = mon.get("id")
        if mid is None:
            continue
        mon_node = f"uptime:monitor:{mid}"
        entities.append(_uptime_monitor_entity(mon, mid, mon_node))
        heartbeat = _uptime_heartbeat_entity(hb_map, mid, mon_node)
        if heartbeat is not None:
            entities.append(heartbeat)
    return entities


def _sync_uptime_kuma(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Uptime Kuma monitors + heartbeat stats as :Monitor / :HeartbeatStat
    (CONCEPT:AU-KG.compute.uptime-kuma-monitors).

    Calls ``uptime_kuma_monitors`` (action=get_monitors → bare list) and
    ``uptime_kuma_status`` (action=get_heartbeats → dict keyed by monitor id), both
    args-shaped, via ``call_tool_once``. Each monitor → a :Monitor; the latest heartbeat
    per monitor → a :HeartbeatStat ``part_of`` it (status/ping). Full snapshot each run;
    the write-layer content-hash makes unchanged monitors a no-op — for service-health and
    failure-pattern analysis over the KG.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    monitor/heartbeat is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`.
    The ``part_of`` edge is self-sourced from the heartbeat's own record and carried on
    the HEARTBEAT's own ``_links`` (it also carries the real per-record ``updatedAt``, so
    the edge is re-asserted whenever the heartbeat actually changes).
    """
    server = _configured_server(("uptime-mcp", "uptime-kuma-mcp", "uptime-kuma-agent"))
    if server is None:
        return {"status": "skipped", "reason": "uptime-kuma server not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once

    monitors = _run_async(
        call_tool_once(
            server=server,
            tool="uptime_kuma_monitors",
            params={"action": "get_monitors"},
            params_style="json",
            action="",
        )
    )
    mon_list = _uptime_monitor_list(monitors)
    entities = _uptime_entities(mon_list, _uptime_heartbeat_map(server))
    ok, failed = _ingest_entities_via_envelope(engine, "uptime_kuma", entities)
    return {
        "status": "ok",
        "source": "uptime_kuma",
        "mode": mode,
        "delta_capable": False,
        "monitors": len(mon_list),
        "nodes_hydrated": ok,
        "failed": failed,
    }


def _hass_entity_rows(doc: Any, devices: set[str]) -> list[dict[str, Any]]:
    """One HA state record as its :Entity, preceded by its :Device on first sight.

    ``devices`` is the running set of device classes already emitted this run; it
    is mutated so the roll-up :Device node is written exactly once.
    """
    eid = getattr(doc, "id", None)
    if not eid:
        return []
    rec = rd if isinstance((rd := _record_of(doc)), dict) else {}
    attrs = a if isinstance((a := rec.get("attributes")), dict) else {}
    device_class = str(eid).split(".", 1)[0]  # light / sensor / switch / …
    ent_node = f"hass:entity:{eid}"
    dev_node = f"hass:device:{device_class}"
    rows: list[dict[str, Any]] = []
    if device_class not in devices:
        rows.append(
            {
                "id": dev_node,
                "type": "device",
                "name": f"HA {device_class}",
                "domain": "home_assistant",
                "source_system": "home_assistant",
            }
        )
        devices.add(device_class)
    rows.append(
        {
            "id": ent_node,
            "type": "entity",
            "name": attrs.get("friendly_name") or str(eid),
            "entity_id": str(eid),
            "state": rec.get("state"),
            "device_class": device_class,
            "domain": "home_assistant",
            "source_system": "home_assistant",
            "externalToolId": str(eid),
            "updatedAt": rec.get("last_updated"),
            "_links": [
                {
                    "source": ent_node,
                    "target": dev_node,
                    "type": "part_of",
                    "domain": "home_assistant",
                }
            ],
        }
    )
    return rows


def _sync_home_assistant(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Home Assistant entities/states as :Device / :Entity (CONCEPT:AU-KG.compute.home-assistant-states).

    Drains the ``home-assistant-states`` preset (action=list_states → bare list) over
    ``home-assistant-mcp``. Each ``entity_id`` → an :Entity (state + attributes); its
    domain prefix (``light``/``sensor``/…) rolls up to a :Device the entity is ``part_of``.
    Delta = the ``last_updated`` watermark.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each HA
    entity is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`. The
    ``part_of`` edge is self-sourced from the entity's own record and carried on the
    ENTITY's own ``_links`` (it also carries the real per-record ``updatedAt``).
    """
    if not _server_configured(("home-assistant-mcp", "home-assistant-agent")):
        return {"status": "skipped", "reason": "home-assistant-mcp not in mcp_config"}
    since = (
        None
        if mode == "full"
        else _read_envelope_watermark(
            engine,
            "home_assistant",
        )
    )
    docs = _drain_preset("home-assistant-states")
    entities: list[dict[str, Any]] = []
    devices: set[str] = set()
    for doc in docs:
        entities.extend(_hass_entity_rows(doc, devices))
    ok, failed = _ingest_entities_via_envelope(engine, "home_assistant", entities)
    return {
        "status": "ok",
        "source": "home_assistant",
        "mode": mode,
        "delta_capable": True,
        "entities": sum(1 for e in entities if e["type"] == "entity"),
        "devices": len(devices),
        "nodes_hydrated": ok,
        "failed": failed,
        "since": since,
    }


def _twenty_company_id(rec: dict[str, Any]) -> str | None:
    """The company a Twenty person/opportunity record references, if any."""
    cid = rec.get("companyId")
    if cid:
        return str(cid)
    company = rec.get("company")
    return (
        str(company["id"]) if isinstance(company, dict) and company.get("id") else None
    )


def _twenty_company_link(node_id: str, cid: str, rel_type: str) -> list[dict[str, Any]]:
    """The self-sourced edge from a person/opportunity to its :Company."""
    return [
        {
            "source": node_id,
            "target": f"twenty:company:{cid}",
            "type": rel_type,
            "domain": "twenty",
        }
    ]


def _twenty_company_entity(doc: Any) -> dict[str, Any] | None:
    """One Twenty CRM company record as a :Company entity."""
    cid = getattr(doc, "id", None)
    if not cid:
        return None
    rec = _record_of(doc)
    return {
        "id": f"twenty:company:{cid}",
        "type": "company",
        "name": rec.get("name") or f"Company {cid}",
        "domain": "twenty",
        "source_system": "twenty",
        "externalToolId": str(cid),
        "updatedAt": rec.get("updatedAt"),
    }


def _twenty_person_name(rec: dict[str, Any]) -> str:
    """A Twenty person's display name from its structured ``name`` block."""
    name = rec.get("name") or {}
    if not isinstance(name, dict):
        return str(name)
    return f"{name.get('firstName', '')} {name.get('lastName', '')}".strip()


def _twenty_person_entity(doc: Any) -> dict[str, Any] | None:
    """One Twenty CRM person record as a :Person ``member_of`` their :Company."""
    pid = getattr(doc, "id", None)
    if not pid:
        return None
    rec = _record_of(doc)
    node_id = f"twenty:person:{pid}"
    person: dict[str, Any] = {
        "id": node_id,
        "type": "person",
        "name": _twenty_person_name(rec) or f"Person {pid}",
        "job_title": rec.get("jobTitle"),
        "domain": "twenty",
        "source_system": "twenty",
        "externalToolId": str(pid),
        "updatedAt": rec.get("updatedAt"),
    }
    if cid := _twenty_company_id(rec):
        person["_links"] = _twenty_company_link(node_id, cid, "member_of")
    return person


def _twenty_opportunity_entity(doc: Any) -> dict[str, Any] | None:
    """One Twenty CRM opportunity record as an :Opportunity ``part_of`` a :Company."""
    oid = getattr(doc, "id", None)
    if not oid:
        return None
    rec = _record_of(doc)
    node_id = f"twenty:opportunity:{oid}"
    opp: dict[str, Any] = {
        "id": node_id,
        "type": "opportunity",
        "name": rec.get("name") or f"Opportunity {oid}",
        "stage": rec.get("stage"),
        "domain": "twenty",
        "source_system": "twenty",
        "externalToolId": str(oid),
        "updatedAt": rec.get("updatedAt"),
    }
    if cid := _twenty_company_id(rec):
        opp["_links"] = _twenty_company_link(node_id, cid, "part_of")
    return opp


def _sync_twenty(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Twenty CRM people/companies/opportunities as :Person / :Company /
    :Opportunity (CONCEPT:AU-KG.compute.twenty-crm-people-companies).

    Drains the ``twenty-people`` / ``twenty-companies`` / ``twenty-opportunities`` presets
    over ``twenty-mcp``. People with a ``companyId`` are linked ``member_of`` their company;
    opportunities with a ``companyId`` are linked ``part_of`` it. Delta = the ``updatedAt``
    watermark across the three object types.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    person/company/opportunity is one ``ChangeEnvelope`` via
    :func:`_ingest_entities_via_envelope`. Both edges are self-sourced (person/opportunity
    are drained in the SAME call as the companies they reference) and carried on the
    person's/opportunity's own ``_links``.
    """
    if not _server_configured(("twenty-mcp", "twenty")):
        return {"status": "skipped", "reason": "twenty-mcp not in mcp_config"}
    since = None if mode == "full" else _read_envelope_watermark(engine, "twenty")
    src = "twenty"

    people = _drain_preset("twenty-people")
    companies = _drain_preset("twenty-companies")
    opps = _drain_preset("twenty-opportunities")
    entities: list[dict[str, Any]] = [
        *_entity_rows(companies, _twenty_company_entity),
        *_entity_rows(people, _twenty_person_entity),
        *_entity_rows(opps, _twenty_opportunity_entity),
    ]
    ok, failed = _ingest_entities_via_envelope(engine, src, entities)
    return {
        "status": "ok",
        "source": "twenty",
        "mode": mode,
        "delta_capable": True,
        "people": len(people),
        "companies": len(companies),
        "opportunities": len(opps),
        "nodes_hydrated": ok,
        "failed": failed,
        "since": since,
    }


# ── Media / finance / document / genealogy connectors as typed OWL entities ──────
# (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors–2.166)
#
# Four more first-class delta connectors reaching their upstream ONLY through a fleet
# ``*-mcp`` server (same contract as jira/dockerhub/twenty) and rebuilding **typed**
# entities mapped to OWL classes — not generic Documents.
#
# CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors — Audiobookshelf libraries/books/authors → :Library / :Book / :Author
# CONCEPT:AU-KG.compute.firefly-iii-accounts-transactions — Firefly III accounts/transactions/budgets → :Account / :Transaction / :Budget
# CONCEPT:AU-KG.compute.paperless-ngx-documents-correspondents — Paperless-ngx documents/correspondents/tags → :Document / :Correspondent / :Tag
# CONCEPT:AU-KG.compute.gramps-web-people-families — Gramps Web people/families/events → :Person / :Family / :Event


def _abs_libraries(libs_res: Any) -> list[Any]:
    """Normalize the Audiobookshelf ``library_operations(action=list)`` payload.

    Some ABS builds return a bare list of libraries, others the documented
    ``{"libraries": [...]}`` envelope; anything else yields no libraries.
    """
    if isinstance(libs_res, list):  # some ABS builds return a bare list
        return libs_res
    if isinstance(libs_res, dict):
        return libs_res.get("libraries") or []
    return []


def _abs_author_node(author: Any) -> str | None:
    """The ``abs:author:<id-or-name>`` node id for one ABS author record."""
    if not isinstance(author, dict):
        return None
    aid = author.get("id")
    aname = author.get("name")
    if not (aid or aname):
        return None
    return f"abs:author:{aid or aname}"


def _abs_author_links(authors: Any, book_node: str) -> list[dict[str, Any]]:
    """``authored_by`` edges from one book to each of its named authors."""
    links: list[dict[str, Any]] = []
    for author in authors or []:
        author_node = _abs_author_node(author)
        if author_node is None:
            continue
        links.append(
            {
                "source": book_node,
                "target": author_node,
                "type": "authored_by",
                "domain": "audiobookshelf",
            }
        )
    return links


def _abs_book_entity(item: Any, lib_node: str) -> dict[str, Any] | None:
    """One ABS library item as a :Book entity, or ``None`` when unidentifiable."""
    if not isinstance(item, dict):
        return None
    item_id = item.get("id")
    if not item_id:
        return None
    media = m if isinstance((m := item.get("media")), dict) else {}
    meta = mm if isinstance((mm := media.get("metadata")), dict) else {}
    title = meta.get("title") or item.get("title") or f"Book {item_id}"
    book_node = f"abs:book:{item_id}"
    book_links: list[dict[str, Any]] = [
        {
            "source": book_node,
            "target": lib_node,
            "type": "part_of",
            "domain": "audiobookshelf",
        }
    ]
    book_links.extend(_abs_author_links(meta.get("authors"), book_node))
    return {
        "id": book_node,
        "type": "book",
        "name": str(title),
        "subtitle": meta.get("subtitle"),
        "isbn": meta.get("isbn"),
        "asin": meta.get("asin"),
        "publisher": meta.get("publisher"),
        "published_year": meta.get("publishedYear"),
        "duration": media.get("duration"),
        "domain": "audiobookshelf",
        "source_system": "audiobookshelf",
        "externalToolId": str(item_id),
        "updatedAt": item.get("updatedAt"),
        "_links": book_links,
    }


def _abs_author_entity(author: Any) -> dict[str, Any] | None:
    """One ABS author record as an :Author entity, or ``None`` when unidentifiable."""
    author_node = _abs_author_node(author)
    if author_node is None:
        return None
    aid = author.get("id")
    aname = author.get("name")
    return {
        "id": author_node,
        "type": "author",
        "name": aname or f"Author {aid}",
        "num_books": author.get("numBooks"),
        "domain": "audiobookshelf",
        "source_system": "audiobookshelf",
        "externalToolId": str(aid or aname),
    }


def _abs_books(call: Callable[..., Any], lib_id: Any, lib_node: str) -> list[dict]:
    """Every :Book in one ABS library; one bad library never aborts the rest."""
    try:
        items_res = call("items", {"id": lib_id, "limit": 500})
    except Exception as exc:  # noqa: BLE001 — one bad library never aborts the rest
        logger.warning(
            "[KG-2.163] audiobookshelf items fetch failed for %s: %s", lib_id, exc
        )
        items_res = {}
    items = (items_res.get("results") if isinstance(items_res, dict) else None) or []
    books: list[dict[str, Any]] = []
    for item in items:
        entity = _abs_book_entity(item, lib_node)
        if entity is not None:
            books.append(entity)
    return books


def _abs_authors(call: Callable[..., Any], lib_id: Any) -> list[dict[str, Any]]:
    """Every :Author in one ABS library (best-effort enrichment)."""
    try:
        authors_res = call("authors", {"id": lib_id})
    except Exception:  # noqa: BLE001 — authors are best-effort enrichment
        authors_res = {}
    records = (
        authors_res.get("authors") if isinstance(authors_res, dict) else None
    ) or []
    authors: list[dict[str, Any]] = []
    for record in records:
        entity = _abs_author_entity(record)
        if entity is not None:
            authors.append(entity)
    return authors


def _abs_library_slice(
    call: Callable[..., Any], lib: Any
) -> tuple[list[dict[str, Any]], int, int]:
    """One ABS library plus its books and authors, as ``(entities, books, authors)``."""
    if not isinstance(lib, dict):
        return [], 0, 0
    lib_id = lib.get("id")
    if not lib_id:
        return [], 0, 0
    lib_node = f"abs:library:{lib_id}"
    entities: list[dict[str, Any]] = [
        {
            "id": lib_node,
            "type": "library",
            "name": lib.get("name") or f"Library {lib_id}",
            "media_type": lib.get("mediaType"),
            "domain": "audiobookshelf",
            "source_system": "audiobookshelf",
            "externalToolId": str(lib_id),
        }
    ]
    books = _abs_books(call, lib_id, lib_node)
    authors = _abs_authors(call, lib_id)
    entities.extend(books)
    entities.extend(authors)
    return entities, len(books), len(authors)


def _sync_audiobookshelf(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Audiobookshelf libraries/books/authors as :Library / :Book / :Author
    (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors).

    Multi-step over ``audiobookshelf-mcp``: ``library_operations(action=list)`` →
    ``{"libraries": [...]}``; per library ``action=items`` → ``{"results": [...]}`` (each
    library item is a :Book ``part_of`` its :Library) and ``action=authors`` →
    ``{"authors": [...]}`` (each :Author, with books linked ``authored_by``). Dict-shaped /
    multi-step → calls the tool directly via ``call_tool_once``. Full snapshot each run; the
    write-layer content-hash makes a re-run a no-op.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    library/book/author is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`.
    Both a book's edges (``part_of`` its library, ``authored_by`` each author) are
    self-sourced from the book's own record (library/book/authors are always drained
    together in this one call) and carried on the BOOK's own ``_links`` — the book also
    carries the real per-record ``updatedAt``, unlike the versionless library/author.
    """
    server = _configured_server(("audiobookshelf-mcp", "audiobookshelf-agent"))
    if server is None:
        return {"status": "skipped", "reason": "audiobookshelf-mcp not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once

    def _call(action: str, params: dict[str, Any]) -> Any:
        return _run_async(
            call_tool_once(
                server=server,
                tool="library_operations",
                action=action,
                params=params,
            )
        )

    libraries = _abs_libraries(_call("list", {}))
    entities: list[dict[str, Any]] = []
    books_total = 0
    authors_total = 0
    for lib in libraries:
        lib_entities, books, authors = _abs_library_slice(_call, lib)
        entities.extend(lib_entities)
        books_total += books
        authors_total += authors
    ok, failed = _ingest_entities_via_envelope(engine, "audiobookshelf", entities)
    return {
        "status": "ok",
        "source": "audiobookshelf",
        "mode": mode,
        "delta_capable": False,
        "libraries": sum(1 for e in entities if e["type"] == "library"),
        "books": books_total,
        "authors": authors_total,
        "nodes_hydrated": ok,
        "failed": failed,
    }


def _firefly_attrs(doc: Any) -> dict[str, Any]:
    """A Firefly III JSON:API record's ``attributes`` block (the real fields)."""
    from ..etl.transforms import dig

    return dig(_record_of(doc), "attributes", default={})


def _firefly_account_entity(doc: Any) -> dict[str, Any] | None:
    """One Firefly III account record as an :Account entity."""
    from ..etl.transforms import coalesce, stable_id

    aid = getattr(doc, "id", None)
    if not aid:
        return None
    attrs = _firefly_attrs(doc)
    return {
        "id": stable_id(aid, prefix="firefly:account"),
        "type": "account",
        "name": coalesce(attrs, "name", default=f"Account {aid}"),
        "account_type": attrs.get("type"),
        "account_role": attrs.get("account_role"),
        "currency_code": attrs.get("currency_code"),
        "current_balance": attrs.get("current_balance"),
        "domain": "firefly_iii",
        "source_system": "firefly_iii",
        "externalToolId": str(aid),
        "updatedAt": attrs.get("updated_at"),
    }


def _firefly_budget_entity(doc: Any) -> dict[str, Any] | None:
    """One Firefly III budget record as a :Budget entity."""
    from ..etl.transforms import coalesce, stable_id

    bid = getattr(doc, "id", None)
    if not bid:
        return None
    attrs = _firefly_attrs(doc)
    return {
        "id": stable_id(bid, prefix="firefly:budget"),
        "type": "budget",
        "name": coalesce(attrs, "name", default=f"Budget {bid}"),
        "active": attrs.get("active"),
        "domain": "firefly_iii",
        "source_system": "firefly_iii",
        "externalToolId": str(bid),
        "updatedAt": attrs.get("updated_at"),
    }


def _firefly_first_split(attrs: dict[str, Any]) -> dict[str, Any]:
    """A transaction's first split — where its real per-transaction fields live."""
    splits = attrs.get("transactions")
    first = splits[0] if isinstance(splits, list) and splits else {}
    return first if isinstance(first, dict) else {}


def _firefly_transaction_links(
    first: dict[str, Any], node_id: str
) -> list[dict[str, Any]]:
    """A transaction's ``part_of`` source account and ``member_of`` budget edges."""
    from ..etl.transforms import stable_id

    tx_links: list[dict[str, Any]] = []
    if src_acct := first.get("source_id"):
        tx_links.append(
            {
                "source": node_id,
                "target": stable_id(src_acct, prefix="firefly:account"),
                "type": "part_of",
                "domain": "firefly_iii",
            }
        )
    if budget_id := first.get("budget_id"):
        tx_links.append(
            {
                "source": node_id,
                "target": stable_id(budget_id, prefix="firefly:budget"),
                "type": "member_of",
                "domain": "firefly_iii",
            }
        )
    return tx_links


def _firefly_transaction_entity(doc: Any) -> dict[str, Any] | None:
    """One Firefly III transaction record as a :Transaction entity."""
    from ..etl.transforms import coalesce, stable_id

    tid = getattr(doc, "id", None)
    if not tid:
        return None
    attrs = _firefly_attrs(doc)
    first = _firefly_first_split(attrs)
    node_id = stable_id(tid, prefix="firefly:transaction")
    return {
        "id": node_id,
        "type": "transaction",
        "name": coalesce(attrs, "group_title")
        or coalesce(first, "description", default=f"Transaction {tid}"),
        "transaction_type": first.get("type"),
        "amount": first.get("amount"),
        "currency_code": first.get("currency_code"),
        "transaction_date": first.get("date"),
        "category_name": first.get("category_name"),
        "domain": "firefly_iii",
        "source_system": "firefly_iii",
        "externalToolId": str(tid),
        "updatedAt": attrs.get("updated_at"),
        "_links": _firefly_transaction_links(first, node_id),
    }


def _sync_firefly_iii(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Firefly III accounts/transactions/budgets as :Account / :Transaction /
    :Budget (CONCEPT:AU-KG.compute.firefly-iii-accounts-transactions).

    Drains the ``firefly-accounts`` / ``firefly-transactions`` / ``firefly-budgets``
    presets over ``firefly-iii-mcp``. Each JSON:API record's ``attributes`` block carries
    the real fields. A transaction's first split is linked ``part_of`` its source :Account.
    Delta = the ``updated_at`` watermark across the three object types.

    Uses the shared transform primitives (CONCEPT:AU-KG.etl.transform-primitives) —
    :func:`~..etl.transforms.dig` for the JSON:API ``attributes`` envelope unwrap
    (replacing the handler-local ``_attrs`` helper), :func:`~..etl.transforms.coalesce`
    for name fallbacks, and :func:`~..etl.transforms.stable_id` for node ids.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    account/budget/transaction is one ``ChangeEnvelope`` via
    :func:`_ingest_entities_via_envelope`. Both edges are self-sourced from the
    transaction's own record (accounts/budgets/transactions are always drained together
    in this one call) and carried on the TRANSACTION's own ``_links``.
    """
    if not _server_configured(("firefly-iii-mcp", "firefly-iii-agent")):
        return {"status": "skipped", "reason": "firefly-iii-mcp not in mcp_config"}

    since = (
        None
        if mode == "full"
        else _read_envelope_watermark(
            engine,
            "firefly_iii",
        )
    )
    src = "firefly_iii"

    accounts = _drain_preset("firefly-accounts")
    transactions = _drain_preset("firefly-transactions")
    budgets = _drain_preset("firefly-budgets")
    entities: list[dict[str, Any]] = [
        *_entity_rows(accounts, _firefly_account_entity),
        *_entity_rows(budgets, _firefly_budget_entity),
        *_entity_rows(transactions, _firefly_transaction_entity),
    ]
    ok, failed = _ingest_entities_via_envelope(engine, src, entities)
    return {
        "status": "ok",
        "source": "firefly_iii",
        "mode": mode,
        "delta_capable": True,
        "accounts": len(accounts),
        "transactions": len(transactions),
        "budgets": len(budgets),
        "nodes_hydrated": ok,
        "failed": failed,
        "since": since,
    }


# The certified zero-PII Paperless-ngx projection's complete node/edge vocabulary.
# Anything outside it is off-contract and fails the sync closed.
_PAPERLESS_NODE_TYPES = frozenset(
    {
        "PaperlessCorrespondentReference",
        "PaperlessDocumentReference",
        "PaperlessDocumentTypeReference",
        "PaperlessStoragePathReference",
        "PaperlessTagReference",
    }
)
_PAPERLESS_RELATIONSHIPS = frozenset(
    {
        "hasCorrespondentReference",
        "hasDocumentTypeReference",
        "hasStoragePathReference",
        "hasTagReference",
    }
)


def _paperless_projection(client: Any) -> dict[str, Any]:
    """Fetch and shape-validate the signed ``paperless-document-structure`` result."""
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_preset_once

    projection = _run_async(
        call_preset_once(
            "paperless-document-structure",
            provider="paperless-ngx-mcp",
            client=client,
        )
    )
    if (
        not isinstance(projection, dict)
        or set(projection) != {"records", "relationships"}
        or not isinstance(projection["records"], list)
        or not isinstance(projection["relationships"], list)
    ):
        raise ValueError("Paperless-ngx projection is malformed")
    return projection


def _paperless_node(record: Any) -> dict[str, Any]:
    """One validated opaque projection node; anything off-contract raises."""
    if not isinstance(record, dict) or set(record) != {"id", "node_type"}:
        raise ValueError("Paperless-ngx projection contains an invalid node")
    node_id = record.get("id")
    node_type = record.get("node_type")
    prefix = f"paperless:{node_type}:"
    if (
        not isinstance(node_id, str)
        or not isinstance(node_type, str)
        or node_type not in _PAPERLESS_NODE_TYPES
        or not node_id.startswith(prefix)
        or re.fullmatch(r"[0-9a-f]{64}", node_id.removeprefix(prefix)) is None
    ):
        raise ValueError("Paperless-ngx projection contains an invalid node")
    return {"id": node_id, "node_type": node_type}


def _paperless_nodes(records: list[Any]) -> tuple[list[dict[str, Any]], set[str]]:
    """Every validated projection node, plus the id set edges must resolve into."""
    entities = [_paperless_node(record) for record in records]
    return entities, {str(entity["id"]) for entity in entities}


def _paperless_relationship(relationship: Any, node_ids: set[str]) -> dict[str, Any]:
    """One validated structural edge; an unreviewed shape or endpoint raises."""
    if not isinstance(relationship, dict) or set(relationship) != {
        "source",
        "target",
        "relationship",
    }:
        raise ValueError("Paperless-ngx projection contains an invalid relationship")
    if (
        relationship.get("source") not in node_ids
        or relationship.get("target") not in node_ids
        or relationship.get("relationship") not in _PAPERLESS_RELATIONSHIPS
    ):
        raise ValueError("Paperless-ngx projection contains an invalid relationship")
    return dict(relationship)


def _sync_paperless_ngx(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Persist Paperless-ngx's certified zero-PII structural projection.

    The connector owns one signed ``paperless-document-structure`` preset.  Its
    MCP tool pseudonymizes provider identifiers in memory and returns only typed
    opaque nodes plus reviewed structural edges.  The complete result is committed
    atomically through the common graph-slice envelope path; the retired central
    three-preset document/correspondent/tag pull is intentionally not used because
    it exposed raw provider fields and was not the contract the manifest certified.
    """
    # CA-22/P11: this handler commits via ``ingest_graph_slice`` directly (a 4th
    # write-path shape the lane brief's classification missed -- caught by
    # check_handler_preflight.py itself, not the shared _ingest_entities_via_
    # envelope tail every other "typed OWL entity" handler uses), so it needs
    # its own reachability call to the preflight chokepoint; batch=[] checks
    # nothing per-record until a manifest declares conflict_policy for
    # "paperless_ngx".
    _apply_with_preflight(engine, "paperless_ngx", [])

    server = _configured_server(("paperless-ngx-mcp", "paperless-ngx-agent"))
    if client is None and server is None:
        return {"status": "skipped", "reason": "paperless-ngx-mcp not in mcp_config"}

    from ..ingestion.envelope_ingest import ingest_graph_slice

    projection = _paperless_projection(client)
    entities, node_ids = _paperless_nodes(projection["records"])
    relationships: list[dict[str, Any]] = [
        _paperless_relationship(relationship, node_ids)
        for relationship in projection["relationships"]
    ]

    result = ingest_graph_slice(
        engine,
        "paperless_ngx",
        entities,
        relationships,
    )
    return {
        "status": result.get("status", "ok"),
        "source": "paperless_ngx",
        "mode": mode,
        "delta_capable": False,
        "nodes_hydrated": len(entities),
        "edges": len(relationships),
    }


def _gramps_type_string(rec: dict[str, Any]) -> Any:
    """A Gramps record's ``type.string``, or ``None`` when ``type`` isn't a dict."""
    type_field = rec.get("type")
    if isinstance(type_field, dict):
        return type_field.get("string")
    return None


def _gramps_surname(name: dict[str, Any]) -> str:
    """The first surname of a Gramps ``primary_name`` block."""
    surnames = name.get("surname_list") or []
    if not isinstance(surnames, list) or not surnames:
        return ""
    first = surnames[0]
    return first.get("surname", "") if isinstance(first, dict) else ""


def _gramps_person_name(rec: dict[str, Any]) -> str:
    """A person's ``<first> <surname>``, falling back to their gramps id/handle."""
    name = rec.get("primary_name")
    if isinstance(name, dict):
        full = f"{name.get('first_name') or ''} {_gramps_surname(name)}".strip()
        if full:
            return full
    return rec.get("gramps_id") or rec.get("handle") or "Person"


def _gramps_person_event_links(
    rec: dict[str, Any], node_id: str
) -> list[dict[str, Any]]:
    """``part_of`` edges from one person to each event they are referenced in."""
    links: list[dict[str, Any]] = []
    for eref in rec.get("event_ref_list") or []:
        if not isinstance(eref, dict):
            continue
        if ev := eref.get("ref"):
            links.append(
                {
                    "source": node_id,
                    "target": f"gramps:event:{ev}",
                    "type": "part_of",
                    "domain": "gramps",
                }
            )
    return links


def _gramps_person_entity(rec: dict[str, Any]) -> dict[str, Any] | None:
    """One Gramps person record as a :Person entity, or ``None`` without a handle."""
    handle = rec.get("handle")
    if not handle:
        return None
    node_id = f"gramps:person:{handle}"
    return {
        "id": node_id,
        "type": "person",
        "name": _gramps_person_name(rec),
        "gramps_id": rec.get("gramps_id"),
        "gender": rec.get("gender"),
        "domain": "gramps",
        "source_system": "gramps",
        "externalToolId": str(handle),
        "updatedAt": rec.get("change"),
        "_links": _gramps_person_event_links(rec, node_id),
    }


def _gramps_family_members(rec: dict[str, Any]) -> list[Any]:
    """The father/mother/child person handles of one family record."""
    members: list[Any] = [rec.get("father_handle"), rec.get("mother_handle")]
    for child in rec.get("child_ref_list") or []:
        if isinstance(child, dict) and child.get("ref"):
            members.append(child["ref"])
    return members


def _gramps_family_entity(rec: dict[str, Any]) -> dict[str, Any] | None:
    """One Gramps family record as a :Family entity, or ``None`` without a handle."""
    handle = rec.get("handle")
    if not handle:
        return None
    fam_node = f"gramps:family:{handle}"
    fam_links = [
        {
            "source": f"gramps:person:{member}",
            "target": fam_node,
            "type": "member_of",
            "domain": "gramps",
        }
        for member in _gramps_family_members(rec)
        if member
    ]
    return {
        "id": fam_node,
        "type": "family",
        "name": rec.get("gramps_id") or f"Family {handle}",
        "gramps_id": rec.get("gramps_id"),
        "relationship": _gramps_type_string(rec),
        "domain": "gramps",
        "source_system": "gramps",
        "externalToolId": str(handle),
        "updatedAt": rec.get("change"),
        "_links": fam_links,
    }


def _gramps_event_entity(rec: dict[str, Any]) -> dict[str, Any] | None:
    """One Gramps event record as an :Event entity, or ``None`` without a handle."""
    handle = rec.get("handle")
    if not handle:
        return None
    named = (
        _gramps_type_string(rec)
        if isinstance(rec.get("type"), dict)
        else rec.get("gramps_id")
    )
    return {
        "id": f"gramps:event:{handle}",
        "type": "event",
        "name": named or f"Event {handle}",
        "gramps_id": rec.get("gramps_id"),
        "description": rec.get("description"),
        "domain": "gramps",
        "source_system": "gramps",
        "externalToolId": str(handle),
        "updatedAt": rec.get("change"),
    }


def _sync_gramps(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest Gramps Web people/families/events as :Person / :Family / :Event
    (CONCEPT:AU-KG.compute.gramps-web-people-families).

    Calls ``gramps_people`` / ``gramps_families`` / ``gramps_events`` (action
    ``get_*``) directly via ``call_tool_once`` — each returns the ``Response`` envelope whose
    ``data`` is the decoded collection. Each person → a :Person; each family → a :Family the
    person is ``member_of`` (father/mother/children handles); each event → an :Event a person
    ``part_of`` (via the person's ``event_ref_list``). Full snapshot each run; the write-layer
    content-hash makes a re-run a no-op. The genealogy graph is the substrate for relationship
    reasoning over the KG.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    person/family/event is one ``ChangeEnvelope`` via :func:`_ingest_entities_via_envelope`.
    A person's own ``part_of`` event edge is carried on the PERSON's own ``_links``
    (self-sourced). A family's father/mother/child ``member_of`` edges — whose ``source``
    is a PERSON id, not the family's — are carried on the FAMILY's own ``_links`` instead:
    people/families/events are always drained together in this one call (self-sourced),
    and the family (not the referenced person) is the entity whose own ``change`` marker
    actually reflects a membership edit, so attaching there is what re-asserts the edge
    whenever membership changes.
    """
    server = _configured_server(("gramps-mcp", "gramps-agent"))
    if server is None:
        return {"status": "skipped", "reason": "gramps-mcp not in mcp_config"}
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ...protocols.source_connectors.connectors.mcp_tool import call_tool_once
    from ...protocols.source_connectors.connectors.rest import _dig

    def _collection(tool: str, action: str) -> list[dict[str, Any]]:
        # Fail-soft per collection: a connector that doesn't expose one list action
        # (if a connector doesn't expose one list action) must not sink the
        # whole sync — skip that collection and ingest the others.
        try:
            res = _run_async(
                call_tool_once(
                    server=server,
                    tool=tool,
                    action=action,
                    params={"pagesize": 500},
                )
            )
        except Exception as exc:
            logger.warning("gramps: %s/%s unavailable, skipping: %s", tool, action, exc)
            return []
        data = _dig(res, "data") if isinstance(res, dict) else res
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []

    people = _collection("gramps_people", "get_people")
    families = _collection("gramps_families", "get_families")
    events = _collection("gramps_events", "get_events")

    entities = [
        *_entity_rows(people, _gramps_person_entity),
        *_entity_rows(families, _gramps_family_entity),
        *_entity_rows(events, _gramps_event_entity),
    ]
    ok, failed = _ingest_entities_via_envelope(
        engine, "gramps", entities, version_field="updatedAt"
    )
    return {
        "status": "ok",
        "source": "gramps",
        "mode": mode,
        "delta_capable": False,
        "people": len(people),
        "families": len(families),
        "events": len(events),
        "nodes_hydrated": ok,
        "failed": failed,
    }


# MCP-backed dedicated trackers (CONCEPT:AU-KG.compute.mcp-backed-dedicated-trackers) — each reaches its upstream ONLY
# through a fleet ``*-mcp`` server (never a direct vendor client / env token), so unlike
# the capability-registry sources (env-token configured) and the always-local feed/fleet
# handlers, their "configured" signal is *"the server is registered in mcp_config.json"*.
# Maps the delta source → the candidate ``server`` keys to probe (the handler's
# ``default_server``; per-instance overrides are unioned in at sweep time). Keep in sync
# with the ``default_server`` of each ``_resolve_tracker_instances`` call.
_MCP_TRACKER_SERVERS: dict[str, tuple[str, ...]] = {
    "jira": ("atlassian-mcp",),
    "confluence": ("atlassian-mcp",),
    "plane": ("plane-mcp",),
    # Ops / platform typed connectors (CONCEPT:AU-KG.compute.dockerhub-repositories–2.161) — server-configured, so the
    # sweep keeps each candidate only when its ``*-mcp`` server is in mcp_config (else drops
    # it, never mis-reporting an unconfigured connector as failed work).
    "dockerhub": ("dockerhub-mcp", "dockerhub-api"),
    "langfuse": ("langfuse-mcp", "langfuse-agent"),
    "technitium": ("technitium-dns-mcp", "technitium-dns"),
    "tunnel_manager": ("tunnel-manager-mcp", "tunnel-manager"),
    "uptime_kuma": ("uptime-mcp", "uptime-kuma-agent", "uptime-kuma-mcp"),
    "home_assistant": ("home-assistant-mcp", "home-assistant-agent"),
    "twenty": ("twenty-mcp", "twenty"),
    # Media / finance / document / genealogy connectors (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors–2.166)
    "audiobookshelf": ("audiobookshelf-mcp", "audiobookshelf-agent"),
    "firefly_iii": ("firefly-iii-mcp", "firefly-iii-agent"),
    "paperless_ngx": ("paperless-ngx-mcp", "paperless-ngx-agent"),
    "gramps": ("gramps-mcp", "gramps-agent"),
}


def _mcp_server_configured(servers: dict[str, Any], name: str) -> bool:
    """True when ``name`` (or ``<name>-mcp``) is registered in the loaded mcp_config
    ``mcpServers`` map — mirrors the connector's own transport resolution
    (:meth:`McpToolSourceConnector` server lookup), so "candidate" and "reachable"
    agree on what counts as configured."""
    if not name:
        return False
    return name in servers or f"{name}-mcp" in servers


def _tracker_instance_servers(field: str, default_server: str) -> tuple[str, ...]:
    """Servers a tracker delta source will actually reach: the per-instance ``server``
    overrides from a configured ``*_instances`` config row, else the ``default_server``.
    Lets a sweep recognise a second Atlassian site / Plane workspace as configured."""
    try:
        from ...core.config import config as cfg

        rows = [r for r in (getattr(cfg, field, None) or []) if isinstance(r, dict)]
        servers = tuple(str(r.get("server") or default_server) for r in rows)
        if servers:
            return servers
    except Exception:  # noqa: BLE001 — config probe is best-effort
        pass
    return (default_server,)


def _mcp_tracker_configured(source: str) -> bool:
    """True when an MCP-backed dedicated tracker (jira/confluence/plane) is configured
    for the sweep — i.e. at least one server it would reach is registered in
    ``mcp_config.json``. Unknown sources default to *configured* (no extra gate)."""
    default_servers = _MCP_TRACKER_SERVERS.get(source)
    if default_servers is None:
        return True
    try:
        from ...protocols.source_connectors.connectors.mcp_package import (
            _load_mcp_config,
        )

        servers = _load_mcp_config() or {}
    except Exception:  # noqa: BLE001 — no config readable → not configured here
        return False
    _INST_FIELD = {
        "jira": "jira_instances",
        "confluence": "confluence_instances",
        "plane": "plane_instances",
    }
    candidate_servers: set[str] = set(default_servers)
    inst_field = _INST_FIELD.get(source)
    if inst_field:
        # Multi-instance trackers union in per-instance ``server`` overrides so a second
        # Atlassian site / Plane workspace counts as configured; the ops/platform connectors
        # (KG-2.155+) are single-server, so their default candidate tuple is authoritative.
        for default_server in default_servers:
            candidate_servers.update(
                _tracker_instance_servers(inst_field, default_server)
            )
    return any(_mcp_server_configured(servers, s) for s in candidate_servers)


# ── ARD registry delta handler (CONCEPT:AU-KG.ingest.source-sync-canonical) ────────────────────────────


def _resolve_ard_registries() -> list[dict[str, Any]]:
    """Resolve configured external ARD registries from ``ARD_REGISTRIES``.

    The value is a JSON list of ``{name, preset|catalog_url, search_url?, media_types?}``
    objects (a bare string item is treated as a preset name), so an operator points the
    consume side at HF + any peer registry with one config key.
    """
    import json as _json

    from ...core.config import setting

    raw = (setting("ARD_REGISTRIES", default="") or "").strip()
    if not raw:
        return []
    try:
        data = _json.loads(raw)
    except Exception:  # noqa: BLE001 — malformed config ⇒ no registries
        return []
    items = data if isinstance(data, list) else [data]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append({"name": item, "preset": item})
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


def _ard_slug(value: str) -> str:
    """A stable, url-safe slug for an ARD registry/resource/capability id."""
    import re as _re

    return _re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "x"


def _ard_capabilities(
    record: dict[str, Any], node_id: str, src: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A resource's tags as ``(:ServiceCapability nodes, providesCapability edges)``."""
    entities: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    for tag in record.get("tags") or []:
        cap = str(tag).strip().lower()
        if not cap:
            continue
        cap_node = f"capability:{_ard_slug(cap)}"
        entities.append(
            {
                "id": cap_node,
                "type": "ServiceCapability",
                "name": cap,
                "domain": "ard",
                "source_system": src,
            }
        )
        links.append(
            {
                "source": node_id,
                "target": cap_node,
                "type": "providesCapability",
                "domain": "ard",
            }
        )
    return entities, links


def _ard_resource_node(
    doc: Any,
    node_id: str,
    media: str,
    src: str,
    record: dict[str, Any],
    resource_links: list[dict[str, Any]],
) -> dict[str, Any]:
    """The typed node for one ARD resource, with its self-sourced ``_links``."""
    eid = getattr(doc, "id", None)
    return {
        "id": node_id,
        "type": "Skill" if media == "application/ai-skill" else "MCPServer",
        "name": getattr(doc, "title", None) or str(eid),
        "description": getattr(doc, "text", "") or "",
        "domain": "ard",
        "source_system": src,
        "externalToolId": str(eid),
        "ardMediaType": media,
        "publisherDomain": str((record.get("publisher") or {}).get("domain", "")),
        "updatedAt": getattr(doc, "updated_at", None),
        "_links": resource_links,
    }


def _ard_resource_rows(
    doc: Any, registry_name: str, registry_node: str, src: str
) -> list[dict[str, Any]]:
    """One ARD resource as its capability nodes plus its own typed node.

    ``application/ai-skill`` → ``:Skill``; anything else → ``:MCPServer``.
    """
    eid = getattr(doc, "id", None)
    if not eid:
        return []
    meta = getattr(doc, "metadata", None) or {}
    record = r if isinstance((r := meta.get("record")), dict) else {}
    media = str((meta or {}).get("ard_media_type") or "")
    node_id = f"ard:{registry_name}:{_ard_slug(eid)}"
    cap_entities, cap_links = _ard_capabilities(record, node_id, src)
    resource_links: list[dict[str, Any]] = [
        {
            "source": node_id,
            "target": registry_node,
            "type": "registeredIn",
            "domain": "ard",
        },
        *cap_links,
    ]
    return [
        *cap_entities,
        _ard_resource_node(doc, node_id, media, src, record, resource_links),
    ]


def _ard_entities(docs: list[Any], registry_name: str) -> list[dict[str, Any]]:
    """Map drained ARD resource docs → typed KG entities (KG-2.188).

    ``application/mcp-server*`` → ``:MCPServer``; ``application/ai-skill`` → ``:Skill``;
    every resource links ``registeredIn`` its ``:ResourceRegistry`` and ``providesCapability``
    a ``:ServiceCapability`` per tag — reusing the a2a/capability ontology terms so an
    ingested external capability is queryable exactly like a native one.

    AU-P1-5 (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): both edges are
    self-sourced from the resource's own record (registry/resource/capabilities are
    always resolved together in this one drain) and carried on the RESOURCE's own
    ``_links`` — the resource also carries the real per-record ``updated_at``, unlike
    the versionless registry/capability nodes.
    """
    src = f"ard:{registry_name}"
    registry_node = f"ard:registry:{_ard_slug(registry_name)}"
    entities: list[dict[str, Any]] = [
        {
            "id": registry_node,
            "type": "ResourceRegistry",
            "name": registry_name,
            "domain": "ard",
            "source_system": src,
        }
    ]
    for doc in docs:
        entities.extend(_ard_resource_rows(doc, registry_name, registry_node, src))
    return entities


def _ard_registry_conn(reg: dict[str, Any], client: Any) -> Any:
    """Build one registry's signature-verified ``ard`` connector.

    ``client`` may inject a fetch function for offline tests.
    """
    from ...protocols.source_connectors.registry import build_connector

    conf = {k: v for k, v in reg.items() if k != "name"}
    if callable(client):
        conf["fetch_fn"] = client
    return build_connector("ard", conf)


def _ard_registry_result(
    engine: Any, reg: dict[str, Any], mode: str, client: Any
) -> tuple[dict[str, Any], int, int, set[str]]:
    """Drain + ingest one ARD registry, as ``(row, nodes, failures, live_ids)``."""
    name = str(reg.get("name") or reg.get("preset") or "ard")
    try:
        conn = _ard_registry_conn(reg, client)
    except Exception as exc:  # noqa: BLE001 — a misconfigured registry is a skip
        return (
            {"registry": name, "status": "skipped", "reason": str(exc)[:160]},
            0,
            0,
            set(),
        )
    since = (
        None
        if mode == "full"
        else _read_envelope_watermark(
            engine,
            "ard",
            source_instance=name,
        )
    )
    docs, fetch_ok = _drain_incremental(conn, since)
    live = {str(getattr(d, "id", "")) for d in docs if getattr(d, "id", None)}
    if mode == "reconcile":
        return (
            _reconcile(
                engine,
                "ard",
                live,
                source_instance=name,
                fetch_ok=fetch_ok,
            )
            | {"registry": name},
            0,
            0,
            live,
        )
    entities = _ard_entities(docs, name)
    ok, failed = _ingest_entities_via_envelope(
        engine, "ard", entities, source_instance=name
    )
    fails = int(getattr(conn, "verify_failures", 0) or 0) + failed
    return (
        {
            "registry": name,
            "resources": len(docs),
            "verify_failures": fails,
            "since": since,
        },
        ok,
        fails,
        live,
    )


def _sync_ard(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest external ARD registries as typed discoverable resources (CONCEPT:AU-KG.ingest.source-sync-canonical).

    For every registry in ``ARD_REGISTRIES`` (e.g. ``[{"name":"hf","preset":"huggingface"}]``)
    this drains the ``ard`` connector (signature-verified), maps each resource to a typed
    ``:MCPServer``/``:Skill`` node linked to its ``:ResourceRegistry`` + capabilities,
    then emits one ChangeEnvelope per typed object. ``mode='reconcile'`` tombstones
    resources no longer present. ``client`` may inject a fetch function for offline tests.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): each
    registry/resource/capability node is one ``ChangeEnvelope`` via
    :func:`_ingest_entities_via_envelope`, ``source_instance=<registry name>`` so the
    per-envelope watermark key (``ard:<name>``) matches this handler's own existing
    ``wm_key`` format exactly.
    """
    registries = _resolve_ard_registries()
    if not registries:
        return {"status": "skipped", "reason": "no ARD_REGISTRIES configured"}

    results: list[dict[str, Any]] = []
    total_e = total_fail = 0
    all_live: set[str] = set()
    for reg in registries:
        row, hydrated, fails, live = _ard_registry_result(engine, reg, mode, client)
        results.append(row)
        total_e += hydrated
        total_fail += fails
        all_live |= live
    return {
        "status": "ok",
        "source": "ard",
        "mode": mode,
        "delta_capable": True,
        "registries": results,
        "nodes_hydrated": total_e,
        "verify_failures": total_fail,
    }


def _parse_memory_file(path: Any) -> tuple[str, str, str, str, str, list[str]]:
    """Parse a Claude Code memory markdown file into
    ``(slug, name, description, memory_type, body, links)``.

    Reads the ``name`` / ``description`` / ``metadata.type`` YAML frontmatter (dependency-
    free — a tiny line scan, no yaml import) and the ``[[other-slug]]`` wiki-links in the
    body. Anything missing falls back to the filename stem / sensible defaults.
    """
    import re

    text = path.read_text(encoding="utf-8", errors="replace")
    slug = path.stem
    name, description, mtype, body = slug, "", "memory", text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if m:
        fm, body = m.group(1), m.group(2)
        for line in fm.splitlines():
            key, _, val = line.partition(":")
            k, v = key.strip(), val.strip()
            if k == "name" and v:
                name = v
            elif k == "description":
                description = v
            elif k == "type" and v:  # ``metadata.type`` (indented) or a top-level type
                mtype = v
    links = re.findall(r"\[\[([a-z0-9][a-z0-9-]*)\]\]", body)
    return slug, name, description, mtype, body.strip(), links


def _sync_package_install(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Auto-extend the KG when a package is installed (CONCEPT:AU-KG.ingest.package-install-autoingest).

    Thin wiring over :func:`~..ingestion.package_install_ingest.sync_package_install`
    (the actual logic lives there, next to ``skill_workflow_ingest``/
    ``change_envelope`` — kept out of this already-large module): reads the
    universal-installer's ``install-manifest.json`` as a change signal and
    re-drives the EXISTING prompt-registry / ontology-federation /
    workflow-skill reloads rather than reimplementing ingestion. See that
    module's docstring for the full design.
    """
    from ..ingestion.package_install_ingest import sync_package_install

    return sync_package_install(engine, mode=mode, ids=ids, client=client)


def _claude_memory_files() -> list[Any]:
    """Every Claude Code memory *topic* file (the MEMORY indexes are excluded).

    The memory dir is ``CLAUDE_MEMORY_DIR`` when set, else every
    ``~/.claude/projects/*/memory`` is swept.
    """
    import glob
    import os
    from pathlib import Path

    from ...core.config import setting

    explicit = (setting("CLAUDE_MEMORY_DIR", default="") or "").strip()
    dirs = (
        [explicit]
        if explicit
        else sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/memory")))
    )
    files: list[Any] = []
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            files.extend(
                f
                for f in sorted(p.glob("*.md"))
                if f.name not in ("MEMORY.md", "MEMORY-ARCHIVE.md")
            )
    return files


def _claude_memory_record(path: Any) -> tuple[str, dict[str, Any]]:
    """One memory topic file as ``(slug, connector record)``."""
    slug, name, description, mtype, body, links = _parse_memory_file(path)
    eid = f"claude_memory:{slug}"
    text = (f"{description}\n\n{body}").strip()
    record: dict[str, Any] = {
        "id": eid,
        "type": "AgentMemory",
        "name": name,
        "slug": slug,
        "memory_type": mtype,
        "description": description,
        "text": text,
        # The durable envelope may identify the configured source class,
        # never the host/user-specific filesystem location.
        "source_uri": "configured-memory",
        # A stable digest of the file's actual content -- NOT the constant
        # ``id`` -- so ``ChangeEnvelope.from_connector_record``'s
        # ``source_version``/idempotency key genuinely varies with content
        # (the documented "content-hash write-delta": an unchanged topic
        # file is skipped, but a changed one under the same slug must
        # produce a new idempotent version rather than colliding with the
        # previous commit under that slug). Full 64-hex-char sha256: the
        # envelope privacy gate's opaque-digest allowlist
        # (envelope_ingest._OPAQUE_DIGEST) only recognizes 24/32/40/64-hex
        # lengths as opaque and exempt from the free-text privacy scan; a
        # truncated digest falls through to that scan and can trip its
        # regex heuristics on ordinary hex material (the same class of
        # false positive already documented there for sha256 digests).
        "content_version": hashlib.sha256(
            f"{name}\n{mtype}\n{text}".encode()
        ).hexdigest(),
    }
    rel_links = [
        {"source": eid, "target": f"claude_memory:{tgt}", "type": "RELATED_TO"}
        for tgt in dict.fromkeys(links)  # de-dup, preserve order
        if tgt != slug
    ]
    if rel_links:
        record["_links"] = rel_links
    return slug, record


def _claude_memory_apply(engine: Any, record: dict[str, Any]) -> tuple[int, int, int]:
    """Commit one memory envelope, as ``(nodes, edges, failed)``."""
    from ..ingestion.change_envelope import ChangeEnvelope
    from ..ingestion.envelope_ingest import ingest_envelope

    env: ChangeEnvelope | None = ChangeEnvelope.from_connector_record(
        record,
        connector="claude_memory",
        id_field="id",
        version_field="content_version",
    )
    env, blocked = _apply_with_preflight_one(engine, "claude_memory", env)
    if env is None:
        logger.warning(
            "claude_memory envelope blocked by backfeed preflight: %s", blocked
        )
        return 0, 0, 1
    result = ingest_envelope(engine, env)
    if result.get("status") not in {"success", "skipped"}:
        logger.warning(
            "claude_memory envelope %s failed: %s",
            env.idempotency_key,
            result.get("error"),
        )
        return 0, 0, 1
    wr = result.get("write_result") or {}
    return wr.get("nodes", 0), wr.get("edges", 0), 0


def _sync_claude_memory(
    engine: Any, *, mode: str, ids: list[str] | None, client: Any
) -> dict[str, Any]:
    """Ingest the Claude Code file-based memory (the ``MEMORY.md`` topic files) into the KG
    as typed ``:AgentMemory`` nodes (CONCEPT:AU-KG.ingest.claude-memory-connector).

    The harness keeps its cross-session memory as flat markdown outside the graph; this
    dogfoods our OWN memory substrate — each topic file becomes a semantically-searchable
    ``:AgentMemory`` node (name/type/description/body embedded, findable via ``graph_search``)
    and its ``[[other-slug]]`` wiki-links become ``RELATED_TO`` edges, so the session
    knowledge is connected to the rest of the ecosystem graph instead of stranded on disk.

    Zero-infra + offline (reads local markdown, no network). The memory dir is
    ``CLAUDE_MEMORY_DIR`` when set, else every ``~/.claude/projects/*/memory`` is swept.
    Delta is the content-hash write-delta (unchanged topic files are skipped even
    on a full sweep); ``ids`` narrows to specific slugs. The ``MEMORY.md`` /
    ``MEMORY-ARCHIVE.md`` indexes themselves are skipped — only the per-memory
    topic files are ingested.

    AU-P1-5 envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction):
    each topic file becomes one ``ChangeEnvelope`` (its ``RELATED_TO`` links
    carried as ``_links``, since a memory's edges are always self-authored —
    ``source`` is always this record's own id, so attaching them to its own
    envelope is lossless) routed through
    :func:`~..ingestion.envelope_ingest.ingest_envelope` — graph material,
    policy, lineage, version, cursor, and outbox per file in one native commit.
    Migrated second (after ``leanix``) as the simplest self-contained offline
    exemplar.
    """
    files = _claude_memory_files()
    if not files:
        return {
            "status": "skipped",
            "reason": "no Claude memory dir (set CLAUDE_MEMORY_DIR) or no *.md topic files",
        }

    id_filter = set(ids or [])
    nodes = 0
    edges = 0
    failed = 0
    for path in files:
        slug, record = _claude_memory_record(path)
        if id_filter and slug not in id_filter:
            continue
        memory_nodes, memory_edges, memory_failed = _claude_memory_apply(engine, record)
        nodes += memory_nodes
        edges += memory_edges
        failed += memory_failed

    return {
        "status": "ok",
        "source": "claude_memory",
        "mode": mode,
        "delta_capable": True,
        "memories_seen": len(files),
        "nodes": nodes,
        "edges": edges,
        "failed": failed,
    }


# Sources with a native delta (watermark/reconcile) handler. Add an entry here to
# make another source incremental (e.g. Camunda once its extractor takes `since`).
_DELTA_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "package_install": _sync_package_install,
    "claude_memory": _sync_claude_memory,
    "leanix": _sync_leanix,
    "archivebox": _sync_archivebox,
    "gitlab": _sync_gitlab,
    "freshrss": _sync_freshrss,
    "rss": _sync_rss,
    "arxiv": _sync_arxiv,
    "jira": _sync_jira,
    "confluence": _sync_confluence,
    "plane": _sync_plane,
    # Ops / platform connectors as typed OWL entities (CONCEPT:AU-KG.compute.dockerhub-repositories–2.161)
    "dockerhub": _sync_dockerhub,
    "langfuse": _sync_langfuse,
    "technitium": _sync_technitium,
    "tunnel_manager": _sync_tunnel_manager,
    "uptime_kuma": _sync_uptime_kuma,
    "home_assistant": _sync_home_assistant,
    "twenty": _sync_twenty,
    # Media / finance / document / genealogy connectors (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors–2.166)
    "audiobookshelf": _sync_audiobookshelf,
    "firefly_iii": _sync_firefly_iii,
    "paperless_ngx": _sync_paperless_ngx,
    "gramps": _sync_gramps,
    # External ARD registries (HF + peers) as typed discoverable resources (KG-2.188).
    "ard": _sync_ard,
    "fleet": _sync_fleet,
    "fleet_connectors": _sync_fleet_connectors,
    # L27 (AU-P1-5): live sync_source call sites for the 6 mandatory-manifest ops
    # connectors that previously had none — see ``_ops_connector_config`` above.
    # Envelope-native from day one (CONCEPT:AU-KG.ingest.envelope-atomic-transaction).
    "microsoft-agent": _sync_microsoft_agent,
    "container-manager-mcp": _sync_container_manager_mcp,
    "documentdb-mcp": _sync_documentdb_mcp,
    "repository-manager": _sync_repository_manager,
    "systems-manager": _sync_systems_manager,
    "vector-mcp": _sync_vector_mcp,
}

# CA-21 -> CA-22 (ordered pair, same wave, this lane merges second): CA-21's
# Debezium/CDC consumer registers its own catchup handler by name via
# ``debezium_envelope.register_envelope_source("cdc", run_cdc_catchup)`` at
# import time (``ingestion/debezium_envelope.py:434``); this is the one-line
# wiring that makes ``_DELTA_HANDLERS`` — and therefore ``ENVELOPE_NATIVE_
# SOURCES`` (the derived-set formula immediately below, unconditionally) —
# pick it up automatically. ``get_envelope_source`` returns
# ``Callable[..., dict[str, Any]] | None``; a ``None`` (module not yet
# imported/registered) is intentionally never inserted — an absent CDC
# handler must fall through to the ordinary full-hydrate path, not a
# ``_DELTA_HANDLERS["cdc"] = None`` entry that would crash ``sync_source``'s
# dispatch on call.
from ..ingestion.debezium_envelope import (
    get_envelope_source as _get_cdc_envelope_source,
)

_cdc_handler = _get_cdc_envelope_source("cdc")
if _cdc_handler is not None:
    _DELTA_HANDLERS["cdc"] = _cdc_handler
del _cdc_handler

# AU-P1-5 (CONCEPT:AU-KG.ingest.envelope-atomic-transaction) — enumerated migration
# status of every ``_DELTA_HANDLERS`` entry so there is no silent gap between "one
# ChangeEnvelope, one atomic ingest_envelope transaction" (the target model) and
# what actually runs today. Object writes route through ``ingest_envelope``
# (validate/lineage/CDC/monotonic per-record watermark, crash-resume safe) — either
# directly (leanix/claude_memory/L27), via the shared
# :func:`_ingest_entities_via_envelope` tail (the "typed OWL entity" handlers:
# dockerhub, langfuse, technitium, tunnel_manager, uptime_kuma, home_assistant,
# twenty, audiobookshelf, firefly_iii, paperless_ngx, gramps, jira, plane, ard), or
# through a document-shaped native pipeline that itself commits via
# ``ApplyChangeEnvelope`` (gitlab / archivebox / freshrss / rss / confluence /
# fleet / fleet_connectors — each may chunk/embed/gate ONE external record into
# MANY derived KG nodes, but still lands through the same atomic engine boundary,
# never the historical ad hoc ``engine.ingest_external_batch`` call). Every
# durable external-ingestion handler is native. The sole
# :data:`ORCHESTRATION_ONLY_SOURCES` entry, ``package_install``, is a write-free
# dispatcher in this module: it never builds an ``entities``/``rels`` batch (or a
# ChangeEnvelope) itself — it re-drives three ALREADY-native ingestion primitives
# (the prompt registry, ontology federation, workflow-skill ingest) that each own
# their own write shape/idempotency and are audited separately.
ORCHESTRATION_ONLY_SOURCES: frozenset[str] = frozenset({"package_install"})
ENVELOPE_NATIVE_SOURCES: frozenset[str] = (
    frozenset(_DELTA_HANDLERS) - ORCHESTRATION_ONLY_SOURCES
)


def _connector_manifest_gate(norm_source: str, mode: str) -> dict[str, Any] | None:
    """Run the compile-before-sync gate; a dict result is a fail-closed refusal.

    ``None`` means the source may dispatch. Missing manifests/providers, drift, or
    a precheck exception all refuse BEFORE dispatch — there is no unowned runtime
    connector pass-through (CONCEPT:AU-KG.ontology.connector-manifest-gate, D17).
    """
    from ..etl.result import EtlResult

    try:
        from ..ontology.connector_manifest_gate import precheck_source

        gate = precheck_source(norm_source)
    except Exception as exc:  # noqa: BLE001 - gate failure must fail closed
        logger.warning(
            "connector-manifest precheck failed closed for %s (%s)",
            norm_source,
            type(exc).__name__,
        )
        return EtlResult(
            status="error",
            source=norm_source or None,
            mode=mode,
            reason=f"connector-manifest precheck failed closed ({type(exc).__name__})",
        ).model_dump()
    if not gate.get("checked") or not gate.get("ok"):
        logger.warning(
            "source_sync: %s refused — connector_manifest.yml failed the "
            "compile-before-sync gate: %s",
            norm_source,
            gate.get("violations"),
        )
        return EtlResult(
            status="error",
            source=norm_source or None,
            mode=mode,
            reason="connector_manifest.yml failed the compile-before-sync "
            f"gate ({gate.get('connector')}): {gate.get('violations')}",
        ).model_dump()
    return None


def _etl_split_fields(res: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a raw connector result into canonical ``EtlResult`` fields + details."""
    from ..etl.result import EtlResult

    canonical_fields = set(EtlResult.model_fields)
    payload = {key: value for key, value in res.items() if key in canonical_fields}
    details = {key: value for key, value in res.items() if key not in canonical_fields}
    return payload, details


def _etl_result_payload(res: Any, norm_source: str, mode: str) -> dict[str, Any]:
    """Project any dispatch result onto the strict ``EtlResult`` wire schema.

    Connector-specific diagnostics are namespaced under ``details`` and are never
    interpreted as canonical counts (CONCEPT:AU-KG.etl.result-contract).
    """
    from ..etl.result import EtlResult

    if isinstance(res, EtlResult):
        return res.model_dump()
    if not isinstance(res, dict):
        return EtlResult(
            status="error",
            source=norm_source or None,
            mode=mode,
            error="connector returned a non-object result",
        ).model_dump()
    payload, details = _etl_split_fields(res)
    payload.setdefault("source", norm_source or None)
    payload.setdefault("mode", mode)
    payload["details"] = {**dict(payload.get("details") or {}), **details}
    return EtlResult.model_validate(payload).model_dump()


def sync_source(
    engine: Any,
    source: str,
    *,
    mode: str = "delta",
    ids: list[str] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Sync one external source into the KG (the single entrypoint).

    ``mode`` ∈ {delta, full, reconcile}. Delta-capable sources (``_DELTA_HANDLERS``)
    do incremental watermark/reconcile; any other registered source falls back to a
    full hydrate via the capability registry.

    Every dispatch path is projected onto the strict
    :class:`..etl.result.EtlResult` wire schema
    (CONCEPT:AU-KG.etl.result-contract). Connector-specific diagnostics are
    namespaced under ``details`` and are not interpreted as canonical counts.

    Before dispatch, a **compile-before-sync** gate (CONCEPT:AU-KG.ontology.connector-manifest-gate,
    D17) requires this source's owned ``connector_manifest.yml``, re-verifies its
    compiled canonical hash and release signature, and requires either an installed
    connector-owned provider or the release-pinned remote-provider snapshot to match
    the signed tool/field maps exactly. Missing manifests/providers, drift, or a
    precheck exception fail closed before dispatch. There is no unowned runtime
    connector pass-through.
    """
    norm_source = (source or "").lower().strip()

    if norm_source not in {"all", "*", "sweep"}:
        refused = _connector_manifest_gate(norm_source, mode)
        if refused is not None:
            return refused

    res = _dispatch_sync_source(engine, norm_source, mode=mode, ids=ids, client=client)
    return _etl_result_payload(res, norm_source, mode)


# An UNCONFIGURED upstream (its MCP server isn't in mcp_config, no creds, etc.) is a
# *skip*, never a task failure — the fleet sweep routinely runs with only a subset of
# connectors provisioned (CONCEPT:AU-KG.ingest.enterprise-source-extractor). Real
# errors still propagate.
#
# ``unknown mcp_tool preset`` is the same class: a source is "configured" (its MCP
# server is registered in mcp_config) but the connector PACKAGE that ships the matching
# contributed mcp_tool preset isn't pip-installed in this process — e.g. FreshRSS /
# ScholarX reached purely over the wire without the freshrss-agent/scholarx package
# co-installed (CONCEPT:AU-KG.ingest.research-connector-presets). The preset genuinely
# isn't resolvable here, so it is a skip too.
_UNCONFIGURED_HANDLER_TOKENS = (
    "not found in mcp_config",
    "not configured",
    "no client",
    "credential",
    "unconfigured",
    "unknown mcp_tool preset",
)


def _run_delta_handler(
    engine: Any,
    source: str,
    handler: Callable[..., dict[str, Any]],
    mode: str,
    ids: list[str] | None,
    client: Any,
) -> dict[str, Any]:
    """Run one registered delta handler; an unconfigured upstream becomes a skip."""
    try:
        return handler(engine, mode=mode, ids=ids, client=client)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if any(token in msg for token in _UNCONFIGURED_HANDLER_TOKENS):
            return {"status": "skipped", "source": source, "reason": str(exc)[:160]}
        raise


def _chunked_drain_handle(engine: Any, source: str) -> dict[str, Any] | None:
    """A handle for a chunked full drain, or ``None`` to keep the sync inline."""
    from .chunked_drain import (
        chunked_drain_enabled,
        start_chunked_drain,
        supports_chunked_drain,
    )

    if chunked_drain_enabled() and supports_chunked_drain(source):
        return start_chunked_drain(engine, source, mode="full")
    return None


def _materialize_or_hydrate(engine: Any, source: str) -> dict[str, Any]:
    """The no-delta-handler fallback: materialize substrate, else generic hydrate."""
    # Extractor/materialize-substrate sources (camunda/aris/egeria) route through the
    # shared materialize core so this stays the one entrypoint for every source.
    from ..enrichment.materialize import MATERIALIZE_SOURCES, run_materialize_source

    if source in MATERIALIZE_SOURCES:
        res = run_materialize_source(engine, source)
        res.setdefault("mode", "full")
        res.setdefault("delta_capable", False)
        return res

    # Otherwise: generic full hydrate via the CAPABILITY_REGISTRY.
    from .hydration import HydrationManager

    res = HydrationManager().hydrate_source(engine, source)
    if isinstance(res, dict):
        res.setdefault("source", source)
        res.setdefault("mode", "full")
        res.setdefault("delta_capable", False)
    return res


def _dispatch_sync_source(
    engine: Any,
    source: str,
    *,
    mode: str = "delta",
    ids: list[str] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """The raw dispatch logic for :func:`sync_source` (pre-``EtlResult`` coercion)."""

    # "all"/"*"/"sweep" → fan out across every configured connector in one pass
    # so the one entrypoint also covers "ingest everything" (CONCEPT:AU-KG.ingest.enterprise-source-extractor).
    if source in {"all", "*", "sweep"}:
        return sweep_all_sources(engine, mode=mode if mode in SYNC_ACTIONS else "delta")

    # CONCEPT:AU-KG.ontology.single-source-full-drain — a single-source FULL drain of a LARGE corpus must NOT run inline:
    # that would block the MCP/REST request until the whole backlog is drained (timeout) or
    # force a human/agent to hand-repeat delta waves. Normalize that ONE call into a stream of
    # capacity-guarded, paginated ``connector_drain`` batch-tasks and return a handle IMMEDIATELY
    # — the "controlled waves" are baked in, not hand-driven. Small/delta syncs stay inline (fast).
    if mode == "full" and hasattr(engine, "submit_task"):
        handle = _chunked_drain_handle(engine, source)
        if handle is not None:
            return handle

    handler = _DELTA_HANDLERS.get(source)
    if handler is not None:
        return _run_delta_handler(engine, source, handler, mode, ids, client)

    if mode == "reconcile":
        return {
            "status": "skipped",
            "reason": f"reconcile not supported for '{source}' (no delta handler)",
        }

    return _materialize_or_hydrate(engine, source)


def _sweep_drop_unconfigured_trackers(candidates: set[str]) -> None:
    """Drop MCP-backed dedicated trackers whose ``*-mcp`` server is absent.

    CONCEPT:AU-KG.compute.mcp-backed-dedicated-trackers — the MCP-backed dedicated
    trackers (jira/confluence/plane) reach their upstream ONLY through a fleet
    ``*-mcp`` server, so their "configured" signal is *"the server is registered in
    mcp_config.json"* — NOT an env token (capability-registry) nor always-on
    (feed/fleet handlers). Keep one as a candidate when its server is in mcp_config
    (the live remote-routed atlassian/plane case the operator runs), and DROP it when
    truly unconfigured so the sweep neither wastes a connector_sync task nor
    misreports a reachable tracker as missing. (Before this gate they were enqueued
    unconditionally, so a tracker whose ``*-mcp`` server was absent under the expected
    key still spawned a task that the connector then aborted with "not found in
    mcp_config" → 0 nodes, never surfacing as configured work.)
    """
    for tracker in _MCP_TRACKER_SERVERS:
        if tracker in candidates and not _mcp_tracker_configured(tracker):
            candidates.discard(tracker)


def _sweep_configured_capability_sources() -> set[str]:
    """Capability-registry sources that env-detect as *configured* (best-effort)."""
    from .hydration import HydrationManager

    try:
        return {
            src
            for src, conf in HydrationManager().get_status().items()
            if isinstance(conf, dict) and conf.get("configured")
        }
    except Exception:  # noqa: BLE001 — status probe is best-effort
        logger.debug("capability status probe failed", exc_info=True)
        return set()


def _sweep_materialize_sources() -> set[str]:
    """Materialize extractor sources whose client provider is importable here."""
    try:
        from ..enrichment.materialize import (
            MATERIALIZE_SOURCES,
            source_client_provider_installed,
        )

        return {
            source
            for source in MATERIALIZE_SOURCES
            if source_client_provider_installed(source)
        }
    except Exception:  # noqa: BLE001
        logger.debug("materialize source list unavailable", exc_info=True)
        return set()


def _sweep_candidate_sources(include_materialize: bool) -> set[str]:
    """The union of delta handlers, configured capability sources and extractors."""
    candidates: set[str] = set(_DELTA_HANDLERS)
    # ``fleet`` capability elevation re-probes ~62 MCP servers; the capability
    # vocabulary is slow-changing, so it runs at boot + on explicit refresh
    # (``source_sync source=fleet``), not on every */20m document sweep.
    candidates.discard("fleet")
    _sweep_drop_unconfigured_trackers(candidates)
    candidates |= _sweep_configured_capability_sources()
    if include_materialize:
        candidates |= _sweep_materialize_sources()
    return candidates


def _sweep_governed_candidates(
    candidates: set[str], *, rejections: dict[str, str] | None = None
) -> set[str]:
    """Filter the candidate union through the signed compile-before-sync contract.

    A registered handler or locally importable extractor is only a candidate
    implementation, not authority to schedule an external pull. Filtering the
    complete union through the same contract used by :func:`sync_source` keeps boot
    from creating guaranteed-failure jobs for stale aliases (for example
    ``freshrss`` or ``homeassistant``) whose provider is neither installed nor
    represented by a valid release-pinned bundle.
    """
    from ..ontology.connector_manifest_gate import precheck_source

    governed: set[str] = set()
    for source in sorted(candidates):
        accepted, rejection = _precheck_sweep_source(precheck_source, source)
        if accepted:
            governed.add(source)
        elif rejections is not None:
            rejections[source] = rejection
    return governed


def _precheck_sweep_source(
    precheck: Callable[[str], dict[str, Any]], source: str
) -> tuple[bool, str]:
    """Run one provider gate and return a stable, content-free rejection."""
    try:
        gate = precheck(source)
    except Exception as exc:  # noqa: BLE001 - fail closed before queue publication
        logger.debug(
            "source sweep contract precheck failed for %s",
            source,
            exc_info=True,
        )
        return False, "provider_precheck_failed:" + type(exc).__name__
    if bool(gate.get("checked")) and bool(gate.get("ok")):
        return True, ""
    logger.debug(
        "source sweep omitted %s because its governed provider contract is unavailable",
        source,
    )
    return False, "provider_contract_unavailable"


def _enqueue_one_sweep_task(
    submit: Callable[..., Any],
    source: str,
    mode: str,
    priority: int | None,
    jobs: list[str],
    jobs_by_source: dict[str, str],
    failures: dict[str, str],
    indeterminate: dict[str, str],
) -> None:
    """Submit and classify one task without leaking provider-controlled text."""
    try:
        handle = submit(
            target_path=source,
            is_codebase=False,
            provenance={"sync_mode": mode},
            task_type="connector_sync",
            **({"priority": priority} if priority is not None else {}),
        )
    except Exception as exc:  # noqa: BLE001 — admission may already be durable
        indeterminate[source] = f"submission_indeterminate:{type(exc).__name__}"
        logger.warning(
            "enqueue connector_sync outcome indeterminate for %s (error_class=%s)",
            source,
            type(exc).__name__,
        )
        return
    if not isinstance(handle, str) or not handle.strip():
        failures[source] = "invalid_job_handle"
        logger.warning(
            "enqueue connector_sync returned an invalid job handle for %s (return_type=%s)",
            source,
            type(handle).__name__,
        )
        return
    jobs.append(handle)
    jobs_by_source[source] = handle


def _enqueue_sweep_tasks(
    submit: Callable[..., Any],
    candidates: set[str],
    mode: str,
    priority: int | None,
) -> tuple[list[str], dict[str, str], dict[str, str], dict[str, str]]:
    """Submit one laned task per source and retain per-source failures.

    A failed submission must remain visible to the sweep caller: a missing job
    handle is not an enqueued task, and a debug-only log cannot tell an operator
    which source needs retrying. An exception has indeterminate admission state:
    ``submit_task`` may have durably admitted the WorkItem before a later
    notification/readiness step raised.
    """
    jobs: list[str] = []
    jobs_by_source: dict[str, str] = {}
    failures: dict[str, str] = {}
    indeterminate: dict[str, str] = {}
    for src in sorted(candidates):
        _enqueue_one_sweep_task(
            submit,
            src,
            mode,
            priority,
            jobs,
            jobs_by_source,
            failures,
            indeterminate,
        )
    return jobs, jobs_by_source, failures, indeterminate


_SWEEP_UNCONFIGURED = (
    "not configured",
    "no client",
    "missing",
    "unconfigured",
    "credential",
)


def _sweep_error_reason(res: Any) -> str:
    """The reason string for a connector result that reported error/failed."""
    # Connector error strings are provider-controlled and may contain credentials,
    # URLs, paths, or upstream payloads. Keep the public sweep contract stable and
    # content-free; the connector's own server-side log retains diagnostics.
    return "connector_error" if isinstance(res, dict) else "error"


def _sweep_exception_text(exc: Exception) -> str:
    """Render an exception only for private classification, never for output."""
    try:
        return str(exc)
    except Exception:  # noqa: BLE001 - malformed exception rendering is still a failure
        return ""


def _sweep_synced_entry(res: Any, status: Any) -> Any:
    """The recorded value for a connector that synced."""
    if not isinstance(res, dict):
        return status
    return {
        "counts": dict(res.get("counts") or {}),
        "details": dict(res.get("details") or {}),
    }


def _classify_sweep_result(res: Any) -> tuple[str, Any]:
    """Bucket one connector's sync result as ``synced`` / ``skipped`` / ``errors``."""
    status = res.get("status") if isinstance(res, dict) else "ok"
    if status in {"skipped", "noop"}:
        return "skipped", "skipped"
    if status in {"error", "failed"}:
        return "errors", _sweep_error_reason(res)
    return "synced", _sweep_synced_entry(res, status)


def _classify_sweep_exception(src: str, exc: Exception) -> tuple[str, str]:
    """Bucket a raised connector failure — an unconfigured upstream is a skip."""
    msg = _sweep_exception_text(exc)
    if any(token in msg.lower() for token in _SWEEP_UNCONFIGURED):
        return "skipped", "unconfigured"
    logger.warning(
        "sweep: source '%s' failed (error_class=%s)",
        src,
        type(exc).__name__,
    )
    return "errors", f"connector_error:{type(exc).__name__}"


def _sweep_inline(
    engine: Any,
    candidates: set[str],
    mode: str,
    rejections: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Sequentially sync every candidate, isolating each connector's failure."""
    buckets: dict[str, dict[str, Any]] = {"synced": {}, "skipped": {}, "errors": {}}
    for src in sorted(candidates):
        bucket, value = _run_inline_sweep_source(engine, src, mode)
        buckets[bucket][src] = value
    rejected = dict(rejections or {})
    status, reason = _inline_sweep_status(candidates, buckets, rejected)
    return _inline_sweep_response(mode, candidates, buckets, rejected, status, reason)


def _run_inline_sweep_source(engine: Any, source: str, mode: str) -> tuple[str, Any]:
    """Run and classify one inline connector without affecting its siblings."""
    try:
        return _classify_sweep_result(sync_source(engine, source, mode=mode))
    except Exception as exc:  # noqa: BLE001 — isolate one bad connector
        return _classify_sweep_exception(source, exc)


def _inline_sweep_status(
    candidates: set[str],
    buckets: dict[str, dict[str, Any]],
    rejected: dict[str, str],
) -> tuple[str, str | None]:
    """Derive truthful inline status from completed and failed source counts."""
    completed = len(buckets["synced"]) + len(buckets["skipped"])
    failed = len(buckets["errors"]) + len(rejected)
    if not candidates and not rejected:
        return "skipped", "no governed source candidates"
    if failed and completed:
        return "partial", "sweep completed partially"
    if failed:
        return "error", "no source completed successfully"
    return "ok", None


def _inline_sweep_response(
    mode: str,
    candidates: set[str],
    buckets: dict[str, dict[str, Any]],
    rejected: dict[str, str],
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    """Build the stable inline response envelope."""
    return {
        "status": status,
        "mode": mode,
        "swept": len(candidates),
        "synced": buckets["synced"],
        "skipped": buckets["skipped"],
        "errors": buckets["errors"],
        "rejected": rejected,
        "reason": reason,
        "counts": {
            "candidates": len(candidates),
            "synced": len(buckets["synced"]),
            "skipped": len(buckets["skipped"]),
            "errors": len(buckets["errors"]),
            "rejected": len(rejected),
        },
    }


def _queued_sweep_status(
    jobs: list[str],
    failures: dict[str, str],
    indeterminate: dict[str, str],
    rejections: dict[str, str],
) -> tuple[str, str | None, str | None]:
    """Derive queue status without claiming an exception rolled admission back."""
    if jobs:
        status = "partial" if failures or indeterminate or rejections else "enqueued"
        reason = "sweep completed partially" if status == "partial" else None
        return status, reason, None
    if indeterminate:
        return (
            "indeterminate",
            "connector_sync submission outcome is indeterminate",
            None,
        )
    if failures or rejections:
        return "error", None, "no connector_sync tasks were enqueued"
    return "skipped", "no governed source candidates", None


def _queued_sweep_response(
    submit: Callable[..., Any],
    candidates: set[str],
    rejections: dict[str, str],
    mode: str,
    priority: int | None,
) -> dict[str, Any]:
    """Submit the governed candidate set and return a truthful queue envelope."""
    jobs, jobs_by_source, failures, indeterminate = _enqueue_sweep_tasks(
        submit, candidates, mode, priority
    )
    status, reason, error = _queued_sweep_status(
        jobs, failures, indeterminate, rejections
    )
    return {
        "status": status,
        "enqueued": len(jobs),
        "candidates": len(candidates),
        "mode": mode,
        "jobs": jobs,
        "jobs_by_source": jobs_by_source,
        "errors": failures,
        "indeterminate": indeterminate,
        "rejected": rejections,
        "counts": {
            "candidates": len(candidates),
            "enqueued": len(jobs),
            "errors": len(failures),
            "indeterminate": len(indeterminate),
            "rejected": len(rejections),
        },
        "reason": reason,
        "error": error,
    }


def sweep_all_sources(
    engine: Any,
    *,
    mode: str = "delta",
    include_materialize: bool = True,
    enqueue: bool = True,
    priority: int | None = None,
) -> dict[str, Any]:
    """Ingest every *configured* connector in one background sweep (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

    The fleet-wide counterpart to :func:`sync_source` — it enumerates the union of

    * the delta-capable handlers (:data:`_DELTA_HANDLERS`),
    * the capability-registry sources that env-detect as *configured*, and
    * (optionally) the materialize extractor sources,

    and dispatches each through :func:`sync_source` so they share the one
    watermark/delta/full machinery. ``mode`` defaults to ``"delta"`` so a
    scheduled sweep only pulls (and, via the write-layer content-hash delta, only
    writes) what changed. Per-source failures are isolated and recorded — a
    background sweep never aborts on one bad connector. Optional unconfigured
    sources are omitted; governed precheck failures are surfaced in ``rejected``.
    Queue admission exceptions are reported as *indeterminate* because the task
    may have been durably admitted before the exception was raised.
    """
    rejections: dict[str, str] = {}
    candidates = _sweep_governed_candidates(
        _sweep_candidate_sources(include_materialize), rejections=rejections
    )
    # CONCEPT:AU-ORCH.dispatch.laned-sweep-fanout — fan the sweep out as LANED
    # ``connector_sync`` tasks (the 'connectors' lane) so every connector syncs in
    # PARALLEL instead of one slow connector (gitlab/servicenow) head-of-line-blocking
    # the rest in the sequential inline loop below. Each task runs
    # ``sync_source(src, mode)`` → the same watermark/delta machinery + content-hash delta.
    submit = getattr(engine, "submit_task", None)
    if enqueue and callable(submit):
        return _queued_sweep_response(submit, candidates, rejections, mode, priority)
    return _sweep_inline(engine, candidates, mode, rejections)
