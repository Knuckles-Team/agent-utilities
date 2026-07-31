"""New engine-surface MCP tools — CONCEPT:AU-KG.coordination.engine-message-broker.

epistemic-graph v2.2.0 grew several engine capabilities that agent-utilities
should expose to the fleet as first-class, purpose-shaped MCP tools rather than
leaving them reachable only through the generic ``engine_<domain>`` 1:1 surface
(``engine_tools.py``). This module ADDS those ergonomic wrappers:

* ``graph_broker``          — the message broker (declare exchange/queue, bind,
  publish, consume, stats — AMQP-style; distinct from the agent-to-agent
  ``graph_bus`` in ``bus_tools.py``).
* ``graph_kvcache``         — the shared, content-addressed KV-cache over the
  EG-187 HTTP surface, driven through the KG-2.306
  :class:`~agent_utilities.kvcache.EpistemicGraphKVBackend` connector.
* ``graph_federated_search``— federated search fanned across registered external
  graph references.
* ``graph_promql``          — PromQL instant/range metric queries (observability).
* ``graph_traces``          — distributed-trace search / fetch (observability).
* ``graph_gis``             — geospatial route / tile / geo-task ops.
* ``graph_memory``          — the EG-318 memory surface: episodic→semantic memory
  (create-summary / consolidate / maintain), the spatial scene graph
  (add-scene-object / world-transform), and RL trajectories (start-trajectory /
  append-step / discounted-return), plus their reads.

Design (matches the rest of ``agent_utilities/mcp/tools``):

* **Reuse the existing engine transport** — every client-backed tool resolves the
  same :class:`~epistemic_graph.client.SyncEpistemicGraphClient` that
  ``engine_tools`` uses (via :func:`engine_tools._client_for`); the KV-cache tool
  reuses the KG-2.306 HTTP connector. No new transport is invented.
* **Additive + gated / graceful degradation** — the v2.2.0 surfaces may not be
  present in the connected engine build (or wired into the ``epistemic_graph``
  client yet). Each tool probes a small set of candidate ``client.<sub>.<method>``
  paths and, when none resolve, returns a clean ``degraded`` payload instead of
  raising — so the fleet can call these today and they light up automatically once
  the engine ships the capability.
* **Two surfaces** — each tool registers a ``/graph/<name>`` REST twin in
  ``ACTION_TOOL_ROUTES`` (auto-mounted by the generic factory), keeping MCP⇄REST
  parity like every other tool.

CONCEPT:AU-KG.coordination.engine-message-broker — MCP surface for the new engine ops (broker / kvcache /
federated-search / promql / traces / gis).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server

# CONCEPT:AU-KG.mining.dsm-forecast-delegation — the fleet write-side connector (call
# a named MCP server's tool once, synchronously, decoded) is the SAME primitive the
# governed ``fleet.write_record`` ontology Action uses
# (``agent_utilities/knowledge_graph/actions/fleet_writeback.py``) — reused here,
# not reinvented, so ``graph_mine_deep`` reaches data-science-mcp exactly like every
# other fleet-delegation call site. Neither module imports torch/transformers.
from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
    _run_async,
)
from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
    call_tool_once,
)
from agent_utilities.security.error_surface import public_error_payload
from agent_utilities.security.identifiers import CYPHER_IDENTIFIER_RE

logger = logging.getLogger(__name__)

# Candidate ``(sub_client_attr, method_attr)`` probe lists per logical action. The
# engine build / client may expose the surface under any of several plausible
# namespaces; the first callable found wins and everything else degrades cleanly.
_FEDERATED_SEARCH_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("search", "federated_search"),
    ("query", "federated_search"),
    ("federation", "search"),
    ("query", "federated"),
)
_PROMQL_INSTANT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("observability", "promql"),
    ("metrics", "promql"),
    ("observability", "query"),
    ("promql", "query"),
)
_PROMQL_RANGE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("observability", "promql_range"),
    ("metrics", "query_range"),
    ("observability", "query_range"),
    ("promql", "query_range"),
)
_TRACES_SEARCH_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("observability", "search_traces"),
    ("traces", "search"),
    ("observability", "query_traces"),
)
_TRACES_GET_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("observability", "get_trace"),
    ("traces", "get"),
    ("observability", "trace"),
)

# graph_memory — EG-318 memory / scene / trajectory ops. Each logical action maps
# to a small probe list of ``(sub_client_attr, method_attr)`` paths the engine
# build / client may plausibly expose the EG-318 wire Method under (the client
# wraps each wire ``Method`` as a snake_case method on a sub-client). The first
# callable found wins; when none resolve the tool degrades cleanly. Any action not
# listed here falls back to probing the three sub-clients with the action name
# directly, so read ops (get_summary / get_scene / get_trajectory / …) light up
# by name once the engine ships them.
_MEMORY_ACTION_CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    # episodic → semantic memory (CreateSummaryNode / Consolidate / Maintain)
    "create_summary": (
        ("memory", "create_summary"),
        ("memory", "create_summary_node"),
    ),
    "consolidate": (("memory", "consolidate"),),
    "maintain": (("memory", "maintain"),),
    # spatial scene graph (AddSceneObject / world transform)
    "add_scene_object": (
        ("scene", "add_object"),
        ("scene", "add_scene_object"),
        ("memory", "add_scene_object"),
    ),
    "world_transform": (
        ("scene", "world_transform"),
        ("scene", "set_world_transform"),
    ),
    # trajectories / RL episodes (StartTrajectory / AppendStep / discounted return)
    "start_trajectory": (
        ("trajectory", "start"),
        ("trajectory", "start_trajectory"),
    ),
    "append_step": (
        ("trajectory", "append_step"),
        ("trajectory", "append"),
    ),
    "discounted_return": (("trajectory", "discounted_return"),),
}


# ── graph_mine action manifest (CONCEPT:EG-KG.mining.frequent-itemset-mining) ─
def _mining_actions() -> frozenset[str]:
    """The valid ``graph_mine`` actions.

    Reuses ``engine_tools.ENGINE_DOMAINS['mining']`` — the SAME
    client-introspected manifest (``inspect.getmembers`` over ``MiningClient``,
    mirroring ``engine_tools._discover_domains``) that already drives the
    granular ``engine_mining`` 1:1 tool — as the one source of truth, so this
    surface can never drift out of sync with the real client again
    (CONCEPT:AU-KG.compute.engine-surface-manifest). Empty only when the
    ``epistemic_graph`` client itself can't be imported (an environment/build
    issue unrelated to action naming); callers then skip the allow-list check
    below and fall through to the normal dispatch/degrade path.
    """
    from agent_utilities.mcp.tools import engine_tools

    return frozenset(engine_tools.ENGINE_DOMAINS.get("mining", ()))


#: Guessable alternate spellings that would otherwise silently miss their real
#: ``MiningClient`` attribute and fall through to a misleading "whole surface
#: degraded" payload — ``entity_resolution`` (the family's doc/CONCEPT name) is
#: actually bound as ``entity_resolve``, and ``process_mining`` as ``process``.
#: Looked up AFTER the hyphen->underscore normalization below, so the
#: hyphenated spellings (``entity-resolution``, ``process-mining``) resolve too.
_MINING_ACTION_ALIASES: dict[str, str] = {
    "entity_resolution": "entity_resolve",
    "process_mining": "process",
}


# ── Transport resolution (reuse the one engine client; injectable for tests) ──
def _client(graph: str) -> Any:
    """Resolve the shared ``SyncEpistemicGraphClient`` for ``graph``.

    Delegates to :func:`engine_tools._client_for` so these tools ride the exact
    same connect path (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision resolver, connection caching) as the
    low-level ``engine_<domain>`` tools — one transport, no reinvention. Isolated
    behind this thin indirection so unit tests can inject a mock client
    (CONCEPT:AU-KG.coordination.engine-message-broker).
    """
    from agent_utilities.mcp.tools import engine_tools

    return engine_tools._client_for(graph)


def _kv_backend() -> Any:
    """Build the KG-2.306 KV-cache connector from the engine's EG-187 environment.

    Isolated behind this indirection so unit tests can inject a fake backend
    (CONCEPT:AU-KG.coordination.engine-message-broker).
    """
    from agent_utilities.kvcache import EpistemicGraphKVBackend

    return EpistemicGraphKVBackend.from_env()


class _EngineComputeAdapter:
    """Minimal ``compute`` shape (``._client`` + ``.graph_name``) the MediaStore-style
    graph-resource stores expect, built from the low-level engine client this module
    already resolves (CONCEPT:AU-KG.memory.kv-checkpoint-resource). No new transport —
    reuses the same :func:`_client` every other tool in this module rides."""

    def __init__(self, client: Any, graph_name: str) -> None:
        self._client = client
        self.graph_name = graph_name


def _checkpoint_error_code(exc: Exception) -> str:
    """Map a :class:`~agent_utilities.kvcache.KVCheckpointError` to one of the
    platform's registered public error codes (CONCEPT:AU-KG.memory.kv-checkpoint-resource).

    Deliberately coarse: :func:`~agent_utilities.security.error_surface.public_error_payload`
    only recognizes a small fixed code vocabulary and collapses anything else to
    ``operation_failed`` — and collapsing here is itself the correct security
    posture, not a limitation. Distinguishing "not found" from "cross-tenant" from
    "stale" on the PUBLIC surface would let a caller enumerate which checkpoint ids
    exist under a tenant they don't own; only ``CrossTenantCheckpointError`` gets
    its own (still generic) ``permission_denied`` code, everything else falls to
    the default ``operation_failed``.
    """
    from agent_utilities.kvcache import CrossTenantCheckpointError

    if isinstance(exc, CrossTenantCheckpointError):
        return "permission_denied"
    return "operation_failed"


def _checkpoint_store(graph: str) -> Any:
    """Build a :class:`~agent_utilities.kvcache.KVCheckpointStore` bound to the
    live engine client for ``graph`` (CONCEPT:AU-KG.memory.kv-checkpoint-resource).

    Isolated behind this indirection so unit tests can inject a fake compute
    object, matching every other resolver in this module.
    """
    from agent_utilities.kvcache import KVCheckpointStore

    return KVCheckpointStore(_EngineComputeAdapter(_client(graph), graph))


#: Process-lifetime RAM tier shared by every ``graph_kv_checkpoint`` call
#: (CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring). It MUST outlive a single tool
#: invocation — "checkpoint to RAM, then decide later whether to persist" is the whole
#: point of the tier — so it is built once and reused, not per call.
_RAM_CHECKPOINT_STORE: Any | None = None


def _checkpoint_manager(graph: str) -> Any:
    """Build a :class:`~agent_utilities.kvcache.TieredCheckpointManager` over the shared
    RAM tier, binding the durable tier when the engine is reachable.

    A durable store that cannot be built is NOT an error here: the manager works
    RAM-only and every promotion is refused with that reason, which is the correct
    degrade for a checkpoint layer whose default tier is RAM anyway.
    """
    global _RAM_CHECKPOINT_STORE
    from agent_utilities.kvcache import RAMCheckpointStore, TieredCheckpointManager

    if _RAM_CHECKPOINT_STORE is None:
        _RAM_CHECKPOINT_STORE = RAMCheckpointStore()
    try:
        disk_store = _checkpoint_store(graph)
    except Exception as exc:  # noqa: BLE001 — engine down degrades to RAM-only
        logger.warning(
            "[CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring] durable checkpoint "
            "tier unavailable (%s: %s) — the manager will serve RAM only and refuse "
            "every promotion",
            type(exc).__name__,
            exc,
        )
        disk_store = None
    return TieredCheckpointManager(
        ram_store=_RAM_CHECKPOINT_STORE, disk_store=disk_store
    )


def _kv_checkpoint_intelligence(
    action: str,
    *,
    graph: str,
    data_b64: str,
    model_identity: str,
    quantization: str,
    serving_engine: str,
    engine_version: str,
    prefix_digest: str,
    tenant: str,
    policy_version: str,
    run_id: str,
    point: str,
    checkpoint_id: str,
    requesting_tenant: str,
    observation_json: str,
    initiator: str,
    persist: bool,
    operator_grant: bool,
) -> str:
    """The worthiness/tiering/eligibility half of ``graph_kv_checkpoint``.

    CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring. Split out of the tool body so
    the three trigger paths can be exercised directly, and so the tool function stays
    argument-marshalling only.
    """
    from agent_utilities.kvcache import (
        CheckpointObservation,
        KVCheckpointError,
        KVCheckpointKey,
    )

    # Validate the initiator AT THE BOUNDARY. It is a Literal on PersistenceRequest /
    # RAMCheckpointRecord, so an unrecognized value would surface deep inside as a raw
    # pydantic ValidationError that the KVCheckpointError handlers below never catch —
    # and, worse, "which initiator is this?" is the input the whole eligibility decision
    # turns on, so it must never be guessed at or coerced.
    if initiator not in {"user", "agent", "system"}:
        return _surface_error(
            ValueError(
                f"initiator must be one of user|agent|system, got {initiator!r}"
            ),
            surface="kv_checkpoint",
            action=action,
            code="invalid_request",
        )

    manager = _checkpoint_manager(graph)

    def _observation() -> Any:
        payload = json.loads(observation_json) if observation_json else {}
        if not isinstance(payload, dict):
            raise ValueError("observation_json must be a JSON object")
        return CheckpointObservation(**payload)

    if action == "ram_stats":
        return json.dumps(
            {
                "surface": "kv_checkpoint",
                "action": action,
                "result": {
                    **manager.ram_store.stats(),
                    "eligibility_gate": manager.eligibility_gate.name,
                },
            }
        )

    if action == "recommend":
        try:
            observation = _observation()
        except Exception as exc:  # noqa: BLE001 — caller-supplied JSON/shape
            return _surface_error(
                exc, surface="kv_checkpoint", action=action, code="invalid_request"
            )
        recommendation = manager.recommend(observation)
        return json.dumps(
            {
                "surface": "kv_checkpoint",
                "action": action,
                "result": {
                    **recommendation.model_dump(mode="json", exclude={"observation"}),
                    "advisory": recommendation.as_advisory(),
                },
            },
            default=_json_default,
        )

    if action == "explain":
        try:
            return json.dumps(
                {
                    "surface": "kv_checkpoint",
                    "action": action,
                    "result": manager.explain(
                        checkpoint_id, requesting_tenant=requesting_tenant
                    ),
                },
                default=_json_default,
            )
        except KVCheckpointError as exc:
            return _surface_error(
                exc,
                surface="kv_checkpoint",
                action=action,
                code=_checkpoint_error_code(exc),
            )

    if action == "promote":
        try:
            outcome = manager.promote(
                checkpoint_id,
                requesting_tenant=requesting_tenant,
                trigger=initiator,  # type: ignore[arg-type]
                operator_grant=operator_grant,
            )
        except KVCheckpointError as exc:
            return _surface_error(
                exc,
                surface="kv_checkpoint",
                action=action,
                code=_checkpoint_error_code(exc),
            )
        return json.dumps(
            {
                "surface": "kv_checkpoint",
                "action": action,
                "result": outcome.model_dump(mode="json", exclude={"recommendation"}),
            },
            default=_json_default,
        )

    # action == "checkpoint_now"
    try:
        data = base64.b64decode(data_b64) if data_b64 else b""
        key = KVCheckpointKey(
            model_identity=model_identity,
            quantization=quantization,
            serving_engine=serving_engine,
            engine_version=engine_version,
            prefix_digest=prefix_digest,
            tenant=tenant,
            policy_version=policy_version,
        )
        observation = (
            _observation() if observation_json.strip() not in {"", "{}"} else None
        )
    except Exception as exc:  # noqa: BLE001 — bad payload/key/observation
        return _surface_error(
            exc, surface="kv_checkpoint", action=action, code="invalid_request"
        )
    try:
        outcome = manager.checkpoint_now(
            data,
            key=key,
            run_id=run_id,
            point=point,
            trigger=initiator,  # type: ignore[arg-type]
            persist=persist,
            operator_grant=operator_grant,
            observation=observation,
        )
    except KVCheckpointError as exc:
        return _surface_error(
            exc,
            surface="kv_checkpoint",
            action=action,
            code=_checkpoint_error_code(exc),
        )
    return json.dumps(
        {
            "surface": "kv_checkpoint",
            "action": action,
            "result": outcome.model_dump(mode="json", exclude={"recommendation"}),
            "advisory": (
                outcome.recommendation.as_advisory() if outcome.recommendation else ""
            ),
        },
        default=_json_default,
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes | bytearray):
        return {"__bytes_b64__": base64.b64encode(bytes(obj)).decode("ascii")}
    return str(obj)


def _degraded(surface: str, action: str, tried: list[str]) -> str:
    """Clean 'this engine build lacks the surface' payload (never raise)."""
    return json.dumps(
        {
            "surface": surface,
            "action": action,
            "degraded": True,
            "error": (
                f"engine surface {surface!r} is not available in this engine build "
                "(no matching client method); this tool degrades cleanly and will "
                "activate once the engine ships the capability"
            ),
            "tried": tried,
        }
    )


def _surface_error(
    exc: BaseException,
    *,
    surface: str,
    action: str = "",
    code: str = "operation_failed",
    context: dict[str, Any] | None = None,
) -> str:
    """Return a stable protocol failure without copying UNTRUSTED request metadata.

    ``surface``/``action`` (and any ``context`` extras, e.g. ``delegated``/
    ``available``) are values the CALLER already supplied to this same
    invocation — not exception-controlled or otherwise untrusted data — so
    merging them into the redacted payload lets a caller correlate which
    surface/action failed (matching the success path's
    ``{"surface", "action", "result"}`` shape) without reintroducing anything
    :func:`~agent_utilities.security.error_surface.public_error_payload`
    exists to keep out: the raw exception text/args, which stays fully
    redacted (only ``type(exc).__name__`` is ever logged or returned).
    """

    payload = public_error_payload(exc, code=code, context=context)
    if context:
        payload.update(context)
    payload["surface"] = surface
    payload["action"] = action
    return json.dumps(payload)


def _require_process_perspective(params: dict[str, Any]) -> Any:
    """Mint the explicit, versioned ``ProcessPerspective`` a flattening needs.

    CONCEPT:AU-KG.mining.governed-perspective-flattening — classical
    single-case flattening (one trace per object) is a real information loss
    over the object-centric source truth, so ``graph_mine(action="process")``
    refuses to derive traces from a bare ``object_type`` string: a caller must
    also disclose WHICH versioned analytical selection produced them via
    ``perspective_id``/``derivation_version``. Pops the three params it
    consumes; raises ``ValueError`` (caught by the caller's existing
    invalid-request handling) when any is missing.
    """
    from agent_utilities.knowledge_graph.ingestion.semantic_event_model import (
        ProcessPerspective,
    )

    object_type = str(params.pop("object_type", "") or "").strip()
    perspective_id = str(params.pop("perspective_id", "") or "").strip()
    derivation_version = str(params.pop("derivation_version", "") or "").strip()
    if not object_type or not perspective_id or not derivation_version:
        raise ValueError(
            "classical single-case flattening requires 'object_type', "
            "'perspective_id', and 'derivation_version' — undisclosed "
            "flattening is refused; declare the versioned ProcessPerspective "
            "these traces are derived under"
        )
    return ProcessPerspective(
        perspective_id=perspective_id,
        object_types=(object_type,),
        derivation_version=derivation_version,
    )


def _resolve(client: Any, candidates: tuple[tuple[str, str], ...]) -> Any:
    """Return the first callable ``client.<sub>.<method>`` among ``candidates``.

    Returns ``None`` when none of the candidate surfaces are present so the caller
    can degrade gracefully (CONCEPT:AU-KG.coordination.engine-message-broker).
    """
    for sub_attr, meth_attr in candidates:
        sub = getattr(client, sub_attr, None)
        if sub is None:
            continue
        fn = getattr(sub, meth_attr, None)
        if callable(fn):
            return fn
    return None


def _invoke(
    *,
    surface: str,
    action: str,
    graph: str,
    candidates: tuple[tuple[str, str], ...],
    params: dict[str, Any],
) -> str:
    """Resolve the client, dispatch to the first present surface, JSON the result.

    Every failure mode is returned as data, never raised: engine unreachable →
    ``error``; surface absent → ``degraded``; bad kwargs / engine error →
    ``error`` (CONCEPT:AU-KG.coordination.engine-message-broker).
    """
    try:
        client = _client(graph)
    except Exception as exc:  # noqa: BLE001 — engine down is a normal degrade
        return _surface_error(
            exc,
            surface=surface,
            action=action,
            code="dependency_unavailable",
        )
    fn = _resolve(client, candidates)
    if fn is None:
        # ".".join(...) rather than an f-string: these are diagnostic
        # attribute-path names for a "degraded" status message (never a
        # query), but the two-part dotted shape is otherwise indistinguishable
        # from a schema-qualified table/label composition at the AST level.
        return _degraded(surface, action, [".".join((a, m)) for a, m in candidates])
    try:
        result = fn(**params)
    except TypeError as exc:
        return _surface_error(
            exc, surface=surface, action=action, code="invalid_request"
        )
    except Exception as exc:  # noqa: BLE001 — surface engine errors as data
        return _surface_error(exc, surface=surface, action=action)
    return json.dumps(
        {"surface": surface, "action": action, "result": result}, default=_json_default
    )


def _drop_empty(**kwargs: Any) -> dict[str, Any]:
    """Keep only kwargs the caller actually supplied (non-empty string / non-None)."""
    return {k: v for k, v in kwargs.items() if v not in ("", None)}


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from a sync MCP handler (loop-running or not)."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


# CONCEPT:AU-KG.memory.unified-memory-crud-core — the unified memory-CRUD core. graph_memory's recall/store/link
# actions route into the SAME ``graph_write`` tool the REST ``/graph/write/memory``
# [``/recall``] twins and the harness ``kg_memory_recall``/``kg_memory_store`` tools
# already use — one core, no fourth memory surface. ``link`` reuses graph_write's
# ``add_edge`` so relating two memories rides the same mutation path.
_MEMORY_CRUD_ACTIONS = ("recall", "store", "link")


def _memory_crud(action: str, params: dict[str, Any]) -> str:
    """Dispatch recall/store/link into the shared ``graph_write`` memory core."""
    if action == "store":
        call = dict(
            action="store_memory",
            agent_id=params.get("agent_id", params.get("agent", "")),
            node_type=params.get("memory_type", params.get("type", "")),
            properties=params.get("content", params.get("properties", "")),
            nodes=json.dumps(params.get("tags", [])),
        )
    elif action == "recall":
        call = dict(
            action="recall_memory",
            properties=params.get("query", params.get("content", "")),
            node_type=params.get("memory_type", params.get("type", "")),
        )
    else:  # link
        src = params.get("source", params.get("source_id", ""))
        tgt = params.get("target", params.get("target_id", ""))
        if not src or not tgt:
            return json.dumps(
                {
                    "surface": "memory",
                    "action": "link",
                    "error": "link requires 'source' and 'target' (memory node ids)",
                }
            )
        call = dict(
            action="add_edge",
            source_id=src,
            target_id=tgt,
            rel_type=params.get("rel_type", "RELATES_TO"),
            properties=json.dumps(params.get("properties", {})),
        )
    try:
        result = _run_coro(kg_server._execute_tool("graph_write", **call))
    except Exception as exc:  # noqa: BLE001 — surface engine/core errors as data
        return _surface_error(exc, surface="memory", action=action)
    return json.dumps(
        {"surface": "memory", "action": action, "result": result}, default=_json_default
    )


def _pick_warm_fork_sandbox(preferred: str = "") -> Any:
    """Return the cheapest available warm-fork rung, or ``None`` (CONCEPT:AU-KG.coordination.warm-fork-fanout).

    Reuses the ORCH-1.86 sandbox registry (``default_sandboxes()``, cheapest-first)
    and selects the first backend whose capabilities advertise ``warm_fork`` and
    which is available on this host. ``preferred`` pins a rung by name when set.
    """
    try:
        from agent_utilities.rlm.sandboxes.registry import default_sandboxes
    except Exception:  # noqa: BLE001 — subsystem unimportable ⇒ degrade cleanly
        return None

    forkable = [
        sb
        for sb in default_sandboxes()
        if getattr(getattr(sb, "capabilities", None), "warm_fork", False)
    ]
    if preferred:
        forkable = [sb for sb in forkable if sb.name == preferred] or forkable
    for sb in forkable:
        try:
            if sb.is_available():
                return sb
        except Exception:  # noqa: BLE001 — an unprobeable rung is simply skipped
            continue
    return None


def _fork_fanout(branches: list[Any], seed_vars: dict[str, Any], preferred: str) -> str:
    """Warm a fork parent once and fan out ``branches``, returning per-branch results.

    Backed by the ORCH-1.86..93 warm-fork primitive: the base
    :class:`ForkableSandbox.execute` warms-or-reuses one parent (copy-on-write) and
    forks a child per branch. Fan-out is concurrent ``execute`` calls sharing that
    one warm parent. Degrades cleanly to a structured ``unavailable`` payload when no
    warm-fork rung is available on this host (CONCEPT:AU-KG.coordination.warm-fork-fanout).
    """
    sb = _pick_warm_fork_sandbox(preferred)
    if sb is None:
        return json.dumps(
            {
                "surface": "fork",
                "degraded": True,
                "error": (
                    "no warm-fork rung available on this host (firecracker needs a "
                    "reachable governed forkd controller + KVM), and the "
                    "epistemic-graph engine client exposes no warm-fork primitive"
                ),
                "followup": (
                    "spike: surface a first-class engine warm-fork/KV-cache-fork op "
                    "on the epistemic_graph client (LMCacheMPConnector snapshot → "
                    "branch), then route graph_fork to that governed primitive"
                ),
                "branch_count": len(branches),
            }
        )

    from agent_utilities.rlm.sandboxes.base import SandboxEnv

    async def _run_all() -> list[dict[str, Any]]:
        import asyncio

        async def _one(idx: int, snippet: Any) -> dict[str, Any]:
            try:
                result = await sb.execute(
                    str(snippet), SandboxEnv(vars=dict(seed_vars))
                )
                return {
                    "index": idx,
                    "ok": result.error is None,
                    "stdout": result.stdout,
                    "error": result.error,
                    "vars": result.updated_vars,
                }
            except Exception as exc:  # noqa: BLE001 — one branch never fails the set
                return {
                    "index": idx,
                    "ok": False,
                    **public_error_payload(exc),
                }

        return await asyncio.gather(*(_one(i, s) for i, s in enumerate(branches)))

    try:
        results = _run_coro(_run_all())
    except Exception as exc:  # noqa: BLE001 — infra death → structured error, no crash
        return _surface_error(exc, surface="fork")
    return json.dumps(
        {
            "surface": "fork",
            "sandbox": sb.name,
            "branch_count": len(results),
            "branches": results,
        },
        default=_json_default,
    )


def _crossmodal_fork_fanout(
    branches: list[Any],
    seed_vars: dict[str, Any],
    preferred: str,
    context_query: str,
    candidate_var: str,
) -> str:
    """Retrieve an engine cross-modal candidate set ONCE, then warm-fork ``branches`` over it.

    The agent-utilities side of the epistemic-graph cross-modal seam
    (CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout): a vector+graph+text candidate set is
    retrieved once for ``context_query`` and forked into every branch as ``candidate_var`` —
    branches reuse that one context with no recompute. Falls back to the structured degraded
    payload when no warm-fork rung is available or the engine retriever is unreachable.
    """
    try:
        from agent_utilities.runtime.crossmodal_fork import CrossModalForkFanout
    except Exception as exc:  # noqa: BLE001 — capability unimportable ⇒ degrade cleanly
        return _surface_error(exc, surface="fork", code="dependency_unavailable")

    fanout = CrossModalForkFanout()

    async def _run() -> Any:
        return await fanout.fan_out(
            context_query,
            [str(b) for b in branches],
            preferred=preferred,
            candidate_var=candidate_var,
            extra_vars=seed_vars or None,
        )

    try:
        res = _run_coro(_run())
    except Exception as exc:  # noqa: BLE001 — engine/infra death → structured error, no crash
        return _surface_error(exc, surface="fork")
    return json.dumps(
        {
            "surface": "fork",
            "context_query": context_query,
            "candidate_var": candidate_var,
            "candidate_count": res.candidate_count,
            "retrieval_calls": res.retrieval_calls,
            "reused_without_recompute": res.reused_without_recompute,
            "degraded": res.degraded,
            "error": res.error,
            "sandbox": res.sandbox,
            "branch_count": len(res.branches),
            "branches": [
                {
                    "index": b.index,
                    "ok": b.ok,
                    "stdout": b.stdout,
                    "error": b.error,
                    "output": b.output,
                }
                for b in res.branches
            ],
        },
        default=_json_default,
    )


# ══════════════════════════════════════════════════════════════════
# graph_mine_deep — Phase-6 heavy-dep delegation to data-science-mcp
# ══════════════════════════════════════════════════════════════════
# CONCEPT:AU-KG.mining.dsm-forecast-delegation — the data-mining plan's Phase 6: the engine
# core stays pure-Rust (never torch/GPU); the deep-learning / heavy-Python family
# (LSTM/RNN sequence forecasting, MLP/deep classifiers, autoencoders, an XGBoost-
# family boosting classifier) is ORCHESTRATED over MCP to ``agents/data-science-mcp``
# — this module ships features out, that service trains/infers, and the result
# folds back into the KG as typed nodes (CONCEPT:AU-KG.mining.foldback-typed-nodes). Word/text
# embeddings are deliberately NOT re-homed here — they already live on the remote
# vLLM embedder (``core/embedding_utilities``); the 'embed' action below embeds
# arbitrary NUMERIC feature rows via a small neural autoencoder, a distinct
# capability from text embedding.
_DSM_SERVER_NAME = "data-science-mcp"
_DSM_TOOL_NAME = "deep_train_predict"

#: One delegated ``graph_mine_deep`` action → the data-science-mcp
#: ``deep_train_predict`` algo it maps to.
_DEEP_ALGO_BY_ACTION: dict[str, str] = {
    "deep_forecast": "lstm_forecast",
    "deep_classify": "mlp_classify",
    "autoencoder_anomaly": "autoencoder_anomaly",
    "xgboost": "histgbm_classify",
    "embed": "autoencoder_embed",
}

#: KG node type a delegated action's result materializes as when writeback=true.
_DEEP_NODE_TYPE: dict[str, str] = {
    "deep_forecast": "Forecast",
    "deep_classify": "Classification",
    "autoencoder_anomaly": "Anomaly",
    "xgboost": "Classification",
    "embed": "Embedding",
}


def _gather_kg_feature_rows(
    source: dict[str, Any], graph: str
) -> tuple[list[str], list[list[float]]]:
    """Gather a feature-row RowSet from the KG for a ``{node_label, fields, limit}`` source spec.

    Runs one read-only Cypher projection through the existing ``graph_query`` tool
    (compute-near-data — no bespoke second engine client) and returns
    ``(node_ids, rows)`` so a caller can ship ``rows`` to data-science-mcp and fold
    the result back onto the SAME ``node_ids`` (CONCEPT:AU-KG.mining.dsm-forecast-delegation).
    """
    node_label = source.get("node_label")
    if not isinstance(node_label, str) or not CYPHER_IDENTIFIER_RE.fullmatch(
        node_label
    ):
        raise ValueError("source.node_label is required")
    fields = source.get("fields") or []
    if (
        not isinstance(fields, list)
        or not fields
        or len(fields) > 64
        or any(
            not isinstance(field, str) or not CYPHER_IDENTIFIER_RE.fullmatch(field)
            for field in fields
        )
    ):
        raise ValueError("source.fields (a list of property names) is required")
    raw_limit = source.get("limit", 200)
    if isinstance(raw_limit, bool):
        raise ValueError("source.limit must be between 1 and 10000")
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("source.limit must be between 1 and 10000") from exc
    if not 1 <= limit <= 10_000:
        raise ValueError("source.limit must be between 1 and 10000")
    projections = ", ".join(f"n.{f} AS f{i}" for i, f in enumerate(fields))
    cypher = f"MATCH (n:{node_label}) RETURN n.id AS id, {projections} LIMIT {limit}"
    raw = _run_coro(
        kg_server._execute_tool(
            "graph_query", cypher=cypher, params="{}", scope="local", target=graph or ""
        )
    )
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError("graph feature query failed")
    if not isinstance(payload, list):
        raise RuntimeError(
            f"unexpected graph_query result shape: {type(payload).__name__}"
        )
    node_ids = [str(row.get("id")) for row in payload]
    rows = [
        [float(row.get(f"f{i}") or 0.0) for i in range(len(fields))] for row in payload
    ]
    return node_ids, rows


def _deep_write_node(node_type: str, properties: dict[str, Any], graph: str) -> str:
    """Materialize one delegated-deep-mining result row as a typed KG node
    (CONCEPT:AU-KG.mining.foldback-typed-nodes). Best-effort — never raises, since a foldback
    failure must not fail the already-completed delegated call. Returns the new node id."""
    node_id = f"{node_type.lower()}_dsm_{uuid.uuid4().hex}"
    try:
        _run_coro(
            kg_server._execute_tool(
                "graph_write",
                action="add_node",
                node_id=node_id,
                node_type=node_type,
                properties=json.dumps(properties, default=_json_default),
                target=graph or "",
            )
        )
    except Exception:  # noqa: BLE001 — writeback is best-effort
        pass
    return node_id


def _deep_write_edge(source_id: str, target_id: str, rel_type: str, graph: str) -> None:
    """Link a delegated-deep-mining node back to the KG node it was derived from
    (best-effort — see :func:`_deep_write_node`)."""
    try:
        _run_coro(
            kg_server._execute_tool(
                "graph_write",
                action="add_edge",
                source_id=source_id,
                target_id=target_id,
                rel_type=rel_type,
                target=graph or "",
            )
        )
    except Exception:  # noqa: BLE001 — writeback is best-effort
        pass


def _prepare_deep_delegation(
    action: str, params: dict[str, Any], graph: str
) -> tuple[dict[str, Any], list[str]]:
    """Marshal ``params`` into the ``deep_train_predict`` kwargs for ``action``.

    Gathers a KG RowSet via ``source`` when raw ``x``/``values`` are not given
    directly. Returns ``(tool_params, node_ids)`` — ``node_ids`` is only populated
    when a ``source`` was used (so the caller can fold results back onto them).
    """
    tool_params: dict[str, Any] = {"algo": _DEEP_ALGO_BY_ACTION[action]}
    node_ids: list[str] = []
    source = params.pop("source", None)

    if action == "deep_forecast":
        values = params.pop("values", None)
        if values is None and source:
            node_ids, rows = _gather_kg_feature_rows(source, graph)
            values = [row[0] for row in rows]
        if not values:
            raise ValueError("provide 'values' (a 1-D series) or a 'source'")
        tool_params["values_json"] = json.dumps(values)
    elif action in ("deep_classify", "xgboost"):
        x = params.pop("x", None)
        y = params.pop("y", None)
        if x is None and source:
            node_ids, x = _gather_kg_feature_rows(source, graph)
        if x is None or y is None:
            raise ValueError("provide 'x' + 'y', or a 'source' + 'y'")
        tool_params["x_json"] = json.dumps(x)
        tool_params["y_json"] = json.dumps(y)
        x_predict = params.pop("x_predict", None)
        if x_predict is not None:
            tool_params["x_predict_json"] = json.dumps(x_predict)
    else:  # autoencoder_anomaly, embed
        x = params.pop("x", None)
        if x is None and source:
            node_ids, x = _gather_kg_feature_rows(source, graph)
        if x is None:
            raise ValueError("provide 'x' or a 'source'")
        tool_params["x_json"] = json.dumps(x)

    tool_params["params_json"] = json.dumps(params)
    return tool_params, node_ids


def register_engine_surface_tools(mcp) -> None:
    """Register the KG-2.310 engine-surface tools + their REST twins.

    Each tool is added to ``REGISTERED_TOOLS`` and mapped to a ``/graph/<name>``
    route in ``ACTION_TOOL_ROUTES`` (auto-mounted by the generic REST factory), so
    MCP and REST stay in lockstep. CONCEPT:AU-KG.coordination.engine-message-broker.
    """

    # ══════════════════════════════════════════════════════════════════
    # graph_broker — engine message broker (exchanges / queues / streams)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_broker",
        description=(
            "CONCEPT:AU-KG.coordination.engine-message-broker — the epistemic-graph engine message broker "
            "(AMQP-style exchanges + queues + streams), distinct from the "
            "agent-to-agent 'graph_bus'. Action-routed 1:1 over the engine broker "
            "surface: set 'action' to the broker method — e.g. 'declare_exchange' "
            "(exchange [+exchange_type]), 'declare_queue' (queue), 'bind' (queue + "
            "exchange [+routing_key]), 'publish' (exchange + routing_key + payload), "
            "'consume' (queue [+max_messages,+ack]), 'stats' / 'list_queues' / "
            "'list_exchanges'. Extra kwargs go via params_json. Degrades cleanly when "
            "the engine build has no broker surface."
        ),
        tags=["graph-os", "engine", "broker", "messaging"],
    )
    def graph_broker(
        action: str = Field(
            default="stats",
            description="Broker method: declare_exchange | declare_queue | bind | "
            "publish | consume | stats | list_queues | list_exchanges | ...",
        ),
        exchange: str = Field(default="", description="Exchange name."),
        queue: str = Field(default="", description="Queue name."),
        routing_key: str = Field(default="", description="Routing/binding key."),
        payload: str = Field(default="", description="Message body (publish)."),
        exchange_type: str = Field(
            default="", description="Exchange type: direct | fanout | topic (declare)."
        ),
        params_json: str = Field(
            default="{}",
            description='JSON object of extra kwargs, e.g. {"max_messages":10,'
            '"ack":true,"durable":true}. Merged over the typed fields.',
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine broker surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        if not action:
            return json.dumps({"surface": "broker", "error": "action is required"})
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="broker", code="invalid_request")
        if not isinstance(extra, dict):
            return json.dumps(
                {"surface": "broker", "error": "params_json must decode to an object"}
            )
        params = _drop_empty(
            exchange=exchange,
            queue=queue,
            routing_key=routing_key,
            payload=payload,
            exchange_type=exchange_type,
        )
        params.update(extra)
        return _invoke(
            surface="broker",
            action=action,
            graph=graph,
            candidates=(("broker", action),),
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_broker"] = graph_broker
    kg_server.ACTION_TOOL_ROUTES["graph_broker"] = "/graph/broker"

    # ══════════════════════════════════════════════════════════════════
    # graph_kvcache — shared content-addressed KV-cache (EG-187 / KG-2.306)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_kvcache",
        description=(
            "CONCEPT:AU-KG.coordination.engine-message-broker — the engine's shared, content-addressed KV-cache over "
            "the EG-187 HTTP surface, driven through the KG-2.306 "
            "EpistemicGraphKVBackend connector. Actions: 'get' (key → base64 block "
            "bytes or miss), 'put' (key + value_b64 → stored bool), 'contains'/'exists' "
            "(key → bool), 'stats' (occupancy + dedup counters). The connector already "
            "degrades every transport error to a cache miss, so this tool never raises."
        ),
        tags=["graph-os", "engine", "kvcache"],
    )
    def graph_kvcache(
        action: str = Field(
            default="stats", description="get | put | contains | exists | stats"
        ),
        key: str = Field(
            default="", description="Opaque block key (get/put/contains/exists)."
        ),
        value_b64: str = Field(
            default="", description="Base64-encoded block bytes to store (put)."
        ),
    ) -> str:
        """Thin wrapper over the KG-2.306 KV-cache connector (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        try:
            backend = _kv_backend()
        except Exception as exc:  # noqa: BLE001 — mis-config degrades, never raises
            return _surface_error(
                exc,
                surface="kvcache",
                action=action,
                code="dependency_unavailable",
            )
        try:
            if action == "get":
                if not key:
                    return json.dumps({"surface": "kvcache", "error": "key required"})
                blob = backend.get(key)
                return json.dumps(
                    {
                        "surface": "kvcache",
                        "action": action,
                        "hit": blob is not None,
                        "value_b64": (
                            base64.b64encode(blob).decode("ascii")
                            if blob is not None
                            else None
                        ),
                    }
                )
            if action == "put":
                if not key:
                    return json.dumps({"surface": "kvcache", "error": "key required"})
                try:
                    raw = base64.b64decode(value_b64) if value_b64 else b""
                except (ValueError, TypeError) as exc:
                    return _surface_error(
                        exc, surface="kvcache", code="invalid_request"
                    )
                return json.dumps(
                    {
                        "surface": "kvcache",
                        "action": action,
                        "stored": bool(backend.put(key, raw)),
                    }
                )
            if action in ("contains", "exists"):
                if not key:
                    return json.dumps({"surface": "kvcache", "error": "key required"})
                probe = backend.exists if action == "exists" else backend.contains
                return json.dumps(
                    {
                        "surface": "kvcache",
                        "action": action,
                        "present": bool(probe(key)),
                    }
                )
            if action == "stats":
                stats = backend.stats()
                data = (
                    stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)
                )
                return json.dumps(
                    {"surface": "kvcache", "action": action, "result": data},
                    default=_json_default,
                )
            return json.dumps(
                {"surface": "kvcache", "error": f"unknown action {action!r}"}
            )
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass

    kg_server.REGISTERED_TOOLS["graph_kvcache"] = graph_kvcache

    # ══════════════════════════════════════════════════════════════════
    # graph_kv_checkpoint — KV-cache checkpoints as graph resources
    # (CONCEPT:AU-KG.memory.kv-checkpoint-resource)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_kv_checkpoint",
        description=(
            "CONCEPT:AU-KG.memory.kv-checkpoint-resource — checkpoint a KV-cache at a point of "
            "ideal understanding, then initialise a new agent from it or reset a "
            "conversation back to it. The heavy KV blob lives behind the engine's "
            "content-addressed blob store; the response never inlines it — only "
            "provenance (checkpoint_id/digest/blob_id) so a caller with real access to "
            "the blob store can fetch it directly. Actions: 'create' (data_b64 + the "
            "full key: model_identity/quantization/serving_engine/engine_version/"
            "prefix_digest/tenant/policy_version, plus run_id/point) → checkpoint "
            "provenance; 'instantiate_agent' (checkpoint_id + requesting_tenant + "
            "new_run_id) → fail-closed load + an initializedFrom lineage edge from a "
            "fresh AgentRun node; 'restore_conversation' (checkpoint_id + "
            "conversation_id + requesting_tenant) → fail-closed load + a restoredFrom "
            "lineage edge, or set allow_cold_start=true for an EXPLICIT, traced "
            "cold-start fallback instead of raising. A tenant mismatch or a stale "
            "policy_version is ALWAYS refused (cross-tenant checkpoint reuse is a "
            "security boundary, never bypassable). "
            "CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring adds the intelligence "
            "actions: 'recommend' (observation_json → a scored checkpoint-worthiness "
            "verdict with drivers, blockers and the scorers that had no evidence — "
            "advisory only, it takes no checkpoint); 'checkpoint_now' (data_b64 + the "
            "key + initiator; stores to the RAM tier immediately, and with "
            "persist=true ALSO attempts a durable write); 'promote' (checkpoint_id → "
            "RAM→disk promotion); 'explain' (checkpoint_id → why this checkpoint "
            "exists and why it is where it is); 'ram_stats'. Durable persistence "
            "ALWAYS passes the CONCEPT:AU-OS.governance.checkpoint-persistence-eligibility "
            "gate, whose default DENIES unless initiator='user' AND operator_grant=true "
            "— an agent may recommend durable persistence but cannot authorize it, and "
            "a checkpoint already living in RAM is NOT consent to write it to disk."
        ),
        tags=["graph-os", "engine", "kvcache", "checkpoint", "memory"],
    )
    def graph_kv_checkpoint(
        action: str = Field(
            default="", description="create | instantiate_agent | restore_conversation"
        ),
        data_b64: str = Field(
            default="", description="Base64 KV-cache blob bytes (create only)."
        ),
        model_identity: str = Field(default="", description="create: key component."),
        quantization: str = Field(default="", description="create: key component."),
        serving_engine: str = Field(default="", description="create: key component."),
        engine_version: str = Field(default="", description="create: key component."),
        prefix_digest: str = Field(default="", description="create: key component."),
        tenant: str = Field(
            default="", description="create: owning tenant (mandatory key component)."
        ),
        policy_version: str = Field(
            default="", description="create: key component (defaults to '')."
        ),
        run_id: str = Field(default="", description="create: the source run's id."),
        point: str = Field(
            default="", description="create: label for the checkpointed point."
        ),
        provenance_json: str = Field(
            default="{}", description="create: free-form provenance JSON object."
        ),
        checkpoint_id: str = Field(
            default="",
            description="instantiate_agent/restore_conversation: the checkpoint to load.",
        ),
        requesting_tenant: str = Field(
            default="",
            description="instantiate_agent/restore_conversation: the caller's tenant "
            "(checked against the checkpoint's stored tenant, fail-closed).",
        ),
        current_policy_version: str = Field(
            default="",
            description="instantiate_agent/restore_conversation: current policy "
            "version to validate against (empty ⇒ skip the staleness check).",
        ),
        new_run_id: str = Field(
            default="", description="instantiate_agent: the new run's id."
        ),
        conversation_id: str = Field(
            default="", description="restore_conversation: the conversation to reset."
        ),
        allow_cold_start: bool = Field(
            default=False,
            description="restore_conversation: on an invalid/foreign/stale "
            "checkpoint, return an explicit traced cold-start result instead of "
            "raising.",
        ),
        observation_json: str = Field(
            default="{}",
            description="recommend/checkpoint_now: a CheckpointObservation as JSON "
            "(rebuild cost, sibling/queued tasks, retrieved/novel items, claim/"
            "evidence counts, contradictions, churn, phase, model_self_report). Every "
            "field is optional; an omitted field means NOT MEASURED and its scorer "
            "abstains rather than guessing.",
        ),
        initiator: str = Field(
            default="agent",
            description="checkpoint_now/promote: who is asking — 'user' (a human "
            "operator), 'agent' (the model decided), or 'system'. Only 'user' with "
            "operator_grant=true can authorize durable persistence under the default "
            "eligibility gate.",
        ),
        persist: bool = Field(
            default=False,
            description="checkpoint_now: also attempt durable (cross-session) "
            "persistence. Always subject to the eligibility gate; a refusal still "
            "leaves the RAM checkpoint in place.",
        ),
        operator_grant: bool = Field(
            default=False,
            description="checkpoint_now/promote: a human operator explicitly "
            "authorized THIS durable persistence. Never inferred, and never satisfied "
            "by the checkpoint already existing in RAM.",
        ),
        graph: str = Field(
            default="", description="Target graph (default engine graph)."
        ),
    ) -> str:
        """Thin verb over :class:`~agent_utilities.kvcache.KVCheckpointStore` (CONCEPT:AU-KG.memory.kv-checkpoint-resource)
        and :class:`~agent_utilities.kvcache.TieredCheckpointManager`
        (CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring)."""
        from agent_utilities.kvcache import KVCheckpointError, KVCheckpointKey

        # ── the intelligence actions: worthiness, tiering, eligibility ──────
        # These route through TieredCheckpointManager (shared RAM tier) rather than
        # the durable store directly, because the RAM tier is the DEFAULT and disk is
        # a gated promotion off it.
        if action in {
            "recommend",
            "checkpoint_now",
            "promote",
            "explain",
            "ram_stats",
        }:
            return _kv_checkpoint_intelligence(
                action,
                graph=graph,
                data_b64=data_b64,
                model_identity=model_identity,
                quantization=quantization,
                serving_engine=serving_engine,
                engine_version=engine_version,
                prefix_digest=prefix_digest,
                tenant=tenant,
                policy_version=policy_version,
                run_id=run_id,
                point=point,
                checkpoint_id=checkpoint_id,
                requesting_tenant=requesting_tenant,
                observation_json=observation_json,
                initiator=initiator,
                persist=persist,
                operator_grant=operator_grant,
            )

        try:
            store = _checkpoint_store(graph)
        except Exception as exc:  # noqa: BLE001 — engine down is a normal degrade
            return _surface_error(
                exc,
                surface="kv_checkpoint",
                action=action,
                code="dependency_unavailable",
            )

        if action == "create":
            try:
                data = base64.b64decode(data_b64) if data_b64 else b""
                provenance = json.loads(provenance_json) if provenance_json else {}
            except (ValueError, TypeError) as exc:
                return _surface_error(
                    exc, surface="kv_checkpoint", action=action, code="invalid_request"
                )
            try:
                key = KVCheckpointKey(
                    model_identity=model_identity,
                    quantization=quantization,
                    serving_engine=serving_engine,
                    engine_version=engine_version,
                    prefix_digest=prefix_digest,
                    tenant=tenant,
                    policy_version=policy_version,
                )
            except Exception as exc:  # noqa: BLE001 — bad/missing key component
                return _surface_error(
                    exc, surface="kv_checkpoint", action=action, code="invalid_request"
                )
            try:
                record = store.create_checkpoint(
                    data,
                    key=key,
                    run_id=run_id,
                    point=point,
                    provenance=provenance if isinstance(provenance, dict) else {},
                )
            except Exception as exc:  # noqa: BLE001 — surface engine errors as data
                return _surface_error(exc, surface="kv_checkpoint", action=action)
            if record is None:
                return json.dumps(
                    {
                        "surface": "kv_checkpoint",
                        "action": action,
                        "error": "checkpoint creation failed (empty payload or engine write failure)",
                    }
                )
            return json.dumps(
                {
                    "surface": "kv_checkpoint",
                    "action": action,
                    "result": record.model_dump(),
                },
                default=_json_default,
            )

        if action == "instantiate_agent":
            try:
                record = store.instantiate_agent(
                    checkpoint_id,
                    requesting_tenant=requesting_tenant,
                    new_run_id=new_run_id,
                    current_policy_version=current_policy_version or None,
                )
            except KVCheckpointError as exc:
                return _surface_error(
                    exc,
                    surface="kv_checkpoint",
                    action=action,
                    code=_checkpoint_error_code(exc),
                )
            return json.dumps(
                {
                    "surface": "kv_checkpoint",
                    "action": action,
                    "result": record.model_dump(),
                },
                default=_json_default,
            )

        if action == "restore_conversation":
            try:
                res = store.restore_conversation(
                    checkpoint_id,
                    conversation_id=conversation_id,
                    requesting_tenant=requesting_tenant,
                    current_policy_version=current_policy_version or None,
                    allow_cold_start=allow_cold_start,
                )
            except KVCheckpointError as exc:
                return _surface_error(
                    exc,
                    surface="kv_checkpoint",
                    action=action,
                    code=_checkpoint_error_code(exc),
                )
            # Never inline the heavy blob bytes over the JSON tool surface — only
            # provenance; a caller that needs the bytes fetches them directly from
            # the engine's own blob store by digest (CONCEPT:AU-KG.memory.kv-checkpoint-resource).
            payload = res.model_dump(exclude={"data"})
            payload["size_bytes"] = len(res.data) if res.data is not None else 0
            return json.dumps(
                {"surface": "kv_checkpoint", "action": action, "result": payload},
                default=_json_default,
            )

        return json.dumps(
            {"surface": "kv_checkpoint", "error": f"unknown action {action!r}"}
        )

    kg_server.REGISTERED_TOOLS["graph_kv_checkpoint"] = graph_kv_checkpoint
    kg_server.ACTION_TOOL_ROUTES["graph_kv_checkpoint"] = "/graph/kv_checkpoint"
    kg_server.ACTION_TOOL_ROUTES["graph_kvcache"] = "/graph/kvcache"

    # ══════════════════════════════════════════════════════════════════
    # graph_federated_search — search across registered external graphs
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_federated_search",
        description=(
            "CONCEPT:AU-KG.coordination.engine-message-broker — federated search fanned across registered external "
            "graph references. Provide a natural-language / keyword 'query'; optionally "
            "scope to specific 'references' (comma-separated ExternalGraphReference ids) "
            "and cap with 'top_k'. Extra engine kwargs via params_json. Degrades cleanly "
            "when the engine build has no federated-search surface."
        ),
        tags=["graph-os", "engine", "search", "federated"],
    )
    def graph_federated_search(
        query: str = Field(description="Search query (natural language or keywords)."),
        references: str = Field(
            default="",
            description="Comma-separated external graph reference ids (empty ⇒ all).",
        ),
        top_k: int = Field(default=10, description="Max results to return."),
        params_json: str = Field(
            default="{}", description="JSON object of extra engine kwargs."
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine federated-search surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(
                exc, surface="federated_search", code="invalid_request"
            )
        if not isinstance(extra, dict):
            return json.dumps(
                {
                    "surface": "federated_search",
                    "error": "params_json must decode to an object",
                }
            )
        refs = [r.strip() for r in references.split(",") if r.strip()]
        params: dict[str, Any] = {"query": query, "top_k": int(top_k)}
        if refs:
            params["references"] = refs
        params.update(extra)
        return _invoke(
            surface="federated_search",
            action="search",
            graph=graph,
            candidates=_FEDERATED_SEARCH_CANDIDATES,
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_federated_search"] = graph_federated_search
    kg_server.ACTION_TOOL_ROUTES["graph_federated_search"] = "/graph/federated-search"

    # ══════════════════════════════════════════════════════════════════
    # graph_promql — observability: PromQL metric queries
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_promql",
        description=(
            "CONCEPT:AU-KG.coordination.engine-message-broker — query the engine's observability metrics with PromQL. "
            "action='instant' (a single evaluation at 'time', default now) or 'range' "
            "(over start..end at 'step'). Extra engine kwargs via params_json. Degrades "
            "cleanly when the engine build has no metrics/PromQL surface."
        ),
        tags=["graph-os", "engine", "observability", "metrics"],
    )
    def graph_promql(
        query: str = Field(description="A PromQL expression."),
        action: str = Field(default="instant", description="instant | range"),
        time: str = Field(
            default="", description="Evaluation time (instant), RFC3339/unix."
        ),
        start: str = Field(default="", description="Range start (range)."),
        end: str = Field(default="", description="Range end (range)."),
        step: str = Field(default="", description="Range step, e.g. '30s' (range)."),
        params_json: str = Field(
            default="{}", description="JSON object of extra engine kwargs."
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine PromQL surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="promql", code="invalid_request")
        if not isinstance(extra, dict):
            return json.dumps(
                {"surface": "promql", "error": "params_json must decode to an object"}
            )
        if action == "range":
            params = _drop_empty(query=query, start=start, end=end, step=step)
            candidates = _PROMQL_RANGE_CANDIDATES
        elif action == "instant":
            params = _drop_empty(query=query, time=time)
            candidates = _PROMQL_INSTANT_CANDIDATES
        else:
            return json.dumps(
                {"surface": "promql", "error": f"unknown action {action!r}"}
            )
        params.update(extra)
        return _invoke(
            surface="promql",
            action=action,
            graph=graph,
            candidates=candidates,
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_promql"] = graph_promql
    kg_server.ACTION_TOOL_ROUTES["graph_promql"] = "/graph/promql"

    # ══════════════════════════════════════════════════════════════════
    # graph_traces — observability: distributed-trace search / fetch
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_traces",
        description=(
            "CONCEPT:AU-KG.coordination.engine-message-broker — search or fetch distributed traces from the engine's "
            "observability surface. action='search' (filter by 'service'/'operation'/"
            "free-form 'query', capped by 'limit') or 'get' (a single 'trace_id'). Extra "
            "engine kwargs via params_json. Degrades cleanly when the engine build has "
            "no trace surface."
        ),
        tags=["graph-os", "engine", "observability", "traces"],
    )
    def graph_traces(
        action: str = Field(default="search", description="search | get"),
        trace_id: str = Field(default="", description="Trace id (action='get')."),
        service: str = Field(default="", description="Service name filter (search)."),
        operation: str = Field(
            default="", description="Operation/span name filter (search)."
        ),
        query: str = Field(
            default="", description="Free-form filter expression (search)."
        ),
        limit: int = Field(default=20, description="Max traces to return (search)."),
        params_json: str = Field(
            default="{}", description="JSON object of extra engine kwargs."
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine trace surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        try:
            extra = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="traces", code="invalid_request")
        if not isinstance(extra, dict):
            return json.dumps(
                {"surface": "traces", "error": "params_json must decode to an object"}
            )
        if action == "get":
            if not trace_id:
                return json.dumps({"surface": "traces", "error": "trace_id required"})
            params: dict[str, Any] = {"trace_id": trace_id}
            candidates = _TRACES_GET_CANDIDATES
        elif action == "search":
            params = _drop_empty(service=service, operation=operation, query=query)
            params["limit"] = int(limit)
            candidates = _TRACES_SEARCH_CANDIDATES
        else:
            return json.dumps(
                {"surface": "traces", "error": f"unknown action {action!r}"}
            )
        params.update(extra)
        return _invoke(
            surface="traces",
            action=action,
            graph=graph,
            candidates=candidates,
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_traces"] = graph_traces
    kg_server.ACTION_TOOL_ROUTES["graph_traces"] = "/graph/traces"

    # ══════════════════════════════════════════════════════════════════
    # graph_gis — geospatial route / tile / geo-task ops
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_gis",
        description=(
            "CONCEPT:AU-KG.coordination.engine-message-broker — the engine's GIS surface. Action-routed 1:1 over the "
            "engine geo methods: e.g. 'route' (from + to [+profile]), 'tile' (z/x/y), "
            "'nearest' (lat + lon [+limit]), 'geo_task' (a named geospatial job). All "
            'structured args go via params_json (e.g. {"from":[lat,lon],"to":[lat,'
            "lon]}). Degrades cleanly when the engine build has no GIS surface."
        ),
        tags=["graph-os", "engine", "gis", "geospatial"],
    )
    def graph_gis(
        action: str = Field(
            default="route",
            description="GIS method: route | tile | nearest | geo_task | ...",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of kwargs for the GIS method (coordinates, "
            "profile, tile z/x/y, task name, ...).",
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine GIS surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        if not action:
            return json.dumps({"surface": "gis", "error": "action is required"})
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="gis", code="invalid_request")
        if not isinstance(params, dict):
            return json.dumps(
                {"surface": "gis", "error": "params_json must decode to an object"}
            )
        return _invoke(
            surface="gis",
            action=action,
            graph=graph,
            candidates=(("gis", action), ("geo", action)),
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_gis"] = graph_gis
    kg_server.ACTION_TOOL_ROUTES["graph_gis"] = "/graph/gis"

    # ══════════════════════════════════════════════════════════════════
    # graph_memory — EG-318 memory / scene / trajectory engine ops
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_memory",
        description=(
            "CONCEPT:AU-KG.coordination.engine-message-broker — the engine's EG-318 memory surface: episodic→semantic "
            "memory, the spatial scene graph, and RL trajectories. Action-routed 1:1 "
            "over the engine memory methods (dashes normalize to underscores): "
            "'create_summary' (episodic nodes → a summary node — node_ids [+window]), "
            "'consolidate' (roll episodic into semantic memory), 'maintain' (decay / "
            "prune / re-index the memory store), 'add_scene_object' (object_id [+pose/"
            "transform/parent]), 'world_transform' (object_id + transform → world pose), "
            "'start_trajectory' (agent/episode → trajectory_id), 'append_step' "
            "(trajectory_id + step {state,action,reward,...}), 'discounted_return' "
            "(trajectory_id [+gamma]). Read ops (e.g. 'get_summary', 'get_scene', "
            "'get_trajectory') route by action name too. UNIFIED memory-CRUD "
            "(CONCEPT:AU-KG.memory.unified-memory-crud-core) — 'store' (agent_id + content [+memory_type,+tags]), "
            "'recall' (query [+memory_type]), 'link' (source + target [+rel_type]) — "
            "route into the SAME graph_write memory core as the REST "
            "/graph/write/memory[/recall] twins and the harness kg_memory_recall/store "
            "tools (one core, no separate surface). Structured args go via params_json. "
            "Degrades cleanly when the engine build has no memory surface."
        ),
        tags=["graph-os", "engine", "memory", "scene", "trajectory"],
    )
    def graph_memory(
        action: str = Field(
            default="consolidate",
            description="Memory method: create_summary | consolidate | maintain | "
            "add_scene_object | world_transform | start_trajectory | append_step | "
            "discounted_return | get_* | ...",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of kwargs for the memory method, e.g. "
            '{"node_ids":["n1","n2"]}, {"object_id":"o1","transform":[...]}, '
            '{"trajectory_id":"t1","step":{"state":...,"action":...,"reward":1.0}}, '
            'or {"trajectory_id":"t1","gamma":0.99}.',
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin wrapper over the engine EG-318 memory surface (CONCEPT:AU-KG.coordination.engine-message-broker)."""
        action = (action or "").strip().replace("-", "_")
        if not action:
            return json.dumps({"surface": "memory", "error": "action is required"})
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="memory", code="invalid_request")
        if not isinstance(params, dict):
            return json.dumps(
                {"surface": "memory", "error": "params_json must decode to an object"}
            )
        # CONCEPT:AU-KG.memory.unified-memory-crud-core — unified memory-CRUD short-circuit: recall/store/link go
        # to the shared graph_write memory core (same as REST + harness), not the
        # engine EG-318 surface.
        if action in _MEMORY_CRUD_ACTIONS:
            return _memory_crud(action, params)
        candidates = _MEMORY_ACTION_CANDIDATES.get(
            action,
            (("memory", action), ("scene", action), ("trajectory", action)),
        )
        return _invoke(
            surface="memory",
            action=action,
            graph=graph,
            candidates=candidates,
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_memory"] = graph_memory
    kg_server.ACTION_TOOL_ROUTES["graph_memory"] = "/graph/memory"

    # ══════════════════════════════════════════════════════════════════
    # graph_mine — data-mining surface (CONCEPT:EG-KG.mining.frequent-itemset-mining)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_mine",
        description=(
            "CONCEPT:EG-KG.mining.frequent-itemset-mining — the unified data-mining surface "
            "over the engine, compute-near-data (mining runs where the graph lives). "
            "Actions (18): 'associate' (association rules), 'cluster' (clustering), "
            "'anomaly' (outlier detection), 'classify_fit'/'classify_predict' "
            "(classification), 'reduce' (dimensionality reduction), 'sequence' "
            "(sequential-pattern mining), 'forecast' (classical time-series forecasting), "
            "'text' (TF-IDF / topic modeling), 'subgraph' (frequent-subgraph mining / "
            "motif counting — mines the RESIDENT GRAPH's own topology, no input rows), "
            "'entity_resolve' (record linkage / entity resolution; alias "
            "'entity_resolution'), 'causal_impact' (interrupted-time-series / "
            "difference-in-differences causal effect estimation), 'process' (process "
            "mining — directly-follows graph + alpha-algorithm footprint; alias "
            "'process_mining'), 'root_cause' (root-cause propagation over a dependency "
            "graph), 'risk_propagation' (seeded personalized-PageRank risk propagation), "
            "'ontology_gap' (ontology completeness-gap detection), 'retrieval_quality' "
            "(precision@k / recall@k / MRR over retrieval traces), 'community' (Louvain / "
            "label-propagation community detection with epistemic writeback). An "
            "unrecognized action returns a clear error listing every valid action "
            "(introspected from the connected MiningClient) rather than silently "
            "degrading as if the whole mining surface were unavailable. "
            "• associate — frequent-itemset + rules (Apriori/FP-Growth/Eclat; support, "
            "confidence, lift). Provide 'transactions' (baskets of item labels) OR a "
            "graph-derived 'source' {node_label, direction(out|in|any), "
            "item_field(label|prop:<key>), relation, limit}. writeback ⇒ :AssociationRule nodes. "
            "• cluster — DBSCAN(default)/hierarchical/gmm/kmedoids over 'features' (a row "
            "matrix) OR a vector 'source' {node_label, limit} (the stored embeddings of "
            "those nodes — cross-modal 'cluster the vectors of these nodes'). Params: "
            "eps/min_pts (dbscan), k, linkage(single|complete|average), max_iter, seed. "
            "writeback ⇒ :Cluster nodes linked to members. Returns "
            "{clusters:[{cluster_id,members,centroid,score}], labels, ...} (gmm adds "
            "responsibilities). "
            "• anomaly — zscore(default)/isoforest/lof/ocsvm over 'features', a 1-D 'values' "
            "series (tsdb RCA), OR a vector 'source'. Params: k(lof), n_trees/sample_size/seed "
            "(isoforest), nu/kernel(rbf|linear)/gamma (ocsvm), threshold. writeback ⇒ "
            ":Anomaly nodes linked to their source. Returns {rows:[{id,anomaly_score,"
            "is_anomaly}], n_anomalies, threshold, ...}. "
            "• classify_fit — PREDICTIVE fit → model blob: gaussiannb(default)/multinomialnb/"
            "knn/logistic/svc over 'x' (rows) OR a vector 'source', plus integer 'y' labels. "
            "Params: k(knn), alpha(mnb), lr/epochs/l2(logistic), c(svc). Returns {model, "
            "classes, ...} — pass 'model' to classify_predict. "
            "• classify_predict — apply a fitted 'model' to 'x' OR a vector 'source' "
            "(cross-modal 'classify these nodes by their embeddings'). writeback ⇒ "
            ":Classification nodes linked to source. Returns {rows:[{id,label,proba}], classes}. "
            "• reduce — DESCRIPTIVE row transform: svd(default)/lda(supervised — needs "
            "'labels')/umap/tsne over 'x' OR a vector 'source' (reduce node vectors for the "
            "graphviz). Params: n_components, n_neighbors/min_dist(umap), perplexity/lr(tsne), "
            "epochs, seed. writeback ⇒ :Embedding2D nodes. Returns {rows:[{id,coords}], "
            "n_components, ...} (svd adds singular_values). UMAP/t-SNE are approximate, small-N. "
            "• sequence — frequent ORDERED subsequences: prefixspan(default)/gsp (both agree) "
            "over 'sequences' (time-ordered lists of item labels, repeats allowed) OR a "
            "graph-derived 'source' {node_label, direction(out|in|any), item_field, relation, "
            "limit} (each node's chronological neighbor history becomes one sequence — "
            "'what reliably follows what'). Params: min_support. writeback ⇒ :SequentialPattern "
            "nodes linked to their item nodes. Returns {patterns:[{items,support,count}], "
            "n_sequences, n_patterns, ...}. "
            "• forecast — classical forecasting over a 1-D 'values' series (a tsdb window "
            "handed in by the caller): arima(default, p/d/q — Hannan-Rissanen AR/MA)/"
            "holtwinters(alpha/beta/gamma, seasonal period — degrades to Holt linear-trend at "
            "period=0)/stl(classical decomposition + extrapolation, also returns trend/"
            "seasonal/residual). Params: horizon, confidence (band level). writeback ⇒ "
            ":Forecast node linked FORECAST_OF a resident node named 'series_id'. Returns "
            "{forecast, lower, upper, horizon, ...}. "
            "• text — TF-IDF(default, descriptive, read-only)/lda(collapsed Gibbs sampling — "
            "alpha/beta priors, iterations)/nmf(multiplicative updates on the TF-IDF matrix) "
            "over 'docs' (pre-tokenized word lists) OR a graph-derived 'source' {node_label, "
            "field, limit} (tokenizes a text property per node — no Tantivy dependency). "
            "Params: k (topic count for lda/nmf), top_n (terms kept per row). writeback ⇒ "
            "(lda/nmf only) :Topic nodes linked HAS_TOPIC from each doc's dominant topic. "
            "Returns {doc_terms:[...]} (tfidf) or {topics:[...], doc_topics:[...]} (lda/nmf). "
            "• subgraph — GRAPH-NATIVE (no input rows — mines the graph itself): "
            "gspan(default — level-wise frequent connected-subgraph pattern growth up to "
            "'max_edges', canonicalized + exactly re-counted; 'min_support' is a fraction of "
            "total host edges; support = raw embedding count, not min-node-image support)/"
            "motif(label-agnostic topological census: wedges/triangles/directed-3-cycles; "
            "min_support/max_edges ignored). Optional 'label' restricts the scanned host "
            "graph to one node type (None = whole resident graph). writeback ⇒ (gspan only) "
            ":FrequentSubgraph nodes linked SUBGRAPH_MEMBER to every node in any embedding. "
            "Returns {patterns:[{nodes,edges,support,count}],...} (gspan) or "
            "{motifs:{wedge,triangle,directed_cycle3},...} (motif). "
            "• entity_resolve — record linkage: which record PAIRS refer to the same "
            "real-world entity, over token 'records' (Jaccard, blocked by 'block_keys') "
            "OR embeddings ('vectors' or a vector 'source' {node_label, field}, Cosine, "
            "blocked by 'bucket_precision'). Params: threshold. writeback ⇒ :EntityMatch "
            "nodes linked to both members. Returns {matches:[{a,b,score}],...}. "
            "• causal_impact — estimate an intervention's causal effect in a time series "
            "at 'intervention_index': interrupted-time-series (post_mean − pre_mean) over "
            "'series' alone, or difference-in-differences when a non-empty 'control' "
            "series is also given (isolates the treatment effect from a shared trend). "
            "Returns {effect_size, standard_error, confidence,...}. writeback (+ "
            "'series_id') ⇒ :CausalEffect node. "
            "• process — process mining over ordered event 'traces' (activity-label "
            "sequences, repeats allowed): mines the directly-follows graph + the "
            "alpha-algorithm footprint (causal 'a>b' / parallel 'a||b' / choice 'a#b') "
            "plus start/end activity sets. Alternatively provide provenance-rich "
            "'events' plus an explicit 'object_type' + 'perspective_id' + "
            "'derivation_version' to deterministically project one NAMED, VERSIONED "
            "object-centric case perspective without an LLM — classical single-case "
            "flattening always requires and discloses this triple; there is no bare "
            "'object_type'-only path (undisclosed flattening is refused). Or provide a "
            "governed 'ocel_json' JSON-OCEL 2.0 document with an authorized 'tenant', "
            "optional source_ref/mapping_version/provenance transport metadata, and "
            "ocel_mode='mine' (default) or 'validate'. 'validate' only validates + exports a "
            "canonical deterministic OCEL document and never writes. 'mine' additionally "
            "requires 'object_type'/'perspective_id'/'derivation_version', commits the OCEL "
            "source truth PLUS the disclosed ProcessPerspective as one tenant-scoped tEKG "
            "ChangeEnvelope (real graph write — not just a plan), and mines traces under that "
            "perspective. Event projection writeback is fail-closed until the native "
            "ProcessModel can retain source-event lineage. "
            "Trace writeback (+ 'process_id') ⇒ :ProcessModel node. "
            "Or provide 'traces'/'object_ids'/'allowed_edges' (a reference model's edge "
            "set, e.g. a discovered directly-follows graph) + 'model_ref' + "
            "'graph_as_of'/'mapping_version'/'export_digest' + the disclosed perspective "
            "triple ('object_type'/'perspective_id'/'derivation_version') + 'tenant' to run "
            "CONFORMANCE CHECKING — 'does THIS run's behavior fit a GIVEN model?', formally "
            "separate from discovery (the model is NEVER re-derived from the traces being "
            "checked). Commits one :ConformanceRun node (linked CHECKED_UNDER_PERSPECTIVE to "
            "the perspective) + one :Deviation node per mismatch (linked HAS_DEVIATION), "
            "querying which is what discovery's own writeback cannot answer. "
            "• root_cause — bounded-depth ('max_hops'), decaying ('decay') backward "
            "search over a weighted dependency graph ('nodes','edges' cause→effect, "
            "per-node anomaly 'scores') to rank the most-likely upstream root cause of a "
            "flagged 'symptom' node. writeback ⇒ :RootCause node linked to the symptom. "
            "• risk_propagation — personalized PageRank over a weighted dependency graph "
            "('nodes','edges'), restarting to a 'seed' risk distribution instead of "
            "teleporting uniformly — propagates risk from seeded nodes along dependency "
            "edges. Params: damping, tolerance, max_iterations. writeback ⇒ one "
            ":RiskScore node per input node. "
            "• ontology_gap — GRAPH-NATIVE completeness scan of the resident graph's own "
            "node-type/edge-relationship class shape (no rdf/OWL-reasoner dependency): a "
            "class with no declared properties, an unresolved subClassOf parent, or a "
            "fully disconnected class. Optional 'label' restricts the scan to one class. "
            "writeback ⇒ :OntologyGap node per gap. "
            "• retrieval_quality — precision@k / recall@k / MRR over stored retrieval "
            "'traces' ([{retrieved,relevant}]) — audits a RAG/search pipeline's own "
            "quality. Params: k. writeback (+ 'query_id') ⇒ :RetrievalQuality node. "
            "• community — wraps the EXISTING Louvain / label-propagation GDS kernels "
            "(adds no new algorithm, only epistemic writeback) over the resident graph, "
            "optional 'label' restriction. Params: algorithm(louvain|label_propagation), "
            "resolution, weighted. writeback ⇒ one :Community node per community, linked "
            "to every member. "
            "REST twins: POST /api/mining/{associate,cluster,anomaly,classify_fit,"
            "classify_predict,reduce,sequence,forecast,text,subgraph,entity_resolve,"
            "causal_impact,process,root_cause,risk_propagation,ontology_gap,"
            "retrieval_quality,community} (same _execute_tool core). Degrades cleanly "
            "on a no-mining engine build."
        ),
        tags=["graph-os", "engine", "mining", "clustering", "anomaly", "data-mining"],
    )
    def graph_mine(
        action: str = Field(
            default="associate",
            description="Mining action (18): 'associate' | 'cluster' | 'anomaly' | "
            "'classify_fit' | 'classify_predict' | 'reduce' | 'sequence' | 'forecast' | "
            "'text' | 'subgraph' | 'entity_resolve' (alias 'entity_resolution') | "
            "'causal_impact' | 'process' (alias 'process_mining') | 'root_cause' | "
            "'risk_propagation' | 'ontology_gap' | 'retrieval_quality' | 'community'. "
            "An unrecognized action returns an error listing every valid action.",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of mining kwargs, e.g. "
            '{"transactions":[["bread","milk"],["bread","butter"]],'
            '"min_support":0.5,"algorithm":"fpgrowth"} (associate); '
            '{"features":[[0,0],[10,10]],"algorithm":"dbscan","eps":1.0,"min_pts":2} '
            'or {"source":{"node_label":"Doc"},"algorithm":"kmedoids","k":3,'
            '"writeback":true} (cluster); '
            '{"values":[1,1,1,100],"algorithm":"zscore"} or '
            '{"source":{"node_label":"Metric"},"algorithm":"isoforest"} (anomaly); '
            '{"x":[[0,0],[10,10]],"y":[0,1],"algorithm":"logistic"} (classify_fit); '
            '{"model":{...},"x":[[0.1,0.1]]} (classify_predict); '
            '{"x":[[..]],"algorithm":"svd","n_components":2} or '
            '{"source":{"node_label":"Doc"},"algorithm":"umap","writeback":true} (reduce); '
            '{"sequences":[["login","browse","purchase"]],"min_support":0.5} or '
            '{"source":{"node_label":"Session"},"algorithm":"gsp","writeback":true} (sequence); '
            '{"values":[5,8,11,14],"algorithm":"arima","p":1,"d":1,"horizon":5} or '
            '{"values":[...],"algorithm":"holtwinters","period":12,"horizon":12} (forecast); '
            '{"docs":[["the","cat","sat"]],"algorithm":"tfidf"} or '
            '{"source":{"node_label":"Doc","field":"body"},"algorithm":"lda","k":5,'
            '"writeback":true} (text); '
            '{"min_support":0.1,"max_edges":2,"writeback":true} or '
            '{"label":"Concept","algorithm":"motif"} (subgraph); '
            '{"records":[["john","smith"],["jon","smith"]],"block_keys":["smith","smith"],'
            '"threshold":0.5} or {"source":{"node_label":"Person","field":"embedding"},'
            '"threshold":0.85,"writeback":true} (entity_resolve); '
            '{"series":[1,1,1,1,5,5,5,5],"intervention_index":4} or '
            '{"series":[...],"control":[...],"intervention_index":10,"writeback":true} '
            "(causal_impact); "
            '{"traces":[["login","browse","checkout"]]} or '
            '{"traces":[...],"process_id":"checkout-flow","writeback":true} or '
            '{"events":[{"event_id":"e1","activity":"login",'
            '"occurred_at":"2026-01-01T00:00:00Z","source_ref":"src:1",'
            '"objects":[{"id":"u1","type":"User","qualifier":"actor"}]}],'
            '"object_type":"User","perspective_id":"case:user-view",'
            '"derivation_version":"v1"} (process); '
            '{"ocel_json":{"eventTypes":[],"objectTypes":[],"events":[],"objects":[]},'
            '"tenant":"acme",'
            '"object_type":"Order","perspective_id":"case:order-view",'
            '"derivation_version":"v1","ocel_mode":"mine|validate"} (process); '
            '{"traces":[["create","ship"]],"object_ids":["order-1"],'
            '"allowed_edges":[["create","pack"]],"model_ref":"model:v1",'
            '"graph_as_of":"2026-01-01T00:00:00Z","mapping_version":"ocel-json-2.0",'
            '"export_digest":"abc123","object_type":"Order",'
            '"perspective_id":"case:order-view","derivation_version":"v1",'
            '"tenant":"acme"} (process, conformance check); '
            '{"nodes":["a","b"],"scores":[0.1,0.9],"edges":[["a","b",1.0]],"symptom":"b"} '
            "(root_cause); "
            '{"nodes":["a","b"],"seed":[1.0,0.0],"edges":[["a","b",1.0]]} '
            "(risk_propagation); "
            '{"label":"Concept"} or {"writeback":true} (ontology_gap); '
            '{"traces":[{"retrieved":["d1","d2"],"relevant":["d1"]}],"k":2} '
            "(retrieval_quality); "
            '{"label":"Concept","algorithm":"louvain"} or '
            '{"algorithm":"label_propagation","writeback":true} (community).',
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin action-router over the engine mining surface (CONCEPT:EG-KG.mining.frequent-itemset-mining)."""
        action = (action or "").strip().replace("-", "_") or "associate"
        action = _MINING_ACTION_ALIASES.get(action, action)
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="mining", code="invalid_request")
        if not isinstance(params, dict):
            return json.dumps(
                {"surface": "mining", "error": "params_json must decode to an object"}
            )
        # CONCEPT:AU-KG.compute.engine-surface-manifest — an action that isn't one of the
        # 18 real MiningClient methods is a NAME error (typo/guess), not "this engine
        # build lacks mining" — report it as such, with the introspected valid-action
        # list, instead of falling through to _invoke's generic degraded payload (which
        # previously made a guessed name like 'entity_resolution' or 'process_mining'
        # look identical to the whole mining surface being absent).
        valid_actions = _mining_actions()
        if valid_actions and action not in valid_actions:
            return json.dumps(
                {
                    "surface": "mining",
                    "action": action,
                    "error": f"unknown action {action!r} for graph_mine; choose one of "
                    f"{sorted(valid_actions)}",
                    "actions": sorted(valid_actions),
                }
            )
        if action == "process" and "ocel_json" in params:
            if "traces" in params or "events" in params:
                return json.dumps(
                    {
                        "surface": "mining",
                        "action": action,
                        "code": "invalid_request",
                        "error": "provide OCEL input instead of events or traces",
                    }
                )
            from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
                ingest_graph_slice,
            )
            from agent_utilities.knowledge_graph.ingestion.event_log_adapter import (
                project_object_centric_slice,
            )
            from agent_utilities.knowledge_graph.ingestion.ocel_adapter import (
                export_ocel_json,
                import_ocel_json,
            )
            from agent_utilities.usage.authorization import resolve_usage_tenant

            try:
                tenant = resolve_usage_tenant(
                    str(params.pop("tenant", "") or "") or None
                )
                if not tenant:
                    raise ValueError("tenant is required for governed OCEL import")
                slice_, provenance = import_ocel_json(
                    params.pop("ocel_json"),
                    tenant=tenant,
                    source_ref=str(params.pop("source_ref", "") or ""),
                    mapping_version=str(params.pop("mapping_version", "") or ""),
                    provenance=params.pop("provenance", None),
                )
                ocel_mode = str(params.pop("ocel_mode", "mine") or "mine").strip()
                if ocel_mode not in {"mine", "validate"}:
                    raise ValueError("ocel_mode must be 'mine' or 'validate'")
                exported = export_ocel_json(slice_)
                if ocel_mode == "validate":
                    envelope = slice_.to_change_envelope(
                        tenant=tenant,
                        provenance=provenance,
                    )
                    evidence = {
                        "mode": "ocel_2.0",
                        "tenant": tenant,
                        "content_hash": slice_.canonical_digest(),
                        "idempotency_key": envelope.idempotency_key,
                        "mapping_version": slice_.mapping_version,
                        "node_count": len(envelope.typed_payload["entities"]),
                        "relationship_count": len(
                            envelope.typed_payload["relationships"]
                        ),
                    }
                    return json.dumps(
                        {
                            "surface": "mining",
                            "action": action,
                            "ocel": exported,
                            "tekg": evidence,
                        },
                        default=_json_default,
                    )
                # ``mine`` mode always materializes source truth AND discloses
                # any case-notion flattening it derives from that same truth
                # (CONCEPT:AU-KG.mining.governed-perspective-flattening) — the
                # perspective used for the trace projection below is folded
                # into the SAME committed slice as a real, versioned
                # ``ProcessPerspective`` node, never a silent side channel.
                perspective = _require_process_perspective(params)
                projection = project_object_centric_slice(
                    slice_,
                    perspective=perspective,
                )
                committed_slice = slice_.model_copy(
                    update={"perspectives": (*slice_.perspectives, perspective)}
                )
                # ``to_change_envelope`` still stamps tenant_id/ocel_provenance
                # onto every entity/link and computes the digest-derived
                # idempotency key — reuse that rendering — but a
                # multi-entity slice's ``{"entities": [...], "relationships":
                # [...]}`` typed_payload is NOT the single-row (+ optional
                # ``_nodes``/``_links`` auxiliary) shape ``ingest_envelope``'s
                # ``to_entity_dict``/``_prepare_node_rows`` understand: handing
                # that envelope to ``ingest_envelope`` directly silently
                # collapses the whole slice onto ONE untyped node (verified
                # against a real engine — the "success" status did not mean
                # the ProcessEvent/BusinessObject/ProcessPerspective nodes
                # were ever created). ``ingest_graph_slice`` is the existing
                # writer built for exactly this multi-node shape (first
                # entity primary, the rest as governed ``_nodes``/``_links``
                # auxiliaries) — the same one ``IngestionEngine`` already uses
                # for its concepts/facts passes.
                envelope = committed_slice.to_change_envelope(
                    tenant=tenant,
                    provenance=provenance,
                )
                # THE single commit of this slice. Three lanes touched this one
                # spot: feat/ocel-roundtrip-and-derivation and
                # feat/wire-first-reachability-gate each independently fixed the
                # discarded ChangeEnvelope (kept ONE commit, the ocel form, which
                # also commits the disclosed ProcessPerspective), and
                # feat/wave6-followups-ocel then fixed the commit ITSELF: routing
                # a {entities, relationships} payload through ingest_envelope
                # silently collapsed every entity onto ONE untyped node while
                # still returning status="success" (D-61-4). ingest_graph_slice is
                # the correct writer. The degradation wrapper is wire-first's: a
                # write-path outage surfaces as a structured error, never a crash.
                try:
                    engine = kg_server._get_engine()
                    applied = ingest_graph_slice(
                        engine,
                        envelope.connector,
                        envelope.typed_payload["entities"],
                        envelope.typed_payload["relationships"],
                        source_instance=envelope.source_instance,
                        checkpoint=envelope.checkpoint,
                    )
                except Exception as exc:  # noqa: BLE001 — a write-path outage degrades graph_mine, never crashes it
                    return _surface_error(
                        exc,
                        surface="mining",
                        action=action,
                        code="dependency_unavailable",
                    )
                if applied.get("status") not in {"success", "skipped"}:
                    return json.dumps(
                        {
                            "surface": "mining",
                            "action": action,
                            "code": "write_failed",
                            "error": (
                                "governed OCEL ChangeEnvelope commit failed: "
                                f"{applied.get('error') or applied.get('status')}"
                            ),
                            "ocel": exported,
                            "tekg": {"idempotency_key": envelope.idempotency_key},
                        },
                        default=_json_default,
                    )
                evidence = {
                    "mode": "ocel_2.0",
                    "tenant": tenant,
                    "content_hash": committed_slice.canonical_digest(),
                    "idempotency_key": applied.get(
                        "idempotency_key", envelope.idempotency_key
                    ),
                    "mapping_version": committed_slice.mapping_version,
                    "node_count": len(envelope.typed_payload["entities"]),
                    "relationship_count": len(envelope.typed_payload["relationships"]),
                    "commit_status": applied.get("status"),
                    "write_status": applied.get("status"),
                }
            except (PermissionError, TypeError, ValueError) as exc:
                return _surface_error(
                    exc,
                    surface="mining",
                    action=action,
                    code="invalid_request",
                )
            except RuntimeError as exc:
                return _surface_error(
                    exc,
                    surface="mining",
                    action=action,
                    code="commit_failed",
                )
            params["traces"] = projection.engine_traces()
            response = json.loads(
                _invoke(
                    surface="mining",
                    action=action,
                    graph=graph,
                    candidates=(("mining", action),),
                    params=params,
                )
            )
            response["projection"] = projection.public_metadata()
            response["ocel"] = exported
            response["tekg"] = evidence
            return json.dumps(response, default=_json_default)
        if action == "process" and "allowed_edges" in params:
            # CONCEPT:AU-KG.mining.process-conformance-checking (D-61-1) — "does
            # THIS run's behavior fit a GIVEN model?", never re-discovering the
            # model from the same traces being checked (that would make it
            # discovery wearing conformance's name). Pure Python compute + a
            # graph writeback; the real engine's mining client is never called
            # here (there is no engine-side conformance primitive to dispatch
            # to — the whole point of the ``ConformanceWorker`` seam is that the
            # native/default worker needs none).
            from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
                ingest_graph_slice,
            )
            from agent_utilities.knowledge_graph.ingestion.process_conformance import (
                ConformanceRun,
                conformance_run_graph_slice,
                run_conformance_check,
            )
            from agent_utilities.usage.authorization import resolve_usage_tenant

            try:
                tenant = resolve_usage_tenant(
                    str(params.pop("tenant", "") or "") or None
                )
                if not tenant:
                    raise ValueError(
                        "tenant is required for governed conformance checking"
                    )
                perspective = _require_process_perspective(params)
                traces = [tuple(trace) for trace in params.pop("traces")]
                object_ids = list(params.pop("object_ids"))
                allowed_edges = [
                    (str(pair[0]), str(pair[1])) for pair in params.pop("allowed_edges")
                ]
                model_ref = str(params.pop("model_ref", "") or "").strip()
                graph_as_of_raw = str(params.pop("graph_as_of", "") or "").strip()
                mapping_version = str(params.pop("mapping_version", "") or "").strip()
                export_digest = str(params.pop("export_digest", "") or "").strip()
                if not model_ref or not graph_as_of_raw or not mapping_version:
                    raise ValueError(
                        "conformance checking requires 'model_ref', 'graph_as_of', "
                        "and 'mapping_version'"
                    )
                if not export_digest:
                    raise ValueError(
                        "conformance checking requires 'export_digest' — the export "
                        "digest of the source data this run executed over"
                    )
                graph_as_of = datetime.fromisoformat(
                    graph_as_of_raw.replace("Z", "+00:00")
                )
                source_ref = str(params.pop("source_ref", "") or "")
                run_id = (
                    str(params.pop("run_id", "") or "")
                    or hashlib.sha256(
                        "\x1f".join(
                            [
                                source_ref,
                                perspective.perspective_id,
                                model_ref,
                                export_digest,
                                graph_as_of.isoformat(),
                            ]
                        ).encode("utf-8")
                    ).hexdigest()[:32]
                )
                start_activities = params.pop("start_activities", None)
                end_activities = params.pop("end_activities", None)
                run = ConformanceRun(
                    run_id=run_id,
                    perspective=perspective,
                    graph_as_of=graph_as_of,
                    mapping_version=mapping_version,
                    model_ref=model_ref,
                    export_digest=export_digest,
                )
                run, deviations = run_conformance_check(
                    traces,
                    object_ids,
                    allowed_edges,
                    run=run,
                    start_activities=start_activities,
                    end_activities=end_activities,
                )
                entities, links = conformance_run_graph_slice(
                    run, deviations, source_ref=source_ref
                )
                for entity in entities:
                    entity["tenant_id"] = tenant
                for link in links:
                    link["tenant_id"] = tenant
                engine = kg_server._get_engine()
                applied = ingest_graph_slice(
                    engine,
                    "conformance",
                    entities,
                    links,
                    source_instance=run_id,
                )
                if applied.get("status") not in {"success", "skipped"}:
                    raise RuntimeError(
                        "ConformanceRun ChangeEnvelope commit failed: "
                        f"{applied.get('error') or applied.get('status')}"
                    )
            except (PermissionError, TypeError, ValueError) as exc:
                return _surface_error(
                    exc,
                    surface="mining",
                    action=action,
                    code="invalid_request",
                )
            except RuntimeError as exc:
                return _surface_error(
                    exc,
                    surface="mining",
                    action=action,
                    code="commit_failed",
                )
            return json.dumps(
                {
                    "surface": "mining",
                    "action": action,
                    "conformance_run": {
                        "run_id": run.run_id,
                        "run_digest": run.run_digest(),
                        "model_ref": run.model_ref,
                    },
                    "deviations": [
                        deviation.model_dump(mode="json") for deviation in deviations
                    ],
                    "tekg": {
                        "commit_status": applied.get("status"),
                        "node_count": len(entities),
                        "relationship_count": len(links),
                    },
                },
                default=_json_default,
            )
        if action == "process" and "events" in params:
            if "traces" in params:
                return json.dumps(
                    {
                        "surface": "mining",
                        "action": action,
                        "code": "invalid_request",
                        "error": "provide either events or traces, not both",
                    }
                )
            if params.get("writeback") is True:
                return json.dumps(
                    {
                        "surface": "mining",
                        "action": action,
                        "code": "lineage_required",
                        "error": (
                            "object-centric event projection writeback is disabled "
                            "until ProcessModel writeback retains source-event lineage"
                        ),
                    }
                )
            from agent_utilities.knowledge_graph.ingestion.event_log_adapter import (
                project_object_centric_events,
            )

            try:
                perspective = _require_process_perspective(params)
                projection = project_object_centric_events(
                    params.pop("events"),
                    perspective=perspective,
                )
            except (TypeError, ValueError) as exc:
                return _surface_error(
                    exc,
                    surface="mining",
                    action=action,
                    code="invalid_request",
                )
            params["traces"] = projection.engine_traces()
            response = json.loads(
                _invoke(
                    surface="mining",
                    action=action,
                    graph=graph,
                    candidates=(("mining", action),),
                    params=params,
                )
            )
            response["projection"] = projection.public_metadata()
            return json.dumps(response, default=_json_default)
        return _invoke(
            surface="mining",
            action=action,
            graph=graph,
            candidates=(("mining", action),),
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_mine"] = graph_mine
    # REST twin path: POST {prefix}/mining/associate (mounted bespoke in kg_server so
    # a natural mining body works while dispatching the SAME _execute_tool core).
    kg_server.ACTION_TOOL_ROUTES["graph_mine"] = "/mining/associate"

    # ══════════════════════════════════════════════════════════════════
    # graph_mine_deep — Phase-6 heavy-dep delegation to data-science-mcp
    # (CONCEPT:AU-KG.mining.dsm-forecast-delegation)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_mine_deep",
        description=(
            "CONCEPT:AU-KG.mining.dsm-forecast-delegation — the deep-learning / heavy-Python family the "
            "engine core deliberately does NOT implement (no torch/GPU in the "
            "pure-Rust engine): this tool DISPATCHES to agents/data-science-mcp over "
            "MCP (the fleet call_tool_once connector, same one 'fleet.write_record' "
            "uses) and folds the result back into the KG as typed nodes "
            "(CONCEPT:AU-KG.mining.foldback-typed-nodes). Actions: 'deep_forecast' (LSTM sequence "
            "forecaster — the delegated Prophet/LSTM family; Prophet itself needs an "
            "unvendored Stan toolchain, ARIMA/Holt-Winters/STL in graph_mine remain "
            "the native classical default), 'deep_classify' (MLP deep classifier), "
            "'autoencoder_anomaly' (reconstruction-error outlier detection), "
            "'xgboost' (histogram gradient-boosting classifier — the documented "
            "xgboost substitute; no separate xgboost package is vendored), 'embed' "
            "(neural embedding of arbitrary NUMERIC feature rows via an autoencoder "
            "bottleneck — NOT text/word embeddings, those stay on the remote vLLM "
            "embedder). Every action accepts either raw rows ('x'/'values') OR a "
            "graph-derived 'source' {node_label, fields(list of properties), limit} "
            "gathered from the KG as a RowSet via one read-only Cypher projection "
            "(compute-near-data — the row-gathering read and the delegated call are "
            "the only two round trips). writeback=true materializes the result as "
            "typed nodes tagged provider='data-science-mcp': ':Forecast' (deep_forecast, "
            "linked FORECAST_OF a 'series_id' node when given), ':Classification' "
            "(deep_classify/xgboost), ':Anomaly' (autoencoder_anomaly), ':Embedding' "
            "(embed) — each row-level node linked DEEP_RESULT_OF its source node when "
            "a 'source' was used. Degrades cleanly (never crashes) when "
            "data-science-mcp is unreachable or its [training] extra (torch) is not "
            "installed: returns {available:false, error:...}. "
            "REST twins: POST /api/mining/deep/{deep_forecast,deep_classify,"
            "autoencoder_anomaly,xgboost,embed} (same _execute_tool core)."
        ),
        tags=[
            "graph-os",
            "engine",
            "mining",
            "deep-learning",
            "data-science-mcp",
            "delegation",
        ],
    )
    def graph_mine_deep(
        action: str = Field(
            default="deep_forecast",
            description="Delegated mining action: 'deep_forecast' | 'deep_classify' "
            "| 'autoencoder_anomaly' | 'xgboost' | 'embed'.",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of kwargs, e.g. "
            '{"values":[5,8,11,14,18],"horizon":5,"lookback":3,"series_id":"metric:cpu",'
            '"writeback":true} (deep_forecast); '
            '{"x":[[0,0],[10,10]],"y":[0,1],"epochs":100,"writeback":true} or '
            '{"source":{"node_label":"Doc","fields":["f1","f2"],"limit":200},"y":[0,1,...]} '
            "(deep_classify/xgboost); "
            '{"x":[[0,0],[0,1],[50,50]],"bottleneck":2,"writeback":true} or '
            '{"source":{"node_label":"Metric","fields":["v"],"limit":500}} '
            "(autoencoder_anomaly/embed).",
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin delegation adapter: ship features to data-science-mcp, fold predictions back (CONCEPT:AU-KG.mining.dsm-forecast-delegation)."""
        action = (action or "").strip().replace("-", "_") or "deep_forecast"
        if action not in _DEEP_ALGO_BY_ACTION:
            return json.dumps(
                {
                    "surface": "mining_deep",
                    "action": action,
                    "error": f"unknown action {action!r}; choose one of "
                    f"{sorted(_DEEP_ALGO_BY_ACTION)}",
                }
            )
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(
                exc,
                surface="mining_deep",
                action=action,
                code="invalid_request",
            )
        if not isinstance(params, dict):
            return json.dumps(
                {
                    "surface": "mining_deep",
                    "action": action,
                    "error": "params_json must decode to an object",
                }
            )

        writeback = bool(params.pop("writeback", False))
        series_id = str(params.pop("series_id", "") or "")

        try:
            tool_params, node_ids = _prepare_deep_delegation(action, params, graph)
        except Exception as exc:  # noqa: BLE001 — bad input / feature-gathering failure is data
            return _surface_error(exc, surface="mining_deep", action=action)

        try:
            raw = _run_async(
                call_tool_once(
                    server=_DSM_SERVER_NAME,
                    tool=_DSM_TOOL_NAME,
                    params=tool_params,
                    params_style="args",
                )
            )
        except Exception as exc:  # noqa: BLE001 — the delegate being unreachable degrades cleanly
            return _surface_error(
                exc,
                surface="mining_deep",
                action=action,
                code="dependency_unavailable",
                context={"delegated": True, "available": False},
            )

        # BUG-7: ``call_tool_once``'s decoder prefers a FastMCP result's
        # structured ``.data`` verbatim (``mcp_package._decode``) — when the
        # delegate's tool itself returns an already-JSON-encoded string (this
        # repo's own tool convention: ``return json.dumps(...)``), ``.data`` IS
        # that raw string, not the parsed object, so ``raw`` arrives here as a
        # ``str`` even though the delegate answered normally. Align the parse
        # here instead of failing on a shape mismatch that isn't a real outage.
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                pass  # genuinely not JSON — falls through to the shape error below
        if not isinstance(raw, dict):
            return json.dumps(
                {
                    "surface": "mining_deep",
                    "action": action,
                    "provider": _DSM_SERVER_NAME,
                    "delegated": True,
                    "available": False,
                    "error": f"unexpected data-science-mcp response shape: {type(raw).__name__}",
                }
            )
        if not raw.get("available", True):
            return json.dumps(
                {
                    "surface": "mining_deep",
                    "action": action,
                    "provider": _DSM_SERVER_NAME,
                    "delegated": True,
                    "available": False,
                    "error": raw.get(
                        "error", "data-science-mcp reported the algo unavailable"
                    ),
                }
            )

        result = raw.get("result") or {}
        node_type = _DEEP_NODE_TYPE[action]
        written: list[str] = []
        if writeback:
            if action == "deep_forecast":
                props = {
                    "provider": _DSM_SERVER_NAME,
                    "algo": _DEEP_ALGO_BY_ACTION[action],
                    **result,
                }
                node_id = _deep_write_node(node_type, props, graph)
                if series_id:
                    _deep_write_edge(node_id, series_id, "FORECAST_OF", graph)
                written = [node_id]
            else:
                for i, row in enumerate(result.get("rows") or []):
                    props = {
                        "provider": _DSM_SERVER_NAME,
                        "algo": _DEEP_ALGO_BY_ACTION[action],
                        **row,
                    }
                    node_id = _deep_write_node(node_type, props, graph)
                    if i < len(node_ids):
                        _deep_write_edge(node_id, node_ids[i], "DEEP_RESULT_OF", graph)
                    written.append(node_id)

        return json.dumps(
            {
                "surface": "mining_deep",
                "action": action,
                "provider": _DSM_SERVER_NAME,
                "delegated": True,
                "available": True,
                "result": result,
                "written_node_ids": written,
            },
            default=_json_default,
        )

    kg_server.REGISTERED_TOOLS["graph_mine_deep"] = graph_mine_deep
    # REST twin path: POST {prefix}/mining/deep/deep_forecast (mounted bespoke in
    # kg_server so a natural body works while dispatching the SAME _execute_tool core).
    kg_server.ACTION_TOOL_ROUTES["graph_mine_deep"] = "/mining/deep/deep_forecast"

    # ══════════════════════════════════════════════════════════════════
    # graph_learn — graph-learning / neuro-symbolic surface (CONCEPT:EG-KG.graphlearn.link-predictor)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_learn",
        description=(
            "CONCEPT:EG-KG.graphlearn.link-predictor — a pure-Rust KAN (Kolmogorov-"
            "Arnold) link-predictor over the resident graph, whose learned per-feature "
            "edge functions are THEMSELVES queryable KG nodes (interpretability, not raw "
            "accuracy). Actions: 'fit' (learn a model) and 'predict' (score links). "
            "• fit — learn over a graph-derived subgraph: every node with 'node_label' is "
            "a vertex; edges among them (direction out|in|any, optional 'relation') are "
            "positives, non-edges are sampled negatives. Params: basis(chebyshev|jacobi), "
            "degree, hidden(0 ⇒ one interpretable edge fn per feature), epochs, lr, "
            "neg_ratio, seed, alpha, limit, writeback. writeback ⇒ :EdgeFunction nodes "
            "(the learned per-feature curves — Cypher-queryable). Returns {model, n_nodes, "
            "n_edges, train_auc, edge_functions:[{feature,coefficients}], ...}. Pass the "
            "returned 'model' to predict. "
            "• predict — score candidate links with a fitted 'model' over the same "
            "'node_label' subgraph: explicit 'candidate_pairs' [[src,dst],...] OR the "
            "'top_k' highest-probability MISSING links. writeback ⇒ :PredictedEdge nodes "
            "linked to their endpoints. Returns {predicted:[{src,dst,score}], n_predicted, "
            "model, ...}. "
            "Structural features: common-neighbors, Jaccard, Adamic-Adar, preferential "
            "attachment, PageRank-product, neighbor-cosine, 1-hop-aggregated node-feature "
            "dot. Heavy multi-layer KAN-GNN training stays a data-science-mcp/torch job. "
            "REST twins: POST /api/graphlearn/{fit,predict} (same _execute_tool core). "
            "Degrades cleanly on a no-graphlearn engine build."
        ),
        tags=[
            "graph-os",
            "engine",
            "graphlearn",
            "link-prediction",
            "kan",
            "neuro-symbolic",
        ],
    )
    def graph_learn(
        action: str = Field(
            default="fit",
            description="Graph-learning action: 'fit' | 'predict'.",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of graph-learning kwargs, e.g. "
            '{"node_label":"Person","direction":"any","degree":4,"epochs":200,'
            '"writeback":true} (fit); '
            '{"model":{...},"node_label":"Person","top_k":20,"writeback":true} '
            'or {"model":{...},"node_label":"Person",'
            '"candidate_pairs":[["a","b"],["c","d"]]} (predict).',
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin action-router over the engine graph-learning surface (CONCEPT:EG-KG.graphlearn.link-predictor)."""
        action = (action or "").strip().replace("-", "_") or "fit"
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="graphlearn", code="invalid_request")
        if not isinstance(params, dict):
            return json.dumps(
                {
                    "surface": "graphlearn",
                    "error": "params_json must decode to an object",
                }
            )
        return _invoke(
            surface="graphlearn",
            action=action,
            graph=graph,
            candidates=((("graphlearn", action),)),
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_learn"] = graph_learn
    # REST twin path: POST {prefix}/graphlearn/fit (bespoke natural-body mount in
    # kg_server → the SAME _execute_tool core as the MCP verb).
    kg_server.ACTION_TOOL_ROUTES["graph_learn"] = "/graphlearn/fit"

    # ══════════════════════════════════════════════════════════════════
    # graph_pipeline — composable ML pipeline surface (CONCEPT:EG-KG.mining.ml-pipeline)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_pipeline",
        description=(
            "CONCEPT:EG-KG.mining.ml-pipeline — a composable train→eval→serve→predict "
            "ML pipeline over a VERSIONED ':Model' artifact that GENERALIZES the KAN "
            "one-off: a spec of 'feature steps → split → a pluggable model family'. "
            "Families: 'classify' (node classification — gaussiannb/multinomialnb/knn/"
            "logistic/svc), 'estimator' (regression — ridge/lasso/elasticnet/"
            "decisiontree/randomforest/gradientboosting/adaboost/svr), 'graphlearn' "
            "(the KAN link-predictor). Actions: "
            "• train — fit over composed features, evaluate on a held-out split, and "
            "persist a versioned :Model node. params: {name, spec:{features:[{step:"
            "'embedding',method:'fastrp'|'node2vec',dim,...} | {step:'node_vector'} | "
            "{step:'normalize'}], split:{test_ratio,shuffle,seed}, label_property, "
            "model:{family,algorithm,params}}, source:{node_label,direction,relation,"
            "limit} OR x:[[...]], y:[...], writeback}. Returns {name, version, model_id, "
            "family, metrics:{train,test}, ...}. Re-running train bumps the version. "
            "• eval — score a stored model (version; 0⇒served) against a labeled set → "
            "{metrics}. "
            "• serve — deploy version N as the served :ServedModel (predict-by-name "
            "then resolves it). "
            "• predict — apply a stored model to a source/x; writeback ⇒ :Prediction "
            "nodes. Returns {rows:[{id,label|value,proba}], ...}. "
            "• compare — diff two versions' held-out metrics → {metrics_a, metrics_b, "
            "diff}. "
            "Node classification: 'source' selects the subgraph, an 'embedding' feature "
            "step builds structural node vectors (fastrp/node2vec), 'label_property' is "
            "the per-node integer class. Heavy/deep training stays a data-science-mcp "
            "job. REST twins: POST /api/pipeline/{train,eval,serve,predict,compare}. "
            "Degrades cleanly on a no-ml-pipeline engine build."
        ),
        tags=[
            "graph-os",
            "engine",
            "ml-pipeline",
            "pipeline",
            "node-classification",
            "model-registry",
        ],
    )
    def graph_pipeline(
        action: str = Field(
            default="train",
            description="Pipeline action: 'train' | 'eval' | 'serve' | 'predict' | 'compare'.",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of pipeline kwargs, e.g. "
            '{"name":"community","spec":{"features":[{"step":"embedding",'
            '"method":"fastrp","dim":32}],"split":{"test_ratio":0.3},'
            '"label_property":"label","model":{"family":"classify",'
            '"algorithm":"logistic"}},"source":{"node_label":"Person"}} (train); '
            '{"name":"community","version":1} (serve); '
            '{"name":"community","version":0,"source":{"node_label":"Person"},'
            '"writeback":true} (predict); '
            '{"name":"community","version_a":1,"version_b":2} (compare).',
        ),
        graph: str = Field(
            default="", description="Target graph (empty ⇒ deployment default)."
        ),
    ) -> str:
        """Thin action-router over the engine ML-pipeline surface (CONCEPT:EG-KG.mining.ml-pipeline)."""
        # MCP verb → client method (the 'eval' alias maps to the client's evaluate()).
        method_by_action = {
            "train": "train",
            "eval": "evaluate",
            "evaluate": "evaluate",
            "serve": "serve",
            "predict": "predict",
            "compare": "compare",
        }
        action = (action or "").strip().replace("-", "_").lower() or "train"
        method = method_by_action.get(action)
        if method is None:
            return json.dumps(
                {
                    "surface": "pipeline",
                    "error": f"unknown action {action!r} for graph_pipeline; choose one of "
                    "train | eval | serve | predict | compare",
                }
            )
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="pipeline", code="invalid_request")
        if not isinstance(params, dict):
            return json.dumps(
                {"surface": "pipeline", "error": "params_json must decode to an object"}
            )
        return _invoke(
            surface="pipeline",
            action=method,
            graph=graph,
            candidates=((("pipeline", method),)),
            params=params,
        )

    kg_server.REGISTERED_TOOLS["graph_pipeline"] = graph_pipeline
    # REST twin path: POST {prefix}/pipeline/train (natural-body mount in kg_server →
    # the SAME _execute_tool core as the MCP verb).
    kg_server.ACTION_TOOL_ROUTES["graph_pipeline"] = "/pipeline/train"

    # ══════════════════════════════════════════════════════════════════
    # graph_fork — warm-fork / KV-cache fan-out (CONCEPT:AU-KG.coordination.warm-fork-fanout)
    # ══════════════════════════════════════════════════════════════════
    @mcp.tool(
        name="graph_fork",
        description=(
            "CONCEPT:AU-KG.coordination.warm-fork-fanout — warm-fork fan-out over the ORCH-1.86..93 warm-fork "
            "primitive (LMCache KV / copy-on-write sandboxes): pay warm-up ONCE for a "
            "parent context, then fork N copy-on-write branches to run per-branch "
            "computations concurrently and return each branch's result. Provide either "
            "'branches_json' (a JSON list of per-branch code snippets) or 'code' + 'n' "
            "(run the same snippet across n branches); 'vars_json' seeds the shared "
            "namespace forked into every branch; 'sandbox' optionally pins the "
            "firecracker rung, else the cheapest available "
            "warm-fork rung is used. Set 'context_query' to retrieve an engine cross-modal "
            "candidate set (vector+graph+text fusion) ONCE and fork it into every branch as "
            "'candidate_var' (default 'candidates') — the branches reuse that one context with "
            "no recompute (CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout). Degrades cleanly "
            "(structured 'unavailable') when no warm-fork rung is available on this host."
        ),
        tags=["graph-os", "engine", "fork", "warm-fork", "fanout"],
    )
    def graph_fork(
        code: str = Field(
            default="",
            description="A single code snippet run on each of 'n' branches (ignored "
            "when 'branches_json' is provided).",
        ),
        n: int = Field(
            default=0, description="Fan-out count when using 'code' (branches to fork)."
        ),
        branches_json: str = Field(
            default="[]",
            description="JSON list of per-branch code snippets; overrides code/n.",
        ),
        vars_json: str = Field(
            default="{}",
            description="JSON object seeding the namespace forked into every branch.",
        ),
        sandbox: str = Field(
            default="",
            description="Preferred warm-fork rung name (empty ⇒ cheapest available).",
        ),
        context_query: str = Field(
            default="",
            description="Optional: retrieve an engine cross-modal candidate set (vector+graph"
            "+text) for this query ONCE and fork it into every branch (reused, no recompute).",
        ),
        candidate_var: str = Field(
            default="candidates",
            description="Namespace name the cross-modal candidate set is bound to in each branch "
            "(only used when context_query is set).",
        ),
    ) -> str:
        """Thin verb over the warm-fork primitive (CONCEPT:AU-KG.coordination.warm-fork-fanout)."""
        try:
            branches = json.loads(branches_json) if branches_json else []
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="fork", code="invalid_request")
        if not isinstance(branches, list):
            return json.dumps(
                {"surface": "fork", "error": "branches_json must decode to a list"}
            )
        if not branches:
            if code and int(n) > 0:
                branches = [code] * int(n)
            else:
                return json.dumps(
                    {
                        "surface": "fork",
                        "error": "provide branches_json (list) or code + n (>0)",
                    }
                )
        try:
            seed_vars = json.loads(vars_json) if vars_json else {}
        except (TypeError, ValueError) as exc:
            return _surface_error(exc, surface="fork", code="invalid_request")
        if not isinstance(seed_vars, dict):
            return json.dumps(
                {"surface": "fork", "error": "vars_json must decode to an object"}
            )
        if context_query.strip():
            return _crossmodal_fork_fanout(
                branches,
                seed_vars,
                sandbox.strip(),
                context_query.strip(),
                candidate_var.strip() or "candidates",
            )
        return _fork_fanout(branches, seed_vars, sandbox.strip())

    kg_server.REGISTERED_TOOLS["graph_fork"] = graph_fork
    kg_server.ACTION_TOOL_ROUTES["graph_fork"] = "/graph/fork"
