#!/usr/bin/python
"""Ontology integrity-policy activation — CONCEPT:AU-KG.ontology.integrity-bootstrap.

**The defect this module closes.** The engine's RDF write guard
(``EG-KG.ontology.rdf-update-guard``) rejects EVERY ``AddTriples``/
``RemoveTriples`` against a graph — even a zero-triple call — until that graph
has a registered SHACL/ICV integrity policy (``IcvConfigure``). There is no
"no policy configured" pass-through; this is deliberate, correct engine
behavior. Every caller in this codebase that loads ontology axioms into a
dedicated ontology graph used to call ``add_triples`` directly, so every one
of them received the guard's rejection and silently reported a failed
activation while the process kept serving — a permanent half-initialised
state (see :mod:`agent_utilities.knowledge_graph.ontology.lifecycle`'s
``_load_axioms`` and :mod:`agent_utilities.knowledge_graph.ontology.evolution`'s
``materialize_shadow``, the two chokepoints this module now gates).

**What this module does — and does not.** It uses ONLY the engine's own
existing policy-registration authority
(:meth:`~agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine.icv_configure`)
to register a graph's integrity policy, then verifies the registration took
effect with a zero-triple ``AddTriples`` probe — the guard evaluates graph
authority/policy presence before it ever looks at the triples supplied, so an
empty probe exercises the SAME check a real load would hit, using no second
validator and no bypass. It never relaxes, skips, or races around the guard.

**Ordering contract.** :func:`ensure_ontology_graph_activated` MUST be called
— and must return successfully — before any ``add_triples`` call against the
same ``(tenant, graph)`` pair. Callers that skip this and call
``add_triples`` directly will be rejected by the engine exactly as before.

**Binding.** Each graph's FIRST successful activation durably records
``(tenant, graph, policy_digest)`` as a marker node in that same graph (the
established ``:HostedOntology``-registry convention —
CONCEPT:AU-KG.ontology.dedicated-tbox-graph). Every later activation call for
that graph must supply the SAME tenant/graph/policy digest or it is rejected
with :class:`OntologyPolicyBindingMismatchError` — never silently
reconfigured. The ontology content digest is recorded for provenance but is
NOT part of the match (loading a new ontology *version* over an
already-activated graph is expected and must not be treated as tampering).

**Idempotence.** A binding match makes the call a no-op success — no second
``IcvConfigure``, no error. (The engine's own ``IcvConfigure`` is additionally
policy-idempotent server-side, so even a network-level replay of the exact
same call is itself absorbed safely — this module's own idempotence check is
belt-and-suspenders, testable without a live engine.)

**Retry/backoff.** Registration + verification is retried with bounded
exponential backoff (jittered) up to ``max_attempts`` or ``time_ceiling_s``,
whichever comes first. Only the first failure and the final give-up are
logged — never one line per attempt.

**Readiness.** Every outcome (success or give-up) is recorded in a process-
wide, per-graph status table via :func:`get_activation_status`, which
:mod:`agent_utilities.knowledge_graph.readiness` reads to fail closed: a
graph that has never been activated, or whose activation gave up, reads
``unavailable`` — never silently "ready".
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ICV_SHAPES_TTL",
    "ActivationRecord",
    "OntologyActivationError",
    "OntologyActivationTimeoutError",
    "OntologyPolicyBindingMismatchError",
    "ensure_ontology_graph_activated",
    "get_activation_status",
    "reset_activation_state_for_tests",
]

#: The mandatory integrity policy every dedicated ontology (TBox) graph
#: registers before the engine will accept any ``AddTriples`` against it.
#: Real, minimal invariant: every ``owl:Class``/``owl:ObjectProperty``/
#: ``owl:DatatypeProperty`` this platform hosts is a stable, named IRI
#: resource — never a blank node — since cross-graph OWL reasoning, SPARQL
#: federation, and the hosted-ontology registry all address TBox entities by
#: IRI. A shape whose ``sh:targetClass`` matches nothing in a given graph
#: validates successfully without constraining anything (engine behavior),
#: so this is safe to register uniformly across every tenant's graph.
DEFAULT_ICV_SHAPES_TTL = """@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix icv: <urn:agent-utilities:ontology-integrity-policy#> .

icv:OntologyClassShape a sh:NodeShape ;
    sh:targetClass owl:Class ;
    sh:nodeKind sh:IRI .

icv:OntologyObjectPropertyShape a sh:NodeShape ;
    sh:targetClass owl:ObjectProperty ;
    sh:nodeKind sh:IRI .

icv:OntologyDatatypePropertyShape a sh:NodeShape ;
    sh:targetClass owl:DatatypeProperty ;
    sh:nodeKind sh:IRI .
"""

#: Bounded retry/backoff defaults (requirement: "Bounded retry/backoff").
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_BASE_DELAY_S = 0.25
DEFAULT_MAX_DELAY_S = 8.0
DEFAULT_TIME_CEILING_S = 30.0

_ACTIVATION_NODE_TYPE = "OntologyActivation"
_ACTIVATION_NODE_PREFIX = "ontactv"

#: Process-local fallback store, used only when no real engine node-write
#: surface is attached (offline/dev/tests) — mirrors
#: ``ontology.lifecycle._MEMORY_STORE``'s existing engine-free fallback
#: convention exactly (never a durability guarantee, just parity of behavior).
_MEMORY_BINDINGS: dict[str, dict[str, Any]] = {}
_MEMORY_LOCK = threading.Lock()

#: Process-wide LAST KNOWN activation outcome per graph, consulted by
#: ``readiness.py``. Deliberately process-local, non-durable: readiness must
#: reflect what THIS process actually observed, not a peer's/replica's state
#: (a stale "ready" read from another process would defeat the fail-closed
#: contract this exists for).
_ACTIVATION_STATUS: dict[str, dict[str, Any]] = {}
_STATUS_LOCK = threading.Lock()


class OntologyActivationError(RuntimeError):
    """Base class for ontology integrity-policy activation failures."""


class OntologyPolicyBindingMismatchError(OntologyActivationError):
    """An activation request's (tenant, graph, policy digest) does not match
    the durably recorded binding for this ontology graph.

    Raised instead of silently reconfiguring or auto-correcting — a mismatch
    here means either a caller bug (wrong tenant/graph resolved) or a policy
    change being pushed at the wrong chokepoint; both must be surfaced, never
    swallowed.
    """


class OntologyActivationTimeoutError(OntologyActivationError):
    """Activation did not succeed within the bounded attempt/time ceiling."""


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    """One graph's integrity-policy activation outcome."""

    tenant: str
    graph: str
    policy_digest: str
    ontology_digest: str
    activated_at: str
    attempts: int
    idempotent: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "tenant": self.tenant,
            "graph": self.graph,
            "policy_digest": self.policy_digest,
            "ontology_digest": self.ontology_digest,
            "activated_at": self.activated_at,
            "attempts": self.attempts,
            "idempotent": self.idempotent,
        }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _record_status(
    graph_name: str,
    *,
    state: str,
    reason: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    with _STATUS_LOCK:
        _ACTIVATION_STATUS[graph_name] = {
            "state": state,
            "reason": reason,
            "detail": detail or {},
            "observed_at": _now_iso(),
        }


def get_activation_status(graph_name: str) -> dict[str, Any] | None:
    """Return the last known activation outcome for ``graph_name``, or
    ``None`` if activation has never been attempted in this process.

    Consumed by :mod:`agent_utilities.knowledge_graph.readiness` — ``None``
    and a give-up outcome both mean "not ready"; only a recorded
    ``state == "ready"`` does.
    """
    with _STATUS_LOCK:
        record = _ACTIVATION_STATUS.get(graph_name)
        return dict(record) if record is not None else None


def reset_activation_state_for_tests() -> None:
    """Clear all in-process activation state. Test-only."""
    with _STATUS_LOCK:
        _ACTIVATION_STATUS.clear()
    with _MEMORY_LOCK:
        _MEMORY_BINDINGS.clear()


# ── durable (tenant, graph) -> policy binding ─────────────────────────────────
#
# Stored the SAME way ``ontology.lifecycle._EngineRegistryStore`` stores
# ``:HostedOntology`` records — a typed marker NODE in the dedicated ontology
# graph itself, addressed via the engine's native property-graph node surface
# (``add_node``/``has_node``/``client.nodes.properties``). That surface is
# NOT gated by the RDF write guard (the guard only gates the RDF/``AddTriples``
# projection; native property-graph writes are a separate, opt-in-only gate —
# see ``ontology/lifecycle.py``'s module docstring), so there is no
# chicken-and-egg problem recording the binding before the policy is active.


def _node_id(graph_name: str) -> str:
    return f"{_ACTIVATION_NODE_PREFIX}:{graph_name}"


def _is_real_engine(gc: Any) -> bool:
    return (
        gc is not None
        and hasattr(gc, "add_node")
        and hasattr(gc, "has_node")
        and hasattr(gc, "client")
    )


def _binding_get(gc: Any, graph_name: str) -> dict[str, Any] | None:
    if not _is_real_engine(gc):
        with _MEMORY_LOCK:
            record = _MEMORY_BINDINGS.get(graph_name)
            return dict(record) if record is not None else None
    node_id = _node_id(graph_name)
    if not gc.has_node(node_id):
        return None
    props = gc.client.nodes.properties(node_id) or {}
    data = props.get("data") if isinstance(props, dict) else None
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception as exc:  # noqa: BLE001 — a corrupt record reads as absent, never crashes activation
        logger.warning(
            "Corrupt ontology-activation binding record for graph %s: %s",
            graph_name,
            type(exc).__name__,
        )
        return None


def _binding_set(gc: Any, graph_name: str, record: dict[str, Any]) -> None:
    if not _is_real_engine(gc):
        with _MEMORY_LOCK:
            _MEMORY_BINDINGS[graph_name] = dict(record)
        return
    gc.add_node(
        _node_id(graph_name),
        node_type=_ACTIVATION_NODE_TYPE,
        graph=graph_name,
        tenant=record.get("tenant", ""),
        policy_digest=record.get("policy_digest", ""),
        activated_at=record.get("activated_at", ""),
        data=json.dumps(record, default=str),
    )


# ── register + verify, with bounded retry/backoff ─────────────────────────────


def _register_and_verify(
    gc: Any,
    *,
    graph_name: str,
    shapes_ttl: str,
    max_attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    deadline: float,
) -> int:
    """(Re)register ``shapes_ttl`` on ``graph_name`` and verify it took effect.

    Verification is a zero-triple ``add_triples`` probe: the engine's write
    guard evaluates graph authority/policy presence BEFORE it inspects any
    supplied triples, so an empty addition set exercises the exact same check
    a real ontology load would hit — using only the engine's existing
    authority, never a second validator.

    Retries transient failures (engine starting, shard moving) with bounded
    exponential backoff + jitter. Logs only the FIRST failure and the FINAL
    give-up — never one line per attempt. Returns the attempt count that
    finally succeeded; raises :class:`OntologyActivationTimeoutError` (from
    the last underlying exception) on exhaustion.
    """
    attempt = 0
    logged_first_failure = False
    while True:
        attempt += 1
        try:
            ok = gc.icv_configure(shapes_ttl, graph=graph_name, mode="enforce")
            if not ok:
                raise OntologyActivationError(
                    f"engine declined IcvConfigure for graph {graph_name!r} "
                    "(returned a falsy result without raising)"
                )
            # Read back / verify using only the engine's own existing
            # AddTriples authority (CONCEPT:EG-KG.ontology.rdf-update-guard).
            gc.add_triples(turtle="")
            return attempt
        except Exception as exc:  # noqa: BLE001 — retried below, never swallowed
            if not logged_first_failure:
                logger.warning(
                    "Ontology activation attempt %d failed for graph %s "
                    "(will retry with bounded backoff): %s",
                    attempt,
                    graph_name,
                    exc,
                )
                logged_first_failure = True
            now = time.monotonic()
            if attempt >= max_attempts or now >= deadline:
                logger.error(
                    "Ontology activation gave up for graph %s after %d attempt(s): %s",
                    graph_name,
                    attempt,
                    exc,
                )
                raise OntologyActivationTimeoutError(
                    f"ontology activation failed for graph {graph_name!r} "
                    f"after {attempt} attempt(s): {exc}"
                ) from exc
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random() / 2  # nosec B311 - retry jitter, not crypto
            remaining = deadline - now
            time.sleep(max(0.0, min(delay, remaining)))


# ── public entry point ────────────────────────────────────────────────────────


def ensure_ontology_graph_activated(
    gc: Any,
    *,
    tenant: str | None,
    graph_name: str,
    ontology_turtle: str,
    shapes_ttl: str = DEFAULT_ICV_SHAPES_TTL,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
    time_ceiling_s: float = DEFAULT_TIME_CEILING_S,
) -> ActivationRecord:
    """Register + verify ``graph_name``'s SHACL/ICV integrity policy.

    MUST be called, and must return successfully, before any ``add_triples``
    call against the same graph — this is the ordering fix for
    CONCEPT:AU-KG.ontology.integrity-bootstrap. Raises (never returns a
    "failed but here's a dict" sentinel) so a caller cannot mistake a failed
    activation for a successful one:

    * :class:`OntologyPolicyBindingMismatchError` — ``tenant``/``graph_name``/
      the policy digest don't match this graph's recorded first activation.
      Not retried (not transient).
    * :class:`OntologyActivationTimeoutError` — transient retries exhausted
      the bounded attempt/time ceiling.
    * :class:`OntologyActivationError` — no engine RDF/ICV surface attached.

    Idempotent: calling this again for an already-activated graph with the
    SAME tenant/graph/policy digest is a no-op success (no second
    ``IcvConfigure``, no error) — restart-safe by construction.
    """
    if gc is None or not hasattr(gc, "icv_configure") or not hasattr(gc, "add_triples"):
        _record_status(
            graph_name,
            state="unavailable",
            reason="no_engine_rdf_icv_surface",
        )
        raise OntologyActivationError(
            "no engine RDF/ICV surface is attached; cannot activate an "
            f"integrity policy for ontology graph {graph_name!r}"
        )

    tenant_key = tenant or ""
    policy_digest = _digest(shapes_ttl)
    ontology_digest = _digest(ontology_turtle)

    existing = _binding_get(gc, graph_name)
    if existing is not None:
        mismatch = (
            existing.get("tenant", "") != tenant_key
            or existing.get("graph") != graph_name
            or existing.get("policy_digest") != policy_digest
        )
        if mismatch:
            _record_status(
                graph_name,
                state="unavailable",
                reason="policy_binding_mismatch",
                detail={
                    "recorded": {
                        "tenant": existing.get("tenant"),
                        "graph": existing.get("graph"),
                        "policy_digest": existing.get("policy_digest"),
                    },
                    "requested": {
                        "tenant": tenant_key,
                        "graph": graph_name,
                        "policy_digest": policy_digest,
                    },
                },
            )
            logger.error(
                "Ontology activation binding mismatch for graph %s: recorded "
                "tenant=%r graph=%r policy_digest=%r; requested tenant=%r "
                "graph=%r policy_digest=%r",
                graph_name,
                existing.get("tenant"),
                existing.get("graph"),
                existing.get("policy_digest"),
                tenant_key,
                graph_name,
                policy_digest,
            )
            raise OntologyPolicyBindingMismatchError(
                "ontology activation binding mismatch for graph "
                f"{graph_name!r}: recorded tenant={existing.get('tenant')!r} "
                f"graph={existing.get('graph')!r} policy_digest="
                f"{existing.get('policy_digest')!r}; requested tenant="
                f"{tenant_key!r} graph={graph_name!r} policy_digest="
                f"{policy_digest!r}"
            )
        record = ActivationRecord(
            tenant=tenant_key,
            graph=graph_name,
            policy_digest=policy_digest,
            ontology_digest=existing.get("ontology_digest", ontology_digest),
            activated_at=str(existing.get("activated_at", "")),
            attempts=int(existing.get("attempts", 0)),
            idempotent=True,
        )
        _record_status(
            graph_name,
            state="ready",
            detail={"idempotent": True, "attempts": record.attempts},
        )
        return record

    deadline = time.monotonic() + max(0.0, time_ceiling_s)
    try:
        attempts = _register_and_verify(
            gc,
            graph_name=graph_name,
            shapes_ttl=shapes_ttl,
            max_attempts=max(1, max_attempts),
            base_delay_s=max(0.0, base_delay_s),
            max_delay_s=max(base_delay_s, max_delay_s),
            deadline=deadline,
        )
    except OntologyActivationTimeoutError:
        _record_status(graph_name, state="unavailable", reason="activation_timeout")
        raise
    except Exception as exc:  # noqa: BLE001 — recorded, then re-raised, never swallowed
        _record_status(
            graph_name,
            state="unavailable",
            reason=f"activation_error:{type(exc).__name__}",
        )
        raise

    record = ActivationRecord(
        tenant=tenant_key,
        graph=graph_name,
        policy_digest=policy_digest,
        ontology_digest=ontology_digest,
        activated_at=_now_iso(),
        attempts=attempts,
        idempotent=False,
    )
    _binding_set(gc, graph_name, record.as_dict())
    _record_status(
        graph_name,
        state="ready",
        detail={"idempotent": False, "attempts": attempts},
    )
    return record
