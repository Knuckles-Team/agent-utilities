#!/usr/bin/python
from __future__ import annotations

"""Universal synthesize-from-KG context plane (CONCEPT:AU-KG.retrieval.route-question-its-domain).

ONE pattern for the whole system: ``synthesize_context(domain, query, intent)``
returns a grounded, cited answer about some *domain* — composed from the KG (and
live signals) — with a stable ``{answer, citations, capability_id,
used_primitives}`` shape and a measurable outcome hook. ``code_context``
(KG-2.134) was the first instance; this generalizes it so **any** domain — code,
ops/health, deployment, tickets, processes — registers a provider and inherits the
surface, the ``file:line``/id citation contract, and the action-outcome reward
loop (AHE-3.62).

A provider is ``fn(engine, *, query, intent, **opts) -> dict`` returning the
standard answer dict (``status=ok``). Built-ins (lazy-imported to avoid cycles):
``code`` → :func:`code_context.build_code_context`, ``ops`` →
:func:`ops_context.diagnose_ops`. External/future domains register via
:func:`register_context_provider`. The point: the enterprise cockpit is not a new
subsystem — it is *more providers on this one plane*.
"""

from collections.abc import Callable
from typing import Any

#: domain -> (module, attribute) lazy import, so the plane has zero import-time
#: dependency on heavy provider modules (and no cycle: providers import the plane).
_BUILTIN_PROVIDERS: dict[str, tuple[str, str]] = {
    "code": (
        "agent_utilities.knowledge_graph.retrieval.code_context",
        "build_code_context",
    ),
    "ops": ("agent_utilities.knowledge_graph.retrieval.ops_context", "diagnose_ops"),
    # Cross-layer error diagnosis: pulls the run's :RunTrace/:ToolCall and emits
    # the app-trace→container→system→host→cross-cutting tool playbook (KG-2.297).
    "troubleshoot": (
        "agent_utilities.knowledge_graph.retrieval.troubleshoot_context",
        "diagnose_symptom",
    ),
    "deploy": (
        "agent_utilities.knowledge_graph.retrieval.deploy_context",
        "deploy_status",
    ),
    "entity": (
        "agent_utilities.knowledge_graph.retrieval.entity_context",
        "entity_context",
    ),
    # Enterprise domains are the entity provider with a label filter — registered
    # here so the cockpit grows with ingested data (CONCEPT:AU-KG.retrieval.kg-3).
    "tickets": (
        "agent_utilities.knowledge_graph.retrieval.entity_context",
        "entity_context",
    ),
    "deploys": (
        "agent_utilities.knowledge_graph.retrieval.entity_context",
        "entity_context",
    ),
    "process": (
        "agent_utilities.knowledge_graph.retrieval.entity_context",
        "entity_context",
    ),
    # Seam 8 Phase 1 (CONCEPT:AU-KG.retrieval.capability-power-descriptor) — "what
    # can this tool do, when should I use it, how reliable is it" for ANY
    # graph-os capability, via the same universal plane every other domain
    # uses: target="capability:<tool name>" (or "capability:list").
    "capability": (
        "agent_utilities.knowledge_graph.retrieval.capability_context",
        "capability_power_context",
    ),
}

#: Registered (dynamic) providers win over built-ins.
_PROVIDERS: dict[str, Callable[..., dict[str, Any]]] = {}
_PROVIDER_META: dict[str, dict[str, Any]] = {}

# Keyword → domain inference when the caller does not name a domain.
_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "ops": (
        "lane",
        "queue",
        "task",
        "tasks",
        "backlog",
        "pending",
        "dead_letter",
        "dead-letter",
        "failed",
        "stuck",
        "breaker",
        "worker",
        "health",
        "drain",
        "scheduler",
        "why is",
        "backing up",
        "throughput",
    ),
    "troubleshoot": (
        "error",
        "failed run",
        "crash",
        "crashloop",
        "unreachable",
        "502",
        "exit 137",
        "oom",
        "traceback",
        "why did",
        "session terminated",
        "no route",
        "troubleshoot",
        "diagnose",
        "container log",
        "runtrace",
        "toolcall",
    ),
}


def read_rows(
    engine: Any, cypher: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Run a read-only query tolerant of either engine read API; never raises.

    The shared KG read used by every context provider (``query_cypher`` first,
    then ``backend.execute``) so a degraded backend yields ``[]``, not a crash.
    """
    params = params or {}
    try:
        qc = getattr(engine, "query_cypher", None)
        if callable(qc):
            return list(qc(cypher, params) or [])
        backend = getattr(engine, "backend", None)
        if backend is not None and hasattr(backend, "execute"):
            return list(backend.execute(cypher, params) or [])
    except Exception:  # pragma: no cover - reads are best-effort by design
        return []
    return []


def register_context_provider(
    domain: str,
    fn: Callable[..., dict[str, Any]],
    *,
    intents: tuple[str, ...] = ("how",),
    description: str = "",
) -> None:
    """Register a context provider for ``domain`` (overrides any built-in)."""
    _PROVIDERS[domain] = fn
    _PROVIDER_META[domain] = {"intents": list(intents), "description": description}


def _resolve_provider(domain: str) -> Callable[..., dict[str, Any]] | None:
    if domain in _PROVIDERS:
        return _PROVIDERS[domain]
    spec = _BUILTIN_PROVIDERS.get(domain)
    if spec is None:
        return None
    import importlib

    try:
        mod = importlib.import_module(spec[0])
        return getattr(mod, spec[1])
    except Exception:  # pragma: no cover - provider import best-effort
        return None


def infer_domain(query: str) -> str:
    """Pick a domain from the question text; defaults to ``code``."""
    low = (query or "").lower()
    for domain, hints in _DOMAIN_HINTS.items():
        if any(h in low for h in hints):
            return domain
    return "code"


def list_context_domains() -> list[dict[str, Any]]:
    """Every registered/built-in domain + its declared intents (for discovery)."""
    out: list[dict[str, Any]] = []
    for domain in sorted(set(_BUILTIN_PROVIDERS) | set(_PROVIDERS)):
        meta = _PROVIDER_META.get(domain, {})
        out.append(
            {
                "domain": domain,
                "builtin": domain in _BUILTIN_PROVIDERS,
                "intents": meta.get("intents", ["how"]),
                "description": meta.get("description", ""),
            }
        )
    return out


def synthesize_context(
    engine: Any,
    *,
    domain: str = "",
    query: str = "",
    intent: str = "",
    **opts: Any,
) -> dict[str, Any]:
    """Route a question to its domain provider and return the cited answer.

    ``domain`` empty → inferred from ``query``. The provider returns the standard
    answer dict; the plane guarantees ``domain``/``capability_id`` are present so
    every answer is feedable into the action-outcome loop (AHE-3.62).
    """
    resolved = (domain or infer_domain(query)).strip().lower()
    provider = _resolve_provider(resolved)
    if provider is None:
        return {
            "status": "error",
            "domain": resolved,
            "answer": f"No context provider for domain '{resolved}'.",
            "available_domains": [d["domain"] for d in list_context_domains()],
        }
    # Pass the resolved domain so a parameterized provider (e.g. ``entity`` serving
    # tickets/deploys/process) knows which slice it was asked for; single-domain
    # providers accept and ignore it via **opts.
    opts.setdefault("domain", resolved)
    result = provider(engine, query=query, intent=intent or "how", **opts)
    if not isinstance(result, dict):  # pragma: no cover - provider contract
        result = {"status": "error", "answer": str(result)}
    result.setdefault("domain", resolved)
    result.setdefault("capability_id", f"context:{resolved}:{(query or '')[:48]}")
    return result
