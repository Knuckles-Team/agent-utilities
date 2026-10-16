"""Direct epistemic-graph client calls for harness mechanisms that need a result.

CONCEPT:AU-KG.coordination.engine-message-broker

The harness serves no engine tools. When one of its in-run mechanisms needs a
result from epistemic-graph it calls the process engine client directly through
these two helpers; every failure is returned as data, never raised.
"""

from __future__ import annotations

import base64
import inspect
import json
from typing import Any

from agent_utilities.security.error_surface import public_error_payload

# Engine client sub-client class per domain used by harness callers.
_DOMAIN_CLASSES: dict[str, str] = {
    "placement": "PlacementClient",
    "resharding": "ReshardingClient",
}


def engine_client(graph: str = "") -> Any:
    """The process epistemic-graph client, optionally graph-scoped."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    return GraphComputeEngine.get_or_create(graph_name=graph or None).client


def domain_methods(domain: str) -> frozenset[str]:
    """Public async methods the installed engine client exposes for ``domain``."""
    try:
        from epistemic_graph import client as client_module
    except Exception:  # noqa: BLE001 — no engine client installed ⇒ no methods
        return frozenset()
    cls = getattr(client_module, _DOMAIN_CLASSES.get(domain, ""), None)
    if cls is None:
        return frozenset()
    return frozenset(
        name
        for name, _member in inspect.getmembers(cls, inspect.iscoroutinefunction)
        if not name.startswith("_")
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes | bytearray):
        return {"__bytes_b64__": base64.b64encode(bytes(obj)).decode("ascii")}
    return str(obj)


def _error(exc: BaseException, *, surface: str, action: str, code: str) -> str:
    payload = public_error_payload(exc, code=code)
    payload["surface"] = surface
    payload["action"] = action
    return json.dumps(payload)


def invoke_surface(
    *,
    surface: str,
    action: str,
    graph: str,
    candidates: tuple[tuple[str, str], ...],
    params: dict[str, Any],
) -> str:
    """Call the first present ``client.<sub>.<method>`` and JSON the result.

    Engine unreachable → ``error``; surface absent → ``degraded``; bad kwargs or
    an engine error → ``error``.
    """
    try:
        client = engine_client(graph)
    except Exception as exc:  # noqa: BLE001 — engine down is a normal degrade
        return _error(exc, surface=surface, action=action, code="dependency_unavailable")
    fn = None
    for sub_attr, method_attr in candidates:
        candidate = getattr(getattr(client, sub_attr, None), method_attr, None)
        if callable(candidate):
            fn = candidate
            break
    if fn is None:
        return json.dumps(
            {
                "surface": surface,
                "action": action,
                "degraded": True,
                "error": f"engine surface {surface!r} is not available in this engine build",
                "tried": [".".join(pair) for pair in candidates],
            }
        )
    try:
        result = fn(**params)
    except TypeError as exc:
        return _error(exc, surface=surface, action=action, code="invalid_request")
    except Exception as exc:  # noqa: BLE001 — surface engine errors as data
        return _error(exc, surface=surface, action=action, code="operation_failed")
    return json.dumps(
        {"surface": surface, "action": action, "result": result}, default=_json_default
    )


def dispatch(
    domain: str, methods: set[str] | frozenset[str], action: str, params_json: str, graph: str
) -> str:
    """Call ``client.<domain>.<action>`` with JSON params; the engine's RBAC decides.

    Returns the raw JSON result, or an ``{"error": ...}`` payload for an unknown
    action, undecodable params or an engine failure.
    """
    if action not in methods:
        return json.dumps(
            {"error": f"unknown action {action!r} for {domain}", "actions": sorted(methods)}
        )
    try:
        params = json.loads(params_json) if params_json else {}
    except (TypeError, ValueError) as exc:
        return json.dumps(public_error_payload(exc, code="invalid_request"))
    if not isinstance(params, dict):
        return json.dumps({"error": "params_json must decode to a JSON object"})
    try:
        fn = getattr(getattr(engine_client(graph), domain), action)
        result = fn(**params)
    except Exception as exc:  # noqa: BLE001 — engine errors surface as data
        return json.dumps(public_error_payload(exc, code="operation_failed"))
    return json.dumps(result, default=_json_default)
