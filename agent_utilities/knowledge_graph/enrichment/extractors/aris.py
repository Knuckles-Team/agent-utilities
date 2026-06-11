"""ARIS process / enterprise-architecture source extractor (CONCEPT:KG-2.9).

Self-registering extractor that maps Software AG **ARIS** models into the uniform
``ExtractionBatch`` shape (typed ``GraphNode`` + ``EnrichmentEdge``) so they
persist through the one generic writer with no edits to any shared hub file.

Emitted node types are the **canonical** ArchiMate concepts so ARIS folds into the
same cross-vendor crosswalk as Camunda (BPM) and LeanIX/ArchiMate (EA):

    process model       -> ``BusinessProcess``      id=aris_model:{id}
    architecture model  -> ``ApplicationComponent`` id=aris_model:{id}

The ARIS client is **injected** (duck-typed) via ``config["client"]`` and is
expected to expose ``list_models()`` returning dict records with ``id``/``name``/
``type``. All field access is tolerant and this module performs **no** network
calls itself.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "aris"

# Model-type substrings marking a process (BPM) model vs an architecture model.
_BPM_HINTS = ("process", "epc", "bpmn", "value", "vad")


def _get(record: Any, key: str, default: Any = None) -> Any:
    """Tolerant field access for dict records (or attr-style objects)."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _first(record: Any, *keys: str) -> Any:
    """Return the first present, non-empty value among ``keys``."""
    for key in keys:
        val = _get(record, key)
        if val is not None and val != "":
            return val
    return None


def _call(client: Any, name: str) -> list:
    """Call a client method if present, returning a list (tolerant)."""
    method = getattr(client, name, None)
    if not callable(method):
        return []
    try:
        result = method()
    except TypeError:
        try:
            result = method({})
        except Exception:
            return []
    except Exception:
        return []
    if isinstance(result, dict):
        result = result.get("items") or result.get("models") or result.get("data") or []
    return list(result) if result else []


def _is_process(model_type: str) -> bool:
    t = (model_type or "").lower()
    return any(h in t for h in _BPM_HINTS)


def extract(config: Any) -> ExtractionBatch:
    """Extract ARIS models into a uniform ``ExtractionBatch``.

    ``config`` carries an injected ``client``. Process models become canonical
    ``BusinessProcess`` nodes; architecture models become ``ApplicationComponent``
    nodes — so they cross-link with the BPM and EA cohorts respectively.
    """
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for rec in _call(client, "list_models"):
        mid = _first(rec, "id", "modelId", "name")
        if not mid:
            continue
        mtype = _first(rec, "type", "modelType") or "Model"
        label = "BusinessProcess" if _is_process(mtype) else "ApplicationComponent"
        nodes.append(
            GraphNode(
                id=f"aris_model:{mid}",
                type=label,
                props={
                    "name": _first(rec, "name", "id"),
                    "model_type": mtype,
                    "capability": "bpm"
                    if label == "BusinessProcess"
                    else "enterprise-architecture",
                },
            )
        )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY,
    extract,
    description="ARIS models (process/architecture) → KG",
)
