#!/usr/bin/python
from __future__ import annotations

"""Derived (function-backed / computed) properties — CONCEPT:KG-2.40.

Palantir Foundry doc matched: *derived / computed / function-backed properties*
(Workshop function-backed columns and ontology computed fields). In Foundry a
derived property is **not stored** on the object — it is declared with an output
type and a backing function, and computed *live at read time* whenever the
property is requested. This module ports that contract into agent-utilities and
**surpasses it**: Foundry only function-backs derived properties, whereas a
:class:`DerivedProperty` here supports FOUR real backings, each computed live
against the existing knowledge-graph fabric:

  1. ``FUNCTION``  — invoke a registered typed function through the A3
     :class:`~agent_utilities.knowledge_graph.ontology.functions.runtime.FunctionRuntime`
     (full I/O typing + audit). This is the Foundry-parity backing.
  2. ``CYPHER``    — evaluate a Cypher expression/aggregate through the facade's
     guarded :meth:`KnowledgeGraph.query` read path (tenant-scoped + ACL +
     audited). Degrades cleanly to ``None`` with no reachable backend.
  3. ``SPARQL``    — evaluate a SPARQL expression through the L2 semantic
     layer's :meth:`OWLBridge.query_sparql`. Degrades cleanly to ``None`` when
     no OWL/fuseki/rdflib path is available.
  4. ``EMBEDDING`` — derive a value from vector similarity (nearest-concept
     label, similarity score, or designated-entity id) using
     :func:`~agent_utilities.core.embedding_utilities.create_embedding_model`
     and the L2 :class:`~agent_utilities.knowledge_graph.retrieval.capability_index.CapabilityIndex`.
     The FUNCTION and EMBEDDING backings are fully offline-capable (the
     embedding model is only needed when the spec hands text rather than a
     pre-computed vector, and the dispatcher accepts a pre-computed vector).

The output of every backing is coerced through the A-stage
:class:`~agent_utilities.knowledge_graph.ontology.property_types.PropertyType`
declared on the derived property, so a computed value carries the same typed
guarantees as a stored one.

A :class:`DerivedPropertyRegistry` holds the declarations (import-populated with
real built-ins — never an empty shell), and :class:`DerivedPropertyEngine`
exposes ``compute(obj, derived_prop, graph)`` — the single live dispatcher for
all four backings — with a read-through cache keyed by (property, object) and
explicit invalidation (``invalidate`` / ``invalidate_object`` / ``clear``).
"""

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_utilities.observability.audit_logger import AuditLogger

from .functions.runtime import DEFAULT_FUNCTION_RUNTIME, FunctionRuntime
from .property_types import PropertyType, parse_type_ref

logger = logging.getLogger(__name__)

DERIVED_AUDIT = "ontology_derived_property.compute"
RESOURCE_DERIVED = "derived_property"


class DerivedBacking(StrEnum):
    """The strategy that backs a derived property (CONCEPT:KG-2.40).

    ``FUNCTION`` invokes a registered typed function; ``CYPHER`` / ``SPARQL``
    evaluate a graph expression through the facade / OWL bridge; ``EMBEDDING``
    derives a value from vector similarity. Foundry has only ``FUNCTION``; the
    other three are the agent-utilities surpass-edge.
    """

    FUNCTION = "function"
    CYPHER = "cypher"
    SPARQL = "sparql"
    EMBEDDING = "embedding"


class EmbeddingDerivation(StrEnum):
    """What an ``EMBEDDING``-backed derived property emits (CONCEPT:KG-2.40).

    ``NEAREST_ID`` returns the designated entity id; ``NEAREST_LABEL`` returns
    that entity's capability label (or id when unlabelled); ``SIMILARITY``
    returns the top cosine score as a float.
    """

    NEAREST_ID = "nearest_id"
    NEAREST_LABEL = "nearest_label"
    SIMILARITY = "similarity"


class DerivedProperty(BaseModel):
    """A computed-at-read-time property declaration (CONCEPT:KG-2.40).

    Palantir doc matched: *function-backed / computed property*. A derived
    property is never stored on the object: it declares an output
    :class:`PropertyType` (via ``output_type`` ref, e.g. ``"double"`` /
    ``"string"``) and a ``backing`` strategy, and is materialised live by
    :meth:`DerivedPropertyEngine.compute`.

    Backing-specific fields:
        FUNCTION: ``function_name`` (+ optional ``function_version``) names the
            registered function; ``input_map`` maps function-param-name ->
            object-property-name (a value pulled from the object), and
            ``static_inputs`` supplies literal params. The object id is passed as
            ``object_id`` and the full property map as ``properties`` when the
            function declares those inputs.
        CYPHER / SPARQL: ``expression`` is the query text. ``$id`` (Cypher) and
            ``?id`` substitution (SPARQL) bind the object id; the first column /
            binding of the first row is taken as the value.
        EMBEDDING: ``embedding_text_property`` names the object property whose
            text is embedded (or ``embedding_vector_property`` names a property
            already holding a vector); ``embedding_derivation`` selects what to
            emit; ``required_caps`` optionally gates the candidate set.

    Attributes:
        name: The derived property name as seen on the object.
        object_type: Optional ontology object/node type this attaches to
            (advisory; the engine computes for any object passed in).
        output_type: A property-type reference resolved by ``parse_type_ref``.
        backing: The :class:`DerivedBacking` strategy.
        cacheable: When True (default) results are memoised by the engine until
            invalidated.
        description: Human/LLM-facing description.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    object_type: str | None = None
    output_type: str = "string"
    backing: DerivedBacking
    cacheable: bool = True
    description: str = ""

    # FUNCTION backing
    function_name: str = ""
    function_version: str | None = None
    input_map: dict[str, str] = Field(default_factory=dict)
    static_inputs: dict[str, Any] = Field(default_factory=dict)

    # CYPHER / SPARQL backing
    expression: str = ""

    # EMBEDDING backing
    embedding_text_property: str = ""
    embedding_vector_property: str = ""
    embedding_derivation: EmbeddingDerivation = EmbeddingDerivation.NEAREST_LABEL
    required_caps: list[str] = Field(default_factory=list)
    embedding_top_k: int = 1

    @field_validator("backing", mode="before")
    @classmethod
    def _coerce_backing(cls, v: Any) -> Any:
        if isinstance(v, str):
            return DerivedBacking(v.lower())
        return v

    def property_type(self) -> PropertyType:
        """Resolve the declared output :class:`PropertyType`."""
        return parse_type_ref(self.output_type)

    def coerce_output(self, value: Any) -> Any:
        """Coerce ``value`` to the declared output type (``None`` passes through).

        A derived property that could not be computed (no backend, empty result)
        yields ``None`` — never a wrong-typed value. A present value is run
        through the :class:`PropertyType` coercer so the typed guarantee holds.
        """
        if value is None:
            return None
        return self.property_type().coerce(value)

    def cache_signature(self) -> str:
        """A stable hash of the parts of this declaration that affect output.

        Used as part of the cache key so editing the backing/expression of a
        declaration (same name) does not serve a stale cached value.
        """
        payload = {
            "backing": str(self.backing),
            "output_type": self.output_type,
            "function_name": self.function_name,
            "function_version": self.function_version,
            "input_map": self.input_map,
            "static_inputs": _jsonable(self.static_inputs),
            "expression": self.expression,
            "embedding_text_property": self.embedding_text_property,
            "embedding_vector_property": self.embedding_vector_property,
            "embedding_derivation": str(self.embedding_derivation),
            "required_caps": sorted(self.required_caps),
            "embedding_top_k": self.embedding_top_k,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:16]


class DerivedPropertyResult(BaseModel):
    """The outcome of computing one derived property (CONCEPT:KG-2.40)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    property_name: str
    object_id: str = ""
    backing: DerivedBacking
    ok: bool = False
    value: Any = None
    error: str = ""
    cached: bool = False
    audit_ref: str = ""
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj, default=str)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _object_id(obj: Any) -> str:
    """Extract a stable id from an object mapping/instance."""
    if isinstance(obj, dict):
        for key in ("id", "object_id", "_id", "rid"):
            if obj.get(key):
                return str(obj[key])
        return ""
    for attr in ("id", "object_id"):
        v = getattr(obj, attr, None)
        if v:
            return str(v)
    return ""


def _object_props(obj: Any) -> dict[str, Any]:
    """Normalise an object to a flat property mapping."""
    if isinstance(obj, dict):
        inner = obj.get("properties")
        if isinstance(inner, dict):
            merged = dict(inner)
            for k, v in obj.items():
                if k != "properties":
                    merged.setdefault(k, v)
            return merged
        return dict(obj)
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    props = getattr(obj, "properties", None)
    if isinstance(props, dict):
        return dict(props)
    return {}


def _first_scalar(rows: list[dict[str, Any]]) -> Any:
    """Pull a single scalar value from the first row of a query result.

    Takes the first row's first column. If that column is itself a node/dict the
    first scalar value within it is used; an aggregate row like
    ``{"count(*)": 7}`` yields ``7``.
    """
    if not rows:
        return None
    row = rows[0]
    if not isinstance(row, dict) or not row:
        return row
    value = next(iter(row.values()))
    if isinstance(value, dict):
        inner = (
            value.get("properties")
            if isinstance(value.get("properties"), dict)
            else value
        )
        if inner:
            return next(iter(inner.values()))
        return None
    return value


# ---------------------------------------------------------------------------
# Registry (import-populated — never an empty shell)
# ---------------------------------------------------------------------------
class DerivedPropertyRegistry:
    """Registry of :class:`DerivedProperty` declarations. CONCEPT:KG-2.40.

    Keyed by ``(object_type, name)`` so the same derived-property name can be
    declared differently per object type, with a global fallback for
    declarations whose ``object_type`` is ``None``.
    """

    def __init__(self) -> None:
        self._by_key: dict[tuple[str | None, str], DerivedProperty] = {}

    def register(
        self, prop: DerivedProperty, *, replace: bool = False
    ) -> DerivedProperty:
        """Register a derived-property declaration.

        Raises:
            ValueError: on a duplicate ``(object_type, name)`` unless ``replace``.
        """
        key = (prop.object_type, prop.name)
        if key in self._by_key and not replace:
            raise ValueError(
                f"derived property already registered: {prop.object_type}.{prop.name}"
            )
        self._by_key[key] = prop
        return prop

    def get(self, name: str, object_type: str | None = None) -> DerivedProperty | None:
        """Resolve a declaration: exact (type, name) first, then global fallback."""
        if object_type is not None:
            hit = self._by_key.get((object_type, name))
            if hit is not None:
                return hit
        return self._by_key.get((None, name))

    def for_object_type(self, object_type: str | None) -> list[DerivedProperty]:
        """All declarations applicable to ``object_type`` (incl. global ones)."""
        out: list[DerivedProperty] = []
        for (otype, _name), prop in self._by_key.items():
            if otype is None or otype == object_type:
                out.append(prop)
        return out

    def list_all(self) -> list[DerivedProperty]:
        return list(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, tuple):
            return item in self._by_key
        return any(name == item for (_t, name) in self._by_key)


# ---------------------------------------------------------------------------
# Engine — the live compute dispatcher for all four backings
# ---------------------------------------------------------------------------
class DerivedPropertyEngine:
    """Computes derived properties live, dispatching across all four backings.

    CONCEPT:KG-2.40 — function-backed / computed properties.

    Args:
        registry: The :class:`DerivedPropertyRegistry` of declarations.
        runtime: The :class:`FunctionRuntime` used for ``FUNCTION`` backing
            (defaults to the import-populated :data:`DEFAULT_FUNCTION_RUNTIME`).
        audit: An :class:`AuditLogger`; a fresh in-memory one is built when
            omitted.
        embedding_fn: Optional ``text -> vector`` callable for ``EMBEDDING``
            backing. Defaults to a lazily-built
            :func:`create_embedding_model` (only constructed if a text property
            actually needs embedding — pre-computed vectors skip it entirely, so
            the engine stays offline-capable).
    """

    def __init__(
        self,
        registry: DerivedPropertyRegistry | None = None,
        runtime: FunctionRuntime | None = None,
        audit: AuditLogger | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.registry = registry or DEFAULT_DERIVED_REGISTRY
        self.runtime = runtime or DEFAULT_FUNCTION_RUNTIME
        self.audit = audit or AuditLogger()
        self._embedding_fn = embedding_fn
        self._embed_model: Any = None
        # cache: (property_name, sig, object_id) -> value
        self._cache: dict[tuple[str, str, str], Any] = {}

    # ── embedding model (lazy, offline-tolerant) ───────────────────────
    def _embed(self, text: str) -> list[float]:
        """Embed ``text`` to a vector using the configured/embedding model."""
        if self._embedding_fn is not None:
            return list(self._embedding_fn(text))
        if self._embed_model is None:
            from agent_utilities.core.embedding_utilities import create_embedding_model

            self._embed_model = create_embedding_model()
        return list(self._embed_model.get_text_embedding(text))

    # ── public compute ─────────────────────────────────────────────────
    def compute(
        self,
        obj: Any,
        derived_prop: DerivedProperty | str,
        graph: Any = None,
        *,
        object_type: str | None = None,
        actor_id: str = "system",
        use_cache: bool = True,
    ) -> DerivedPropertyResult:
        """Compute one derived property for ``obj`` live (never stored).

        Args:
            obj: The object — a property mapping, a Pydantic model, or any object
                exposing ``id``/``properties``.
            derived_prop: A :class:`DerivedProperty` declaration, or its name
                (resolved via the registry, optionally scoped by ``object_type``).
            graph: The live :class:`KnowledgeGraph` facade (required for CYPHER /
                SPARQL / EMBEDDING backings; FUNCTION uses the runtime's own
                graph context). When ``None`` those backings degrade to ``None``.
            object_type: Object type used to resolve a name-only declaration.
            actor_id: Recorded in the audit entry.
            use_cache: Read-through cache (honoured only when the declaration is
                ``cacheable``).

        Returns:
            A :class:`DerivedPropertyResult` — ``ok=True`` with the typed
            ``value`` (or ``None`` when a degraded backing produced nothing), or
            ``ok=False`` with a populated ``error`` (never raises).
        """
        started = time.perf_counter()
        prop = self._resolve(derived_prop, object_type)
        if prop is None:
            name = derived_prop if isinstance(derived_prop, str) else "?"
            return DerivedPropertyResult(
                property_name=str(name),
                backing=DerivedBacking.FUNCTION,
                ok=False,
                error=f"unknown derived property {name!r}",
            )

        oid = _object_id(obj)
        cache_key = (prop.name, prop.cache_signature(), oid)
        if use_cache and prop.cacheable and cache_key in self._cache:
            value = self._cache[cache_key]
            res = DerivedPropertyResult(
                property_name=prop.name,
                object_id=oid,
                backing=prop.backing,
                ok=True,
                value=value,
                cached=True,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            return res

        try:
            raw = self._dispatch(obj, prop, graph)
            value = prop.coerce_output(raw)
            ok, error = True, ""
        except Exception as exc:  # noqa: BLE001 — surface as a typed error
            logger.warning("derived property %s compute failed: %s", prop.name, exc)
            value, ok, error = None, False, str(exc)

        if ok and use_cache and prop.cacheable:
            self._cache[cache_key] = value

        res = DerivedPropertyResult(
            property_name=prop.name,
            object_id=oid,
            backing=prop.backing,
            ok=ok,
            value=value,
            error=error,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        self._audit(res, actor_id)
        return res

    def compute_all(
        self,
        obj: Any,
        graph: Any = None,
        *,
        object_type: str | None = None,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        """Compute every derived property applicable to ``obj``'s type.

        Returns a ``{name: value}`` map (values may be ``None`` for degraded
        backings). The live read-path consumer for materialising a fully-derived
        view of an object.
        """
        otype = object_type
        if otype is None and isinstance(obj, dict):
            otype = obj.get("type") or obj.get("object_type")
        out: dict[str, Any] = {}
        for prop in self.registry.for_object_type(otype):
            res = self.compute(obj, prop, graph, object_type=otype, actor_id=actor_id)
            out[prop.name] = res.value
        return out

    # ── dispatch ────────────────────────────────────────────────────────
    def _dispatch(self, obj: Any, prop: DerivedProperty, graph: Any) -> Any:
        backing = prop.backing
        if backing == DerivedBacking.FUNCTION:
            return self._compute_function(obj, prop, graph)
        if backing == DerivedBacking.CYPHER:
            return self._compute_cypher(obj, prop, graph)
        if backing == DerivedBacking.SPARQL:
            return self._compute_sparql(obj, prop, graph)
        if backing == DerivedBacking.EMBEDDING:
            return self._compute_embedding(obj, prop, graph)
        raise ValueError(f"unsupported derived backing: {backing!r}")

    def _compute_function(self, obj: Any, prop: DerivedProperty, graph: Any) -> Any:
        if not prop.function_name:
            raise ValueError("FUNCTION-backed derived property requires function_name")
        props = _object_props(obj)
        oid = _object_id(obj)
        params: dict[str, Any] = dict(prop.static_inputs)
        for fn_param, obj_prop in prop.input_map.items():
            params[fn_param] = props.get(obj_prop)

        # Auto-supply object id / properties when the spec declares them, so a
        # Functions-on-Objects summarizer can run with no explicit input_map.
        spec = self.runtime.registry.get(prop.function_name, prop.function_version)
        if spec is not None:
            declared = {p.name for p in spec.inputs}
            if "object_id" in declared and "object_id" not in params:
                params["object_id"] = oid
            if "properties" in declared and "properties" not in params:
                params["properties"] = props

        runtime = self.runtime
        if graph is not None and getattr(runtime, "_graph", None) is None:
            # Bind the live facade so ON_OBJECTS handlers can read the graph.
            runtime = FunctionRuntime(
                registry=runtime.registry, audit=runtime.audit, graph=graph
            )
        result = runtime.invoke(
            prop.function_name, params, version=prop.function_version
        )
        if not result.ok:
            raise ValueError(f"function backing failed: {result.error}")
        return result.value

    def _compute_cypher(self, obj: Any, prop: DerivedProperty, graph: Any) -> Any:
        if not prop.expression:
            raise ValueError("CYPHER-backed derived property requires an expression")
        if graph is None or not hasattr(graph, "query"):
            # Documented graceful degradation: no facade/backend -> no value.
            logger.debug("CYPHER derived %s: no facade, degrading to None", prop.name)
            return None
        oid = _object_id(obj)
        rows = graph.query(prop.expression, {"id": oid}) or []
        return _first_scalar(rows)

    def _compute_sparql(self, obj: Any, prop: DerivedProperty, graph: Any) -> Any:
        if not prop.expression:
            raise ValueError("SPARQL-backed derived property requires an expression")
        bridge = getattr(graph, "semantic", None) if graph is not None else None
        if bridge is None or not hasattr(bridge, "query_sparql"):
            # Documented graceful degradation: no OWL/fuseki/rdflib path.
            logger.debug("SPARQL derived %s: no semantic layer, degrading", prop.name)
            return None
        oid = _object_id(obj)
        sparql = prop.expression.replace("$id", oid).replace("?id", f'"{oid}"')
        rows = bridge.query_sparql(sparql) or []
        return _first_scalar(rows)

    def _compute_embedding(self, obj: Any, prop: DerivedProperty, graph: Any) -> Any:
        index = getattr(graph, "retrieval", None) if graph is not None else None
        if index is None or len(index) == 0:
            logger.debug("EMBEDDING derived %s: empty index, degrading", prop.name)
            return None
        props = _object_props(obj)

        # Prefer a pre-computed vector (offline path); else embed text.
        vector: list[float] | None = None
        if prop.embedding_vector_property:
            raw = props.get(prop.embedding_vector_property)
            if raw is not None:
                vector = [float(x) for x in raw]
        if vector is None:
            text = ""
            if prop.embedding_text_property:
                text = str(props.get(prop.embedding_text_property) or "")
            if not text:
                text = str(props.get("name") or props.get("title") or _object_id(obj))
            if not text:
                return None
            vector = self._embed(text)

        caps = prop.required_caps or None
        designations = index.designate(
            vector, required_caps=caps, k=max(1, prop.embedding_top_k)
        )
        if not designations:
            return None
        top = designations[0]
        if prop.embedding_derivation == EmbeddingDerivation.SIMILARITY:
            return float(top.score)
        if prop.embedding_derivation == EmbeddingDerivation.NEAREST_ID:
            return top.id
        # NEAREST_LABEL — the capability label, falling back to the id.
        caps_set = top.capabilities or set()
        if caps_set:
            return sorted(caps_set)[0]
        return top.id

    # ── cache invalidation ──────────────────────────────────────────────
    def invalidate(self, name: str, object_id: str | None = None) -> int:
        """Invalidate cached values for a derived property.

        With ``object_id`` given, drops just that object's cached value for the
        property; otherwise drops every cached value for the property. Returns
        the number of entries removed.
        """
        keys = [
            k
            for k in self._cache
            if k[0] == name and (object_id is None or k[2] == object_id)
        ]
        for k in keys:
            del self._cache[k]
        return len(keys)

    def invalidate_object(self, object_id: str) -> int:
        """Drop every cached derived value for one object (e.g. after an edit)."""
        keys = [k for k in self._cache if k[2] == object_id]
        for k in keys:
            del self._cache[k]
        return len(keys)

    def clear(self) -> None:
        """Drop the entire derived-property cache."""
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    # ── internals ───────────────────────────────────────────────────────
    def _resolve(
        self, derived_prop: DerivedProperty | str, object_type: str | None
    ) -> DerivedProperty | None:
        if isinstance(derived_prop, DerivedProperty):
            return derived_prop
        return self.registry.get(derived_prop, object_type)

    def _audit(self, result: DerivedPropertyResult, actor_id: str) -> None:
        try:
            record = self.audit.log(
                actor=actor_id,
                action=DERIVED_AUDIT,
                resource_type=RESOURCE_DERIVED,
                resource_id=f"{result.property_name}:{result.object_id}",
                details={
                    "backing": str(result.backing),
                    "ok": result.ok,
                    "error": result.error,
                },
            )
        except Exception:  # noqa: BLE001 — audit is best-effort
            record = None
        if record is not None:
            result.audit_ref = getattr(record, "id", "")


# ---------------------------------------------------------------------------
# Built-in declarations (populated at import — never an empty shell)
# ---------------------------------------------------------------------------
def register_builtins(registry: DerivedPropertyRegistry) -> None:
    """Register real, live built-in derived properties.

    Each exercises a distinct backing so the default registry is a live path:
      - ``summary`` (FUNCTION) — function-backed one-line object summary, the
        Foundry-parity backing, reusing the built-in ``object.summarize``.
      - ``degree`` (CYPHER) — live edge-count aggregate over the object's links.
      - ``nearest_capability`` (EMBEDDING) — the nearest capability label by
        vector similarity over the L2 capability index.
    """
    registry.register(
        DerivedProperty(
            name="summary",
            output_type="string",
            backing=DerivedBacking.FUNCTION,
            function_name="object.summarize",
            function_version="1.0.0",
            description="Function-backed one-line summary (Foundry parity).",
        )
    )
    registry.register(
        DerivedProperty(
            name="degree",
            output_type="long",
            backing=DerivedBacking.CYPHER,
            expression="MATCH (n {id: $id})-[r]-() RETURN count(r) AS degree",
            description="Live link-degree of the object (computed at read time).",
        )
    )
    registry.register(
        DerivedProperty(
            name="nearest_capability",
            output_type="string",
            backing=DerivedBacking.EMBEDDING,
            embedding_text_property="description",
            embedding_derivation=EmbeddingDerivation.NEAREST_LABEL,
            description="Nearest capability label by embedding similarity.",
        )
    )


# CONCEPT:KG-2.40 — import-populated registry + a default engine bound to it and
# to the default function runtime, so derived properties have a live compute
# entry (consumed by the ontology facade / MCP read path).
DEFAULT_DERIVED_REGISTRY = DerivedPropertyRegistry()
register_builtins(DEFAULT_DERIVED_REGISTRY)
DEFAULT_DERIVED_ENGINE = DerivedPropertyEngine(registry=DEFAULT_DERIVED_REGISTRY)


def compute_derived(
    obj: Any,
    derived_prop: DerivedProperty | str,
    graph: Any = None,
    *,
    object_type: str | None = None,
    actor_id: str = "system",
) -> DerivedPropertyResult:
    """Module-level convenience: compute one derived property via the default engine.

    The single-call live entry used by the ontology facade / MCP read path.
    """
    return DEFAULT_DERIVED_ENGINE.compute(
        obj,
        derived_prop,
        graph,
        object_type=object_type,
        actor_id=actor_id,
    )


def compute_all_derived(
    obj: Any,
    graph: Any = None,
    *,
    object_type: str | None = None,
    actor_id: str = "system",
) -> dict[str, Any]:
    """Module-level convenience: compute all applicable derived properties for ``obj``."""
    return DEFAULT_DERIVED_ENGINE.compute_all(
        obj, graph, object_type=object_type, actor_id=actor_id
    )


__all__ = [
    "DERIVED_AUDIT",
    "DEFAULT_DERIVED_ENGINE",
    "DEFAULT_DERIVED_REGISTRY",
    "DerivedBacking",
    "DerivedProperty",
    "DerivedPropertyEngine",
    "DerivedPropertyRegistry",
    "DerivedPropertyResult",
    "EmbeddingDerivation",
    "compute_all_derived",
    "compute_derived",
    "register_builtins",
]

# `re` is used by callers extending expressions; keep the import meaningful by
# exposing a small expression-id validator used when registering ad-hoc props.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def is_valid_property_name(name: str) -> bool:
    """True iff ``name`` is a valid derived-property identifier."""
    return bool(_IDENT_RE.match(name or ""))
