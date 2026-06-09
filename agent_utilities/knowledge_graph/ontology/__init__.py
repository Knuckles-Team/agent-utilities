#!/usr/bin/python
from __future__ import annotations

"""Ontology layer — the Palantir-Foundry-parity type/link/function system.

This package composes the first-class ontology primitives the execution plane
reaches through :class:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph`:

  - **Property types** (CONCEPT:KG-2.47, ``property_types``) — the Palantir-style
    type vocabulary (scalars, geo, vector/embedding, array/struct) that drives
    node-table column DDL and ontology write-path coercion.
  - **Value types** (CONCEPT:KG-2.39, ``value_types``) — constrained semantic
    types (EmailAddress, Percentage, …) that compile to SHACL/OWL and gate
    writes via the SHACL validator.
  - **Interfaces** (CONCEPT:KG-2.38, ``interfaces``) — abstract shapes a concrete
    object type can implement; the programmatic-targeting resolver expands an
    interface name to its implementing types.
  - **Links** (CONCEPT:KG-2.26, ``links``) — first-class typed links and
    many-to-many junction reification onto the existing graph-write path.
  - **Functions** (CONCEPT:KG-2.41, ``functions``) — typed, versioned, governed
    user functions (PLAIN | ON_OBJECTS | QUERY) with a single audited runtime.
  - **Derived properties** (CONCEPT:KG-2.40, ``derived_properties``) — computed
    read-time properties dispatched across FUNCTION/CYPHER/SPARQL/EMBEDDING.

:class:`OntologySystem` binds these registries together and (optionally) to a
live :class:`KnowledgeGraph` facade, so Functions-on-Objects, derived-property
compute, and interface targeting resolve against the real store/semantic/
retrieval layers. It is the object :pyattr:`KnowledgeGraph.ontology` returns.

The composition is **import-populated, never an empty shell**: every registry it
exposes ships real built-ins at import (the base/complex/geo/vector property
types, six constrained value types, three built-in interfaces with live
implementers, two built-in link types, two released built-in functions, and
three built-in derived properties).
"""

from typing import Any

from .derived_properties import (
    DEFAULT_DERIVED_ENGINE,
    DEFAULT_DERIVED_REGISTRY,
    DerivedBacking,
    DerivedProperty,
    DerivedPropertyEngine,
    DerivedPropertyRegistry,
    DerivedPropertyResult,
    EmbeddingDerivation,
    compute_all_derived,
    compute_derived,
)
from .document_processing import (
    CHUNK_NODE_TYPE,
    CHUNK_OF_EDGE,
    DEFAULT_EMBEDDING_DIM,
    DOCUMENT_NODE_TYPE,
    HAS_CHUNK_EDGE,
    ChunkingConfig,
    ChunkSpan,
    DocumentChunk,
    DocumentExtractionError,
    DocumentProcessor,
    ProcessedDocument,
    chunk_text,
    process_document,
)
from .edits import (
    EDIT_NODE_TYPE,
    Edit,
    EditLedger,
    EditSink,
    EditType,
    JsonlEditSink,
    WriteBackRouter,
    invert_edit,
    object_type_of,
    revert_edit,
    revert_edits,
    revert_object,
)
from .functions import (
    DEFAULT_FUNCTION_REGISTRY,
    DEFAULT_FUNCTION_RUNTIME,
    FunctionKind,
    FunctionParameter,
    FunctionRegistry,
    FunctionResult,
    FunctionRuntime,
    FunctionSpec,
    ObjectFunctionContext,
)
from .indexing import (
    DataRestriction,
    FunnelDelta,
    ObjectIndexFunnel,
    ObjectVersion,
    StalenessLedger,
    StalenessReport,
    SyncResult,
    content_hash,
)
from .interfaces import (
    DEFAULT_INTERFACE_REGISTRY,
    ImplementationReport,
    Interface,
    InterfaceLinkConstraint,
    InterfaceProperty,
    InterfaceRegistry,
    register_builtin_interfaces,
    target_object_types,
)
from .links import (
    DEFAULT_LINK_REGISTRY,
    JunctionLinkType,
    LinkCardinality,
    LinkType,
    LinkTypeRegistry,
    endpoints_of,
    is_junction_node,
    junctions_for,
    neighbors_via,
    register_builtin_links,
)
from .object_set import (
    DEFAULT_SEARCH_AROUND_CAP,
    AggregationResult,
    GraphView,
    ObjectSet,
    ObjectSetKind,
    PivotResult,
    Predicate,
    PropertyFilter,
    dynamic_object_set,
    object_set_from_ids,
    object_set_of_type,
)
from .permissioning import (
    MARKING_REGISTRY,
    MASK_TOKEN,
    Marking,
    apply_marking,
    build_acl,
    enforce,
    markings_for,
    propagate_markings,
    propagate_over_edges,
    redact_object,
    restricted_view,
)
from .property_types import (
    DEFAULT_VECTOR_DIM,
    PROPERTY_TYPES,
    PropertyType,
    coerce_value,
    column_type_for,
    get_property_type,
    list_property_types,
    parse_type_ref,
    validate_value,
)
from .value_types import (
    SHAPES_PREFIXES,
    VALUE_TYPE_NS,
    VALUE_TYPES,
    ValueConstraints,
    ValueType,
    coerce_value_type,
    get_value_type,
    list_value_types,
    register_value_type,
    validate_value_type,
    value_types_owl_ttl,
    value_types_shapes_ttl,
    write_value_shapes_ttl,
)


class OntologySystem:
    """Composition root over the ontology layer (Palantir Foundry parity).

    Binds the six import-populated registries — property types, value types,
    interfaces, links, functions, derived properties — into one object the
    :class:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph` exposes as
    :pyattr:`KnowledgeGraph.ontology`.

    When constructed with a live ``graph`` facade, the Functions runtime and
    derived-property compute resolve against that graph's store/semantic/
    retrieval layers (Functions-on-Objects, CYPHER/SPARQL/EMBEDDING backings).
    Constructing without a graph still yields a fully usable system (offline
    functions, type/value validation, interface targeting).

    Args:
        graph: Optional live :class:`KnowledgeGraph` facade. Passing it makes the
            object-aware paths read the real graph; omitting it keeps the system
            usable for pure type/link/function operations.
    """

    def __init__(self, graph: Any = None) -> None:
        self._graph = graph
        # Shared import-populated registries (the live discovery surfaces).
        self.property_types: dict[str, PropertyType] = PROPERTY_TYPES
        self.value_types: dict[str, ValueType] = VALUE_TYPES
        self.interfaces: InterfaceRegistry = DEFAULT_INTERFACE_REGISTRY
        self.links: LinkTypeRegistry = DEFAULT_LINK_REGISTRY
        self.function_registry: FunctionRegistry = DEFAULT_FUNCTION_REGISTRY
        self.derived_registry: DerivedPropertyRegistry = DEFAULT_DERIVED_REGISTRY
        # A runtime bound to THIS graph so ON_OBJECTS functions read the live store.
        self.functions: FunctionRuntime = (
            FunctionRuntime(graph=graph)
            if graph is not None
            else DEFAULT_FUNCTION_RUNTIME
        )
        self.derived: DerivedPropertyEngine = DEFAULT_DERIVED_ENGINE
        # Durable edit ledger (CONCEPT:KG-2.43) — bound to the live graph so
        # recorded edits persist as durable object_edit nodes + EDITED_BY/EDITS
        # edges and revert is itself durable + audited.
        self.edits: EditLedger = EditLedger(graph=graph)
        # Object Index Funnel (CONCEPT:KG-2.44) — drives the SAME live search
        # index the router ranks against (graph.retrieval), never a second one.
        self.index_funnel: ObjectIndexFunnel = ObjectIndexFunnel(
            index=getattr(graph, "retrieval", None) if graph is not None else None
        )

    # ── Property / value type validation (write-path coercion) ───────────────
    def column_type_for(self, type_ref: str) -> str:
        """Map an ontology property type ref to a node-table column-type string."""
        return column_type_for(type_ref)

    def coerce_property(self, type_ref: str, value: Any) -> Any:
        """Coerce ``value`` to the property type named by ``type_ref``."""
        return coerce_value(type_ref, value)

    def validate_property(self, type_ref: str, value: Any) -> bool:
        """Whether ``value`` is valid for the property type ``type_ref``."""
        return validate_value(type_ref, value)

    def validate_value(self, value_type_name: str, value: Any) -> bool:
        """Whether ``value`` satisfies the named constrained value type."""
        return validate_value_type(value_type_name, value)

    def coerce_value(self, value_type_name: str, value: Any) -> Any:
        """Coerce + constrain ``value`` through the named value type."""
        return coerce_value_type(value_type_name, value)

    # ── Functions ────────────────────────────────────────────────────────────
    def invoke_function(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        version: str | None = None,
        *,
        actor_id: str = "system",
    ) -> FunctionResult:
        """Validate, run, coerce, and audit a typed function via the bound runtime."""
        return self.functions.invoke(name, params, version, actor_id=actor_id)

    # ── Derived properties ───────────────────────────────────────────────────
    def derive(
        self,
        obj: Any,
        name: str,
        *,
        object_type: str | None = None,
        actor_id: str = "system",
    ) -> DerivedPropertyResult:
        """Compute one derived property for ``obj`` against the bound graph."""
        return compute_derived(
            obj, name, self._graph, object_type=object_type, actor_id=actor_id
        )

    def derive_all(
        self,
        obj: Any,
        *,
        object_type: str | None = None,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        """Compute all derived properties applicable to ``obj``'s type."""
        return compute_all_derived(
            obj, self._graph, object_type=object_type, actor_id=actor_id
        )

    # ── Interfaces (programmatic targeting) ──────────────────────────────────
    def resolve_target(self, type_or_interface: str) -> list[str]:
        """Expand an interface name to its implementing object types (else pass-through)."""
        return self.interfaces.resolve_target(type_or_interface)

    def conforms(self, object_dict: dict[str, Any], interface: str) -> bool:
        """Whether a concrete object dict satisfies a named interface's shape."""
        return self.interfaces.conforms(object_dict, interface)

    # ── Links (junction reification) ─────────────────────────────────────────
    def materialize_link(
        self,
        link_name: str,
        source_id: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, Any]:
        """Reify a M:N link as a ``(junction_node, edge_a, edge_b)`` write triple.

        Raises:
            KeyError: if ``link_name`` is not a registered junction link type.
        """
        link = self.links.get(link_name)
        if link is None or not isinstance(link, JunctionLinkType):
            raise KeyError(f"no junction link type registered as {link_name!r}")
        return link.materialize_junction(source_id, target_id, properties)

    # ── Durable edit ledger (CONCEPT:KG-2.43) ────────────────────────────────
    def record_edit(self, edit: Edit) -> Edit:
        """Apply + durably persist a structured object edit via the ledger."""
        return self.edits.record(edit)

    def set_property_edit(
        self,
        object_id: str,
        properties: dict[str, Any],
        *,
        actor: str = "system",
        provenance: str = "",
        invocation_ref: str = "",
    ) -> Edit:
        """Record a PROPERTY_SET edit (snapshotting prior values it overwrites)."""
        return self.edits.set_property(
            object_id,
            properties,
            actor=actor,
            provenance=provenance,
            invocation_ref=invocation_ref,
        )

    def history(self, object_id: str) -> list[Edit]:
        """Per-object edit history (oldest first) — CONCEPT:KG-2.43."""
        return self.edits.history(object_id)

    def as_of(self, object_id: str, ts: float) -> dict[str, Any] | None:
        """Reconstruct an object's property snapshot as of ``ts``."""
        return self.edits.as_of(object_id, ts)

    def revert_edit(self, edit_id: str, *, actor: str = "system") -> Edit:
        """Undo a recorded edit, recording a durable compensating edit."""
        return revert_edit(self.edits, edit_id, actor=actor)

    # ── Object sets (CONCEPT:KG-2.45 / KG-2.38) ──────────────────────────────
    def object_set(self, ids: Any) -> ObjectSet:
        """A STATIC object set over fixed ids, bound to the live graph."""
        return object_set_from_ids(self._graph, ids)

    def object_set_of_type(self, type_or_interface: str) -> ObjectSet:
        """A DYNAMIC object set of all live objects of a type/interface."""
        return object_set_of_type(self._graph, type_or_interface)

    def dynamic_object_set(
        self, predicate: Any = None, *, filters: Any = None
    ) -> ObjectSet:
        """A DYNAMIC object set from a predicate / typed filters."""
        return dynamic_object_set(self._graph, predicate, filters=filters)

    # ── Document processing (CONCEPT:KG-2.48) ────────────────────────────────
    def process_document(self, document: Any, **kwargs: Any) -> dict[str, Any]:
        """Process a document into Document + Chunk objects through the graph."""
        return process_document(document, self._graph, **kwargs)


def build_ontology_system(graph: Any = None) -> OntologySystem:
    """Construct an :class:`OntologySystem` bound to an optional live graph facade."""
    return OntologySystem(graph=graph)


__all__ = [
    # System facade
    "OntologySystem",
    "build_ontology_system",
    # Property types (KG-2.47)
    "PropertyType",
    "PROPERTY_TYPES",
    "DEFAULT_VECTOR_DIM",
    "get_property_type",
    "parse_type_ref",
    "list_property_types",
    "column_type_for",
    "coerce_value",
    "validate_value",
    # Value types (KG-2.39)
    "ValueType",
    "ValueConstraints",
    "VALUE_TYPES",
    "get_value_type",
    "register_value_type",
    "list_value_types",
    "coerce_value_type",
    "validate_value_type",
    "value_types_shapes_ttl",
    "value_types_owl_ttl",
    "write_value_shapes_ttl",
    "SHAPES_PREFIXES",
    "VALUE_TYPE_NS",
    # Interfaces (KG-2.38)
    "Interface",
    "InterfaceProperty",
    "InterfaceLinkConstraint",
    "ImplementationReport",
    "InterfaceRegistry",
    "DEFAULT_INTERFACE_REGISTRY",
    "register_builtin_interfaces",
    "target_object_types",
    # Links (KG-2.26)
    "LinkCardinality",
    "LinkType",
    "JunctionLinkType",
    "LinkTypeRegistry",
    "DEFAULT_LINK_REGISTRY",
    "register_builtin_links",
    "is_junction_node",
    "endpoints_of",
    "junctions_for",
    "neighbors_via",
    # Functions (KG-2.41)
    "FunctionSpec",
    "FunctionParameter",
    "FunctionKind",
    "FunctionRegistry",
    "FunctionRuntime",
    "FunctionResult",
    "ObjectFunctionContext",
    "DEFAULT_FUNCTION_REGISTRY",
    "DEFAULT_FUNCTION_RUNTIME",
    # Derived properties (KG-2.40)
    "DerivedProperty",
    "DerivedBacking",
    "EmbeddingDerivation",
    "DerivedPropertyRegistry",
    "DerivedPropertyEngine",
    "DerivedPropertyResult",
    "DEFAULT_DERIVED_REGISTRY",
    "DEFAULT_DERIVED_ENGINE",
    "compute_derived",
    "compute_all_derived",
    # Durable edit ledger (KG-2.43)
    "Edit",
    "EditType",
    "EditLedger",
    "EDIT_NODE_TYPE",
    "EditSink",
    "JsonlEditSink",
    "WriteBackRouter",
    "invert_edit",
    "revert_edit",
    "revert_edits",
    "revert_object",
    "object_type_of",
    # Object Index Lifecycle (KG-2.44)
    "ObjectIndexFunnel",
    "DataRestriction",
    "FunnelDelta",
    "SyncResult",
    "StalenessLedger",
    "StalenessReport",
    "ObjectVersion",
    "content_hash",
    # Object permissioning (KG-2.46)
    "Marking",
    "MARKING_REGISTRY",
    "MASK_TOKEN",
    "apply_marking",
    "markings_for",
    "redact_object",
    "restricted_view",
    "enforce",
    "build_acl",
    "propagate_markings",
    "propagate_over_edges",
    # Object sets (KG-2.45 / KG-2.38)
    "ObjectSet",
    "ObjectSetKind",
    "Predicate",
    "PropertyFilter",
    "AggregationResult",
    "PivotResult",
    "GraphView",
    "object_set_from_ids",
    "object_set_of_type",
    "dynamic_object_set",
    "DEFAULT_SEARCH_AROUND_CAP",
    # Document processing (KG-2.48)
    "ChunkingConfig",
    "ChunkSpan",
    "chunk_text",
    "DocumentChunk",
    "ProcessedDocument",
    "DocumentExtractionError",
    "DocumentProcessor",
    "process_document",
    "DEFAULT_EMBEDDING_DIM",
    "DOCUMENT_NODE_TYPE",
    "CHUNK_NODE_TYPE",
    "HAS_CHUNK_EDGE",
    "CHUNK_OF_EDGE",
]
