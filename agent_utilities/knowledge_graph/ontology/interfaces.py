#!/usr/bin/python
from __future__ import annotations

"""Ontology Interfaces — abstract shape contracts implemented by object types.

CONCEPT:KG-2.38 — Ontology Interfaces.

Palantir Foundry doc matched: *interfaces/interface-overview*. A Foundry
**interface** is an *abstract* schema element — it has no instances of its own.
It declares a **shape**: a set of *shared properties* (interface-properties) and
*shared link constraints* every implementing object type must satisfy. Object
types declare which interfaces they **implement** (and may implement several);
interfaces can **extend** other interfaces (single- and multi-inheritance), so a
sub-interface inherits its parents' required properties and link constraints.
Crucially, Functions, Actions and object queries can be **programmatically
targeted at an interface** rather than a concrete type — at runtime the platform
resolves the interface to the set of concrete object types that implement it.

This module ports that abstract-contract layer into agent-utilities, reusing the
existing fabric rather than reinventing it:

  - :class:`InterfaceProperty` — a required interface-property, typed by a real
    :class:`~agent_utilities.knowledge_graph.ontology.property_types.PropertyType`
    (Stage-A property vocabulary), so an object type "satisfies" the property
    only when its declared/observed value *coerces* to that type.
  - :class:`InterfaceLinkConstraint` — a required shared link: the implementer
    must expose a link of a given :class:`RegistryEdgeType` (optionally to a
    target object/interface type, with a minimum cardinality). Mirrors a
    Foundry interface *link constraint*.
  - :class:`Interface` — the abstract shape: required properties + link
    constraints + ``extends`` (multi-extend) parents. ``to_owl`` emits it as an
    ``owl:Class`` plus a SHACL ``sh:NodeShape`` reusing the ``owl_bridge``
    namespace/IRI conventions, so an implementing type becomes
    ``rdfs:subClassOf`` the interface class and ``sh:node`` its shape.
  - :class:`InterfaceRegistry` — import-populated with real built-in interfaces
    (``HasProvenance`` and ``Locatable``), never an empty shell.
  - :func:`InterfaceRegistry.implement` — records that an object/node type
    implements an interface and **validates the contract** (collecting the
    missing properties / unsatisfied link constraints as *gaps*), returning an
    :class:`ImplementationReport`.
  - :func:`InterfaceRegistry.conforms` / :func:`InterfaceRegistry.find_implementers`
    — the *programmatic-targeting* feature: a Function/Action/query that names an
    interface resolves, via ``find_implementers``, to the concrete object types
    that implement it, and ``conforms`` checks a concrete object dict against the
    interface's shape at runtime.

The registry follows the import-populated-registry idiom (mirroring
``actions.registry`` / ``ontology.links``): :data:`DEFAULT_INTERFACE_REGISTRY`
is populated at import with live built-ins.
"""

import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from .property_types import PropertyType, parse_type_ref

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .functions.registry import FunctionRegistry

# OWL/SHACL namespaces — identical bindings to the ontology *.ttl files and the
# owl_bridge RDF materialization, so interface classes/shapes resolve in the same
# graph (CONCEPT:KG-2.38).
KG = "http://knuckles.team/kg#"
OWL = "http://www.w3.org/2002/07/owl#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
SH = "http://www.w3.org/ns/shacl#"
XSD = "http://www.w3.org/2001/XMLSchema#"


def _camel(name: str) -> str:
    """Turn ``has_provenance``/``hasProvenance`` into a ``HasProvenance`` IRI local.

    Matches the casing ``owl_bridge._build_rdf_graph`` uses for type classes
    (``.title().replace("_", "")``) so an interface OWL class lines up with how
    node-type classes are minted elsewhere.
    """
    return name.replace(" ", "_").title().replace("_", "")


class InterfaceProperty(BaseModel):
    """A required interface-property: a name plus the :class:`PropertyType` it must satisfy.

    CONCEPT:KG-2.38 — a Foundry interface *shared property*. An implementing
    object type satisfies this when it carries a property ``name`` whose value
    coerces to ``type_ref`` (via Stage-A :func:`parse_type_ref`). Optional
    properties (``required=False``) participate in the shape (and OWL/SHACL
    emission) but never produce a *gap* when absent.

    Attributes:
        name: The shared property name implementers must expose (e.g. ``"lat"``).
        type_ref: A Palantir-style property-type reference resolved by
            :func:`~agent_utilities.knowledge_graph.ontology.property_types.parse_type_ref`
            (e.g. ``"double"``, ``"timestamp"``, ``"array<string>"``).
        required: Whether the property is mandatory for conformance (default True).
        description: Human/LLM-facing description of the property's contract.
    """

    name: str
    type_ref: str
    required: bool = True
    description: str = ""

    def property_type(self) -> PropertyType:
        """Resolve and return the underlying :class:`PropertyType` (Stage-A)."""
        return parse_type_ref(self.type_ref)

    def is_satisfied_by_value(self, value: Any) -> bool:
        """Whether ``value`` satisfies this property's typed contract.

        A missing value (``None``) satisfies an *optional* property and fails a
        *required* one; a present value must coerce to the declared
        :class:`PropertyType`.
        """
        if value is None:
            return not self.required
        return self.property_type().validate(value)


class InterfaceLinkConstraint(BaseModel):
    """A required shared link the implementer must expose. CONCEPT:KG-2.38.

    Mirrors a Foundry interface *link constraint*: every implementing object type
    must declare a link of ``edge_type`` (optionally to a specific
    ``target_type`` object/interface) with at least ``min_count`` cardinality.
    Conformance is checked against an object's declared link set (the
    ``links``/``edges`` portion of an object dict, or its outgoing edge types).

    Attributes:
        name: Constraint name (the local key within the interface).
        edge_type: The :class:`RegistryEdgeType` the implementer must support.
        target_type: Optional required target object type (a
            :class:`RegistryNodeType`) or the *name* of another interface the
            target must implement.
        min_count: Minimum number of such links a conforming object must have
            (``0`` = "the type must declare the link, instances need not have
            one"; ``>=1`` = "instances must carry at least N").
        description: Human/LLM-facing description of the link contract.
    """

    name: str
    edge_type: RegistryEdgeType
    target_type: RegistryNodeType | str | None = None
    min_count: int = 1
    description: str = ""


class Interface(BaseModel):
    """An abstract ontology interface — a shape contract, never instantiated.

    CONCEPT:KG-2.38 — a Foundry *interface*. Declares the shared properties and
    link constraints that any implementing object type must satisfy, plus the
    parent interfaces it ``extends`` (single- or multi-inheritance). Because an
    interface is abstract it has no own instances; instead concrete object types
    *implement* it, and Functions/Actions/queries target it (resolving to the
    implementing types at runtime).

    Attributes:
        name: Unique interface name (the registry key), e.g. ``"HasProvenance"``.
        properties: The directly-declared required interface-properties.
        link_constraints: The directly-declared required shared link constraints.
        extends: Names of parent interfaces this one extends (multi-extend).
        description: Human/LLM-facing description of the contract.
    """

    model_config = ConfigDict(frozen=False)

    name: str
    properties: list[InterfaceProperty] = Field(default_factory=list)
    link_constraints: list[InterfaceLinkConstraint] = Field(default_factory=list)
    extends: list[str] = Field(default_factory=list)
    description: str = ""

    # ── Effective (inheritance-resolved) shape ─────────────────────────────

    def all_properties(
        self, registry: InterfaceRegistry | None = None
    ) -> dict[str, InterfaceProperty]:
        """Return the effective property map including inherited ones.

        CONCEPT:KG-2.38 — interface *inheritance*: a sub-interface's shape is the
        union of its own properties and every ancestor's. A locally-declared
        property overrides an inherited one of the same name (sub-interface
        refinement). Resolution is cycle-safe.
        """
        resolved: dict[str, InterfaceProperty] = {}
        for parent in self._iter_ancestors(registry):
            for p in parent.properties:
                resolved[p.name] = p
        for p in self.properties:
            resolved[p.name] = p
        return resolved

    def all_link_constraints(
        self, registry: InterfaceRegistry | None = None
    ) -> dict[str, InterfaceLinkConstraint]:
        """Return the effective link-constraint map including inherited ones."""
        resolved: dict[str, InterfaceLinkConstraint] = {}
        for parent in self._iter_ancestors(registry):
            for c in parent.link_constraints:
                resolved[c.name] = c
        for c in self.link_constraints:
            resolved[c.name] = c
        return resolved

    def _iter_ancestors(
        self, registry: InterfaceRegistry | None
    ) -> list[Interface]:
        """Return ancestor interfaces (parents first), cycle-safe.

        Without a registry, an interface has no resolvable parents (its declared
        shape is taken as-is); with one, ``extends`` names are resolved through
        it. Excludes ``self``.
        """
        if registry is None or not self.extends:
            return []
        seen: set[str] = {self.name}
        ordered: list[Interface] = []

        def visit(iface_name: str) -> None:
            if iface_name in seen:
                return
            seen.add(iface_name)
            parent = registry.get(iface_name)
            if parent is None:
                return
            # Depth-first: grandparents before parents so nearer overrides win.
            for grand in parent.extends:
                visit(grand)
            ordered.append(parent)

        for name in self.extends:
            visit(name)
        return ordered

    # ── Conformance ────────────────────────────────────────────────────────

    def gaps_for(
        self,
        object_dict: dict[str, Any],
        *,
        registry: InterfaceRegistry | None = None,
    ) -> list[str]:
        """Return the list of unmet contract requirements for ``object_dict``.

        CONCEPT:KG-2.38 — the conformance check. ``object_dict`` is a concrete
        object's property map (with an optional ``links``/``edges`` collection or
        ``link_types`` set describing its outgoing edge types). An empty list
        means the object satisfies the (inheritance-resolved) interface shape.

        Property gaps: a required interface-property is missing or holds a value
        that does not coerce to its :class:`PropertyType`. Link gaps: a required
        link constraint's edge type is absent from the object's declared links
        (or present fewer than ``min_count`` times).
        """
        gaps: list[str] = []
        props = self.all_properties(registry)
        for name, iface_prop in props.items():
            if iface_prop.required and name not in object_dict:
                gaps.append(f"missing required property '{name}' ({iface_prop.type_ref})")
                continue
            if name in object_dict and not iface_prop.is_satisfied_by_value(
                object_dict[name]
            ):
                gaps.append(
                    f"property '{name}' does not satisfy type {iface_prop.type_ref!r}"
                )

        link_types, link_counts = _object_link_view(object_dict)
        for cname, constraint in self.all_link_constraints(registry).items():
            et = constraint.edge_type.value
            have = link_counts.get(et, 1 if et in link_types else 0)
            if constraint.min_count <= 0:
                if et not in link_types:
                    gaps.append(
                        f"link constraint '{cname}' requires a {et!r} link to be declared"
                    )
            elif have < constraint.min_count:
                gaps.append(
                    f"link constraint '{cname}' requires >= {constraint.min_count} "
                    f"{et!r} link(s), found {have}"
                )
        return gaps

    def conforms(
        self,
        object_dict: dict[str, Any],
        *,
        registry: InterfaceRegistry | None = None,
    ) -> bool:
        """Whether ``object_dict`` satisfies this interface's full shape."""
        return not self.gaps_for(object_dict, registry=registry)

    # ── OWL / SHACL emission ───────────────────────────────────────────────

    def owl_class_iri(self) -> str:
        """Return the interface's ``owl:Class`` IRI (``kg:`` namespace)."""
        return KG + _camel(self.name)

    def shape_iri(self) -> str:
        """Return the interface's SHACL ``sh:NodeShape`` IRI."""
        return KG + _camel(self.name) + "Shape"

    def to_owl(
        self, *, registry: InterfaceRegistry | None = None
    ) -> str:
        """Emit the interface as an ``owl:Class`` plus a SHACL ``sh:NodeShape``.

        CONCEPT:KG-2.38 — OWL/SHACL projection, reusing the ``owl_bridge``
        namespace conventions (``kg:`` = ``http://knuckles.team/kg#``):

          - The interface becomes an ``owl:Class`` (abstract — no individuals are
            promoted for it). ``extends`` parents map to ``rdfs:subClassOf`` so
            interface inheritance is OWL subsumption.
          - A ``sh:NodeShape`` (``<Name>Shape``) carries one ``sh:property`` per
            *required* interface-property (``sh:minCount 1`` + ``sh:datatype``
            from the :class:`PropertyType`'s ``xsd_iri``) and one ``sh:property``
            per required link constraint (``sh:path`` = the edge type, ``sh:minCount``
            = ``min_count``). An object type that ``implement``\\ s this interface
            is emitted ``rdfs:subClassOf`` the class and ``sh:node`` the shape.

        The returned Turtle is self-contained (prefixes included) so it can be
        appended to a pack's ``owl_extensions`` ``.ttl`` and loaded by the OWL
        reasoner / SHACL gate alongside the hand-authored shapes in
        ``knowledge_graph/shapes/``.
        """
        local = _camel(self.name)
        lines: list[str] = [
            f"@prefix : <{KG}> .",
            f"@prefix owl: <{OWL}> .",
            f"@prefix rdfs: <{RDFS}> .",
            f"@prefix sh: <{SH}> .",
            f"@prefix xsd: <{XSD}> .",
            "",
        ]

        # owl:Class for the interface, with rdfs:subClassOf for each parent.
        class_lines = [f":{local} a owl:Class ;"]
        class_lines.append(f'    rdfs:label "{self.name}" ;')
        if self.description:
            comment = self.description.replace('"', "'").replace("\n", " ")
            class_lines.append(f'    rdfs:comment "{comment}" ;')
        class_lines.append(
            "    rdfs:comment "
            '"Abstract ontology interface (CONCEPT:KG-2.38); '
            'no own individuals — implemented by object types." ;'
        )
        for parent in self.extends:
            class_lines.append(f"    rdfs:subClassOf :{_camel(parent)} ;")
        class_lines[-1] = class_lines[-1].rstrip(" ;") + " ."
        lines.extend(class_lines)
        lines.append("")

        # SHACL NodeShape capturing the contract.
        shape_lines = [f":{local}Shape a sh:NodeShape ;"]
        shape_lines.append(f"    sh:targetClass :{local} ;")
        shape_lines.append(
            f'    sh:name "{self.name} Shape" ;'
        )
        prop_blocks: list[str] = []
        for name, iface_prop in self.all_properties(registry).items():
            if not iface_prop.required:
                continue
            xsd_iri = iface_prop.property_type().xsd_iri
            prop_blocks.append(
                "    sh:property [ sh:path :%s ;\n"
                "            sh:datatype <%s> ;\n"
                "            sh:minCount 1 ;\n"
                '            sh:message "%s must declare interface-property %s." ]'
                % (name, xsd_iri, self.name, name)
            )
        for cname, constraint in self.all_link_constraints(registry).items():
            mincount = max(0, int(constraint.min_count))
            prop_blocks.append(
                "    sh:property [ sh:path :%s ;\n"
                "            sh:minCount %d ;\n"
                '            sh:message "%s must expose link %s." ]'
                % (constraint.edge_type.value, mincount, self.name, constraint.edge_type.value)
            )
        if prop_blocks:
            shape_lines.append(",\n".join(prop_blocks) + " .")
        else:
            shape_lines[-1] = shape_lines[-1].rstrip(" ;") + " ."
        lines.extend(shape_lines)
        lines.append("")
        return "\n".join(lines)

    def signature(self) -> str:
        """Return a stable content hash of the interface shape (cache-key use)."""
        return hashlib.sha256(
            self.model_dump_json(exclude_none=False).encode("utf-8")
        ).hexdigest()[:16]


def _object_link_view(
    object_dict: dict[str, Any],
) -> tuple[set[str], dict[str, int]]:
    """Derive ``(declared_edge_types, per_edge_counts)`` from an object dict.

    Accepts several shapes for an object's links so conformance works against
    real data:

      - ``object_dict["link_types"]`` — an explicit set/list of edge-type
        strings the *type* declares (cardinality unknown → counted as present);
      - ``object_dict["links"]`` / ``["edges"]`` — a list of link dicts, each
        with a ``type``/``edge_type``/``rel`` key (counted per edge type).

    Edge-type strings are normalized to their lowercase ``RegistryEdgeType``
    values for comparison.
    """
    declared: set[str] = set()
    counts: dict[str, int] = {}

    raw_types = object_dict.get("link_types")
    if isinstance(raw_types, (list, tuple, set)):
        for t in raw_types:
            if isinstance(t, str) and t:
                declared.add(t.lower())

    for key in ("links", "edges"):
        raw = object_dict.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        for link in raw:
            if not isinstance(link, dict):
                continue
            et = link.get("type") or link.get("edge_type") or link.get("rel")
            if isinstance(et, str) and et:
                low = et.lower()
                declared.add(low)
                counts[low] = counts.get(low, 0) + 1
    return declared, counts


class ImplementationReport(BaseModel):
    """The outcome of validating that an object type implements an interface.

    CONCEPT:KG-2.38 — returned by :meth:`InterfaceRegistry.implement`. ``ok`` is
    True only when ``gaps`` is empty (the type satisfies the full inheritance-
    resolved shape). Even a gap-bearing implementation is *recorded* (so the
    targeting/index reflects intent), but ``ok=False`` flags the contract breach
    for the caller / SHACL gate.

    Attributes:
        object_type: The implementing object/node type value (e.g. ``"document"``).
        interface: The interface name implemented.
        ok: Whether the type fully satisfies the interface contract.
        gaps: Human-readable list of unmet requirements (empty when ``ok``).
    """

    object_type: str
    interface: str
    ok: bool
    gaps: list[str] = Field(default_factory=list)


def _type_value(object_type: RegistryNodeType | str) -> str:
    """Normalize an object-type reference to its lowercase string value."""
    if isinstance(object_type, RegistryNodeType):
        return object_type.value
    return str(object_type).lower()


class InterfaceRegistry:
    """Registry of ontology interfaces + the type→interface implementation graph.

    CONCEPT:KG-2.38. Mirrors :class:`~agent_utilities.knowledge_graph.ontology.links.LinkTypeRegistry`
    and the actions registry: name-keyed, rejects duplicates, import-populated
    with real built-ins. Holds both the interface *definitions* and the
    *implementations* (which object types implement which interfaces), and is the
    resolution point for programmatic targeting (:meth:`find_implementers`).
    """

    def __init__(self) -> None:
        self._interfaces: dict[str, Interface] = {}
        # interface name -> set of implementing object-type values
        self._implementers: dict[str, set[str]] = {}
        # object-type value -> declared property/link "shape" used for validation
        self._type_shapes: dict[str, dict[str, Any]] = {}

    # ── Definition management ──────────────────────────────────────────────

    def register(self, interface: Interface) -> Interface:
        """Register an interface definition; raises on a duplicate name."""
        if interface.name in self._interfaces:
            raise ValueError(f"Interface already registered: {interface.name!r}")
        # Validate that any declared parents exist (or will be registered first).
        for parent in interface.extends:
            if parent not in self._interfaces:
                raise ValueError(
                    f"Interface {interface.name!r} extends unknown interface "
                    f"{parent!r}; register parents first."
                )
        self._interfaces[interface.name] = interface
        self._implementers.setdefault(interface.name, set())
        return interface

    def get(self, name: str) -> Interface | None:
        """Return the interface named ``name``, or ``None``."""
        return self._interfaces.get(name)

    def list_interfaces(self) -> list[Interface]:
        """Return all registered interfaces."""
        return list(self._interfaces.values())

    def __contains__(self, name: object) -> bool:
        return name in self._interfaces

    def __len__(self) -> int:
        return len(self._interfaces)

    # ── Type shapes (what a concrete object type declares) ─────────────────

    def declare_type_shape(
        self,
        object_type: RegistryNodeType | str,
        *,
        properties: dict[str, str] | None = None,
        link_types: Iterable[str | RegistryEdgeType] | None = None,
    ) -> None:
        """Record a concrete object type's declared property/link shape.

        Used by :meth:`implement` to validate the type against an interface
        *at the schema level* (does the type declare the required properties and
        links?). ``properties`` maps property name → a property-type reference;
        ``link_types`` is the set of edge types the type exposes.
        """
        tv = _type_value(object_type)
        shape = self._type_shapes.setdefault(tv, {"properties": {}, "link_types": set()})
        if properties:
            shape["properties"].update({k: str(v) for k, v in properties.items()})
        if link_types:
            for lt in link_types:
                val = lt.value if isinstance(lt, RegistryEdgeType) else str(lt).lower()
                shape["link_types"].add(val)

    def type_shape(self, object_type: RegistryNodeType | str) -> dict[str, Any]:
        """Return the declared shape of an object type as a conformance-ready dict.

        The returned dict has the property names as keys (with a *type-valid
        sentinel* value so required-property presence is satisfied) plus a
        ``link_types`` set — exactly the shape :meth:`Interface.gaps_for`
        consumes.
        """
        tv = _type_value(object_type)
        shape = self._type_shapes.get(tv, {"properties": {}, "link_types": set()})
        obj: dict[str, Any] = {}
        for pname, type_ref in shape.get("properties", {}).items():
            obj[pname] = _sentinel_for(type_ref)
        obj["link_types"] = set(shape.get("link_types", set()))
        return obj

    # ── Implementation + contract validation ───────────────────────────────

    def implement(
        self,
        object_type: RegistryNodeType | str,
        interface: str | Interface,
        *,
        type_shape: dict[str, Any] | None = None,
    ) -> ImplementationReport:
        """Record that ``object_type`` implements ``interface`` and validate it.

        CONCEPT:KG-2.38 — the *implemented-by* relation plus contract checking. If
        the type previously declared a shape (via :meth:`declare_type_shape`) or
        an explicit ``type_shape`` is passed, it is validated against the
        interface's inheritance-resolved property + link contract. Gaps are
        collected into the returned :class:`ImplementationReport`. The
        implementation is recorded regardless (so targeting reflects declared
        intent), with ``ok`` flagging whether the contract is fully met.

        Args:
            object_type: The concrete object/node type implementing the interface.
            interface: The interface (or its name) being implemented.
            type_shape: Optional explicit conformance dict (overrides the
                registered type shape) describing the type's properties/links.

        Raises:
            ValueError: if the interface name is unknown.
        """
        iface = (
            interface
            if isinstance(interface, Interface)
            else self.get(interface)
        )
        if iface is None:
            raise ValueError(f"unknown interface: {interface!r}")

        tv = _type_value(object_type)
        shape = type_shape if type_shape is not None else self.type_shape(object_type)
        gaps = iface.gaps_for(shape, registry=self)

        self._implementers.setdefault(iface.name, set()).add(tv)
        return ImplementationReport(
            object_type=tv,
            interface=iface.name,
            ok=not gaps,
            gaps=gaps,
        )

    def implementations_of_type(
        self, object_type: RegistryNodeType | str
    ) -> list[str]:
        """Return the interface names a given object type implements."""
        tv = _type_value(object_type)
        return [
            name
            for name, impls in self._implementers.items()
            if tv in impls
        ]

    # ── Programmatic targeting (the headline feature) ──────────────────────

    def find_implementers(self, interface: str | Interface) -> list[str]:
        """Resolve an interface to the concrete object types that implement it.

        CONCEPT:KG-2.38 — *programmatic targeting*. A Function/Action/object query
        that names an interface where a concrete type is expected calls this to
        expand the interface to the live set of implementing object-type values.
        Sub-interface implementers are included transitively: implementing a
        sub-interface implies implementing every interface it ``extends``.

        Raises:
            ValueError: if the interface name is unknown.
        """
        iface = (
            interface
            if isinstance(interface, Interface)
            else self.get(interface)
        )
        if iface is None:
            raise ValueError(f"unknown interface: {interface!r}")

        result: set[str] = set(self._implementers.get(iface.name, set()))
        # Transitive: any interface that extends this one contributes its
        # implementers (an implementer of the sub-interface satisfies the parent).
        for other in self._interfaces.values():
            if iface.name in self._ancestor_names(other):
                result |= self._implementers.get(other.name, set())
        return sorted(result)

    def _ancestor_names(self, iface: Interface) -> set[str]:
        """Return the set of all ancestor interface names of ``iface``."""
        return {a.name for a in iface._iter_ancestors(self)}

    def conforms(
        self,
        object_dict: dict[str, Any],
        interface: str | Interface,
    ) -> bool:
        """Whether a concrete object satisfies an interface's shape (targeting check).

        CONCEPT:KG-2.38 — the runtime conformance gate used when a function/query
        receives a concrete object and an interface target: it confirms the object
        actually conforms before the interface-typed handler runs.

        Raises:
            ValueError: if the interface name is unknown.
        """
        iface = (
            interface
            if isinstance(interface, Interface)
            else self.get(interface)
        )
        if iface is None:
            raise ValueError(f"unknown interface: {interface!r}")
        return iface.conforms(object_dict, registry=self)

    def resolve_target(self, type_or_interface: str) -> list[str]:
        """Resolve a target reference (object type *or* interface) to object types.

        CONCEPT:KG-2.38 — the single entry the functions runtime (A3) / object
        queries call when a target name may be *either* a concrete object type or
        an interface. An interface name expands to its implementers
        (:meth:`find_implementers`); any other string is returned as-is (a plain
        concrete object type), so callers can pass an interface name wherever a
        type was expected without branching.
        """
        if type_or_interface in self._interfaces:
            return self.find_implementers(type_or_interface)
        return [type_or_interface]

    def to_owl(self) -> str:
        """Emit OWL/SHACL for every registered interface and its implementers.

        CONCEPT:KG-2.38 — concatenates each interface's :meth:`Interface.to_owl`
        and appends the ``rdfs:subClassOf`` / ``sh:node`` assertions linking each
        implementing object-type class to the interface class + shape. The result
        is loadable alongside the hand-authored ``knowledge_graph/shapes/`` TTL.
        """
        chunks: list[str] = []
        for iface in self._interfaces.values():
            chunks.append(iface.to_owl(registry=self))
        # Implements assertions: <Type> rdfs:subClassOf :<Iface> ; sh:node :<Iface>Shape .
        impl_lines: list[str] = [
            f"@prefix : <{KG}> .",
            f"@prefix rdfs: <{RDFS}> .",
            f"@prefix sh: <{SH}> .",
            "",
        ]
        for iface_name, impls in self._implementers.items():
            iface = self._interfaces.get(iface_name)
            if iface is None:
                continue
            for tv in sorted(impls):
                type_local = _camel(tv)
                impl_lines.append(
                    f":{type_local} rdfs:subClassOf :{_camel(iface_name)} ; "
                    f"sh:node :{_camel(iface_name)}Shape ."
                )
        chunks.append("\n".join(impl_lines))
        return "\n\n".join(chunks)


def _sentinel_for(type_ref: str) -> Any:
    """Return a value that validly coerces to ``type_ref`` (for type-shape checks).

    Used when validating a *type's* declared shape (not a concrete instance):
    declaring a property of a given type means the type guarantees a conforming
    value, so we materialize a canonical valid example for that property type.
    Falls back to a non-empty string for any type whose canonical example is
    string-coercible.
    """
    try:
        pt = parse_type_ref(type_ref)
    except ValueError:
        return "x"
    name = pt.name
    examples: dict[str, Any] = {
        "string": "x",
        "boolean": True,
        "byte": 1,
        "short": 1,
        "integer": 1,
        "long": 1,
        "float": 1.0,
        "double": 1.0,
        "decimal": "1.0",
        "date": "2020-01-01",
        "timestamp": "2020-01-01T00:00:00Z",
        "geohash": "u4pruydqqvj",
        "geo_point": {"lat": 0.0, "lon": 0.0},
    }
    if name in examples:
        return examples[name]
    if name.startswith("array") or name in ("vector", "embedding"):
        # An empty array/vector is dimension-checked; use a sentinel the
        # required-presence check accepts without asserting element validity.
        return [] if name.startswith("array") else "[]"
    # Complex JSON-backed types accept a dict/string; default to a present marker.
    return "x"


# ── Built-in interfaces (import-populated; live, not an empty shell) ──────────


def register_builtin_interfaces(registry: InterfaceRegistry) -> None:
    """Register the built-in interfaces into ``registry``.

    Two real, reusable interfaces mirroring common Foundry shared shapes:

      - ``HasProvenance`` — anything that records where it came from: a required
        ``timestamp`` provenance time plus a ``was_derived_from`` link to its
        source. Implemented by ``document`` (which declares both).
      - ``Locatable`` — anything with a geographic position: required ``lat`` /
        ``lon`` ``double`` properties (Foundry interface-properties). Extended by
        ``GeoTracked`` to demonstrate interface inheritance (adds an ``occurred_at``
        link). ``place`` implements ``Locatable``.
    """
    registry.register(
        Interface(
            name="HasProvenance",
            description=(
                "Abstract shape for any object that records its provenance: a "
                "creation/observation time and a link to the source it was "
                "derived from (Foundry interface — shared properties + link "
                "constraint)."
            ),
            properties=[
                InterfaceProperty(
                    name="timestamp",
                    type_ref="timestamp",
                    description="When the object was created/observed.",
                ),
            ],
            link_constraints=[
                InterfaceLinkConstraint(
                    name="derivation",
                    edge_type=RegistryEdgeType.WAS_DERIVED_FROM,
                    min_count=0,
                    description="Must declare a was_derived_from provenance link.",
                ),
            ],
        )
    )
    registry.register(
        Interface(
            name="Locatable",
            description=(
                "Abstract shape for any object with a geographic position: "
                "latitude and longitude interface-properties."
            ),
            properties=[
                InterfaceProperty(
                    name="lat", type_ref="double", description="Latitude in degrees."
                ),
                InterfaceProperty(
                    name="lon", type_ref="double", description="Longitude in degrees."
                ),
            ],
        )
    )
    # GeoTracked extends Locatable (interface inheritance, multi-extend capable):
    # everything Locatable, plus a temporal occurred_at link.
    registry.register(
        Interface(
            name="GeoTracked",
            description=(
                "A Locatable object whose position is timestamped — demonstrates "
                "interface inheritance (extends Locatable, HasProvenance)."
            ),
            extends=["Locatable", "HasProvenance"],
            link_constraints=[
                InterfaceLinkConstraint(
                    name="when",
                    edge_type=RegistryEdgeType.HAS_TEMPORAL_EXTENT,
                    min_count=0,
                    description="Must declare a has_temporal_extent temporal link.",
                ),
            ],
        )
    )

    # Declare built-in type shapes and record the implements-by relations so the
    # registry ships with live implementers (programmatic targeting works out of
    # the box, not just after a caller wires it).
    registry.declare_type_shape(
        RegistryNodeType.DOCUMENT,
        properties={"timestamp": "timestamp"},
        link_types=[RegistryEdgeType.WAS_DERIVED_FROM],
    )
    registry.implement(RegistryNodeType.DOCUMENT, "HasProvenance")
    registry.declare_type_shape(
        RegistryNodeType.PLACE,
        properties={"lat": "double", "lon": "double"},
    )
    registry.implement(RegistryNodeType.PLACE, "Locatable")


# CONCEPT:KG-2.38 — populated at import with real built-ins, never empty.
DEFAULT_INTERFACE_REGISTRY = InterfaceRegistry()
register_builtin_interfaces(DEFAULT_INTERFACE_REGISTRY)


def target_object_types(
    type_or_interface: str,
    *,
    registry: InterfaceRegistry | None = None,
) -> list[str]:
    """Module-level convenience resolving a target name to concrete object types.

    CONCEPT:KG-2.38 — the function the A3 functions runtime / object queries call
    to accept an interface name where a concrete type is expected. Delegates to
    :meth:`InterfaceRegistry.resolve_target` on the default registry (or a
    supplied one). A plain object-type name passes through unchanged; an
    interface name expands to its (transitive) implementers.
    """
    reg = registry or DEFAULT_INTERFACE_REGISTRY
    return reg.resolve_target(type_or_interface)


__all__ = [
    "Interface",
    "InterfaceProperty",
    "InterfaceLinkConstraint",
    "ImplementationReport",
    "InterfaceRegistry",
    "DEFAULT_INTERFACE_REGISTRY",
    "register_builtin_interfaces",
    "target_object_types",
]
