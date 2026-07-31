#!/usr/bin/python
"""Hosted-ontology lifecycle management (CRUD) — CONCEPT:AU-KG.ontology.manage-arbitrary.

The bundled ``ontology*.ttl`` library is the platform's *static* TBox. This
module adds **dynamic, hosted** ontologies: an agent or HTTP client can load an
arbitrary OWL/RDF ontology (from a file path, a URL, or raw turtle text) into the
running KG, list/inspect what is hosted, replace it with a new version, validate a
candidate without committing, and unload it again — all through one service core
that both the ``graph_ontology`` MCP tool and the ``/graph/ontology`` REST route
dispatch into.

Design:

* Parsing / validation / counting is pure ``rdflib`` (+ optional ``pyshacl`` /
  ``owlrl`` — shipped in the ``ontology-guardrails`` serving extra, CONCEPT:AU-KG.ontology.activation-icv-fallback)
  so the whole surface works with **no engine** (unit-testable).
* When a live engine is present, ``load``/``update`` push the ontology's axioms
  into a **dedicated, durable, per-tenant ontology graph** — never the mixed
  property graph an agent's ABox instances live in (CONCEPT:AU-KG.ontology.dedicated-tbox-graph) — via
  ``GraphComputeEngine.add_triples``, so the native OWL reasoner (``owl_reason``)
  and SPARQL surface immediately operate over them — that is what "active for
  reasoning" means here. Physically separating TBox axioms from ABox property-graph
  instances (whose node identifiers are arbitrary opaque strings, not IRIs) is what
  keeps the engine's SHACL/ICV write guard (``EPISTEMIC_GRAPH_ICV_NATIVE_WRITES``)
  from tripping on non-RDF property-graph identifiers when it evaluates the
  ontology graph's RDF projection.
* The registry of hosted ontologies is keyed by ``(tenant, graph, iri, version)``.
  When a live engine is attached, records are durable, engine-native
  ``:HostedOntology`` nodes in that SAME dedicated per-tenant ontology graph (see
  :class:`_EngineRegistryStore`) — not a process-local structure, so every
  ``graph-os`` process/replica for a tenant sees the same hosted set and it
  survives a process restart. With no engine attached (offline/dev/tests), the
  registry degrades honestly to an in-process, non-durable store
  (:class:`_InMemoryRegistryStore`) — the same behavior this module has always
  had for the engine-free path.

Engine gap (documented, not worked around — eg-rdf is owned by another agent):
the engine RDF surface exposes ``add_triples`` (load) and ``get_rdf`` /
``sparql`` (read) but **no remove-triples / drop-named-graph op**, so ``delete``
deactivates an ontology and drops it from the hosted registry but cannot
physically retract its axioms from the engine's RDF dataset until the engine
reloads. See :meth:`OntologyLifecycle.delete`.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Namespaces whose subjects we count as ontology "classes"/"properties".
_OWL = "http://www.w3.org/2002/07/owl#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

_CLASS_TYPES = (f"{_OWL}Class", f"{_RDFS}Class")
_PROPERTY_TYPES = (
    f"{_OWL}ObjectProperty",
    f"{_OWL}DatatypeProperty",
    f"{_OWL}AnnotationProperty",
    f"{_RDF}Property",
)
_ONTOLOGY_TYPE = f"{_OWL}Ontology"

#: Node type/label for durable hosted-ontology registry records
#: (CONCEPT:AU-KG.ontology.dedicated-tbox-graph) — one per ``(iri, version)`` in the
#: tenant's dedicated ontology graph, mirroring the ``:Server`` registry pattern
#: (``knowledge_graph/core/engine_ingestion.py``) rather than a Cypher-string MERGE.
_ONTOLOGY_NODE_TYPE = "HostedOntology"

#: Base name for the dedicated per-tenant ontology graph — resolved through
#: :func:`shard_topology.tenant_graph_name` so it lands on the SAME tenant
#: namespace as everything else, just under its own base rather than the
#: mixed ABox default graph.
_ONTOLOGY_GRAPH_BASE = "ontology"

#: Process-wide memo of ontology graphs already confirmed to exist on the
#: live engine, so repeated ``OntologyLifecycle()`` construction (one per MCP
#: call — see ``mcp/tools/ontology_tools.py``) doesn't round-trip
#: ``tenants.list()``/``tenants.create()`` on every call. Losing this cache
#: (process restart) only costs one extra round-trip on next use, never
#: correctness — the engine itself is the durability boundary.
_KNOWN_ONTOLOGY_GRAPHS: set[str] = set()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _key(iri: str, version: str) -> str:
    return f"{iri}@@{version}"


class OntologyError(ValueError):
    """A candidate ontology failed to parse or validate."""


def _parse_graph(source: str, source_type: str = "auto") -> Any:
    """Parse ``source`` into an ``rdflib.Graph``.

    ``source_type``: ``file`` (path on disk, resolves ``owl:imports``), ``url``
    (HTTP/HTTPS fetch), ``text`` (raw turtle/RDF), or ``auto`` (sniff).
    """
    import rdflib

    st = (source_type or "auto").lower()
    if st == "auto":
        s = source.strip()
        if s.startswith(("http://", "https://")) and "\n" not in s:
            st = "url"
        elif "\n" not in s and len(s) < 4096 and Path(s).expanduser().exists():
            st = "file"
        else:
            st = "text"

    try:
        if st == "file":
            # File path: use the import-resolving loader so owl:imports are merged.
            from ..core.ontology_loader import OntologyLoader

            return OntologyLoader().load_with_imports(Path(source).expanduser())
        if st == "url":
            g = rdflib.Graph()
            g.parse(source.strip())  # rdflib content-negotiates the URL
            return g
        # text
        g = rdflib.Graph()
        g.parse(data=source, format="turtle")
        return g
    except OntologyError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface a clean parse failure
        raise OntologyError(f"could not parse ontology ({st}): {exc}") from exc


def _typed_subjects(graph: Any, type_iris: tuple[str, ...]) -> list[str]:
    import rdflib

    out: set[str] = set()
    for t in type_iris:
        for s in graph.subjects(predicate=rdflib.RDF.type, object=rdflib.URIRef(t)):
            out.add(str(s))
    return sorted(out)


def summarize(graph: Any) -> dict[str, Any]:
    """Compute lifecycle metadata for a parsed ontology graph."""
    classes = _typed_subjects(graph, _CLASS_TYPES)
    properties = _typed_subjects(graph, _PROPERTY_TYPES)
    ontology_iris = _typed_subjects(graph, (_ONTOLOGY_TYPE,))
    return {
        "ontology_iri": ontology_iris[0] if ontology_iris else None,
        "declared_ontology_iris": ontology_iris,
        "n_axioms": len(graph),
        "n_classes": len(classes),
        "n_properties": len(properties),
        "classes": classes,
        "properties": properties,
    }


def validate_graph(graph: Any, *, run_shacl: bool = True) -> dict[str, Any]:
    """Run the valid/connected/SHACL-style checks over a parsed ontology.

    Mirrors the bundled-library gate (CONCEPT:AU-KG.maintenance.canonical-ontology-library) at the granularity of a
    single candidate: it must parse (already done), declare something
    addressable, and survive OWL-RL closure; bundled SHACL shapes (if present and
    ``pyshacl`` installed) must load and run without error.
    """
    errors: list[str] = []
    warnings: list[str] = []
    summary = summarize(graph)
    # Populated only when SHACL actually ran against bundled shapes (below) —
    # the literal sh:ValidationReport (CONCEPT:AU-KG.ontology.shacl-report-passthrough),
    # surfaced so a caller (the ontology_api.py /ontology/validate REST twin, the
    # agent-webui SHACL validation-report view) gets the real pyshacl report
    # instead of just the derived valid/errors/warnings summary.
    shacl_report: dict[str, Any] | None = None

    if summary["n_classes"] == 0 and not summary["declared_ontology_iris"]:
        errors.append(
            "ontology declares no owl:Class/rdfs:Class and no owl:Ontology IRI — "
            "nothing addressable to host"
        )

    # OWL-RL closure must not break (reasoning safety).
    try:
        import owlrl  # type: ignore
        import rdflib

        merged = rdflib.Graph()
        for triple in graph:
            merged.add(triple)
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(merged)
    except ImportError:
        warnings.append("owlrl not installed — OWL-RL closure check skipped")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ontology breaks OWL-RL closure: {exc}")

    # Bundled SHACL shapes well-formedness + run against the candidate.
    if run_shacl:
        try:
            import pyshacl  # type: ignore
            import rdflib

            shapes_dir = (
                Path(__file__).resolve().parent.parent / "shapes"
            )  # knowledge_graph/shapes
            shapes = rdflib.Graph()
            if shapes_dir.exists():
                for shape_file in sorted(shapes_dir.glob("*.ttl")):
                    try:
                        shapes.parse(str(shape_file), format="turtle")
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(
                            f"skipped unparseable shape {shape_file.name}: {exc}"
                        )
            if len(shapes) > 0:
                conforms, results_graph, results_text = pyshacl.validate(
                    data_graph=graph,
                    shacl_graph=shapes,
                    inference="none",
                    abort_on_first=False,
                )
                shacl_report = {"conforms": bool(conforms), "text": results_text}
                try:
                    shacl_report["turtle"] = results_graph.serialize(format="turtle")
                except Exception:  # noqa: BLE001 — report text above is enough
                    pass
                if not conforms:
                    # Shapes target instance data, not a TBox — a non-conformance
                    # is advisory for an ontology load, not a hard reject.
                    warnings.append(
                        "candidate does not conform to bundled SHACL shapes"
                    )
        except ImportError:
            warnings.append("pyshacl not installed — SHACL check skipped")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"SHACL validation error: {exc}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            k: summary[k]
            for k in ("ontology_iri", "n_axioms", "n_classes", "n_properties")
        },
        "shacl_report": shacl_report,
    }


# ── Registry stores ──────────────────────────────────────────────────────────
#
# Two implementations behind the same tiny dict-like interface
# (get/set/delete/values), selected per :class:`OntologyLifecycle` instance by
# whether a live, node-write-capable engine is attached:
#
# * :class:`_EngineRegistryStore` — durable, per-tenant, engine-native. Used
#   whenever a real ``GraphComputeEngine`` is attached.
# * :class:`_InMemoryRegistryStore` — the pre-existing behavior, unchanged, for
#   the engine-free offline/dev/test path. A single process-wide instance
#   (``_MEMORY_STORE``) so state still survives across the repeated
#   ``OntologyLifecycle()`` construction every MCP tool call does (this is NOT
#   a regression: it was already a process-global dict before this change —
#   the fix is that a REAL engine no longer falls back to it).


class _InMemoryRegistryStore:
    """Non-durable, process-local registry (offline/no-engine fallback)."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    def set(self, key: str, record: dict[str, Any]) -> None:
        self._records[key] = record

    def delete(self, key: str) -> dict[str, Any] | None:
        return self._records.pop(key, None)

    def values(self) -> list[dict[str, Any]]:
        return list(self._records.values())

    def clear(self) -> None:
        self._records.clear()


#: The one process-wide in-memory fallback (mirrors the previous module-level
#: ``_REGISTRY`` dict exactly — see class docstring above).
_MEMORY_STORE = _InMemoryRegistryStore()


class _EngineRegistryStore:
    """Durable, per-tenant, engine-native hosted-ontology registry.

    Each record is a typed ``:HostedOntology`` node in the tenant's dedicated
    ontology graph (CONCEPT:AU-KG.ontology.dedicated-tbox-graph), addressed by the SAME
    ``iri@@version`` key :func:`_key` already uses — tenant and graph isolation
    come from WHICH graph this store's ``gc`` view is bound to (see
    :func:`_ontology_graph_name`/``tenant_graph_name``), so the key itself
    doesn't need to repeat the tenant. Uses the engine's native typed node
    surface (``add_node``/``get_nodes_by_label``/``remove_node``) rather than
    compiled Cypher — the record's full JSON is a ``data`` property so no
    schema migration is needed as fields evolve.
    """

    def __init__(
        self, gc: Any, node_type: str = _ONTOLOGY_NODE_TYPE, prefix: str = "ont"
    ) -> None:
        self._gc = gc
        self._node_type = node_type
        self._prefix = prefix

    def _node_id(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def get(self, key: str) -> dict[str, Any] | None:
        node_id = self._node_id(key)
        if not self._gc.has_node(node_id):
            return None
        props = self._gc.client.nodes.properties(node_id) or {}
        data = props.get("data") if isinstance(props, dict) else None
        if not data:
            return None
        try:
            return json.loads(data)
        except Exception as exc:  # noqa: BLE001 — a corrupt record reads as absent
            logger.warning("Corrupt %s record at %s: %s", self._node_type, node_id, exc)
            return None

    def set(self, key: str, record: dict[str, Any]) -> None:
        self._gc.add_node(
            self._node_id(key),
            node_type=self._node_type,
            iri=record.get("iri", ""),
            version=record.get("version", ""),
            active=bool(record.get("active", False)),
            loaded_at=record.get("loaded_at", ""),
            data=json.dumps(record, default=str),
        )

    def delete(self, key: str) -> dict[str, Any] | None:
        record = self.get(key)
        if record is not None:
            self._gc.remove_node(self._node_id(key))
        return record

    def values(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _node_id, props in self._gc.get_nodes_by_label(self._node_type):
            if not isinstance(props, dict):
                continue
            data = props.get("data")
            if not data:
                continue
            try:
                out.append(json.loads(data))
            except Exception as exc:  # noqa: BLE001 — skip one corrupt record
                logger.warning("Skipping corrupt %s record: %s", self._node_type, exc)
        return out


def _ontology_graph_name(tenant: str | None) -> str:
    """The dedicated per-tenant graph TBox axioms + the registry live in.

    Deliberately a DIFFERENT named graph from the tenant's default/ABox graph
    (CONCEPT:AU-KG.ontology.dedicated-tbox-graph) — see the module docstring for why mixing TBox
    axioms into the property graph trips the engine's SHACL/ICV write guard on
    opaque, non-IRI-safe property-graph node identifiers.
    """
    from ..core.shard_topology import tenant_graph_name

    return tenant_graph_name(tenant, base=_ONTOLOGY_GRAPH_BASE)


def _ensure_ontology_graph(gc: Any, graph_name: str) -> None:
    """Best-effort create-if-absent for the dedicated ontology graph.

    Mirrors ``GraphComputeEngine._ensure_local_session_graph``'s create-then-
    reverify-on-race pattern, without that method's local-session-only ``kg:admin``
    scope gate (this runs for any tenant's ontology graph, not just the one
    packaged local session graph). Raises :class:`OntologyError` on a genuine,
    non-race failure so the caller can fail closed rather than silently writing
    into a graph that was never actually provisioned.
    """
    if graph_name in _KNOWN_ONTOLOGY_GRAPHS:
        return
    client = getattr(gc, "client", None)
    tenants = getattr(client, "tenants", None)
    if (
        tenants is None
        or not hasattr(tenants, "create")
        or not hasattr(tenants, "list")
    ):
        # Not a real engine client (e.g. a unit-test fake) — nothing to provision.
        return

    def _listed() -> set[str]:
        try:
            entries = tenants.list() or []
        except Exception as exc:  # noqa: BLE001 — surfaced to the caller below
            raise OntologyError(
                f"could not list engine graphs while provisioning {graph_name!r}: {exc}"
            ) from exc
        return {str(e.get("name") if isinstance(e, dict) else e) for e in entries}

    if graph_name in _listed():
        _KNOWN_ONTOLOGY_GRAPHS.add(graph_name)
        return
    try:
        tenants.create(graph_name, "Ontology")
    except Exception as exc:
        # Another process/writer may have won a create race — reverify before
        # treating this as fatal; otherwise fail closed (do not silently
        # proceed to write axioms into a graph that was never provisioned).
        if graph_name not in _listed():
            raise OntologyError(
                f"could not provision dedicated ontology graph {graph_name!r}: {exc}"
            ) from exc
    _KNOWN_ONTOLOGY_GRAPHS.add(graph_name)


class OntologyLifecycle:
    """CRUD lifecycle for ontologies hosted in the running KG (CONCEPT:AU-KG.ontology.manage-arbitrary).

    Args:
        engine: Optional live engine exposing ``graph_compute`` with the native
            RDF surface (``add_triples`` / ``owl_reason`` / ``sparql``). When
            ``None`` the lifecycle still parses/validates/inspects/registers
            ontologies (offline), it just cannot push axioms into a reasoner.
        tenant: Optional tenant id. Resolves the dedicated ontology graph via
            the SAME :func:`shard_topology.tenant_graph_name` convention every
            other tenant-scoped engine access uses. ``None`` resolves the
            ambient :func:`current_actor` tenant (mirrors
            ``TenantEnginePool._graph_for``); pass ``""`` explicitly for the
            single-tenant default graph regardless of the ambient actor.
        graph_name: Escape hatch overriding the resolved graph name outright
            (tests / callers that already know the exact graph).
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        tenant: str | None = None,
        graph_name: str | None = None,
    ) -> None:
        self._engine = engine
        self._tenant = tenant if tenant is not None else self._ambient_tenant()
        self._ontology_graph = graph_name or _ontology_graph_name(self._tenant)
        self._gc = self._resolve_graph_compute()
        self._store = self._make_store()

    @staticmethod
    def _ambient_tenant() -> str | None:
        try:
            from ...security.brain_context import current_actor

            return current_actor().tenant_id
        except Exception as exc:  # noqa: BLE001 — no session context (e.g. offline/tests)
            logger.debug("No ambient tenant context: %s", exc)
            return None

    def _resolve_graph_compute(self) -> Any:
        """A view of the attached engine scoped to the dedicated ontology graph.

        Falls back to the engine's default view (today's behavior) when it
        exposes no ``for_graph`` (e.g. a unit-test fake), so existing
        engine-free/fake-engine tests are unaffected.
        """
        gc0 = getattr(self._engine, "graph_compute", None) if self._engine else None
        if gc0 is None:
            return None
        for_graph = getattr(gc0, "for_graph", None)
        if callable(for_graph):
            try:
                return for_graph(self._ontology_graph)
            except Exception as exc:  # noqa: BLE001 — degrade to the unscoped view
                logger.debug(
                    "for_graph(%s) unavailable, using the engine's default graph "
                    "view: %s",
                    self._ontology_graph,
                    exc,
                )
        return gc0

    def _make_store(self) -> Any:
        """Durable per-tenant store when a real engine is attached, else the
        process-local in-memory fallback (see the module docstring)."""
        gc = self._gc
        is_real_engine = (
            gc is not None
            and hasattr(gc, "add_node")
            and hasattr(gc, "get_nodes_by_label")
            and hasattr(gc, "has_node")
            and hasattr(gc, "remove_node")
            and hasattr(gc, "client")
        )
        if is_real_engine:
            try:
                _ensure_ontology_graph(gc, self._ontology_graph)
                return _EngineRegistryStore(gc)
            except OntologyError as exc:
                logger.warning(
                    "Durable per-tenant ontology registry unavailable for graph "
                    "%s (falling back to the process-local, non-durable "
                    "registry): %s",
                    self._ontology_graph,
                    exc,
                )
        return _MEMORY_STORE

    # ── internals ────────────────────────────────────────────────────────────
    def _load_axioms(self, turtle: str) -> dict[str, Any]:
        """Push an ontology's axioms into the engine's native RDF dataset."""
        gc = self._gc
        if gc is None or not hasattr(gc, "add_triples"):
            return {
                "loaded_to_engine": False,
                "reason": "no engine RDF surface",
                "engine_attached": False,
            }
        try:
            report = gc.add_triples(turtle=turtle)
            return {
                "loaded_to_engine": True,
                "engine_attached": True,
                "graph": self._ontology_graph,
                **(report or {}),
            }
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            logger.warning(
                "Ontology activation failed on graph %s: %s", self._ontology_graph, exc
            )
            return {
                "loaded_to_engine": False,
                "reason": str(exc),
                "engine_attached": True,
            }

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        """A record minus its bulky stored turtle (for list/summary views)."""
        return {k: v for k, v in record.items() if k != "turtle"}

    # ── load / register ──────────────────────────────────────────────────────
    def load(
        self,
        source: str,
        *,
        source_type: str = "auto",
        version: str | None = None,
        iri: str | None = None,
        activate: bool = True,
        force: bool = False,
        category: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Parse, validate, register, and (if a live engine) activate an ontology.

        Idempotent on ``(iri, version)``: loading the same IRI+version twice
        returns the existing record (``idempotent: true``) unless ``force``.

        ``category``/``tags`` are optional catalogue metadata (CONCEPT:AU-KG.ontology.catalogue-browse) —
        free-form curation labels with no effect on parsing/validation/reasoning,
        stored on the record so :meth:`list_ontologies` can facet/filter the
        hosted set into a browsable gallery (Ontology-Playground coverage row #4).
        """
        graph = _parse_graph(source, source_type)
        report = validate_graph(graph)
        if not report["valid"]:
            return {"status": "rejected", **report}

        summary = summarize(graph)
        resolved_iri = (
            iri
            or summary["ontology_iri"]
            or f"urn:hosted-ontology:{abs(hash(source)) & 0xFFFFFFFF:08x}"
        )
        resolved_version = version or "1.0.0"
        key = _key(resolved_iri, resolved_version)

        existing = self._store.get(key)
        if existing and not force:
            return {
                "status": "ok",
                "idempotent": True,
                "ontology": self._public(existing),
            }

        turtle = graph.serialize(format="turtle")
        if isinstance(turtle, bytes):
            turtle = turtle.decode("utf-8")

        engine_report = (
            self._load_axioms(turtle) if activate else {"loaded_to_engine": False}
        )
        # Fail closed: with a live engine attached, "active" means the engine
        # actually absorbed the axioms, not merely that activation was
        # requested (CONCEPT:AU-KG.ontology.activation-fails-closed) — an
        # add_triples error (e.g. the engine's SHACL/ICV write guard rejecting
        # the candidate) must not be reported as a successful activation. With
        # NO engine attached (offline/dev), "active" stays the requested
        # intent flag — there is nothing to fail.
        activated = (
            bool(activate)
            if self._gc is None
            else bool(activate) and bool(engine_report.get("loaded_to_engine"))
        )

        record: dict[str, Any] = {
            "iri": resolved_iri,
            "version": resolved_version,
            "source": source if len(source) < 256 else f"{source[:240]}…",
            "source_type": source_type,
            "n_axioms": summary["n_axioms"],
            "n_classes": summary["n_classes"],
            "n_properties": summary["n_properties"],
            "loaded_at": _now(),
            "active": activated,
            "tenant": self._tenant or "",
            "graph": self._ontology_graph,
            "warnings": report["warnings"],
            "engine": engine_report,
            "category": category,
            "tags": list(tags) if tags else [],
            "deprecated": False,
            "turtle": turtle,
        }
        self._store.set(key, record)
        logger.info(
            "Hosted ontology loaded: %s v%s (%d axioms, active=%s, graph=%s)",
            resolved_iri,
            resolved_version,
            summary["n_axioms"],
            activated,
            self._ontology_graph,
        )
        return {"status": "ok", "idempotent": False, "ontology": self._public(record)}

    # ── list / catalogue ─────────────────────────────────────────────────────
    def list_ontologies(
        self,
        *,
        active_only: bool = False,
        deprecated_only: bool = False,
        search: str = "",
        category: str = "",
        source_type: str = "",
        tag: str = "",
    ) -> dict[str, Any]:
        """All hosted ontologies with metadata (newest first).

        The optional ``search``/``category``/``source_type``/``tag`` filters turn
        this into a browsable catalogue/gallery over the hosted set
        (CONCEPT:AU-KG.ontology.catalogue-browse, Ontology-Playground coverage row #4)
        — every filter defaults to unset, so a plain ``list_ontologies()`` call is
        unchanged. ``search`` is a case-insensitive substring match over
        ``iri``/``version``/``source``; ``category``/``source_type``/``tag`` are
        case-insensitive exact matches against those stored record fields (a
        record matches ``tag`` if it appears anywhere in its ``tags`` list).
        ``deprecated_only`` mirrors ``active_only`` for the ``deprecated`` flag
        (CONCEPT:AU-KG.ontology.deprecation-workflow, D-75-5) — the two are
        independent axes (a version can be deprecated and still active, e.g.
        during a migration window), so neither implies the other.
        """
        records = [
            self._public(r)
            for r in self._store.values()
            if (not active_only or r.get("active"))
            and (not deprecated_only or r.get("deprecated"))
        ]
        if search:
            needle = search.lower()
            records = [
                r
                for r in records
                if needle
                in f"{r.get('iri', '')} {r.get('version', '')} {r.get('source', '')}".lower()
            ]
        if category:
            records = [
                r for r in records if r.get("category", "").lower() == category.lower()
            ]
        if source_type:
            records = [
                r
                for r in records
                if r.get("source_type", "").lower() == source_type.lower()
            ]
        if tag:
            needle_tag = tag.lower()
            records = [
                r
                for r in records
                if needle_tag in [t.lower() for t in r.get("tags", [])]
            ]
        records.sort(key=lambda r: r.get("loaded_at", ""), reverse=True)
        return {"count": len(records), "ontologies": records}

    # ── get / inspect ────────────────────────────────────────────────────────
    def _resolve(
        self, iri: str, version: str | None
    ) -> tuple[str, dict[str, Any]] | None:
        if version is not None:
            key = _key(iri, version)
            rec = self._store.get(key)
            return (key, rec) if rec else None
        # No version → newest loaded version of this IRI.
        candidates = [
            (_key(r["iri"], r["version"]), r)
            for r in self._store.values()
            if r.get("iri") == iri
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda kr: kr[1].get("loaded_at", ""), reverse=True)
        return candidates[0]

    def get(
        self, iri: str, *, version: str | None = None, serialize: bool = False
    ) -> dict[str, Any]:
        """Inspect a hosted ontology: classes, properties, axiom count.

        With ``serialize=True`` also returns the ontology re-serialized to turtle.
        """
        resolved = self._resolve(iri, version)
        if resolved is None:
            return {"error": f"ontology not hosted: {iri} (version={version})"}
        _key_, record = resolved
        turtle = record.get("turtle", "")
        detail = self._public(record)
        if turtle:
            import rdflib

            g = rdflib.Graph()
            try:
                g.parse(data=turtle, format="turtle")
                s = summarize(g)
                detail["classes"] = s["classes"]
                detail["properties"] = s["properties"]
            except Exception as exc:  # noqa: BLE001
                detail["inspect_error"] = str(exc)
        if serialize:
            detail["turtle"] = turtle
        return {"ontology": detail}

    # ── update / replace ─────────────────────────────────────────────────────
    def update(
        self,
        source: str,
        *,
        iri: str,
        version: str,
        source_type: str = "auto",
        supersede: bool = True,
        activate: bool = True,
    ) -> dict[str, Any]:
        """Load a NEW version of an ontology IRI.

        With ``supersede`` (default) every prior version of the same IRI is
        deactivated (kept for history — bi-temporal); the new version becomes the
        active one. The hosted set is therefore versioned, not destructively
        overwritten.
        """
        if supersede:
            for r in self._store.values():
                if r.get("iri") == iri and r.get("version") != version:
                    r["active"] = False
                    self._store.set(_key(r["iri"], r["version"]), r)
        result = self.load(
            source,
            source_type=source_type,
            version=version,
            iri=iri,
            activate=activate,
            force=True,
        )
        result["superseded_prior"] = supersede
        return result

    # ── delete / unload ──────────────────────────────────────────────────────
    def _retract_axioms(self, turtle: str) -> dict[str, Any]:
        """Physically retract an ontology's axioms from the engine RDF dataset.

        The retract counterpart to :meth:`_load_axioms` (CONCEPT:AU-KG.ontology.ontology-lifecycle — wires
        KG-2.265's unload to the engine's ``remove_triples`` op). Feeds the stored
        serialized ``turtle`` back through ``GraphComputeEngine.remove_triples`` so the
        unloaded ontology's triples leave the engine's RDF dataset (stop being reasoned
        over / queried), not just the registry record. Degrades honestly when the
        engine / op is unavailable.
        """
        gc = self._gc
        if gc is None or not hasattr(gc, "remove_triples"):
            return {
                "retracted_from_engine": False,
                "reason": "no engine retract surface",
            }
        if not turtle:
            return {"retracted_from_engine": False, "reason": "no stored axioms"}
        try:
            report = gc.remove_triples(turtle=turtle)
            return {"retracted_from_engine": True, **(report or {})}
        except Exception as exc:  # noqa: BLE001 — engine optional / feature-gated
            logger.debug("remove_triples failed: %s", exc)
            return {"retracted_from_engine": False, "reason": str(exc)}

    def delete(
        self, iri: str, *, version: str | None = None, drop_inferences: bool = False
    ) -> dict[str, Any]:
        """Unload an ontology: retract its axioms from the engine + drop the registry record.

        CONCEPT:AU-KG.ontology.ontology-lifecycle — wires KG-2.265's unload to the engine's native
        ``remove_triples`` retract op. The stored serialized turtle for each matched
        version is fed back through :meth:`_retract_axioms` so the ontology's triples
        physically leave the engine's RDF dataset (no longer reasoned over / SPARQL-
        queryable), then the hosted-registry record is removed. When no engine is
        attached (or the op is unavailable) it degrades to the registry-only behaviour
        and reports the gap honestly.
        """
        if version is not None:
            keys = [_key(iri, version)] if self._store.get(_key(iri, version)) else []
        else:
            keys = [
                _key(r["iri"], r["version"])
                for r in self._store.values()
                if r.get("iri") == iri
            ]
        if not keys:
            return {"error": f"ontology not hosted: {iri} (version={version})"}

        removed = []
        retractions: list[dict[str, Any]] = []
        for k in keys:
            rec = self._store.delete(k)
            if rec is None:
                continue
            removed.append({"iri": rec["iri"], "version": rec["version"]})
            if self._gc is not None:
                retractions.append(self._retract_axioms(rec.get("turtle", "")))

        retracted = bool(retractions) and all(
            r.get("retracted_from_engine") for r in retractions
        )
        if self._gc is None:
            engine_note = "no engine attached"
        elif retracted:
            engine_note = (
                "axioms retracted from the engine RDF dataset (remove_triples)"
            )
        else:
            engine_note = (
                "; ".join(
                    r.get("reason", "retract failed")
                    for r in retractions
                    if not r.get("retracted_from_engine")
                )
                or "retract unavailable"
            )
        result: dict[str, Any] = {
            "status": "ok",
            "removed": removed,
            "axioms_retracted_from_engine": retracted,
            "engine_note": engine_note,
        }
        if retractions:
            result["retractions"] = retractions
        if drop_inferences and self._gc is not None:
            # Materialized entailments are derived facts in the live graph, not RDF
            # axioms; retracting the source axioms removes the basis but does not
            # re-run the reasoner. A full inference sweep needs a re-classify pass.
            result["inferences_dropped"] = False
            result["inferences_note"] = (
                "source axioms retracted; materialized inferences clear on the next "
                "owl_reason pass (no incremental un-materialize op)"
            )
        return result

    # ── activate / deactivate ────────────────────────────────────────────────
    def set_active(
        self, iri: str, *, version: str | None = None, active: bool = True
    ) -> dict[str, Any]:
        """Flip an ontology's participation in reasoning.

        Activating with a live engine (re)loads its axioms into the engine RDF
        dataset; deactivating flips the flag (axioms are not retracted — see the
        :meth:`delete` engine gap).
        """
        resolved = self._resolve(iri, version)
        if resolved is None:
            return {"error": f"ontology not hosted: {iri} (version={version})"}
        key, record = resolved
        if active:
            engine_report = self._load_axioms(record.get("turtle", ""))
            record["engine"] = engine_report
            # Same fail-closed rule as `load()`: only claim "active" when a
            # live engine confirms it, else keep the offline intent semantics.
            record["active"] = (
                True
                if self._gc is None
                else bool(engine_report.get("loaded_to_engine"))
            )
        else:
            record["active"] = False
        self._store.set(key, record)
        return {"status": "ok", "ontology": self._public(record)}

    # ── deprecate (advisory) ─────────────────────────────────────────────────
    def set_deprecated(
        self, iri: str, *, version: str | None = None, deprecated: bool = True
    ) -> dict[str, Any]:
        """Mark/unmark a hosted version as deprecated (CONCEPT:AU-KG.ontology.deprecation-workflow, D-75-5).

        Purely advisory: unlike :meth:`set_active`, this never touches the
        engine or reasoning participation — a superseded-and-inactive version
        was already indistinguishable from a deliberately-sunset one before
        this flag existed; this makes that distinction explicit and queryable
        (:meth:`list_ontologies`'s ``deprecated_only``) without changing what
        "active" means.
        """
        resolved = self._resolve(iri, version)
        if resolved is None:
            return {"error": f"ontology not hosted: {iri} (version={version})"}
        key, record = resolved
        record["deprecated"] = bool(deprecated)
        self._store.set(key, record)
        return {"status": "ok", "ontology": self._public(record)}

    # ── validate (no commit) ─────────────────────────────────────────────────
    def validate(self, source: str, *, source_type: str = "auto") -> dict[str, Any]:
        """Run the valid/connected/SHACL gate on a candidate WITHOUT hosting it."""
        try:
            graph = _parse_graph(source, source_type)
        except OntologyError as exc:
            return {"valid": False, "errors": [str(exc)], "warnings": [], "summary": {}}
        return validate_graph(graph)


def reset_registry() -> None:
    """Clear the in-process hosted-ontology registry (tests / clean-slate).

    Only the non-durable, in-memory fallback (:data:`_MEMORY_STORE`) is
    process-local and needs clearing between tests — a durable
    :class:`_EngineRegistryStore` is backed by the live engine and cleaning it
    is the caller's/fixture's responsibility (e.g. dropping the test graph),
    exactly like every other engine-backed KG fixture in this test suite.
    """
    _MEMORY_STORE.clear()
    _KNOWN_ONTOLOGY_GRAPHS.clear()
