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
  ``owlrl``) so the whole surface works with **no engine** (unit-testable).
* When a live engine is present, ``load``/``update`` push the ontology's axioms
  into the engine's native RDF dataset via ``GraphComputeEngine.add_triples`` so
  the native OWL reasoner (``owl_reason``) and SPARQL surface immediately operate
  over them — that is what "active for reasoning" means here.
* The registry of hosted ontologies is a process-wide singleton keyed by
  ``(iri, version)``; ``load`` is idempotent on that key. Records carry the
  metadata (#classes/#properties/#axioms, source, loaded_at, active) the ``list``
  surface reports.

Engine gap (documented, not worked around — eg-rdf is owned by another agent):
the engine RDF surface exposes ``add_triples`` (load) and ``get_triples`` /
``sparql`` (read) but **no remove-triples / drop-named-graph op**, so ``delete``
deactivates an ontology and drops it from the hosted registry but cannot
physically retract its axioms from the engine's RDF dataset until the engine
reloads. See :meth:`OntologyLifecycle.delete`.
"""

from __future__ import annotations

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


# Process-wide hosted-ontology registry (authoritative for the live process).
# key = f"{iri}@@{version}" → record dict. A module singleton so a sequence of
# graph_ontology calls (load → list → get → delete) is consistent within one
# graph-os process without a round-trip per call.
_REGISTRY: dict[str, dict[str, Any]] = {}


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
                conforms, _g, _txt = pyshacl.validate(
                    data_graph=graph,
                    shacl_graph=shapes,
                    inference="none",
                    abort_on_first=False,
                )
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
    }


class OntologyLifecycle:
    """CRUD lifecycle for ontologies hosted in the running KG (CONCEPT:AU-KG.ontology.manage-arbitrary).

    Args:
        engine: Optional live engine exposing ``graph_compute`` with the native
            RDF surface (``add_triples`` / ``owl_reason`` / ``sparql``). When
            ``None`` the lifecycle still parses/validates/inspects/registers
            ontologies (offline), it just cannot push axioms into a reasoner.
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine

    # ── internals ────────────────────────────────────────────────────────────
    @property
    def _graph_compute(self) -> Any:
        return getattr(self._engine, "graph_compute", None) if self._engine else None

    def _load_axioms(self, turtle: str) -> dict[str, Any]:
        """Push an ontology's axioms into the engine's native RDF dataset."""
        gc = self._graph_compute
        if gc is None or not hasattr(gc, "add_triples"):
            return {"loaded_to_engine": False, "reason": "no engine RDF surface"}
        try:
            report = gc.add_triples(turtle=turtle)
            return {"loaded_to_engine": True, **(report or {})}
        except Exception as exc:  # noqa: BLE001 — engine optional / feature-gated
            logger.debug("add_triples failed: %s", exc)
            return {"loaded_to_engine": False, "reason": str(exc)}

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
    ) -> dict[str, Any]:
        """Parse, validate, register, and (if a live engine) activate an ontology.

        Idempotent on ``(iri, version)``: loading the same IRI+version twice
        returns the existing record (``idempotent: true``) unless ``force``.
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

        existing = _REGISTRY.get(key)
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

        record: dict[str, Any] = {
            "iri": resolved_iri,
            "version": resolved_version,
            "source": source if len(source) < 256 else f"{source[:240]}…",
            "source_type": source_type,
            "n_axioms": summary["n_axioms"],
            "n_classes": summary["n_classes"],
            "n_properties": summary["n_properties"],
            "loaded_at": _now(),
            "active": bool(activate),
            "warnings": report["warnings"],
            "engine": engine_report,
            "turtle": turtle,
        }
        _REGISTRY[key] = record
        logger.info(
            "Hosted ontology loaded: %s v%s (%d axioms, active=%s)",
            resolved_iri,
            resolved_version,
            summary["n_axioms"],
            activate,
        )
        return {"status": "ok", "idempotent": False, "ontology": self._public(record)}

    # ── list ─────────────────────────────────────────────────────────────────
    def list_ontologies(self, *, active_only: bool = False) -> dict[str, Any]:
        """All hosted ontologies with metadata (newest first)."""
        records = [
            self._public(r)
            for r in _REGISTRY.values()
            if not active_only or r.get("active")
        ]
        records.sort(key=lambda r: r.get("loaded_at", ""), reverse=True)
        return {"count": len(records), "ontologies": records}

    # ── get / inspect ────────────────────────────────────────────────────────
    def _resolve(
        self, iri: str, version: str | None
    ) -> tuple[str, dict[str, Any]] | None:
        if version is not None:
            key = _key(iri, version)
            rec = _REGISTRY.get(key)
            return (key, rec) if rec else None
        # No version → newest loaded version of this IRI.
        candidates = [(k, r) for k, r in _REGISTRY.items() if r.get("iri") == iri]
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
            for r in _REGISTRY.values():
                if r.get("iri") == iri and r.get("version") != version:
                    r["active"] = False
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
        gc = self._graph_compute
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
            keys = [_key(iri, version)] if _key(iri, version) in _REGISTRY else []
        else:
            keys = [k for k, r in _REGISTRY.items() if r.get("iri") == iri]
        if not keys:
            return {"error": f"ontology not hosted: {iri} (version={version})"}

        removed = []
        retractions: list[dict[str, Any]] = []
        for k in keys:
            rec = _REGISTRY.pop(k)
            removed.append({"iri": rec["iri"], "version": rec["version"]})
            if self._graph_compute is not None:
                retractions.append(self._retract_axioms(rec.get("turtle", "")))

        retracted = bool(retractions) and all(
            r.get("retracted_from_engine") for r in retractions
        )
        if self._graph_compute is None:
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
        if drop_inferences and self._graph_compute is not None:
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
        _key_, record = resolved
        record["active"] = bool(active)
        if active:
            record["engine"] = self._load_axioms(record.get("turtle", ""))
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
    """Clear the in-process hosted-ontology registry (tests / clean-slate)."""
    _REGISTRY.clear()
