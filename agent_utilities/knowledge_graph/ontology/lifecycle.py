"""Thin control-plane lifecycle for Epistemic Graph GraphSchema sources.

Agent Utilities does not parse, validate, register, cache, or reason over
ontology bodies. The generated EG GraphSchema contract owns attachment,
composition validation, persistence, and source metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class OntologyError(ValueError):
    """A GraphSchema lifecycle request is invalid or unsupported."""


def _source_document(source: str, source_type: str) -> str:
    """Read a caller-owned document without interpreting RDF semantics."""
    kind = (source_type or "auto").strip().lower()
    if kind == "auto":
        candidate = Path(source).expanduser()
        kind = "file" if "\n" not in source and candidate.is_file() else "text"
    if kind == "file":
        return Path(source).expanduser().read_text(encoding="utf-8")
    if kind == "text":
        return source
    raise OntologyError(
        "GraphSchema lifecycle accepts source_type='file' or 'text'; fetch remote "
        "content through its owning connector before attachment"
    )


def _source_id(iri: str, version: str) -> str:
    """Derive a stable admin source key without creating a second registry."""
    iri_digest = hashlib.sha256(iri.encode("utf-8")).hexdigest()[:32]
    version_digest = hashlib.sha256(version.encode("utf-8")).hexdigest()[:16]
    return f"admin:ontology:{iri_digest}:{version_digest}"


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    raise OntologyError("generated GraphSchema client returned an untyped result")


def _select_graph_compute(engine: Any, graph_name: str | None) -> Any:
    graph_compute = getattr(engine, "graph_compute", engine)
    for_graph = getattr(graph_compute, "for_graph", None)
    return (
        for_graph(graph_name) if graph_name and callable(for_graph) else graph_compute
    )


def _reject_legacy_filters(filters: tuple[Any, ...]) -> None:
    if any(filters):
        raise OntologyError(
            "legacy local-registry filters are unavailable; filter GraphSchema source metadata"
        )


def _admin_ontology_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in payload.get("dynamic_sources", [])
        if str(item.get("source_id", "")).startswith("admin:ontology:")
    ]


class OntologyLifecycle:
    """Map public ontology lifecycle verbs onto generated GraphSchema calls."""

    def __init__(
        self,
        engine: Any = None,
        *,
        tenant: str | None = None,
        graph_name: str | None = None,
    ) -> None:
        del tenant
        self._graph_compute = _select_graph_compute(engine, graph_name)

    def _require_engine(self) -> Any:
        gc = self._graph_compute
        required = ("graph_schema_attach", "graph_schema_detach", "graph_schema_list")
        if gc is None or any(not hasattr(gc, name) for name in required):
            raise OntologyError("generated EG GraphSchema authority is unavailable")
        return gc

    @staticmethod
    def _require_identity(iri: str | None, version: str | None) -> tuple[str, str]:
        if not iri or not version:
            raise OntologyError(
                "GraphSchema attachment requires explicit iri and version"
            )
        return iri, version

    def load(
        self,
        source: str,
        *,
        source_type: str = "auto",
        version: str | None = None,
        iri: str | None = None,
        activate: bool = True,
        category: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Attach one admin-owned ontology source to the current graph."""
        del category, tags
        if not activate:
            raise OntologyError("inactive local ontology records no longer exist")
        iri, version = self._require_identity(iri, version)
        body = _source_document(source, source_type)
        if not body.strip():
            raise OntologyError("ontology source is empty")
        gc = self._require_engine()
        source_id = _source_id(iri, version)
        receipt = gc.graph_schema_attach(source_id, ontology_ttl=body)
        return {
            "action": "attach",
            "iri": iri,
            "version": version,
            "source_id": source_id,
            **_dump(receipt),
        }

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
        """List EG source metadata; no local registry filters are emulated."""
        _reject_legacy_filters(
            (active_only, deprecated_only, search, category, source_type, tag)
        )
        view = self._require_engine().graph_schema_list()
        payload = _dump(view)
        sources = _admin_ontology_sources(payload)
        return {
            "count": len(sources),
            "ontologies": sources,
            "schema_version": payload.get("schema_version"),
            "graph": payload.get("graph"),
            "composed_digest": payload.get("composed_digest"),
        }

    def get(
        self, iri: str, *, version: str | None = None, serialize: bool = False
    ) -> dict[str, Any]:
        """Return GraphSchema metadata for one exact admin source."""
        if serialize:
            raise OntologyError(
                "GraphSchema is metadata-only and does not export source bodies"
            )
        iri, version = self._require_identity(iri, version)
        wanted = _source_id(iri, version)
        for item in self.list_ontologies()["ontologies"]:
            if item.get("source_id") == wanted:
                return {"iri": iri, "version": version, **item}
        raise OntologyError(f"GraphSchema source not found: {wanted}")

    def update(
        self,
        source: str,
        *,
        iri: str,
        version: str,
        source_type: str = "auto",
    ) -> dict[str, Any]:
        """Replace the exact admin source; EG validates composition atomically."""
        return self.load(
            source,
            source_type=source_type,
            iri=iri,
            version=version,
        )

    def delete(
        self,
        iri: str,
        *,
        version: str | None = None,
        drop_inferences: bool = False,
    ) -> dict[str, Any]:
        """Detach one exact admin source from the current graph."""
        if drop_inferences:
            raise OntologyError(
                "inference cleanup is owned by EG GraphSchema/Datalog materialization"
            )
        iri, version = self._require_identity(iri, version)
        source_id = _source_id(iri, version)
        receipt = self._require_engine().graph_schema_detach(source_id)
        return {
            "action": "detach",
            "iri": iri,
            "version": version,
            "source_id": source_id,
            **_dump(receipt),
        }
