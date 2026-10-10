"""Typed AU declaration publisher feeding the epistemic graph's pack compiler.

AU keeps only typed declarations; the RDF/OWL text is produced by the SDK's
pack compiler (``agent_connector_sdk.manifest.ontology_pack``), never by a
hand-written emitter here (AU-BOUNDARY-R030.7).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_connector_sdk.manifest.model import ConnectorManifest
from agent_connector_sdk.manifest.ontology_pack import compile_manifest_ontology
from pydantic import ValidationError


class OntologyDeclarationError(ValueError):
    """The declaration is not a valid typed connector manifest (fail closed)."""


def publish_declaration(declaration: ConnectorManifest | Mapping[str, Any]) -> str:
    """Validate a typed declaration and publish it as Turtle via pack compilation."""
    if isinstance(declaration, ConnectorManifest):
        manifest = declaration
    else:
        try:
            manifest = ConnectorManifest.model_validate(dict(declaration))
        except ValidationError as exc:
            raise OntologyDeclarationError(
                f"invalid ontology declaration: {exc}"
            ) from exc
    return compile_manifest_ontology(manifest)
