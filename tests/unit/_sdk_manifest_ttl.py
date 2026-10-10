"""Shared helper: compile an AU ``ConnectorManifest`` to Turtle via the SDK pack."""

from __future__ import annotations

from typing import Any

from agent_connector_sdk.manifest.model import ConnectorManifest as SDKManifest
from agent_connector_sdk.manifest.ontology_pack import compile_manifest_ontology


def sdk_manifest_ttl(manifest: Any) -> str:
    """Turtle for ``manifest`` from ``compile_manifest_ontology`` (the SDK owner)."""
    return compile_manifest_ontology(
        SDKManifest.model_validate(manifest.model_dump(mode="python"))
    )
