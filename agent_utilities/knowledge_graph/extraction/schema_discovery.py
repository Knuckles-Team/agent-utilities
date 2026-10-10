"""Schema discovery backed by the SDK manifest ontology-pack compiler.

CONCEPT:AU-BOUNDARY-R030.8

``discover_schema`` validates a connector manifest against the SDK
``ConnectorManifest`` and compiles it with
``agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology_spec``.
The result is a set of frozen models. Any failure raises
``SchemaDiscoveryError``; there is no partial result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict


class SchemaDiscoveryError(Exception):
    """Raised when a manifest cannot be validated or compiled."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DiscoveredClass(_Frozen):
    local: str
    label: str
    parent: str | None
    id_prefix: str


class DiscoveredRelation(_Frozen):
    local: str
    label: str
    domain: str | None
    range: str
    lpg_rel_type: str


class DiscoveredDatatypeProperty(_Frozen):
    local: str
    label: str
    range: str


class DiscoveredSchema(_Frozen):
    classes: tuple[DiscoveredClass, ...]
    relations: tuple[DiscoveredRelation, ...]
    datatype_properties: tuple[DiscoveredDatatypeProperty, ...]


def discover_schema(manifest: Any) -> DiscoveredSchema:
    """Validate ``manifest`` and compile it into a :class:`DiscoveredSchema`."""
    try:
        from agent_connector_sdk.manifest.model import ConnectorManifest
        from agent_connector_sdk.manifest.ontology_pack import (
            compile_manifest_ontology_spec,
        )

        if isinstance(manifest, ConnectorManifest):
            validated = ConnectorManifest.model_validate(manifest.model_dump())
        elif isinstance(manifest, Mapping):
            validated = ConnectorManifest.model_validate(dict(manifest))
        else:
            raise TypeError(
                f"manifest must be a ConnectorManifest or mapping, got {type(manifest).__name__}"
            )
        spec = compile_manifest_ontology_spec(validated)
        return DiscoveredSchema(
            classes=tuple(
                DiscoveredClass(
                    local=c.local,
                    label=c.label,
                    parent=c.parent,
                    id_prefix=c.id_prefix,
                )
                for c in spec.classes
            ),
            relations=tuple(
                DiscoveredRelation(
                    local=r.local,
                    label=r.label,
                    domain=r.domain,
                    range=r.range,
                    lpg_rel_type=r.lpg_rel_type,
                )
                for r in spec.object_properties
            ),
            datatype_properties=tuple(
                DiscoveredDatatypeProperty(local=d.local, label=d.label, range=d.range)
                for d in spec.datatype_properties
            ),
        )
    except SchemaDiscoveryError:
        raise
    except Exception as exc:
        raise SchemaDiscoveryError(f"schema discovery failed: {exc}") from exc
