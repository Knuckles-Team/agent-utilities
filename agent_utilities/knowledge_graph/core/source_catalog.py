"""Atlas's governed external-source catalogue.

The catalogue is deliberately a *read projection*, not another connector
registry.  ``graph_configure`` owns named graph connections and
``source_connectors.registry`` owns executable source adapters; this module
only projects those existing registries together with the reference-only
``AgentConfig`` declarations into a bounded, provider-oriented view for Atlas.
Graph OS owns the typed connector/source control-plane contracts. They are not
live registries in this process, so this projection does not invent
control-plane observations until an approved runtime adapter exists.

No provider client is opened here.  In particular, the connection registry's
``connected`` bit means that a named adapter is already cached, not that this
read path performed a probe.  Generic OpenCypher and PuppyGraph therefore stay
``unverified`` until an explicit read probe has recorded an observation.  The
catalogue never returns resolved endpoint, DSN, user, credential, or probe
exception text.

CONCEPT:AU-KG.backend.connection-registry
CONCEPT:AU-ECO.connector.document-source-framework
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CatalogAvailability = Literal[
    "available", "configured", "unavailable", "unverified", "unsupported"
]
SourceKind = Literal[
    "database", "graph", "virtual_graph", "compute", "lakehouse", "object_store"
]
QueryMode = Literal["sql", "cypher", "graphql", "uql", "compute", "object_store"]
SyncMode = Literal["delta", "full", "reconcile"]

_REF_RE = re.compile(r"^(?:vault|env|secret)://[A-Za-z0-9_./#-]+$")
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_SAFE_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
_SYNC_MODES = frozenset({"delta", "full", "reconcile"})


class SourceSyncSupport(BaseModel):
    """The source-sync contract for one provider.

    ``entrypoint`` remains ``source_sync`` even when ``supported`` is false so
    clients have one canonical execution seam and cannot accidentally invent a
    provider-specific write route.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    modes: tuple[SyncMode, ...] = Field(default=(), max_length=3)
    entrypoint: Literal["source_sync"] = "source_sync"
    reason: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _modes_match_support(self) -> SourceSyncSupport:
        if self.supported and not self.modes:
            raise ValueError("a supported source sync must declare at least one mode")
        if not self.supported and self.modes:
            raise ValueError("an unsupported source sync cannot declare modes")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("source sync modes must be unique")
        return self


class QuerySurface(BaseModel):
    """A query language exposed by the existing Graph-OS read tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal["natural_language", "uql"]
    available: bool
    tool: str = Field(min_length=1, max_length=96)
    action: str | None = Field(default=None, max_length=96)
    reason: str = Field(min_length=1, max_length=512)


class SourceProviderDescriptor(BaseModel):
    """Safe, typed provider metadata consumed by Atlas.

    The reference fields are intentionally the only connection material in
    this model.  A profile reference identifies a governed secret/config
    lookup; it is never resolved by the catalogue.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, str_strip_whitespace=True
    )

    provider: str = Field(min_length=1, max_length=96)
    label: str = Field(min_length=1, max_length=160)
    kind: SourceKind
    availability: CatalogAvailability
    available: bool
    reason: str = Field(min_length=1, max_length=512)
    query_mode: QueryMode = Field(serialization_alias="queryMode")
    dialects: tuple[str, ...] = Field(min_length=1, max_length=16)
    sync: SourceSyncSupport
    capabilities: tuple[str, ...] = Field(min_length=1, max_length=32)
    connection_names: tuple[str, ...] = Field(
        default=(), serialization_alias="connectionNames", max_length=64
    )
    profile_ref: str | None = Field(
        default=None, serialization_alias="profileRef", max_length=192
    )
    connection_profile_ref: str | None = Field(
        default=None, serialization_alias="connectionProfileRef", max_length=192
    )
    auth_profile_ref: str | None = Field(
        default=None, serialization_alias="authProfileRef", max_length=192
    )
    tls_profile_ref: str | None = Field(
        default=None, serialization_alias="tlsProfileRef", max_length=192
    )
    mapping_policy_ref: str | None = Field(
        default=None, serialization_alias="mappingPolicyRef", max_length=192
    )
    variables_ref: str | None = Field(
        default=None, serialization_alias="variablesRef", max_length=192
    )

    @model_validator(mode="after")
    def _reference_fields_are_refs(self) -> SourceProviderDescriptor:
        for name in (
            "profile_ref",
            "connection_profile_ref",
            "auth_profile_ref",
            "tls_profile_ref",
            "mapping_policy_ref",
            "variables_ref",
        ):
            value = getattr(self, name)
            if value is not None and not _REF_RE.fullmatch(value):
                # ``profile_ref`` is a neutral profile identity, while every
                # other ref is a secret/config lookup.  Keep that distinction
                # explicit rather than accepting arbitrary sensitive strings.
                if name == "profile_ref" and _ALIAS_RE.fullmatch(value):
                    continue
                raise ValueError(f"{name} must be a runtime reference")
        if len(set(self.dialects)) != len(self.dialects):
            raise ValueError("provider dialects must be unique")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("provider capabilities must be unique")
        if len(set(self.connection_names)) != len(self.connection_names):
            raise ValueError("provider connection names must be unique")
        return self


class SourceCatalog(BaseModel):
    """Wire contract shared by the MCP and REST graph-catalog surfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["atlas-source-catalog.v1"]
    sources: tuple[SourceProviderDescriptor, ...] = Field(min_length=1, max_length=64)
    query_surfaces: tuple[QuerySurface, ...] = Field(
        min_length=1, max_length=8, serialization_alias="querySurfaces"
    )
    source_connector_types: tuple[str, ...] = Field(
        default=(), max_length=256, serialization_alias="sourceConnectorTypes"
    )

    @model_validator(mode="after")
    def _source_ids_are_unique(self) -> SourceCatalog:
        providers = [item.provider for item in self.sources]
        if len(set(providers)) != len(providers):
            raise ValueError("source provider ids must be unique")
        return self

    def wire(self) -> dict[str, Any]:
        """Return the JSON-safe transport projection."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


@dataclass(frozen=True, slots=True)
class _ProviderDefinition:
    provider: str
    label: str
    kind: SourceKind
    query_mode: QueryMode
    dialects: tuple[str, ...]
    capabilities: tuple[str, ...]
    aliases: frozenset[str]
    unsupported_reason: str | None = None


_DEFINITIONS: tuple[_ProviderDefinition, ...] = (
    _ProviderDefinition(
        "postgresql",
        "PostgreSQL",
        "database",
        "sql",
        ("postgresql", "sql"),
        ("query", "schema_discovery", "read_only"),
        frozenset({"postgres", "postgresql", "pg"}),
    ),
    _ProviderDefinition(
        "database",
        "Relational database (generic)",
        "database",
        "sql",
        ("sql",),
        ("query", "schema_discovery", "read_only"),
        frozenset({"database", "relational_database", "databases"}),
    ),
    _ProviderDefinition(
        "neo4j",
        "Neo4j graph",
        "graph",
        "cypher",
        ("neo4j", "cypher", "opencypher"),
        ("query", "schema_discovery", "read_only", "external_graph"),
        frozenset({"neo4j"}),
    ),
    _ProviderDefinition(
        "age",
        "Apache AGE graph",
        "graph",
        "cypher",
        ("age", "opencypher", "cypher"),
        ("query", "schema_discovery", "read_only", "external_graph"),
        frozenset({"age"}),
    ),
    _ProviderDefinition(
        "ladybug",
        "Ladybug graph",
        "graph",
        "cypher",
        ("ladybug", "opencypher", "cypher"),
        ("query", "schema_discovery", "read_only", "external_graph"),
        frozenset({"ladybug"}),
    ),
    _ProviderDefinition(
        "epistemic_graph",
        "Epistemic Graph",
        "graph",
        "cypher",
        ("epistemic_graph", "cypher", "uql"),
        ("query", "schema_discovery", "read_only", "external_graph"),
        frozenset({"epistemic_graph"}),
    ),
    _ProviderDefinition(
        "opencypher",
        "OpenCypher graph (generic)",
        "graph",
        "cypher",
        ("opencypher", "cypher"),
        ("query", "schema_discovery", "read_only", "external_graph", "read_probe"),
        frozenset({"opencypher", "cypher"}),
    ),
    _ProviderDefinition(
        "puppygraph",
        "PuppyGraph (OpenCypher)",
        "virtual_graph",
        "cypher",
        ("opencypher", "cypher"),
        ("query", "federation", "read_only", "read_probe"),
        frozenset({"puppygraph", "puppy_graph"}),
    ),
    _ProviderDefinition(
        "graphql",
        "GraphQL graph source",
        "graph",
        "graphql",
        ("graphql",),
        ("query", "schema_discovery", "read_only", "external_graph"),
        frozenset({"graphql"}),
    ),
    _ProviderDefinition(
        "virtual_graph",
        "Virtual graph (generic)",
        "virtual_graph",
        "uql",
        ("uql", "opencypher"),
        ("federation", "query", "read_only", "read_probe"),
        frozenset({"virtual_graph", "virtualgraph"}),
    ),
    _ProviderDefinition(
        "spark",
        "Apache Spark",
        "compute",
        "compute",
        ("spark-sql", "spark-connect"),
        ("compute", "jobs", "read_only"),
        frozenset({"spark", "spark_connect", "spark_compute"}),
    ),
    _ProviderDefinition(
        "iceberg",
        "Apache Iceberg",
        "lakehouse",
        "sql",
        ("iceberg-rest", "sql"),
        ("catalog", "query", "schema_discovery", "read_only"),
        frozenset({"iceberg", "iceberg_rest", "lakehouse"}),
    ),
    _ProviderDefinition(
        "trino",
        "Trino",
        "lakehouse",
        "sql",
        ("trino", "sql"),
        ("catalog", "query", "federation", "read_only"),
        frozenset({"trino"}),
    ),
    _ProviderDefinition(
        "s3",
        "Amazon S3-compatible object store",
        "object_store",
        "object_store",
        ("s3", "http"),
        ("object_listing", "object_read", "metadata", "read_only"),
        frozenset({"s3", "amazon_s3", "minio", "seaweedfs"}),
    ),
    _ProviderDefinition(
        "object_store",
        "Object store (generic)",
        "object_store",
        "object_store",
        ("object_store",),
        ("object_listing", "object_read", "metadata", "read_only"),
        frozenset({"object_store", "object-store", "blob_store"}),
    ),
    _ProviderDefinition(
        "teradata",
        "Teradata",
        "database",
        "sql",
        ("teradata", "tds", "sql"),
        ("read_only",),
        frozenset({"teradata", "td"}),
        unsupported_reason=(
            "unsupported: no certified Teradata source connector or governed "
            "graph connection adapter is registered"
        ),
    ),
)

_PROVIDER_BY_TOKEN = {
    token.replace("-", "_"): definition.provider
    for definition in _DEFINITIONS
    for token in (definition.provider, *definition.aliases)
}
_PROVIDER_PREFIXES = tuple(
    (token.replace("-", "_"), definition.provider)
    for definition in _DEFINITIONS
    for token in (definition.provider, *definition.aliases)
)


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        try:
            rendered = value.model_dump(mode="python", by_alias=False)
        except Exception:  # noqa: BLE001 - a malformed optional profile is absent
            return {}
        return dict(rendered) if isinstance(rendered, Mapping) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _safe_ref(value: Any) -> str | None:
    rendered = str(value or "").strip()
    return rendered if _REF_RE.fullmatch(rendered) else None


def _first_safe_ref(value: Any) -> str | None:
    """Return one deterministic runtime reference from a bounded mapping.

    ``ProviderRuntimeProfile`` stores credential and selector references as
    alias-to-reference mappings.  The catalogue does not need (and must not
    expose) those aliases or their resolved values, but a provider-level
    connection/auth reference is useful to a governed client.  Pick the first
    reference in sorted alias order so the projection is stable while keeping
    the mapping itself private.
    """

    if not isinstance(value, Mapping):
        return None
    for alias in sorted(value, key=lambda item: str(item)):
        reference = _safe_ref(value.get(alias))
        if reference is not None:
            return reference
    return None


def _neutral_alias(value: Any) -> str:
    rendered = str(value or "").strip().lower().replace("-", "_")
    return rendered if _ALIAS_RE.fullmatch(rendered) else ""


def _definition_for(value: Any, *, backend: Any = None) -> str | None:
    """Map a registry/profile token to a canonical definition id."""

    backend_token = _neutral_alias(backend)
    token = _neutral_alias(value)
    # A PuppyGraph declaration uses the generic ``opencypher`` backend in the
    # existing graph registry.  Preserve its provider identity when the
    # neutral alias makes that distinction explicit; otherwise the backend
    # selector would collapse every such declaration into OpenCypher.
    if token in {"puppygraph", "puppy_graph"} or token.startswith(
        ("puppygraph_", "puppy_graph_")
    ):
        return "puppygraph"
    candidates = [item for item in (backend_token, token) if item]
    for candidate in candidates:
        provider = _PROVIDER_BY_TOKEN.get(candidate)
        if provider is not None:
            return provider
    # Neutral profile names commonly carry the provider prefix (for example
    # ``postgres-primary``); accept only a known prefix, never arbitrary text.
    for candidate in candidates:
        for alias, provider in _PROVIDER_PREFIXES:
            if candidate.startswith(alias + "_"):
                return provider
    return None


def _probe_recorded(value: Mapping[str, Any]) -> bool:
    """Read only an explicit, already-recorded probe observation."""

    return bool(value.get("probed")) or str(
        value.get("probe_status") or ""
    ).casefold() in {"ready", "ok", "available"}


def _runtime_config() -> Any | None:
    try:
        from agent_utilities.core.config import config

        return config
    except Exception:  # noqa: BLE001 - static catalogue still remains useful
        return None


def _config_fields(config: Any) -> tuple[Any, Any]:
    values = _dump(config)
    external = getattr(config, "external_graph_connectors", None)
    profiles = getattr(config, "provider_configs", None)
    if external is None:
        external = values.get("external_graph_connectors") or values.get(
            "EXTERNAL_GRAPH_CONNECTORS", []
        )
    if profiles is None:
        profiles = values.get("provider_configs") or values.get("PROVIDER_CONFIGS", {})
    return external, profiles


def _config_values(config: Any = None) -> tuple[list[Any], Mapping[str, Any]]:
    config = config or _runtime_config()
    if config is None:
        return [], {}

    external, profiles = _config_fields(config)
    if not isinstance(external, Sequence) or isinstance(external, (str, bytes)):
        external = []
    if not isinstance(profiles, Mapping):
        profiles = {}
    return list(external), profiles


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _registry_status_items(registry: Any) -> list[Mapping[str, Any]]:
    try:
        raw = registry.status()
    except Exception:  # noqa: BLE001 - one registry health fault must not break reads
        return []
    entries = raw.get("connections", []) if isinstance(raw, Mapping) else []
    return _mapping_items(entries)


def _registry_spec_items(registry: Any) -> list[Mapping[str, Any]]:
    try:
        return _mapping_items(registry.export_specs())
    except Exception:  # noqa: BLE001 - transient/literal specs are simply omitted
        return []


def _registry_item_name(item: Mapping[str, Any]) -> str | None:
    """The named-reference guard both registry projections share
    (CX-DUP-ENFORCE): ``None`` for an unnamed or ``"default"`` entry, else
    the stripped name."""
    name = str(item.get("name") or "").strip()
    if not name or name.casefold() == "default":
        return None
    return name


def _registry_status_record(item: Mapping[str, Any]) -> dict[str, Any] | None:
    name = _registry_item_name(item)
    if name is None:
        return None
    backend = item.get("backend_type")
    return {
        "name": name,
        "backend": str(backend or ""),
        "connected": bool(item.get("connected")),
        "probe_required": _probe_required(backend, name),
        "probed": _probe_recorded(item),
    }


def _registry_spec_record(item: Mapping[str, Any]) -> dict[str, Any] | None:
    name = _registry_item_name(item)
    if name is None:
        return None
    backend = item.get("backend_type") or item.get("backend")
    # Only named references survive this projection.  Never copy
    # endpoint/host/uri/database/user/password values.
    return {
        "name": name,
        "backend": str(backend or ""),
        "probe_required": _probe_required(backend, name),
        "connection_profile_ref": _safe_ref(item.get("connection_profile_ref")),
        "auth_profile_ref": _safe_ref(item.get("auth_profile_ref")),
        "tls_profile_ref": _safe_ref(item.get("tls_profile_ref")),
        "mapping_policy_ref": _safe_ref(item.get("mapping_policy_ref")),
        "variables_ref": _safe_ref(item.get("variables_ref")),
    }


def _project_registry_records(
    items: Sequence[Mapping[str, Any]],
    projector: Callable[[Mapping[str, Any]], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        record = projector(item)
        if record is not None:
            records.append(record)
    return records


def _registry_values(
    registry: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the registry's safe status/spec projection with strict whitelists."""

    statuses = _project_registry_records(
        _registry_status_items(registry), _registry_status_record
    )
    specs = _project_registry_records(
        _registry_spec_items(registry), _registry_spec_record
    )
    return statuses, specs


def _probe_required(backend: Any, name: Any = None) -> bool:
    """Whether a connection uses a generic, not provider-specific adapter."""

    backend_token = _neutral_alias(backend)
    name_token = _neutral_alias(name)
    return backend_token in {"opencypher"} or name_token.startswith(
        ("puppygraph", "puppy_graph", "virtual_graph")
    )


def _source_connector_types() -> tuple[str, ...]:
    """Use the executable source registry as the sync-adapter authority."""

    try:
        from agent_utilities.protocols.source_connectors.registry import list_sources

        values = [str(item).strip() for item in list_sources()]
    except Exception:  # noqa: BLE001 - optional connector discovery is best effort
        values = []
    # Registry keys are source-type aliases, not external coordinates.  Keep
    # this projection on the same neutral grammar as provider/connection names
    # so an accidental URL/DSN-like registration cannot cross the public
    # catalogue boundary.
    return tuple(sorted({item for item in values if _ALIAS_RE.fullmatch(item)}))


def _sync_support(
    definition: _ProviderDefinition, connector_types: Sequence[str]
) -> SourceSyncSupport:
    # ``source_sync`` has no generic pass-through for database/graph/profile
    # entries.  A registered document connector is not silently upgraded into
    # a source_sync adapter: it is reported as its own non-canonical seam.
    if definition.provider in {"spark"}:
        return SourceSyncSupport(
            supported=False,
            reason="compute-only provider; it does not expose source_sync ingestion",
        )
    if definition.provider == "teradata":
        return SourceSyncSupport(
            supported=False,
            reason="unsupported provider; no source_sync adapter is registered",
        )
    if definition.provider in {"s3", "object_store", "iceberg", "trino"}:
        return SourceSyncSupport(
            supported=False,
            reason="no governed source_sync adapter is registered for this provider",
        )
    if definition.provider in {"opencypher", "puppygraph", "graphql", "virtual_graph"}:
        return SourceSyncSupport(
            supported=False,
            reason=(
                "external graph configuration/ingest is owned by graph_configure; "
                "a source_sync adapter is not registered"
            ),
        )
    if "database" in connector_types:
        return SourceSyncSupport(
            supported=False,
            reason=(
                "the database source connector is registered for source_connector; "
                "source_sync has no database adapter yet"
            ),
        )
    return SourceSyncSupport(
        supported=False,
        reason="no governed source_sync adapter is registered for this provider",
    )


def _base_match(definition: _ProviderDefinition) -> dict[str, Any]:
    return {
        "configured": False,
        "unavailable_reason": "no governed profile or named connection is configured",
        "enabled": True,
        "connected": False,
        "probe_required": False,
        "probed": False,
        "names": set(),
        "profile_ref": None,
        "connection_profile_ref": None,
        "auth_profile_ref": None,
        "tls_profile_ref": None,
        "mapping_policy_ref": None,
        "variables_ref": None,
    }


def _merge_match_flags(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    target["configured"] = bool(target["configured"] or value.get("configured"))
    target["enabled"] = bool(target["enabled"] and value.get("enabled", True))
    target["connected"] = bool(target["connected"] or value.get("connected"))
    target["probe_required"] = bool(
        target["probe_required"] or value.get("probe_required")
    )
    target["probed"] = bool(target["probed"] or value.get("probed"))


def _merge_match_name(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    name = str(value.get("name") or "").strip()
    # Registry aliases are operator-controlled and some legacy registrations
    # are less strict than AgentConfig.  Only emit the neutral alias grammar;
    # this prevents a URL/DSN accidentally supplied as a name from becoming a
    # catalog connection identifier.
    if name and _ALIAS_RE.fullmatch(name):
        target["names"].add(name)


def _merge_match_refs(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    for key in (
        "profile_ref",
        "connection_profile_ref",
        "auth_profile_ref",
        "tls_profile_ref",
        "mapping_policy_ref",
        "variables_ref",
    ):
        if target[key] is None and value.get(key):
            target[key] = value[key]


def _merge_match(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    _merge_match_flags(target, value)
    _merge_match_name(target, value)
    _merge_match_refs(target, value)


def _availability(
    definition: _ProviderDefinition, match: Mapping[str, Any]
) -> tuple[CatalogAvailability, bool, str]:
    if definition.unsupported_reason:
        return "unsupported", False, definition.unsupported_reason
    if not match.get("configured"):
        return (
            "unavailable",
            False,
            str(match["unavailable_reason"]),
        )
    if not match.get("enabled"):
        return "configured", False, "governed provider profile is disabled"
    if match.get("probe_required") and not match.get("probed"):
        return (
            "unverified",
            False,
            "profile is configured, but an explicit read probe is required before use",
        )
    if match.get("connected"):
        return "available", True, "registered connection is live"
    return (
        "configured",
        False,
        "governed profile is configured; provider probe has not been recorded",
    )


def _merge_registry_matches(
    matches: dict[str, dict[str, Any]],
    statuses: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
) -> None:
    """Merge registry observations before lower-authority declarations."""

    for item in (*statuses, *specs):
        provider = _definition_for(item.get("name"), backend=item.get("backend"))
        if provider is None:
            continue
        _merge_match(
            matches[provider],
            {**dict(item), "configured": True, "enabled": True},
        )


def _merge_external_entry(matches: dict[str, dict[str, Any]], raw: Any) -> None:
    value = _dump(raw)
    name = value.get("name") or value.get("source_alias")
    provider = _definition_for(
        value.get("source_alias") or name, backend=value.get("backend")
    )
    if provider is None:
        return
    _merge_match(
        matches[provider],
        {
            "configured": True,
            "enabled": True,
            "name": name,
            "probe_required": _probe_required(value.get("backend"), name),
            "probed": _probe_recorded(value),
            "connection_profile_ref": _safe_ref(value.get("connection_profile_ref")),
            "auth_profile_ref": _safe_ref(value.get("auth_profile_ref")),
            "tls_profile_ref": _safe_ref(value.get("tls_profile_ref")),
            "mapping_policy_ref": _safe_ref(value.get("mapping_policy_ref")),
            "variables_ref": _safe_ref(value.get("variables_ref")),
        },
    )


def _merge_external_matches(
    matches: dict[str, dict[str, Any]], external: Sequence[Any]
) -> None:
    for raw in external:
        _merge_external_entry(matches, raw)


def _profile_match(
    raw_name: Any, raw_profile: Any
) -> tuple[str, dict[str, Any]] | None:
    value = _dump(raw_profile)
    provider = _definition_for(raw_name)
    if provider is None:
        return None
    auth_profile_ref = _first_safe_ref(value.get("credential_refs"))
    has_refs = bool(
        any(_safe_ref(value.get(key)) for key in ("endpoint_ref", "tls_profile_ref"))
        or value.get("credential_refs")
        or value.get("selector_refs")
    )
    return provider, {
        "configured": bool(value.get("enabled") or has_refs),
        "enabled": bool(value.get("enabled", False)),
        "name": raw_name,
        "probe_required": provider in {"opencypher", "puppygraph", "virtual_graph"},
        "probed": _probe_recorded(value),
        "profile_ref": raw_name if _ALIAS_RE.fullmatch(str(raw_name)) else None,
        # ProviderRuntimeProfile keeps these as runtime references; projecting
        # only explicit auth/TLS refs lets an Atlas client associate governed
        # policy without exposing values. Its endpoint_ref is deliberately
        # omitted: it identifies a resolved endpoint, not a connection profile,
        # and must never be relabeled as one in this public projection.
        "auth_profile_ref": auth_profile_ref,
        "tls_profile_ref": _safe_ref(value.get("tls_profile_ref")),
    }


def _merge_profile_matches(
    matches: dict[str, dict[str, Any]], profiles: Mapping[str, Any]
) -> None:
    for raw_name, raw_profile in profiles.items():
        result = _profile_match(raw_name, raw_profile)
        if result is not None:
            provider, value = result
            _merge_match(matches[provider], value)


def _mask_unbound_registry(
    matches: Mapping[str, dict[str, Any]], registry_available: bool
) -> dict[str, dict[str, Any]]:
    """Keep config declarations non-executable until composition binds a registry."""
    return {
        provider: {
            **match,
            "configured": bool(registry_available and match["configured"]),
            "unavailable_reason": (
                "no governed profile or named connection is configured"
                if registry_available
                else "connection registry was not supplied by composition"
            ),
        }
        for provider, match in matches.items()
    }


def _catalog_matches(
    statuses: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    external: Sequence[Any],
    profiles: Mapping[str, Any],
    *,
    registry_available: bool = True,
) -> dict[str, dict[str, Any]]:
    matches = {
        definition.provider: _base_match(definition) for definition in _DEFINITIONS
    }
    _merge_registry_matches(matches, statuses, specs)
    _merge_external_matches(matches, external)
    _merge_profile_matches(matches, profiles)
    return _mask_unbound_registry(matches, registry_available)


def _descriptor(
    definition: _ProviderDefinition,
    match: Mapping[str, Any],
    connector_types: Sequence[str],
) -> SourceProviderDescriptor:
    availability, available, reason = _availability(definition, match)
    return SourceProviderDescriptor(
        provider=definition.provider,
        label=definition.label,
        kind=definition.kind,
        availability=availability,
        available=available,
        reason=reason,
        query_mode=definition.query_mode,
        dialects=definition.dialects,
        sync=_sync_support(definition, connector_types),
        capabilities=definition.capabilities,
        connection_names=tuple(sorted(match["names"])),
        profile_ref=match["profile_ref"],
        connection_profile_ref=match["connection_profile_ref"],
        auth_profile_ref=match["auth_profile_ref"],
        tls_profile_ref=match["tls_profile_ref"],
        mapping_policy_ref=match["mapping_policy_ref"],
        variables_ref=match["variables_ref"],
    )


def _query_surfaces() -> tuple[QuerySurface, ...]:
    return (
        QuerySurface(
            id="natural_language",
            available=True,
            tool="graph_ask",
            reason="Graph-OS natural-language planning is exposed by graph_ask/nl_query",
        ),
        QuerySurface(
            id="uql",
            available=True,
            tool="engine_query",
            action="uql",
            reason="the engine_query uql action is the governed cross-modal query seam",
        ),
    )


def build_source_catalog(*, config: Any = None, registry: Any = None) -> dict[str, Any]:
    """Build the bounded Atlas source catalogue without opening a provider.

    ``config`` and ``registry`` are injectable for tests and for callers that
    already resolved the process-owned registries.  Composition must pass the
    process-owned ``ConnectionRegistry`` explicitly; this lower layer never
    imports an MCP module to discover one.  Omitting ``registry`` therefore
    produces an explicitly unavailable catalog state rather than a fabricated
    empty-registry success.
    """

    statuses, specs = _registry_values(registry) if registry is not None else ([], [])
    external, profiles = _config_values(config)
    connector_types = _source_connector_types()
    matches = _catalog_matches(
        statuses,
        specs,
        external,
        profiles,
        registry_available=registry is not None,
    )
    descriptors = tuple(
        _descriptor(definition, matches[definition.provider], connector_types)
        for definition in _DEFINITIONS
    )
    catalog = SourceCatalog(
        schema_version="atlas-source-catalog.v1",
        sources=descriptors,
        query_surfaces=_query_surfaces(),
        source_connector_types=connector_types,
    )
    return catalog.wire()


class SourceSyncPreview(BaseModel):
    """A normalized, non-executable preview of one ``source_sync`` request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["atlas-source-sync-preview.v1"]
    source: str = Field(min_length=1, max_length=128)
    mode: SyncMode
    ids: tuple[str, ...] = Field(default=(), max_length=100)
    connection: str | None = None
    graph: str | None = None
    entrypoint: Literal["source_sync"] = "source_sync"
    would_execute: Literal[False] = Field(
        default=False, serialization_alias="wouldExecute"
    )
    reason: str = Field(
        default="preview only; execute the normalized request through source_sync",
        max_length=256,
    )

    @model_validator(mode="after")
    def _single_source_and_safe_aliases(self) -> SourceSyncPreview:
        if self.source.casefold() in {"all", "*", "sweep"}:
            raise ValueError("preview accepts exactly one source, not a fleet sweep")
        for value in (self.source, self.connection, self.graph):
            if value is not None and not _ALIAS_RE.fullmatch(value):
                raise ValueError(
                    "source_sync preview selectors must be neutral aliases"
                )
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("source_sync preview ids must be unique")
        return self


def _preview_ids(ids_json: str) -> tuple[str, ...]:
    try:
        parsed_ids = json.loads(ids_json) if ids_json else []
    except (TypeError, ValueError) as exc:
        raise ValueError("ids_json must be a JSON list") from exc
    if not isinstance(parsed_ids, list):
        raise ValueError("ids_json must be a JSON list")
    ids: list[str] = []
    for value in parsed_ids:
        rendered = str(value).strip()
        if (
            not rendered
            or len(rendered) > 192
            or not _SAFE_RECORD_ID_RE.fullmatch(rendered)
        ):
            raise ValueError("ids_json contains an invalid id")
        ids.append(rendered)
    return tuple(ids)


def normalize_source_sync_preview(
    *,
    source: str,
    mode: str = "delta",
    ids_json: str = "[]",
    connection: str = "",
    graph: str = "",
) -> dict[str, Any]:
    """Validate and normalize one sync request without dispatching it."""

    rendered_source = str(source or "").strip().lower()
    rendered_mode = str(mode or "delta").strip().lower()
    if rendered_mode not in _SYNC_MODES:
        raise ValueError("mode must be one of delta, full, or reconcile")
    return SourceSyncPreview(
        schema_version="atlas-source-sync-preview.v1",
        source=rendered_source,
        mode=rendered_mode,  # type: ignore[arg-type]
        ids=_preview_ids(ids_json),
        connection=(str(connection or "").strip().lower() or None),
        graph=(str(graph or "").strip().lower() or None),
    ).model_dump(mode="json", by_alias=True, exclude_none=True)


__all__ = [
    "QuerySurface",
    "SourceCatalog",
    "SourceProviderDescriptor",
    "SourceSyncPreview",
    "build_source_catalog",
    "normalize_source_sync_preview",
]
