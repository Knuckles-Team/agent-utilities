"""Governed GraphOS lifecycle for external GraphQL document sources.

This module is the narrow bridge between a named ``graph_configure`` connection
and :class:`GraphQLDocumentConnector`.  Durable configuration contains only a
neutral alias and runtime secret references.  Endpoint, authentication headers,
TLS material, configured or introspection-generated query documents, field
mappings, variables, and discovered schema are resolved only inside this
boundary and are never returned by public actions.

The lifecycle is deliberately proposal based:

* discover a bounded, read-only field graph;
* validate a secret-backed or structurally generated policy and store a
  pseudonymous proposal;
* approve the exact schema and policy digests;
* rediscover immediately before a dry-run or ingest and fail closed on drift;
* drain the native connector through the authoritative ChangeEnvelope path.

CONCEPT:AU-KG.ingest.external-graph-universal-discovery
CONCEPT:AU-KG.ingest.external-graph-mapping-approval
CONCEPT:AU-KG.ingest.change-envelope
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

from .external_graph_schema import (
    BackendCapabilities,
    ExternalGraphSchemaError,
    GraphQLDiscoveredSchema,
    GraphQLDiscoveryAdapter,
    SecretStore,
    canonical_identity_key_ref,
    canonical_profile_ref,
)

_ALIAS_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_SECRET_REF_RE = re.compile(r"^(?:vault|env|secret)://[A-Za-z0-9_./#-]+$")
_HEADER_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")
_PROFILE_FORMAT = "graphql-document-profile/v1"
GRAPHQL_CONNECTION_PROFILE_FORMAT = "graphql-connection/v1"
GRAPHQL_MAPPING_POLICY_FORMAT = "graphql-document-policy/v1"
GRAPHQL_AUTH_PROFILE_FORMAT = "graphql-auth/v1"
_POLICY_FORMAT = GRAPHQL_MAPPING_POLICY_FORMAT
_MAX_SECRET_BYTES = 4 * 1024 * 1024
_MAX_OPERATIONS = 64
_MAX_GENERATED_OPERATIONS = 16
_MAX_GENERATED_FIELDS = 32
_BLOCKED_REQUEST_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_MAPPING_POLICY_PRIVATE_FIELDS = frozenset(
    {
        "access_token",
        "api_key",
        "api_token",
        "auth",
        "auth_profile_ref",
        "authorization",
        "client_key",
        "client_secret",
        "connection_profile_ref",
        "cookie",
        "credential",
        "credentials",
        "endpoint",
        "headers",
        "identity_hmac_key",
        "password",
        "proxy",
        "proxy_url",
        "refresh_token",
        "secret",
        "tls",
        "tls_profile_ref",
        "token",
        "transport_security",
        "uri",
        "url",
        "variables",
        "variables_ref",
    }
)


def _alias(value: str, label: str) -> str:
    result = str(value or "").strip().lower()
    if not _ALIAS_RE.fullmatch(result):
        raise ExternalGraphSchemaError(f"{label} must be a neutral lowercase alias")
    return result


def _runtime_ref(value: str | None, label: str, *, required: bool = False) -> str:
    rendered = str(value or "").strip()
    if not rendered and not required:
        return ""
    if not _SECRET_REF_RE.fullmatch(rendered):
        raise ExternalGraphSchemaError(
            f"{label} must use a supported runtime secret-reference scheme"
        )
    return rendered


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constants are not supported")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON keys are not supported")
        value[key] = item
    return value


def _load_secret_object(
    resolver: Callable[[str], str | None], ref: str, label: str
) -> dict[str, Any]:
    try:
        raw = resolver(ref)
    except Exception as exc:
        raise ExternalGraphSchemaError(
            f"{label} resolution failed ({type(exc).__name__})"
        ) from None
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw.encode("utf-8")) > _MAX_SECRET_BYTES
    ):
        raise ExternalGraphSchemaError(f"{label} is missing or exceeds its bound")
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (TypeError, ValueError, RecursionError):
        raise ExternalGraphSchemaError(f"{label} is not valid JSON") from None
    if not isinstance(value, dict):
        raise ExternalGraphSchemaError(f"{label} must be a JSON object")
    return value


def _validate_headers(value: Any) -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping) or len(value) > 32:
        raise ExternalGraphSchemaError("GraphQL auth profile headers are invalid")
    headers: dict[str, str] = {}
    normalized_names: set[str] = set()
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise ExternalGraphSchemaError("GraphQL auth profile headers are invalid")
        name = str(raw_name)
        rendered = str(raw_value)
        normalized_name = name.lower()
        if (
            not _HEADER_RE.fullmatch(name)
            or len(rendered.encode("utf-8")) > 16_384
            or "\r" in rendered
            or "\n" in rendered
            or normalized_name in _BLOCKED_REQUEST_HEADERS
            or normalized_name in normalized_names
        ):
            raise ExternalGraphSchemaError("GraphQL auth profile headers are invalid")
        normalized_names.add(normalized_name)
        headers[name] = rendered
    return headers


def _reject_private_mapping_fields(value: Mapping[str, Any]) -> None:
    """Reject embedded transport/credential material without recursive descent."""

    pending: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        if visited > 100_000 or depth > 64:
            raise ExternalGraphSchemaError("GraphQL mapping policy is too complex")
        if isinstance(current, Mapping):
            fields = {str(key).strip().lower() for key in current}
            if fields.intersection(_MAPPING_POLICY_PRIVATE_FIELDS):
                raise ExternalGraphSchemaError(
                    "GraphQL mapping policy contains transport or credential material"
                )
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend((item, depth + 1) for item in current)


def _validate_optional_bound(
    value: Mapping[str, Any], key: str, *, minimum: int, maximum: int
) -> None:
    if key not in value:
        return
    candidate = value[key]
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ExternalGraphSchemaError("GraphQL source policy bound is invalid")
    if not minimum <= candidate <= maximum:
        raise ExternalGraphSchemaError("GraphQL source policy bound is invalid")


def _policy_digest(profile: Mapping[str, Any]) -> str:
    """Digest every approval-critical GraphQL policy field."""

    operations = profile.get("operations") or {}
    default_operation = str(profile.get("default_operation") or "")
    if (
        not default_operation
        and isinstance(operations, Mapping)
        and len(operations) == 1
    ):
        default_operation = sorted(str(alias) for alias in operations)[0]
    policy = {
        "policy_format": _POLICY_FORMAT,
        "default_operation": default_operation,
        "discovery": profile.get("discovery") or {},
        "governance": profile.get("governance") or {},
        "limits": profile.get("limits") or {},
        "operations": operations,
    }
    policy["identity_hmac_key_ref"] = str(profile.get("identity_hmac_key_ref") or "")
    return hashlib.sha256(
        json.dumps(
            policy,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def graphql_mapping_policy_digest(profile: Mapping[str, Any]) -> str:
    """Public digest seam used by the shared approval integrity check."""

    return _policy_digest(profile)


@dataclass(frozen=True)
class GraphQLConnectionRefs:
    """Reference-only declaration retained by a named runtime source."""

    connection_profile_ref: str
    mapping_policy_ref: str = ""
    auth_profile_ref: str = ""
    tls_profile_ref: str = ""
    variables_ref: str = ""


class GraphQLSourceAdapter:
    """Read-only GraphQL transport that resolves every source detail at use time.

    It is intentionally not a graph engine and exposes no Cypher/write surface.
    Registering it as a named source therefore cannot create a second GraphOS
    authority or make GraphQL mutations reachable through generic graph tools.
    """

    backend_kind = "graphql"
    read_only = True
    cypher_support = "none"

    def __init__(
        self,
        *,
        connection: str,
        source_alias: str,
        connection_profile_ref: str,
        mapping_policy_ref: str | None = None,
        auth_profile_ref: str | None = None,
        tls_profile_ref: str | None = None,
        variables_ref: str | None = None,
        ingest_operation: str | None = None,
        discovery_max_types: int = 200,
        discovery_max_depth: int = 6,
        ingest_max_records: int = 1_000,
        contextual: bool = True,
        allow_introspection: bool = False,
        allow_empty_snapshot: bool = False,
        resolver: Callable[[str], str | None] | None = None,
        transport: Any = None,
    ) -> None:
        self.connection = _alias(connection, "connection")
        self.source_alias = _alias(source_alias, "source_alias")
        self.refs = GraphQLConnectionRefs(
            connection_profile_ref=_runtime_ref(
                connection_profile_ref, "connection_profile_ref", required=True
            ),
            mapping_policy_ref=_runtime_ref(mapping_policy_ref, "mapping_policy_ref"),
            auth_profile_ref=_runtime_ref(auth_profile_ref, "auth_profile_ref"),
            tls_profile_ref=_runtime_ref(tls_profile_ref, "tls_profile_ref"),
            variables_ref=_runtime_ref(variables_ref, "variables_ref"),
        )
        self.ingest_operation = str(ingest_operation or "").strip().lower()
        if self.ingest_operation and not _OPERATION_RE.fullmatch(self.ingest_operation):
            raise ExternalGraphSchemaError("GraphQL ingest operation alias is invalid")
        self.discovery_max_types = max(1, min(int(discovery_max_types), 500))
        self.discovery_max_depth = max(1, min(int(discovery_max_depth), 12))
        self.ingest_max_records = max(1, min(int(ingest_max_records), 10_000))
        self.contextual = bool(contextual)
        self.allow_introspection = bool(allow_introspection)
        self.allow_empty_snapshot = bool(allow_empty_snapshot)
        self._resolver = resolver
        self._transport = transport

    def _resolve_ref(self, ref: str) -> str | None:
        resolver = self._resolver
        if resolver is None:
            from agent_utilities.security.secrets_client import create_secrets_client

            resolver = create_secrets_client().resolve_ref
            self._resolver = resolver
        return resolver(ref)

    def runtime_snapshot(self) -> GraphQLSourceAdapter:
        """Freeze referenced documents for one discover/propose/ingest operation."""

        cache: dict[str, str | None] = {}

        def load(ref: str) -> str | None:
            try:
                return self._resolve_ref(ref)
            except Exception as exc:
                raise ExternalGraphSchemaError(
                    f"GraphQL runtime reference snapshot failed ({type(exc).__name__})"
                ) from None

        for ref in (
            self.refs.connection_profile_ref,
            self.refs.mapping_policy_ref,
            self.refs.auth_profile_ref,
            self.refs.tls_profile_ref,
            self.refs.variables_ref,
        ):
            if ref and ref not in cache:
                cache[ref] = load(ref)

        def resolve(ref: str) -> str | None:
            if ref not in cache:
                cache[ref] = load(ref)
            return cache[ref]

        return GraphQLSourceAdapter(
            connection=self.connection,
            source_alias=self.source_alias,
            connection_profile_ref=self.refs.connection_profile_ref,
            mapping_policy_ref=self.refs.mapping_policy_ref,
            auth_profile_ref=self.refs.auth_profile_ref or None,
            tls_profile_ref=self.refs.tls_profile_ref or None,
            variables_ref=self.refs.variables_ref or None,
            ingest_operation=self.ingest_operation or None,
            discovery_max_types=self.discovery_max_types,
            discovery_max_depth=self.discovery_max_depth,
            ingest_max_records=self.ingest_max_records,
            contextual=self.contextual,
            allow_introspection=self.allow_introspection,
            allow_empty_snapshot=self.allow_empty_snapshot,
            resolver=resolve,
            transport=self._transport,
        )

    def _connection_profile(self) -> dict[str, Any]:
        value = _load_secret_object(
            self._resolve_ref,
            self.refs.connection_profile_ref,
            "GraphQL connection profile",
        )
        allowed = {"endpoint", "profile_format"}
        if set(value).difference(allowed):
            raise ExternalGraphSchemaError(
                "GraphQL connection profile contains non-transport fields"
            )
        if value.get("profile_format") != GRAPHQL_CONNECTION_PROFILE_FORMAT:
            raise ExternalGraphSchemaError(
                "GraphQL connection profile format is unsupported"
            )
        endpoint = value.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ExternalGraphSchemaError("GraphQL connection profile has no endpoint")
        return {"endpoint": endpoint}

    def mapping_policy(self) -> dict[str, Any]:
        if not self.refs.mapping_policy_ref:
            return {
                "profile_format": _POLICY_FORMAT,
                "discovery": {
                    "enabled": True,
                    "allow_introspection": self.allow_introspection,
                    "max_depth": self.discovery_max_depth,
                },
                "governance": {
                    "classification": "internal",
                    "retention": "P30D",
                    "access": {"markings": ["external-source-quarantine"]},
                },
                "limits": {
                    "max_documents": self.ingest_max_records,
                    "max_entities": self.ingest_max_records,
                    "max_hierarchy_depth": self.discovery_max_depth,
                    "max_pages": 25,
                    "max_total_response_bytes": 25_000_000,
                    "page_size": min(self.ingest_max_records, 100),
                },
                "operations": {},
            }
        value = _load_secret_object(
            self._resolve_ref, self.refs.mapping_policy_ref, "GraphQL mapping policy"
        )
        if value.get("profile_format") != _POLICY_FORMAT:
            raise ExternalGraphSchemaError(
                "GraphQL mapping policy format is unsupported"
            )
        _reject_private_mapping_fields(value)
        allowed = {
            "default_operation",
            "discovery",
            "governance",
            "limits",
            "operations",
            "profile_format",
        }
        if set(value).difference(allowed):
            raise ExternalGraphSchemaError("GraphQL mapping policy has unknown fields")
        operations = value.get("operations") or {}
        if not isinstance(operations, Mapping) or len(operations) > _MAX_OPERATIONS:
            raise ExternalGraphSchemaError(
                "GraphQL mapping policy operations are invalid"
            )
        for operation in operations:
            if not _OPERATION_RE.fullmatch(str(operation)):
                raise ExternalGraphSchemaError(
                    "GraphQL mapping policy operation alias is invalid"
                )
        default_operation = str(value.get("default_operation") or "")
        if default_operation and default_operation not in operations:
            raise ExternalGraphSchemaError(
                "GraphQL default operation is not present in the mapping policy"
            )
        if len(operations) > 1 and not default_operation:
            raise ExternalGraphSchemaError(
                "GraphQL policies with multiple operations require a default alias"
            )
        discovery = value.get("discovery") or {}
        if not isinstance(discovery, Mapping) or set(discovery).difference(
            {
                "accept_bounded_probe",
                "allow_introspection",
                "enabled",
                "max_depth",
                "probe_query",
            }
        ):
            raise ExternalGraphSchemaError("GraphQL discovery policy is invalid")
        for key in ("enabled", "allow_introspection", "accept_bounded_probe"):
            if key in discovery and not isinstance(discovery[key], bool):
                raise ExternalGraphSchemaError("GraphQL discovery policy is invalid")
        if "probe_query" in discovery and not isinstance(discovery["probe_query"], str):
            raise ExternalGraphSchemaError("GraphQL discovery policy is invalid")
        if len(str(discovery.get("probe_query") or "").encode("utf-8")) > 200_000:
            raise ExternalGraphSchemaError("GraphQL discovery policy is invalid")
        _validate_optional_bound(discovery, "max_depth", minimum=1, maximum=12)
        limits = value.get("limits") or {}
        if not isinstance(limits, Mapping) or set(limits).difference(
            {
                "max_documents",
                "max_entities",
                "max_hierarchy_depth",
                "max_pages",
                "max_total_response_bytes",
                "page_size",
            }
        ):
            raise ExternalGraphSchemaError("GraphQL source limits are invalid")
        for key, minimum, maximum in (
            ("max_documents", 1, 10_000),
            ("max_entities", 1, 10_000),
            ("max_hierarchy_depth", 1, 32),
            ("max_pages", 1, 100),
            ("max_total_response_bytes", 1_024, 100_000_000),
            ("page_size", 1, 1_000),
        ):
            _validate_optional_bound(limits, key, minimum=minimum, maximum=maximum)
        return dict(value)

    def _auth_headers(self) -> dict[str, str]:
        if not self.refs.auth_profile_ref:
            return {}
        value = _load_secret_object(
            self._resolve_ref, self.refs.auth_profile_ref, "GraphQL auth profile"
        )
        if value.get("profile_format") != GRAPHQL_AUTH_PROFILE_FORMAT:
            raise ExternalGraphSchemaError("GraphQL auth profile format is unsupported")
        if set(value).difference({"headers", "profile_format"}):
            raise ExternalGraphSchemaError("GraphQL auth profile has unknown fields")
        return _validate_headers(value.get("headers"))

    def runtime_profile(
        self,
        *,
        policy: Mapping[str, Any] | None = None,
        identity_key: str | None = None,
    ) -> dict[str, Any]:
        """Compose a connector profile in memory without copying source details."""

        selected = dict(policy) if policy is not None else self.mapping_policy()
        profile = {
            **self._connection_profile(),
            "headers": self._auth_headers(),
            "identity_hmac_key": identity_key or secrets.token_hex(32),
            "operations": selected.get("operations") or {},
            "governance": selected.get("governance") or {},
            "limits": selected.get("limits") or {},
            "discovery": selected.get("discovery") or {},
        }
        if self.refs.tls_profile_ref:
            profile["tls_profile_ref"] = self.refs.tls_profile_ref
        return profile

    def _discovery_options(self, policy: Mapping[str, Any]) -> dict[str, Any]:
        discovery = policy.get("discovery") or {}
        if not isinstance(discovery, Mapping) or not bool(discovery.get("enabled")):
            raise ExternalGraphSchemaError("GraphQL discovery is not enabled by policy")
        if "probe_variables" in discovery:
            raise ExternalGraphSchemaError(
                "GraphQL discovery variables must use variables_ref"
            )
        return {
            "allow_introspection": bool(discovery.get("allow_introspection", False)),
            "probe_document": str(discovery.get("probe_query") or ""),
            "probe_variables": self.variables(purpose="discovery"),
            "max_depth": max(1, min(int(discovery.get("max_depth") or 6), 12)),
            "accept_bounded_probe": bool(discovery.get("accept_bounded_probe", False)),
        }

    def _connector(self, profile: Mapping[str, Any], operation: str):
        from agent_utilities.protocols.source_connectors.connectors.graphql_document import (
            GraphQLDocumentConnector,
        )

        runtime_ref = "secret://runtime/graphql-document-profile"
        rendered = json.dumps(
            profile, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return GraphQLDocumentConnector(
            source_alias=self.source_alias,
            profile_ref=runtime_ref,
            profile_resolver=(
                lambda ref: rendered if ref == runtime_ref else self._resolve_ref(ref)
            ),
            operation=operation,
            transport=self._transport,
        )

    def execute(self, document: str, variables: Mapping[str, Any]) -> Mapping[str, Any]:
        policy = self.mapping_policy()
        operation = "schema_discovery"
        policy = {
            **policy,
            "operations": {
                operation: {
                    "query": "{ __typename }",
                    "root_path": "__typename",
                    "id_path": "__typename",
                }
            },
        }
        return self._connector(self.runtime_profile(policy=policy), operation).execute(
            document, variables
        )

    def _discover_snapshot(
        self, *, max_types: int = 200, max_depth: int | None = None
    ) -> tuple[GraphQLDiscoveredSchema, BackendCapabilities, bool]:
        policy = self.mapping_policy()
        options = self._discovery_options(policy)
        if max_depth is not None:
            options["max_depth"] = max(1, min(int(max_depth), 12))
        accept_bounded_probe = bool(options.pop("accept_bounded_probe"))
        schema = GraphQLDiscoveryAdapter().discover(
            self.execute,
            max_types=max_types,
            **options,
        )
        return schema, GraphQLDiscoveryAdapter.capabilities, accept_bounded_probe

    def discover(
        self, *, max_types: int = 200, max_depth: int | None = None
    ) -> tuple[GraphQLDiscoveredSchema, BackendCapabilities, bool]:
        """Discover against one in-memory snapshot of every runtime reference."""

        return self.runtime_snapshot()._discover_snapshot(
            max_types=max_types, max_depth=max_depth
        )

    def variables(
        self,
        variables_ref: str | None = None,
        *,
        purpose: str = "",
    ) -> dict[str, Any]:
        ref = _runtime_ref(variables_ref, "variables_ref") or self.refs.variables_ref
        if not ref:
            return {}
        value = _load_secret_object(self._resolve_ref, ref, "GraphQL variables profile")
        selected: Any = value
        if "operations" in value or "discovery" in value:
            if set(value).difference({"discovery", "operations"}):
                raise ExternalGraphSchemaError(
                    "GraphQL variables profile has unknown fields"
                )
            if purpose == "discovery":
                selected = value.get("discovery") or {}
            else:
                operations = value.get("operations") or {}
                if not isinstance(operations, Mapping):
                    raise ExternalGraphSchemaError(
                        "GraphQL operation variables are invalid"
                    )
                selected = operations.get(purpose) or {}
        if not isinstance(selected, Mapping):
            raise ExternalGraphSchemaError("GraphQL variables profile is invalid")
        result = dict(selected)
        try:
            rendered_size = len(
                json.dumps(
                    result,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError, RecursionError):
            raise ExternalGraphSchemaError(
                "GraphQL variables profile is invalid"
            ) from None
        if rendered_size > 1_000_000:
            raise ExternalGraphSchemaError("GraphQL variables exceed their bound")
        return result

    def validate_runtime_profiles(self) -> None:
        """Apply the native connector contract without performing a source read.

        Doctor uses this seam after resolving one immutable runtime snapshot. It
        therefore validates the endpoint, headers, governance, limits, every
        configured operation, and operation-scoped variables with the same parser
        ingestion uses instead of maintaining a weaker format-only copy.
        """

        runtime = self.runtime_snapshot()
        policy = runtime.mapping_policy()
        operations = policy.get("operations") or {}
        if not isinstance(operations, Mapping):
            raise ExternalGraphSchemaError(
                "GraphQL mapping policy operations are invalid"
            )
        validation_policy = dict(policy)
        if not operations:
            operations = {
                "runtime_validation": {
                    "query": "query RuntimeValidation { __typename }",
                    "root_path": "__typename",
                    "id_path": "__typename",
                }
            }
            validation_policy["operations"] = operations
        profile = runtime.runtime_profile(
            policy=validation_policy,
            identity_key="0" * 32,
        )
        from agent_utilities.protocols.source_connectors.connectors.graphql_document import (
            GraphQLDocumentError,
        )

        try:
            for alias in operations:
                connector = runtime._connector(profile, str(alias))
                resolved_profile = connector._resolve_profile()
                connector._transport_security(resolved_profile)
                runtime.variables(purpose=str(alias))
        except GraphQLDocumentError as exc:
            raise ExternalGraphSchemaError(str(exc)) from None
        runtime.variables(purpose="discovery")

    def probe_connection(self) -> bool:
        """Perform one bounded metadata-only read using resolved auth and TLS."""

        runtime = self.runtime_snapshot()
        policy = runtime.mapping_policy()
        discovery = dict(policy.get("discovery") or {})
        discovery.update(enabled=True, allow_introspection=False)
        operation = "runtime_connection_probe"
        probe_policy = {
            **policy,
            "discovery": discovery,
            "operations": {
                operation: {
                    "query": "query RuntimeConnectionProbe { __typename }",
                    "root_path": "__typename",
                    "id_path": "__typename",
                }
            },
        }
        profile = runtime.runtime_profile(
            policy=probe_policy,
            identity_key="0" * 32,
        )
        connector = runtime._connector(profile, operation)
        from agent_utilities.protocols.source_connectors.connectors.graphql_document import (
            GraphQLDocumentError,
        )

        try:
            connector._transport_security(connector._resolve_profile())
            result = connector.execute(
                "query RuntimeConnectionProbe { __typename }", {}
            )
        except GraphQLDocumentError:
            raise ExternalGraphSchemaError("GraphQL connection probe failed") from None
        data = result.get("data")
        return isinstance(data, Mapping) and bool(data.get("__typename"))

    def close(self) -> None:
        """No persistent socket or resolved secret material is retained."""


_GRAPHQL_READ_BOOTSTRAP = """
query AgentUtilitiesReadBootstrap {
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      kind
      name
      fields(includeDeprecated: false) {
        name
        args {
          name
          defaultValue
          type {
            kind name
            ofType {
              kind name
              ofType {
                kind name
                ofType { kind name }
              }
            }
          }
        }
        type {
          kind name
          ofType {
            kind name
            ofType {
              kind name
              ofType { kind name }
            }
          }
        }
      }
    }
  }
}
""".strip()

_IDENTITY_FIELDS = (
    "id",
    "uuid",
    "key",
    "slug",
    "identifier",
    "externalId",
    "external_id",
)
_TITLE_FIELDS = ("title", "name", "label", "displayName", "display_name")
_VERSION_FIELDS = (
    "version",
    "updatedAt",
    "updated_at",
    "modifiedAt",
    "modified_at",
)


def _type_ref(value: Any, *, depth: int = 0) -> tuple[str, str, bool]:
    """Return ``(leaf_kind, leaf_name, contains_list)`` for a bounded type ref."""

    if not isinstance(value, Mapping) or depth > 8:
        return "", "", False
    kind = str(value.get("kind") or "")
    name = str(value.get("name") or "")
    if kind in {"NON_NULL", "LIST"}:
        leaf_kind, leaf_name, nested_list = _type_ref(
            value.get("ofType"), depth=depth + 1
        )
        return leaf_kind, leaf_name, nested_list or kind == "LIST"
    if not re.fullmatch(r"[A-Z_]+", kind) or not re.fullmatch(
        r"[_A-Za-z][_0-9A-Za-z]{0,127}", name
    ):
        return "", "", False
    return kind, name, False


def _render_type_ref(value: Any, *, depth: int = 0) -> str:
    if not isinstance(value, Mapping) or depth > 8:
        raise ExternalGraphSchemaError("GraphQL generated argument type is invalid")
    kind = str(value.get("kind") or "")
    if kind == "NON_NULL":
        return f"{_render_type_ref(value.get('ofType'), depth=depth + 1)}!"
    if kind == "LIST":
        return f"[{_render_type_ref(value.get('ofType'), depth=depth + 1)}]"
    name = str(value.get("name") or "")
    if kind not in {"SCALAR", "ENUM", "INPUT_OBJECT"} or not re.fullmatch(
        r"[_A-Za-z][_0-9A-Za-z]{0,127}", name
    ):
        raise ExternalGraphSchemaError("GraphQL generated argument type is invalid")
    return name


def _required_argument(value: Mapping[str, Any]) -> bool:
    type_ref = value.get("type")
    return (
        isinstance(type_ref, Mapping)
        and type_ref.get("kind") == "NON_NULL"
        and value.get("defaultValue") is None
    )


def _field_has_required_arguments(field: Mapping[str, Any]) -> bool:
    return any(
        _required_argument(argument)
        for argument in field.get("args") or []
        if isinstance(argument, Mapping)
    )


def _privacy_safe_field(name: str) -> bool:
    if not re.fullmatch(r"[_A-Za-z][_0-9A-Za-z]{0,127}", name):
        return False
    clean, report = PersistencePrivacyGuard().sanitize({name: "present"})
    return not report.changed and clean.get(name) == "present"


def _field_index(type_value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        name: field
        for field in type_value.get("fields") or []
        if isinstance(field, Mapping)
        and re.fullmatch(
            r"[_A-Za-z][_0-9A-Za-z]{0,127}",
            name := str(field.get("name") or ""),
        )
    }


def _leaf_fields(type_value: Mapping[str, Any]) -> list[str]:
    fields: list[str] = []
    for name, field in sorted(_field_index(type_value).items()):
        kind, _type_name, is_list = _type_ref(field.get("type"))
        if (
            kind in {"SCALAR", "ENUM"}
            and not is_list
            and not _field_has_required_arguments(field)
            and _privacy_safe_field(name)
        ):
            fields.append(name)
    return fields[:_MAX_GENERATED_FIELDS]


def _select_field(names: list[str], preferences: tuple[str, ...]) -> str:
    return next((candidate for candidate in preferences if candidate in names), "")


def _generated_operation(
    *,
    root_field: Mapping[str, Any],
    types: Mapping[str, Mapping[str, Any]],
    max_depth: int,
    allow_empty_snapshot: bool,
) -> dict[str, Any] | None:
    """Synthesize one structurally safe, bounded query or reject the root."""

    root_name = str(root_field.get("name") or "")
    if not re.fullmatch(r"[_A-Za-z][_0-9A-Za-z]{0,127}", root_name):
        return None
    leaf_kind, leaf_name, returns_list = _type_ref(root_field.get("type"))
    if leaf_kind != "OBJECT" or leaf_name not in types:
        return None

    arguments = {
        str(argument.get("name") or ""): argument
        for argument in root_field.get("args") or []
        if isinstance(argument, Mapping)
    }
    if any(
        not re.fullmatch(r"[_A-Za-z][_0-9A-Za-z]{0,127}", name) for name in arguments
    ):
        return None
    bound_name = next(
        (
            name
            for name in ("first", "limit")
            if name in arguments
            and _type_ref(arguments[name].get("type"))[:2] == ("SCALAR", "Int")
        ),
        "",
    )
    after_argument = arguments.get("after")
    unsupported_required = [
        name
        for name, argument in arguments.items()
        if _required_argument(argument) and name != bound_name
    ]
    if unsupported_required:
        return None

    object_type = types[leaf_name]
    connection_fields = _field_index(object_type)
    records_path = ""
    page_info: tuple[str, str, str] | None = None
    entity_type = leaf_name
    entity_fields = _leaf_fields(object_type)
    query_depth = 2
    record_selection_prefix = ""
    record_selection_suffix = ""

    if not returns_list:
        for candidate in ("nodes", "items"):
            nested = connection_fields.get(candidate)
            if nested is None:
                continue
            if _field_has_required_arguments(nested):
                continue
            nested_kind, nested_name, nested_list = _type_ref(nested.get("type"))
            if nested_kind == "OBJECT" and nested_list and nested_name in types:
                records_path = candidate
                entity_type = nested_name
                entity_fields = _leaf_fields(types[nested_name])
                query_depth = 3
                record_selection_prefix = f"{candidate} {{ "
                record_selection_suffix = " }"
                break

    identity_field = _select_field(entity_fields, _IDENTITY_FIELDS)
    if not identity_field or query_depth > max_depth:
        return None
    if returns_list and not bound_name:
        return None
    if records_path and not bound_name:
        return None

    if records_path:
        page_info_field = connection_fields.get("pageInfo")
        page_kind, page_name, _page_list = _type_ref(
            page_info_field.get("type") if page_info_field else None
        )
        page_fields = (
            _field_index(types[page_name])
            if page_kind == "OBJECT" and page_name in types
            else {}
        )
        if (
            bound_name == "first"
            and after_argument is not None
            and page_info_field is not None
            and not _field_has_required_arguments(page_info_field)
            and not _field_has_required_arguments(page_fields.get("endCursor") or {})
            and not _field_has_required_arguments(page_fields.get("hasNextPage") or {})
            and "endCursor" in page_fields
            and "hasNextPage" in page_fields
        ):
            page_info = ("after", "endCursor", "hasNextPage")
        elif returns_list:
            return None

    selected_fields = sorted(set(entity_fields))
    title_field = _select_field(selected_fields, _TITLE_FIELDS) or identity_field
    version_field = _select_field(selected_fields, _VERSION_FIELDS)
    variable_definitions: list[str] = []
    arguments_rendered: list[str] = []
    if bound_name:
        variable_definitions.append(
            f"${bound_name}: {_render_type_ref(arguments[bound_name].get('type'))}"
        )
        arguments_rendered.append(f"{bound_name}: ${bound_name}")
    if page_info is not None:
        after_name = page_info[0]
        variable_definitions.append(
            f"${after_name}: {_render_type_ref(arguments[after_name].get('type'))}"
        )
        arguments_rendered.append(f"{after_name}: ${after_name}")
    variables_text = (
        f"({', '.join(variable_definitions)})" if variable_definitions else ""
    )
    arguments_text = f"({', '.join(arguments_rendered)})" if arguments_rendered else ""
    page_selection = (
        " pageInfo { endCursor hasNextPage }" if page_info is not None else ""
    )
    query = (
        f"query GeneratedRead{variables_text} {{ {root_name}{arguments_text} {{ "
        f"{record_selection_prefix}{' '.join(selected_fields)}"
        f"{record_selection_suffix}{page_selection} }} }}"
    )
    mapping: dict[str, Any] = {
        "records_path": records_path,
        "id_path": identity_field,
        "title_path": title_field,
        "entity_type": entity_type,
        "property_allowlist": selected_fields,
    }
    if version_field:
        mapping["version_path"] = version_field
    snapshot_authoritative = (not returns_list and page_info is not None) or (
        not returns_list and not records_path
    )
    operation: dict[str, Any] = {
        "query": query,
        "root_path": root_name,
        "mappings": {"entities": mapping},
        "snapshot_authoritative": snapshot_authoritative,
        "allow_empty_snapshot": allow_empty_snapshot and snapshot_authoritative,
    }
    if page_info is not None:
        operation["pagination"] = {
            "cursor_variable": "after",
            "page_size_variable": "first",
            "next_cursor_path": f"{root_name}.pageInfo.{page_info[1]}",
            "has_more_path": f"{root_name}.pageInfo.{page_info[2]}",
        }
    elif bound_name:
        operation["read_bound"] = {
            "variable": bound_name,
            "maximum": 100,
        }
    return operation


def _generate_mapping_policy(
    source: GraphQLSourceAdapter,
    base_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate deterministic operations from introspection, never sample values."""

    if not source.allow_introspection and not bool(
        (base_policy.get("discovery") or {}).get("allow_introspection", False)
    ):
        raise ExternalGraphSchemaError(
            "GraphQL operation generation requires explicit introspection approval"
        )
    response = source.execute(_GRAPHQL_READ_BOOTSTRAP, {})
    schema = (
        (response.get("data") or {}).get("__schema")
        if isinstance(response.get("data"), Mapping)
        else None
    )
    if not isinstance(schema, Mapping):
        raise ExternalGraphSchemaError("GraphQL operation generation found no schema")
    query_name = str((schema.get("queryType") or {}).get("name") or "")
    mutation_name = str((schema.get("mutationType") or {}).get("name") or "")
    subscription_name = str((schema.get("subscriptionType") or {}).get("name") or "")
    if not re.fullmatch(r"[_A-Za-z][_0-9A-Za-z]{0,127}", query_name) or query_name in {
        mutation_name,
        subscription_name,
    }:
        raise ExternalGraphSchemaError("GraphQL read root is missing or ambiguous")
    types = {
        name: item
        for item in schema.get("types") or []
        if isinstance(item, Mapping)
        and str(item.get("kind") or "") == "OBJECT"
        and re.fullmatch(
            r"[_A-Za-z][_0-9A-Za-z]{0,127}",
            name := str(item.get("name") or ""),
        )
    }
    query_type = types.get(query_name)
    if query_type is None:
        raise ExternalGraphSchemaError("GraphQL read root is missing or ambiguous")
    candidates: list[tuple[str, dict[str, Any]]] = []
    for root_name, root_field in sorted(_field_index(query_type).items()):
        operation = _generated_operation(
            root_field=root_field,
            types=types,
            max_depth=source.discovery_max_depth,
            allow_empty_snapshot=source.allow_empty_snapshot,
        )
        if operation is not None:
            candidates.append((root_name, operation))
    if not candidates:
        raise ExternalGraphSchemaError(
            "GraphQL schema exposes no structurally safe bounded read operation"
        )
    if len(candidates) > _MAX_GENERATED_OPERATIONS:
        raise ExternalGraphSchemaError(
            "GraphQL schema has too many ambiguous bounded read operations"
        )
    operations = {
        f"generated_read_{index:03d}": operation
        for index, (_root_name, operation) in enumerate(candidates, start=1)
    }
    policy = {
        **dict(base_policy),
        "default_operation": sorted(operations)[0],
        "operations": operations,
    }
    return policy, {
        "mode": "introspection-structural",
        "candidate_count": len(operations),
        "ambiguous": len(operations) > 1,
        "approval": "exact-digest",
    }


def _effective_mapping_policy(
    source: GraphQLSourceAdapter,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    operations = policy.get("operations")
    if isinstance(operations, Mapping) and operations:
        return dict(policy), {
            "mode": "configured",
            "candidate_count": len(operations),
            "ambiguous": len(operations) > 1,
            "approval": "exact-digest",
        }
    return _generate_mapping_policy(source, policy)


def _identity_key(
    store: SecretStore,
    connection: str,
    *,
    candidate: str = "",
    persist: bool = True,
) -> str:
    key_name = f"external-graphs/{connection}/identity-key"
    existing = store.get(key_name)
    if isinstance(existing, str) and len(existing.encode("utf-8")) >= 32:
        return existing
    generated = (
        candidate
        if len(str(candidate or "").encode("utf-8")) >= 32
        else secrets.token_hex(32)
    )
    if persist:
        store.set(key_name, generated, purpose="external-graph-pseudonymization")
    return generated


def _required_identity_key(
    store: SecretStore,
    connection: str,
    profile: Mapping[str, Any],
) -> str:
    """Resolve approved pseudonymization material without persisting it in policy."""

    if "identity_hmac_key" in profile:
        raise ExternalGraphSchemaError(
            "GraphQL mapping profiles cannot embed identity key material"
        )
    if str(profile.get("identity_hmac_key_ref") or "") != canonical_identity_key_ref(
        connection
    ):
        raise ExternalGraphSchemaError(
            "GraphQL mapping profile has an invalid identity key ref"
        )
    identity_key = store.get(f"external-graphs/{connection}/identity-key")
    if not isinstance(identity_key, str) or len(identity_key.encode("utf-8")) < 32:
        raise ExternalGraphSchemaError("GraphQL mapping profile has no identity key")
    return identity_key


def _profile_key(connection: str) -> str:
    return f"external-graphs/{connection}/mapping-profile"


def _load_profile(store: SecretStore, connection: str) -> dict[str, Any] | None:
    raw = store.get(_profile_key(connection))
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw.encode("utf-8")) > 8 * 1024 * 1024
    ):
        return None
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _token(key: str, namespace: str, value: str, length: int = 24) -> str:
    return hmac.new(
        key.encode("utf-8"),
        f"{namespace}\x1f{value}".encode(),
        hashlib.sha256,
    ).hexdigest()[:length]


def _validate_connector_policy(
    source: GraphQLSourceAdapter,
    policy: Mapping[str, Any],
    identity_key: str,
) -> tuple[list[str], int]:
    """Reuse the native connector validator for every configured operation."""

    from agent_utilities.protocols.source_connectors.connectors.graphql_document import (
        GraphQLDocumentError,
    )

    operations = dict(policy.get("operations") or {})
    if not operations:
        raise ExternalGraphSchemaError(
            "GraphQL mapping proposal requires at least one candidate operation"
        )
    profile = source.runtime_profile(policy=policy, identity_key=identity_key)
    mapping_count = 0
    aliases: list[str] = []
    for raw_alias, raw_operation in operations.items():
        alias = str(raw_alias)
        connector = source._connector(profile, alias)
        try:
            connector._resolve_profile()  # one validation implementation, no network I/O
        except GraphQLDocumentError:
            raise ExternalGraphSchemaError(
                "GraphQL connector policy is invalid"
            ) from None
        mappings = (
            raw_operation.get("mappings")
            if isinstance(raw_operation, Mapping)
            else None
        )
        mapping_count += len(mappings) if isinstance(mappings, Mapping) else 1
        aliases.append(alias)
    return aliases, mapping_count


def propose_graphql_mapping_profile(
    source: GraphQLSourceAdapter,
    *,
    connection: str,
    source_alias: str,
    secret_store: SecretStore,
    max_types: int = 200,
    max_depth: int = 6,
) -> dict[str, Any]:
    """Discover and store an exact, pseudonymous GraphQL mapping proposal."""

    connection = _alias(connection, "connection")
    source_alias = _alias(source_alias, "source_alias")
    if source.connection != connection or source.source_alias != source_alias:
        raise ExternalGraphSchemaError(
            "GraphQL source declaration does not match request"
        )
    runtime = source.runtime_snapshot()
    schema, capabilities, accept_bounded_probe = runtime._discover_snapshot(
        max_types=max_types, max_depth=max_depth
    )
    if schema.partial and not (schema.mode == "bounded-probe" and accept_bounded_probe):
        raise ExternalGraphSchemaError(
            "GraphQL schema discovery was partial and was not explicitly accepted"
        )
    policy, generation = _effective_mapping_policy(runtime, runtime.mapping_policy())
    identity_key = _identity_key(secret_store, connection, persist=False)
    operation_aliases, mapping_count = _validate_connector_policy(
        runtime, policy, identity_key
    )
    identity_key = _identity_key(
        secret_store, connection, candidate=identity_key, persist=True
    )
    identity_key_ref = canonical_identity_key_ref(connection)
    mapping_digest = _policy_digest(
        {**policy, "identity_hmac_key_ref": identity_key_ref}
    )
    previous = _load_profile(secret_store, connection) or {}
    previous_version = int(previous.get("proposal_version") or 0)
    unchanged = (
        previous.get("profile_format") == _PROFILE_FORMAT
        and previous.get("schema_digest") == schema.schema_digest
        and previous.get("mapping_digest") == mapping_digest
    )
    proposal_version = max(1, previous_version if unchanged else previous_version + 1)
    proposal_id = "map-" + _token(
        identity_key,
        "proposal",
        f"{schema.schema_digest}:{mapping_digest}:{proposal_version}",
    )
    status = (
        "approved"
        if unchanged and previous.get("approval_status") == "approved"
        else "proposed"
    )
    public_mappings = [
        {
            "operation_token": "operation-" + _token(identity_key, "operation", alias),
            "mapping_count": (
                len(
                    (policy.get("operations") or {}).get(alias, {}).get("mappings")
                    or {}
                )
                or 1
            ),
        }
        for alias in operation_aliases
    ]
    profile = {
        "profile_format": _PROFILE_FORMAT,
        "proposal_id": proposal_id,
        "proposal_version": proposal_version,
        "approval_status": status,
        "schema_digest": schema.schema_digest,
        "mapping_digest": mapping_digest,
        "backend_kind": "graphql",
        "source_alias": source_alias,
        "identity_hmac_key_ref": identity_key_ref,
        "default_operation": policy.get("default_operation")
        or sorted(operation_aliases)[0],
        "discovery": policy.get("discovery") or {},
        "governance": policy.get("governance") or {},
        "limits": policy.get("limits") or {},
        "operations": policy.get("operations") or {},
        "raw_schema": schema.raw_dict(),
        "public_mappings": public_mappings,
        "generation": generation,
    }
    if status == "approved" and previous.get("approval_token"):
        profile["approval_token"] = str(previous["approval_token"])
    secret_store.set(
        _profile_key(connection),
        json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False),
        purpose="external-graph-mapping-profile",
        proposal_version=proposal_version,
        approval_status=status,
    )
    return {
        "status": status,
        "connection": connection,
        "source_alias": source_alias,
        "profile_ref": canonical_profile_ref(connection),
        "proposal_id": proposal_id,
        "proposal_version": proposal_version,
        "schema_digest": schema.schema_digest,
        "mapping_digest": mapping_digest,
        "mapped": mapping_count,
        "novel": max(0, len(schema.types) - mapping_count),
        "mappings": public_mappings,
        "schema": schema.public_dict(),
        "capabilities": capabilities.public_dict(),
        "generation": generation,
        "semantic_enrichment": (
            "privacy-compiled-proposal" if generation["ambiguous"] else "disabled"
        ),
    }


def graphql_source_readiness(
    source: GraphQLSourceAdapter,
    *,
    connection: str,
    secret_store: SecretStore,
    max_types: int = 200,
    max_depth: int = 6,
) -> dict[str, Any]:
    """Return only lifecycle metadata, never source schema or error text."""

    from .external_graph_schema import mapping_profile_status

    connection = _alias(connection, "connection")
    status = mapping_profile_status(connection, secret_store=secret_store)
    approved_profile = _load_profile(secret_store, connection) or {}
    try:
        runtime = source.runtime_snapshot()
        schema, capabilities, accept_bounded_probe = runtime._discover_snapshot(
            max_types=max_types, max_depth=max_depth
        )
        current_policy, _generation = _effective_mapping_policy(
            runtime, runtime.mapping_policy()
        )
        _required_identity_key(secret_store, connection, approved_profile)
        current_mapping_digest = _policy_digest(
            {
                **current_policy,
                "identity_hmac_key_ref": approved_profile.get("identity_hmac_key_ref"),
            }
        )
        status = mapping_profile_status(
            connection,
            secret_store=secret_store,
            runtime_policy_digest=current_mapping_digest,
        )
    except Exception as exc:
        return {
            "status": "not_ready",
            "connection": connection,
            "backend": "graphql",
            "capabilities": GraphQLDiscoveryAdapter.capabilities.public_dict(),
            "discovery": "failed",
            "approval": status.get("status", "not_found"),
            "schema_drift": "unknown",
            "mapping_drift": "unknown",
            "ready": False,
            "error_type": type(exc).__name__,
        }
    complete = not schema.partial or (
        schema.mode == "bounded-probe" and accept_bounded_probe
    )
    approved_digest = str(status.get("schema_digest") or "")
    approved_mapping_digest = str(status.get("mapping_digest") or "")
    drift = (
        "none"
        if approved_digest and approved_digest == schema.schema_digest
        else "detected"
        if approved_digest
        else "unapproved"
    )
    mapping_drift = (
        "none"
        if approved_mapping_digest and approved_mapping_digest == current_mapping_digest
        else "detected"
        if approved_mapping_digest
        else "unapproved"
    )
    ready = (
        complete
        and status.get("status") == "approved"
        and drift == "none"
        and mapping_drift == "none"
    )
    return {
        "status": "ready" if ready else "not_ready",
        "connection": connection,
        "backend": "graphql",
        "capabilities": capabilities.public_dict(),
        "discovery": "complete" if complete else "partial",
        "schema": schema.public_dict(),
        "approval": status.get("status", "not_found"),
        "schema_drift": drift,
        "mapping_drift": mapping_drift,
        "ready": ready,
    }


def graphql_mapping_profile_status(
    source: GraphQLSourceAdapter,
    *,
    connection: str,
    secret_store: SecretStore,
) -> dict[str, Any]:
    """Return approval and current policy-drift metadata without source reads."""

    from .external_graph_schema import mapping_profile_status

    connection = _alias(connection, "connection")
    approved_profile = _load_profile(secret_store, connection) or {}
    current_policy = source.mapping_policy()
    if (
        not current_policy.get("operations")
        and (approved_profile.get("generation") or {}).get("mode")
        == "introspection-structural"
    ):
        current_policy = {
            **current_policy,
            "default_operation": approved_profile.get("default_operation"),
            "operations": approved_profile.get("operations") or {},
        }
    current_mapping_digest = _policy_digest(
        {
            **current_policy,
            "identity_hmac_key_ref": approved_profile.get("identity_hmac_key_ref"),
        }
    )
    return mapping_profile_status(
        connection,
        secret_store=secret_store,
        runtime_policy_digest=current_mapping_digest,
    )


def ingest_registered_graphql(
    authority_engine: Any,
    source: GraphQLSourceAdapter,
    *,
    connection: str,
    secret_store: SecretStore,
    operation: str = "",
    variables_ref: str = "",
    max_records: int = 1_000,
    max_types: int = 200,
    max_depth: int = 6,
    contextual: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rediscover, fail on drift, then use native connector ingestion."""

    connection = _alias(connection, "connection")
    runtime = source.runtime_snapshot()
    profile = _load_profile(secret_store, connection)
    if not profile or profile.get("profile_format") != _PROFILE_FORMAT:
        raise ExternalGraphSchemaError("GraphQL mapping proposal does not exist")
    if profile.get("approval_status") != "approved":
        raise ExternalGraphSchemaError("GraphQL mapping profile is not approved")
    identity_key = _required_identity_key(secret_store, connection, profile)
    if _policy_digest(profile) != str(profile.get("mapping_digest") or ""):
        raise ExternalGraphSchemaError("GraphQL mapping profile integrity check failed")
    current_policy, _generation = _effective_mapping_policy(
        runtime, runtime.mapping_policy()
    )
    if _policy_digest(
        {
            **current_policy,
            "identity_hmac_key_ref": profile.get("identity_hmac_key_ref"),
        }
    ) != str(profile.get("mapping_digest") or ""):
        raise ExternalGraphSchemaError(
            "GraphQL mapping policy changed and requires a new approval"
        )
    schema, _capabilities, accept_bounded_probe = runtime._discover_snapshot(
        max_types=max_types, max_depth=max_depth
    )
    complete = not schema.partial or (
        schema.mode == "bounded-probe" and accept_bounded_probe
    )
    if not complete or schema.schema_digest != str(profile.get("schema_digest") or ""):
        raise ExternalGraphSchemaError(
            "GraphQL schema drift requires a new proposal and approval"
        )

    selected_operation = str(operation or profile.get("default_operation") or "")
    if not _OPERATION_RE.fullmatch(selected_operation) or selected_operation not in (
        profile.get("operations") or {}
    ):
        raise ExternalGraphSchemaError("GraphQL ingest operation is not approved")
    variables = runtime.variables(variables_ref, purpose=selected_operation)
    runtime_profile = runtime.runtime_profile(
        policy=profile,
        identity_key=identity_key,
    )
    runtime_ref = "secret://runtime/approved-graphql-document-profile"
    rendered_profile = json.dumps(
        runtime_profile,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    from agent_utilities.knowledge_graph.ingestion.engine import (
        ContentType,
        IngestionEngine,
        IngestionManifest,
    )

    ingestion = IngestionEngine(kg_engine=authority_engine)
    bounded_records = max(1, min(int(max_records), 10_000))
    config = {
        "source_alias": source.source_alias,
        "profile_ref": runtime_ref,
        "profile_resolver": (
            lambda ref: (
                rendered_profile if ref == runtime_ref else runtime._resolve_ref(ref)
            )
        ),
        "operation": selected_operation,
        "variables": variables,
        "max_documents": bounded_records,
        "max_entities": bounded_records,
        "dry_run": bool(dry_run),
        "transport": runtime._transport,
    }
    result = asyncio.run(
        ingestion.ingest(
            IngestionManifest(
                content_type=ContentType.CONNECTOR,
                source_uri="graphql_document",
                metadata={
                    "connector_config": config,
                    "connector_id": f"external-graphql:{connection}",
                    "contextual": bool(contextual),
                    "incremental": True,
                },
            )
        )
    )
    if result.status not in {"success", "skipped"}:
        raise ExternalGraphSchemaError("GraphQL connector ingestion failed")
    details = dict(result.details or {})
    safe_details = {
        key: details[key]
        for key in (
            "acl_synced",
            "checkpoint_advanced",
            "connector",
            "documents_failed",
            "documents_ingested",
            "dry_run",
            "envelopes_failed",
            "envelopes_ingested",
            "plan",
        )
        if key in details
    }
    guard = PersistencePrivacyGuard()
    safe_details, _report = guard.sanitize(safe_details)
    return {
        "status": "dry_run" if dry_run else result.status,
        "connection": connection,
        "source_alias": source.source_alias,
        "schema_digest": schema.schema_digest,
        "nodes_created": int(result.nodes_created or 0),
        "edges_created": int(result.edges_created or 0),
        **(safe_details if isinstance(safe_details, dict) else {}),
    }


__all__ = [
    "GRAPHQL_AUTH_PROFILE_FORMAT",
    "GRAPHQL_CONNECTION_PROFILE_FORMAT",
    "GRAPHQL_MAPPING_POLICY_FORMAT",
    "GraphQLConnectionRefs",
    "GraphQLSourceAdapter",
    "graphql_mapping_profile_status",
    "graphql_mapping_policy_digest",
    "graphql_source_readiness",
    "ingest_registered_graphql",
    "propose_graphql_mapping_profile",
]
