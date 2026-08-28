from __future__ import annotations

"""Profile-driven GraphQL document ingestion with a zero-PII persistence gate.

The connector is intentionally source-neutral. Production callers provide only
a neutral ``source_alias``, a secret-backed ``profile_ref``, an operation name,
and variables. The resolved profile (endpoint, headers, query documents, and
field mappings) remains process-local and is never copied into document
metadata, checkpoints, logs, or traces.

CONCEPT:AU-KG.ingest.universal-data-connector
CONCEPT:AU-KG.ingest.external-graph-federation
"""

import hashlib
import hmac
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from graphql import GraphQLError, parse
from graphql.language import (
    ArgumentNode,
    FieldNode,
    FragmentDefinitionNode,
    InlineFragmentNode,
    Node,
    OperationDefinitionNode,
    OperationType,
    SelectionNode,
    VariableNode,
)

from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.models.company_brain import DataClassification
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    ExternalAccess,
    LoadConnector,
    PollConnector,
    SourceDocument,
)
from ..registry import register_source

_ALIAS_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_DOCUMENT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_SECRET_REF_RE = re.compile(r"^(?:vault|secret|env)://[A-Za-z0-9_./#-]+$")
_ERROR_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_FIELD_NAME_RE = re.compile(r"^[_A-Za-z][_0-9A-Za-z]{0,127}$")
_FIELD_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_RETENTION_RE = re.compile(
    r"^(?:P(?:\d+[YMWD])+(?:T(?:\d+[HMS])+)?)|[A-Za-z][A-Za-z0-9_.:-]{1,63}$"
)
_VERSION_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_HEADER_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")
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
_MAX_PROFILE_BYTES = 8 * 1024 * 1024
_MAX_GRAPHQL_TOKENS = 20_000
_CHECKPOINT_FORMAT = "graphql-snapshot-checkpoint/v1"
_ENTITY_KINDS = ("entity", "hierarchy", "document", "application", "dependency")
_MAPPING_KEYS = {
    "entity": ("entity", "entities"),
    "hierarchy": ("hierarchy", "hierarchies"),
    "document": ("document", "documents"),
    "application": ("application", "applications"),
    "dependency": ("dependency", "dependencies"),
}
_DEFAULT_ENTITY_TYPES = {
    "entity": "ExternalEntity",
    "hierarchy": "ExternalHierarchy",
    "document": "Document",
    "application": "Application",
    "dependency": "Dependency",
}


class GraphQLDocumentError(RuntimeError):
    """A source-safe GraphQL connector failure with no upstream payload text."""


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constants are not supported")


@dataclass(frozen=True)
class GraphQLHierarchyBatch:
    """One bounded GraphQL read mapped to embedding documents and envelopes."""

    documents: tuple[SourceDocument, ...] = ()
    envelopes: tuple[ChangeEnvelope, ...] = ()
    checkpoint: ConnectorCheckpoint = field(default_factory=ConnectorCheckpoint)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _dig(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in (segment for segment in path.split(".") if segment):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _is_invalid_next_cursor(next_cursor_text: str, seen_cursors: set[str]) -> bool:
    return (
        not next_cursor_text
        or next_cursor_text != next_cursor_text.strip()
        or len(next_cursor_text.encode("utf-8")) > 4_096
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in next_cursor_text
        )
        or next_cursor_text in seen_cursors
    )


def _dig_resolve_segment(current: list[Any], part: str) -> list[Any]:
    """Resolve one dotted-path segment across all in-flight items."""
    resolved: list[Any] = []
    for item in current:
        values = item if isinstance(item, list) else [item]
        for candidate in values:
            if isinstance(candidate, Mapping) and part in candidate:
                resolved.append(candidate[part])
    return resolved


def _dig_flatten(current: list[Any]) -> list[Any]:
    """Flatten one level of list nesting produced by fan-out resolution."""
    flattened: list[Any] = []
    for item in current:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _dig_many(value: Any, path: str) -> list[Any]:
    """Resolve a dotted path while treating lists as bounded fan-out points."""
    current = [value]
    for part in (segment for segment in str(path or "").split(".") if segment):
        current = _dig_resolve_segment(current, part)
    return _dig_flatten(current)


def _digest(*parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _private_digest(key: str, *parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hmac.new(
        key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _safe_alias(value: str, *, label: str) -> str:
    clean = str(value or "").strip().lower()
    if not _ALIAS_RE.fullmatch(clean):
        raise ValueError(f"{label} must be a neutral lowercase alias")
    return clean


def _safe_document_type(value: Any) -> str:
    clean = str(value or "external_document").strip().lower()
    return clean if _DOCUMENT_TYPE_RE.fullmatch(clean) else "external_document"


def _safe_entity_type(value: Any, *, fallback: str) -> str:
    clean = str(value or fallback).strip()
    return clean if re.fullmatch(r"^[A-Za-z][A-Za-z0-9_-]{1,63}$", clean) else fallback


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _cfg_int(config: Mapping[str, Any], key: str, default: int) -> int:
    return int(config.get(key) or default)


def _cfg_float(config: Mapping[str, Any], key: str, default: float) -> float:
    return float(config.get(key) or default)


def _parse_configured_tls_mapping(
    configured: Mapping[str, Any],
) -> tuple[str | None, str | None, Mapping[str, Any] | None]:
    profile_name = (
        str(configured.get("profile_name") or configured.get("profile") or "").strip()
        or None
    )
    profile_ref = str(configured.get("profile_ref") or "").strip() or None
    settings = configured.get("settings")
    inline: Mapping[str, Any] | None = None
    if isinstance(settings, Mapping):
        inline = settings
    elif not profile_name and not profile_ref:
        inline = configured
    return profile_name, profile_ref, inline


def _parse_configured_tls(
    configured: Any,
) -> tuple[str | None, str | None, Mapping[str, Any] | None]:
    if isinstance(configured, str):
        return configured, None, None
    if isinstance(configured, Mapping):
        return _parse_configured_tls_mapping(configured)
    if configured is not None:
        raise GraphQLDocumentError("GraphQL transport security profile is invalid")
    return None, None, None


def _classification(value: Any) -> DataClassification:
    try:
        return DataClassification(str(value or DataClassification.INTERNAL.value))
    except ValueError:
        raise GraphQLDocumentError(
            "GraphQL governance classification is invalid"
        ) from None


def _validate_governance_classification(
    classification: DataClassification, access: ExternalAccess
) -> None:
    if classification == DataClassification.PUBLIC and not access.is_public:
        raise GraphQLDocumentError(
            "GraphQL public classification requires public source access"
        )
    if classification != DataClassification.PUBLIC and access.is_public:
        raise GraphQLDocumentError(
            "GraphQL non-public classification cannot use public source access"
        )


def _policy_values(
    value: Any, *, label: str, pattern: re.Pattern[str]
) -> tuple[str, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list) or not value or len(value) > 256:
        raise GraphQLDocumentError(f"GraphQL {label} allowlist is invalid")
    result = tuple(str(item).strip() for item in value)
    if any(
        len(item.encode("utf-8")) > 512 or not pattern.fullmatch(item)
        for item in result
    ):
        raise GraphQLDocumentError(f"GraphQL {label} allowlist is invalid")
    return result


def _valid_field_path(value: Any) -> bool:
    rendered = str(value or "")
    return len(rendered.encode("utf-8")) <= 512 and bool(
        _FIELD_PATH_RE.fullmatch(rendered)
    )


def _valid_field_name(value: Any) -> bool:
    return bool(_FIELD_NAME_RE.fullmatch(str(value or "")))


def _error_signature(error: Any) -> tuple[str, str] | None:
    if not isinstance(error, Mapping):
        return None
    extensions = error.get("extensions")
    code = (
        str(extensions.get("code") or "").strip().upper()
        if isinstance(extensions, Mapping)
        else ""
    )
    raw_path = error.get("path")
    if not isinstance(raw_path, list):
        return None
    path = ".".join(str(part) for part in raw_path if isinstance(part, str))
    if not _ERROR_CODE_RE.fullmatch(code) or not _valid_field_path(path):
        return None
    return code, path


def _error_is_allowlisted(
    error: Any, *, codes: tuple[str, ...], paths: tuple[str, ...]
) -> bool:
    signature = _error_signature(error)
    if signature is None:
        return False
    code, path = signature
    if code not in codes:
        return False
    return any(path == allowed or path.startswith(f"{allowed}.") for allowed in paths)


def _errors_are_allowlisted(
    errors: Any, *, codes: tuple[str, ...], paths: tuple[str, ...]
) -> bool:
    if not isinstance(errors, list) or not errors or not codes or not paths:
        return False
    return all(
        _error_is_allowlisted(error, codes=codes, paths=paths) for error in errors
    )


def _parse_bounded_query(query: str) -> Any:
    """Enforce the size/emptiness bounds, then parse via the GraphQL AST."""
    if len(query.encode("utf-8")) > 200_000:
        raise GraphQLDocumentError("GraphQL query exceeds the configured bound")
    if not query:
        raise GraphQLDocumentError("GraphQL operation must be a read query")
    try:
        return parse(
            query,
            no_location=True,
            max_tokens=_MAX_GRAPHQL_TOKENS,
            allow_legacy_fragment_variables=False,
        )
    except (GraphQLError, RecursionError, TypeError, ValueError):
        raise GraphQLDocumentError(
            "GraphQL operation is not a valid document"
        ) from None


def _ensure_single_read_query(document: Any) -> None:
    operations = [
        definition
        for definition in document.definitions
        if isinstance(definition, OperationDefinitionNode)
    ]
    if len(operations) != 1 or operations[0].operation is not OperationType.QUERY:
        raise GraphQLDocumentError("GraphQL operation must be a read query")
    if any(
        not isinstance(definition, OperationDefinitionNode | FragmentDefinitionNode)
        for definition in document.definitions
    ):
        raise GraphQLDocumentError("GraphQL operation contains unsupported definitions")


def _document_selections(document: Any) -> list[SelectionNode]:
    selections: list[SelectionNode] = []
    for definition in document.definitions:
        if isinstance(definition, OperationDefinitionNode | FragmentDefinitionNode):
            selections.extend(definition.selection_set.selections)
    return selections


def _reject_introspection(document: Any, *, allow_introspection: bool) -> None:
    selections = _document_selections(document)
    while selections:
        selection = selections.pop()
        if (
            isinstance(selection, FieldNode)
            and selection.name.value in {"__schema", "__type"}
            and not allow_introspection
        ):
            raise GraphQLDocumentError(
                "GraphQL introspection is not an ingest operation"
            )
        if isinstance(selection, FieldNode | InlineFragmentNode):
            selection_set = selection.selection_set
            if selection_set is not None:
                selections.extend(selection_set.selections)


def _validate_query_document(value: Any, *, allow_introspection: bool = False) -> str:
    """Accept exactly one bounded query operation without echoing its text.

    The AST is the authority.  Keyword regexes are not sufficient here: operation
    names, comments, string literals, fragments, and multi-operation documents can
    all make a lexical classifier disagree with what a GraphQL server executes.
    """
    query = str(value or "").strip()
    document = _parse_bounded_query(query)
    _ensure_single_read_query(document)
    _reject_introspection(document, allow_introspection=allow_introspection)
    return query


def _is_row_bound_argument(node: Node, variable: str) -> bool:
    return (
        isinstance(node, ArgumentNode)
        and node.name.value in {"first", "limit"}
        and isinstance(node.value, VariableNode)
        and node.value.name.value == variable
    )


def _node_children(node: Node) -> list[Node]:
    children: list[Node] = []
    for key in node.keys:
        child = getattr(node, key, None)
        if isinstance(child, tuple):
            children.extend(item for item in child if isinstance(item, Node))
        elif isinstance(child, Node):
            children.append(child)
    return children


def _query_binds_row_bound(query: str, variable: str) -> bool:
    """Prove the variable is used by a conventional row-bound argument."""

    try:
        document = parse(
            query,
            no_location=True,
            max_tokens=_MAX_GRAPHQL_TOKENS,
            allow_legacy_fragment_variables=False,
        )
    except (GraphQLError, RecursionError, TypeError, ValueError):
        return False
    stack: list[Node] = [
        definition
        for definition in document.definitions
        if isinstance(definition, Node)
    ]
    while stack:
        node = stack.pop()
        if _is_row_bound_argument(node, variable):
            return True
        stack.extend(_node_children(node))
    return False


def _collect_bounded_iterator_bytes(iterator: Any, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in iterator():
            if not isinstance(chunk, bytes):
                raise TypeError("response chunk is not bytes")
            total += len(chunk)
            if total > limit:
                raise GraphQLDocumentError(
                    "GraphQL response exceeds the configured bound"
                )
            chunks.append(chunk)
    except GraphQLDocumentError:
        raise
    except Exception:
        raise GraphQLDocumentError("GraphQL transport byte stream is invalid") from None
    return b"".join(chunks)


def _response_raw_bytes(response: Any, limit: int) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    iterator = getattr(response, "iter_bytes", None)
    if not callable(iterator):
        raise GraphQLDocumentError(
            "GraphQL transport must expose a bounded byte response"
        )
    return _collect_bounded_iterator_bytes(iterator, limit)


def _decode_bounded_json(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, RecursionError, UnicodeDecodeError):
        raise GraphQLDocumentError("GraphQL response is not valid JSON") from None
    if not isinstance(payload, dict):
        raise GraphQLDocumentError("GraphQL response is not an object")
    return payload


def _bounded_transport_payload(response: Any, limit: int) -> tuple[dict[str, Any], int]:
    """Decode a transport response only after enforcing its raw byte bound.

    Custom transports are production-capable injection points, not test-only
    shortcuts.  Requiring bytes (or a bounded byte iterator) prevents ``.json()``
    from allocating an arbitrarily large object before the connector can apply its
    configured response limit.
    """

    raw = _response_raw_bytes(response, limit)
    if len(raw) > limit:
        raise GraphQLDocumentError("GraphQL response exceeds the configured bound")
    payload = _decode_bounded_json(raw)
    return payload, len(raw)


def _access_from_config(value: Any) -> ExternalAccess:
    if value is None:
        return ExternalAccess.quarantined()
    access = ExternalAccess.model_validate(value)
    if access.user_emails:
        raise ValueError(
            "GraphQL document ingestion does not persist user-email ACLs; "
            "use non-personal group aliases or a mandatory marking"
        )
    guard = PersistencePrivacyGuard()
    if any(
        guard.sanitize_text(str(principal))[1].changed
        for principal in (*access.group_ids, *access.markings)
    ):
        raise ValueError("GraphQL ACL aliases must be non-personal and location-free")
    if not access.is_public and not (access.group_ids or access.markings):
        return ExternalAccess.quarantined()
    return access


@dataclass(frozen=True)
class _GraphQLDocumentLimits:
    """Bounded numeric limits parsed from ``configure(**config)``."""

    max_documents: int
    max_sections: int
    max_content_chars: int
    max_response_bytes: int
    max_total_response_bytes: int
    max_entities: int
    max_pages: int
    page_size: int
    max_hierarchy_depth: int
    max_fallbacks: int
    timeout_seconds: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> _GraphQLDocumentLimits:
        return cls(
            max_documents=_cfg_int(config, "max_documents", 100),
            max_sections=_cfg_int(config, "max_sections", 500),
            max_content_chars=_cfg_int(config, "max_content_chars", 2_000_000),
            max_response_bytes=_cfg_int(config, "max_response_bytes", 10_000_000),
            max_total_response_bytes=_cfg_int(
                config, "max_total_response_bytes", 25_000_000
            ),
            max_entities=_cfg_int(config, "max_entities", 2_000),
            max_pages=_cfg_int(config, "max_pages", 25),
            page_size=_cfg_int(config, "page_size", 100),
            max_hierarchy_depth=_cfg_int(config, "max_hierarchy_depth", 12),
            max_fallbacks=_cfg_int(config, "max_fallbacks", 2),
            timeout_seconds=_cfg_float(config, "timeout_seconds", 30.0),
        )


@register_source("graphql_document")
class GraphQLDocumentConnector(LoadConnector, PollConnector):
    """Ingest mapped GraphQL responses as governed ``SourceDocument`` objects.

    Required runtime config:

    ``source_alias``
        Stable, non-personal source identifier.
    ``profile_ref``
        ``vault://``/``env://``/engine-secret reference resolving to JSON with
        ``endpoint``, optional ``headers``, ``identity_hmac_key``, and
        ``operations``.
    ``operation``
        Name of one profile operation. An operation contains ``query``,
        ``root_path``, ``id_path``, ``title_path``, and optional document
        mapping fields.

    Inline profiles are accepted only with an injected transport, which keeps
    deterministic unit tests offline while preventing production callers from
    putting endpoints or credentials in MCP arguments.
    """

    provider = "GraphQL"

    def configure(self, **config: Any) -> None:
        source_alias = str(config.get("source_alias") or "")
        operation = str(config.get("operation") or "")
        variables = config.get("variables")
        profile_ref = str(config.get("profile_ref") or "")
        access = config.get("access")
        limits = _GraphQLDocumentLimits.from_config(config)
        dry_run = bool(config.get("dry_run", False))
        profile = config.get("profile")
        profile_resolver = config.get("profile_resolver")
        transport = config.get("transport")
        privacy_guard = config.get("privacy_guard")

        self._configure_identity(
            source_alias, operation, profile_ref, profile, transport
        )
        self._configure_variables_and_access(variables, access)
        self._configure_limits(limits)
        self._configure_runtime(
            dry_run=dry_run,
            profile=profile,
            profile_resolver=profile_resolver,
            transport=transport,
            privacy_guard=privacy_guard,
        )

    def _configure_identity(
        self,
        source_alias: str,
        operation: str,
        profile_ref: str,
        profile: Any,
        transport: Any,
    ) -> None:
        self.source_alias = _safe_alias(source_alias, label="source_alias")
        self.operation = str(operation or "").strip().lower()
        if not _OPERATION_RE.fullmatch(self.operation):
            raise ValueError("operation must be a neutral lowercase identifier")
        if not profile_ref and not (
            isinstance(profile, dict) and transport is not None
        ):
            raise ValueError(
                "graphql_document requires a secret-backed profile_ref; "
                "inline profiles are test-only with an injected transport"
            )
        self.profile_ref = str(profile_ref or "")
        if self.profile_ref and not _SECRET_REF_RE.fullmatch(self.profile_ref):
            raise ValueError(
                "profile_ref must use a supported runtime secret-reference scheme"
            )

    def _configure_variables_and_access(self, variables: Any, access: Any) -> None:
        self.variables = dict(variables or {})
        try:
            variables_size = len(
                json.dumps(
                    self.variables,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError, RecursionError):
            raise ValueError("GraphQL variables must be JSON serializable") from None
        if variables_size > 1_000_000:
            raise ValueError("GraphQL variables exceed the configured bound")
        self.external_access = _access_from_config(access)

    def _configure_limits(self, limits: _GraphQLDocumentLimits) -> None:
        self.max_documents = max(1, min(limits.max_documents, 10_000))
        self.max_sections = max(1, min(limits.max_sections, 10_000))
        self.max_content_chars = max(1_024, min(limits.max_content_chars, 20_000_000))
        self.max_response_bytes = max(1_024, min(limits.max_response_bytes, 50_000_000))
        self.max_total_response_bytes = max(
            self.max_response_bytes,
            min(limits.max_total_response_bytes, 100_000_000),
        )
        self.max_entities = max(1, min(limits.max_entities, 10_000))
        self.max_pages = max(1, min(limits.max_pages, 100))
        self.page_size = max(1, min(limits.page_size, 1_000))
        self.max_hierarchy_depth = max(1, min(limits.max_hierarchy_depth, 32))
        self.max_fallbacks = max(0, min(limits.max_fallbacks, 3))
        self.timeout_seconds = max(1.0, min(limits.timeout_seconds, 120.0))

    def _configure_runtime(
        self,
        *,
        dry_run: bool,
        profile: Any,
        profile_resolver: Any,
        transport: Any,
        privacy_guard: Any,
    ) -> None:
        self.dry_run = dry_run
        self._inline_profile = dict(profile) if isinstance(profile, dict) else None
        self._profile_resolver = profile_resolver
        self._transport = transport
        self._resolved_tls: Any | None = None
        self._privacy = (
            privacy_guard
            if isinstance(privacy_guard, PersistencePrivacyGuard)
            else PersistencePrivacyGuard()
        )
        self.last_envelopes: list[ChangeEnvelope] = []
        self.last_plan: dict[str, Any] | None = None

    def _resolve_profile(self) -> dict[str, Any]:
        profile = self._load_profile_document()
        self._validate_endpoint(profile)
        headers, operations = self._validate_profile_shape(profile)
        self._validate_headers(headers)
        op = self._select_operation(operations)
        validated_query = self._validate_operation_paths_and_query(op)
        self._validate_identity_and_limits(profile)
        pagination = self._validate_pagination(op, validated_query)
        self._validate_read_bound(op, validated_query, pagination)
        self._validate_partial_errors(op)
        self._validate_fallbacks(op)
        mappings = self._validate_snapshot_policy_and_mappings(op)
        self._validate_governance(profile, mappings)
        self._validate_discovery(profile)
        return profile

    def _load_profile_document(self) -> dict[str, Any]:
        if self._inline_profile is not None:
            return dict(self._inline_profile)
        resolver = self._profile_resolver
        if resolver is None:
            from agent_utilities.security.secrets_client import (
                create_secrets_client,
            )

            resolver = create_secrets_client().resolve_ref
        try:
            raw = resolver(self.profile_ref)
        except Exception as exc:
            raise GraphQLDocumentError(
                f"GraphQL profile resolution failed ({type(exc).__name__})"
            ) from None
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw.encode("utf-8")) > _MAX_PROFILE_BYTES
        ):
            raise GraphQLDocumentError("GraphQL profile could not be resolved")
        try:
            parsed = json.loads(raw, parse_constant=_reject_json_constant)
        except (TypeError, ValueError, RecursionError):
            raise GraphQLDocumentError("GraphQL profile is not valid JSON") from None
        if not isinstance(parsed, dict):
            raise GraphQLDocumentError("GraphQL profile must be a JSON object")
        return parsed

    @staticmethod
    def _validate_endpoint(profile: dict[str, Any]) -> None:
        endpoint = str(profile.get("endpoint") or "")
        parsed_endpoint = urlparse(endpoint)
        if (
            parsed_endpoint.scheme != "https"
            or not parsed_endpoint.hostname
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or parsed_endpoint.query
            or parsed_endpoint.fragment
        ):
            raise GraphQLDocumentError("GraphQL endpoint must use HTTPS")

    @staticmethod
    def _validate_profile_shape(profile: dict[str, Any]) -> tuple[Any, Any]:
        headers = profile.get("headers") or {}
        operations = profile.get("operations") or {}
        if not isinstance(headers, dict) or not isinstance(operations, dict):
            raise GraphQLDocumentError("GraphQL profile has an invalid shape")
        return headers, operations

    @staticmethod
    def _validate_header_entry(
        name: Any, value: Any, normalized_names: set[str]
    ) -> None:
        if not isinstance(name, str) or not isinstance(value, str):
            raise GraphQLDocumentError("GraphQL profile headers are invalid")
        rendered_name = str(name)
        rendered_value = str(value)
        normalized_name = rendered_name.lower()
        if (
            not _HEADER_RE.fullmatch(rendered_name)
            or len(rendered_value.encode("utf-8")) > 16_384
            or "\r" in rendered_value
            or "\n" in rendered_value
            or normalized_name in _BLOCKED_REQUEST_HEADERS
            or normalized_name in normalized_names
        ):
            raise GraphQLDocumentError("GraphQL profile headers are invalid")
        normalized_names.add(normalized_name)

    @classmethod
    def _validate_headers(cls, headers: dict[str, Any]) -> None:
        if len(headers) > 32:
            raise GraphQLDocumentError("GraphQL profile headers are invalid")
        normalized_names: set[str] = set()
        for name, value in headers.items():
            cls._validate_header_entry(name, value, normalized_names)

    def _select_operation(self, operations: dict[str, Any]) -> dict[str, Any]:
        op = operations.get(self.operation)
        if not isinstance(op, dict):
            raise GraphQLDocumentError("Requested GraphQL operation is not configured")
        return op

    @staticmethod
    def _validate_operation_paths_and_query(op: dict[str, Any]) -> Any:
        validated_query = _validate_query_document(op.get("query"))
        if not _valid_field_path(op.get("root_path")):
            raise GraphQLDocumentError("GraphQL operation has no response root mapping")
        for key, value in op.items():
            if key.endswith("_path") and value not in (None, ""):
                if not _valid_field_path(value):
                    raise GraphQLDocumentError(
                        "GraphQL operation field path is invalid"
                    )
        for key in (
            "section_content_field",
            "section_level_field",
            "section_title_field",
        ):
            if key in op and not _valid_field_name(op[key]):
                raise GraphQLDocumentError("GraphQL operation field name is invalid")
        return validated_query

    @staticmethod
    def _validate_identity_and_limits(profile: dict[str, Any]) -> None:
        identity_key = str(profile.get("identity_hmac_key") or "")
        if len(identity_key) < 32:
            raise GraphQLDocumentError(
                "GraphQL profile requires a 32-character identity HMAC key"
            )
        limits = profile.get("limits") or {}
        if not isinstance(limits, dict):
            raise GraphQLDocumentError("GraphQL profile limits are invalid")

    @staticmethod
    def _validate_pagination(op: dict[str, Any], validated_query: Any) -> Any:
        pagination = op.get("pagination")
        if pagination is not None:
            if not isinstance(pagination, dict):
                raise GraphQLDocumentError("GraphQL pagination mapping is invalid")
            if any(
                not _valid_field_name(pagination.get(key))
                for key in ("cursor_variable", "page_size_variable")
            ) or any(
                not _valid_field_path(pagination.get(key))
                for key in ("next_cursor_path", "has_more_path")
            ):
                raise GraphQLDocumentError("GraphQL pagination mapping is invalid")
            if not _query_binds_row_bound(
                validated_query, str(pagination["page_size_variable"])
            ):
                raise GraphQLDocumentError(
                    "GraphQL pagination query does not enforce its row bound"
                )
        return pagination

    @staticmethod
    def _validate_read_bound(
        op: dict[str, Any], validated_query: Any, pagination: Any
    ) -> None:
        read_bound = op.get("read_bound")
        if read_bound is not None:
            if pagination is not None or not isinstance(read_bound, dict):
                raise GraphQLDocumentError("GraphQL read bound is invalid")
            variable = str(read_bound.get("variable") or "")
            maximum = read_bound.get("maximum")
            if (
                variable not in {"first", "limit"}
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or not 1 <= maximum <= 1_000
                or not _query_binds_row_bound(validated_query, variable)
            ):
                raise GraphQLDocumentError("GraphQL read bound is invalid")

    @staticmethod
    def _validate_partial_errors(op: dict[str, Any]) -> None:
        partial = op.get("partial_errors")
        if partial is not None:
            if not isinstance(partial, dict):
                raise GraphQLDocumentError("GraphQL partial-error policy is invalid")
            _policy_values(
                partial.get("codes"), label="partial-error code", pattern=_ERROR_CODE_RE
            )
            _policy_values(
                partial.get("paths"), label="partial-error path", pattern=_FIELD_PATH_RE
            )

    @staticmethod
    def _validate_fallbacks(op: dict[str, Any]) -> None:
        fallbacks = op.get("optional_field_fallbacks") or []
        if not isinstance(fallbacks, list) or len(fallbacks) > 3:
            raise GraphQLDocumentError("GraphQL optional-field fallbacks are invalid")
        for fallback in fallbacks:
            if not isinstance(fallback, dict):
                raise GraphQLDocumentError("GraphQL optional-field fallback is invalid")
            _validate_query_document(fallback.get("query"))
            _policy_values(
                fallback.get("codes"), label="fallback code", pattern=_ERROR_CODE_RE
            )
            _policy_values(
                fallback.get("paths"), label="fallback path", pattern=_FIELD_PATH_RE
            )

    @staticmethod
    def _validate_entity_mapping_paths(mapping: dict[str, Any]) -> None:
        id_path = str(mapping.get("id_path") or "id")
        if not _valid_field_path(id_path):
            raise GraphQLDocumentError("GraphQL entity identity mapping is invalid")
        records_path = str(mapping.get("records_path") or "")
        if records_path and not _valid_field_path(records_path):
            raise GraphQLDocumentError("GraphQL entity records mapping is invalid")
        if any(
            not _valid_field_path(value)
            for key, value in mapping.items()
            if key.endswith("_path") and value not in (None, "")
        ):
            raise GraphQLDocumentError("GraphQL entity field path is invalid")

    @staticmethod
    def _validate_entity_allowlist(mapping: dict[str, Any]) -> None:
        allowlist = mapping.get("property_allowlist")
        if not isinstance(allowlist, list) or len(allowlist) > 256:
            raise GraphQLDocumentError("GraphQL entity property allowlist is required")
        if any(not _valid_field_path(field) for field in allowlist):
            raise GraphQLDocumentError("GraphQL entity property allowlist is invalid")

    @staticmethod
    def _validate_hierarchy_children_path(mapping: dict[str, Any]) -> None:
        children_path = str(mapping.get("children_path") or "")
        if children_path and not _valid_field_path(children_path):
            raise GraphQLDocumentError("GraphQL hierarchy children mapping is invalid")

    @classmethod
    def _validate_entity_mapping(cls, kind: str, mapping: Any) -> None:
        if not isinstance(mapping, dict):
            raise GraphQLDocumentError("GraphQL entity mapping is invalid")
        cls._validate_entity_mapping_paths(mapping)
        cls._validate_entity_allowlist(mapping)
        if kind == "hierarchy":
            cls._validate_hierarchy_children_path(mapping)

    @staticmethod
    def _validate_snapshot_authority_flags(op: dict[str, Any]) -> None:
        for key in ("snapshot_authoritative", "allow_empty_snapshot"):
            if key in op and not isinstance(op[key], bool):
                raise GraphQLDocumentError(
                    "GraphQL snapshot authority policy is invalid"
                )
        if op.get("allow_empty_snapshot") is True and not op.get(
            "snapshot_authoritative", False
        ):
            raise GraphQLDocumentError(
                "GraphQL empty snapshot approval requires authoritative mode"
            )

    @classmethod
    def _validate_mappings(cls, mappings: Any) -> None:
        if not isinstance(mappings, dict):
            raise GraphQLDocumentError("GraphQL entity mappings are invalid")
        recognized = False
        for kind, aliases in _MAPPING_KEYS.items():
            mapping = next(
                (mappings.get(alias) for alias in aliases if alias in mappings),
                None,
            )
            if mapping is None:
                continue
            recognized = True
            cls._validate_entity_mapping(kind, mapping)
        if not recognized:
            raise GraphQLDocumentError("GraphQL profile has no entity mappings")

    @classmethod
    def _validate_snapshot_policy_and_mappings(cls, op: dict[str, Any]) -> Any:
        mappings = op.get("mappings")
        cls._validate_snapshot_authority_flags(op)
        if mappings is not None:
            cls._validate_mappings(mappings)
        return mappings

    @staticmethod
    def _validate_governance_shape(governance: Any) -> None:
        if not isinstance(governance, dict):
            raise GraphQLDocumentError("GraphQL governance mapping is invalid")
        _classification(governance.get("classification"))
        if "legal_hold" in governance and not isinstance(
            governance["legal_hold"], bool
        ):
            raise GraphQLDocumentError("GraphQL governance legal hold is invalid")

    @staticmethod
    def _validate_governance_retention(
        governance: dict[str, Any], mappings: Any
    ) -> str:
        retention = str(governance.get("retention") or "").strip()
        if mappings is not None and not retention:
            raise GraphQLDocumentError(
                "GraphQL hierarchy governance requires a retention policy"
            )
        if retention and not _RETENTION_RE.fullmatch(retention):
            raise GraphQLDocumentError("GraphQL governance retention is invalid")
        return retention

    @staticmethod
    def _validate_governance_versions(governance: dict[str, Any]) -> None:
        tenant = str(governance.get("tenant") or "").strip()
        if tenant:
            _safe_alias(tenant, label="tenant")
        for key in ("schema_version", "ontology_mapping_version"):
            if key in governance and not _VERSION_LABEL_RE.fullmatch(
                str(governance[key] or "")
            ):
                raise GraphQLDocumentError("GraphQL governance version is invalid")

    @classmethod
    def _validate_governance(cls, profile: dict[str, Any], mappings: Any) -> None:
        governance = profile.get("governance") or {}
        cls._validate_governance_shape(governance)
        cls._validate_governance_retention(governance, mappings)
        cls._validate_governance_versions(governance)
        _access_from_config(governance.get("access", profile.get("access")))

    @staticmethod
    def _validate_discovery(profile: dict[str, Any]) -> None:
        discovery = profile.get("discovery")
        if discovery is not None and not isinstance(discovery, Mapping):
            raise GraphQLDocumentError("GraphQL discovery policy is invalid")
        if isinstance(discovery, Mapping):
            for key in ("enabled", "allow_introspection"):
                if key in discovery and not isinstance(discovery[key], bool):
                    raise GraphQLDocumentError("GraphQL discovery policy is invalid")

    def _transport_security(self, profile: dict[str, Any]) -> Any:
        """Resolve a runtime-only TLS profile without exposing it downstream."""
        if self._resolved_tls is not None:
            return self._resolved_tls
        from agent_utilities.core.transport_security import (
            TransportSecurityError,
            resolve_configured_tls_profile,
        )

        configured = profile.get("transport_security", profile.get("tls"))
        profile_name, profile_ref, inline = _parse_configured_tls(configured)
        profile_name = str(profile.get("tls_profile") or "").strip() or profile_name
        profile_ref = str(profile.get("tls_profile_ref") or "").strip() or profile_ref
        try:
            self._resolved_tls = resolve_configured_tls_profile(
                "GRAPHQL_DOCUMENT",
                profile_name=profile_name,
                profile_ref=profile_ref,
                profile=inline,
                resolver=self._profile_resolver,
            )
        except TransportSecurityError:
            raise GraphQLDocumentError(
                "GraphQL transport security profile is invalid"
            ) from None
        # HTTPX receives an already-loaded SSLContext, so secret-backed files
        # need not remain on disk for the lifetime of this connector.
        self._resolved_tls.cleanup()
        return self._resolved_tls

    def _effective_limit(
        self,
        profile: dict[str, Any],
        key: str,
        configured: int,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        limits = profile.get("limits") or {}
        profile_value = _bounded_int(
            limits.get(key),
            default=configured,
            minimum=minimum,
            maximum=maximum,
        )
        return min(configured, profile_value)

    def _resolve_governance_access(
        self, value: dict[str, Any], profile: dict[str, Any]
    ) -> ExternalAccess:
        access = _access_from_config(value.get("access", profile.get("access")))
        if value.get("access") is None and profile.get("access") is None:
            return self.external_access
        return access

    def _governance(
        self, profile: dict[str, Any]
    ) -> tuple[ExternalAccess, DataClassification, str | None, bool, str, str, str]:
        value = profile.get("governance") or {}
        access = self._resolve_governance_access(value, profile)
        classification = _classification(value.get("classification"))
        _validate_governance_classification(classification, access)
        retention = str(value.get("retention") or "").strip() or None
        legal_hold = value.get("legal_hold", False)
        tenant = str(value.get("tenant") or "").strip().lower()
        schema_version = str(value.get("schema_version") or "1")
        mapping_version = str(value.get("ontology_mapping_version") or "1")
        return (
            access,
            classification,
            retention,
            legal_hold,
            tenant,
            schema_version,
            mapping_version,
        )

    def _post(
        self,
        profile: dict[str, Any],
        query: str,
        variables: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        endpoint = str(profile["endpoint"])
        try:
            from agent_utilities.core.config import config as runtime_config
            from agent_utilities.protocols.source_connectors.http_safety import (
                require_safe_source_url,
            )

            require_safe_source_url(
                endpoint,
                allowed_private_hosts=runtime_config.source_http_allowed_private_hosts,
                resolve_dns=self._transport is None,
            )
            response_bound = min(
                self.max_response_bytes,
                int(runtime_config.source_http_max_response_bytes),
            )
        except Exception as exc:
            raise GraphQLDocumentError(
                f"GraphQL endpoint is not permitted ({type(exc).__name__})"
            ) from None
        headers = dict(profile.get("headers") or {})
        request_options: dict[str, Any] = {
            "headers": {str(k): str(v) for k, v in headers.items()},
            "json": {"query": query, "variables": variables},
            "timeout": self.timeout_seconds,
        }
        try:
            if self._transport is not None:
                response = self._transport.post(endpoint, **request_options)
                response.raise_for_status()
                payload, response_size = _bounded_transport_payload(
                    response, response_bound
                )
            else:
                from agent_utilities.core.http_client import create_http_client
                from agent_utilities.protocols.source_connectors.http_safety import (
                    _read_bounded,
                )

                tls = self._transport_security(profile)
                with create_http_client(
                    timeout=self.timeout_seconds,
                    follow_redirects=False,
                    **tls.httpx_kwargs(),
                ) as client:
                    with client.stream(
                        "POST",
                        endpoint,
                        headers=request_options["headers"],
                        json=request_options["json"],
                    ) as response:
                        response.raise_for_status()
                        raw_payload = _read_bounded(response, response_bound)
                    payload = json.loads(
                        raw_payload, parse_constant=_reject_json_constant
                    )
        except Exception as exc:
            if isinstance(exc, GraphQLDocumentError):
                raise
            raise GraphQLDocumentError(
                f"GraphQL request failed ({type(exc).__name__})"
            ) from None
        if not isinstance(payload, dict):
            raise GraphQLDocumentError("GraphQL response is not an object")
        if self._transport is None:
            response_size = len(raw_payload)
        return payload, response_size

    def execute(
        self, document: str, variables: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute one bounded discovery read through the configured transport.

        This is the transport-neutral callable consumed by graph discovery
        adapters. It is disabled unless the secret-backed source profile opts in;
        introspection additionally requires ``allow_introspection``. The query,
        endpoint, headers, TLS material, and response never enter checkpoints or
        connector diagnostics.
        """
        profile = self._resolve_profile()
        policy = profile.get("discovery") or {}
        if not isinstance(policy, Mapping) or not bool(policy.get("enabled", False)):
            raise GraphQLDocumentError("GraphQL discovery is not enabled")
        query = _validate_query_document(
            document,
            allow_introspection=bool(policy.get("allow_introspection", False)),
        )
        payload, _size = self._post(profile, query, dict(variables or {}))
        if payload.get("errors"):
            raise GraphQLDocumentError("GraphQL discovery source returned an error")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise GraphQLDocumentError("GraphQL discovery response has no data object")
        return {"data": data}

    def _page_data(
        self,
        *,
        profile: dict[str, Any],
        operation: dict[str, Any],
        variables: dict[str, Any],
    ) -> tuple[dict[str, Any], int, int, int]:
        """Read one page with bounded, allowlisted optional-field recovery."""
        query = _validate_query_document(operation.get("query"))
        payload, response_size = self._post(profile, query, variables)
        fallback_count = 0
        fallbacks = operation.get("optional_field_fallbacks") or []
        for fallback in fallbacks[: self.max_fallbacks]:
            errors = payload.get("errors")
            if not errors:
                break
            codes = _policy_values(
                fallback.get("codes"), label="fallback code", pattern=_ERROR_CODE_RE
            )
            paths = _policy_values(
                fallback.get("paths"), label="fallback path", pattern=_FIELD_PATH_RE
            )
            if not _errors_are_allowlisted(errors, codes=codes, paths=paths):
                break
            payload, size = self._post(
                profile,
                _validate_query_document(fallback.get("query")),
                variables,
            )
            response_size += size
            fallback_count += 1

        errors = payload.get("errors")
        partial_count = 0
        if errors:
            partial = operation.get("partial_errors") or {}
            codes = _policy_values(
                partial.get("codes"),
                label="partial-error code",
                pattern=_ERROR_CODE_RE,
            )
            paths = _policy_values(
                partial.get("paths"),
                label="partial-error path",
                pattern=_FIELD_PATH_RE,
            )
            if not _errors_are_allowlisted(errors, codes=codes, paths=paths):
                # Error messages can echo variables, paths, or upstream identities.
                raise GraphQLDocumentError("GraphQL source returned an error")
            partial_count = len(errors)

        data = payload.get("data")
        if not isinstance(data, dict):
            raise GraphQLDocumentError("GraphQL response has no data object")
        return data, response_size, fallback_count, partial_count

    def _page_variables(
        self,
        pagination: Any,
        read_bound: Any,
        page_size: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        variables = dict(self.variables)
        if isinstance(pagination, dict):
            variables[str(pagination["page_size_variable"])] = page_size
            variables[str(pagination["cursor_variable"])] = cursor
        elif isinstance(read_bound, dict):
            variables[str(read_bound["variable"])] = min(
                page_size, int(read_bound["maximum"])
            )
        return variables

    def _next_page_cursor(
        self,
        data: Any,
        pagination: dict[str, Any],
        seen_cursors: set[str],
        page_index: int,
        max_pages: int,
    ) -> str | None:
        has_more = _dig(data, str(pagination["has_more_path"]), False)
        if not isinstance(has_more, bool):
            raise GraphQLDocumentError(
                "GraphQL pagination continuation flag is not boolean"
            )
        if not has_more:
            return None
        next_cursor = _dig(data, str(pagination["next_cursor_path"]))
        if not isinstance(next_cursor, str):
            raise GraphQLDocumentError(
                "GraphQL pagination returned an invalid continuation"
            )
        next_cursor_text = next_cursor
        if _is_invalid_next_cursor(next_cursor_text, seen_cursors):
            raise GraphQLDocumentError(
                "GraphQL pagination returned an invalid continuation"
            )
        if page_index + 1 >= max_pages:
            raise GraphQLDocumentError(
                "GraphQL pagination exceeds the configured page bound"
            )
        seen_cursors.add(next_cursor_text)
        return next_cursor_text

    def _fetch_roots(
        self, profile: dict[str, Any], operation: dict[str, Any]
    ) -> tuple[list[Any], dict[str, int]]:
        """Fetch all configured pages with strict response and cursor bounds."""
        pagination = operation.get("pagination")
        read_bound = operation.get("read_bound")
        max_pages = self._effective_limit(
            profile, "max_pages", self.max_pages, minimum=1, maximum=100
        )
        page_size = self._effective_limit(
            profile, "page_size", self.page_size, minimum=1, maximum=1_000
        )
        max_total_bytes = self._effective_limit(
            profile,
            "max_total_response_bytes",
            self.max_total_response_bytes,
            minimum=1_024,
            maximum=100_000_000,
        )
        roots: list[Any] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        total_bytes = 0
        fallbacks = 0
        partial_errors = 0

        for page_index in range(max_pages):
            variables = self._page_variables(pagination, read_bound, page_size, cursor)
            data, response_size, page_fallbacks, page_partial = self._page_data(
                profile=profile,
                operation=operation,
                variables=variables,
            )
            total_bytes += response_size
            if total_bytes > max_total_bytes:
                raise GraphQLDocumentError(
                    "GraphQL responses exceed the configured total bound"
                )
            fallbacks += page_fallbacks
            partial_errors += page_partial
            roots.append(_dig(data, str(operation["root_path"])))

            if not isinstance(pagination, dict):
                break
            cursor = self._next_page_cursor(
                data, pagination, seen_cursors, page_index, max_pages
            )
            if cursor is None:
                break

        return roots, {
            "pages": len(roots),
            "response_bytes": total_bytes,
            "fallbacks": fallbacks,
            "partial_errors": partial_errors,
        }

    def _render_title(
        self, record: dict[str, Any], operation: dict[str, Any], raw_id: Any
    ) -> tuple[str, Any]:
        raw_title = _dig(record, str(operation.get("title_path") or "title"), raw_id)
        return self._privacy.sanitize_text(str(raw_title))

    def _render_content_lines(
        self, record: dict[str, Any], operation: dict[str, Any]
    ) -> tuple[list[str], Any]:
        content_path = str(operation.get("content_path") or "")
        if not content_path:
            return [], None
        clean_content, content_report = self._privacy.sanitize_text(
            str(_dig(record, content_path, "") or "")
        )
        lines: list[str] = []
        if clean_content.strip():
            lines = ["", clean_content.strip()]
        return lines, content_report

    def _render_frontmatter_lines(self, frontmatter: Any) -> tuple[list[str], Any]:
        if frontmatter in (None, "", {}, []):
            return [], None
        clean_frontmatter, report = self._privacy.sanitize(frontmatter)
        if isinstance(clean_frontmatter, str):
            return ["", clean_frontmatter], report
        return (
            [
                "",
                json.dumps(
                    clean_frontmatter,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                ),
            ],
            report,
        )

    def _render_one_section(
        self,
        section: Any,
        *,
        title_field: str,
        level_field: str,
        content_field: str,
    ) -> tuple[list[str] | None, Any]:
        if not isinstance(section, dict):
            return None, None
        clean_section, report = self._privacy.sanitize(section)
        if not isinstance(clean_section, dict):
            return None, report
        section_title = str(clean_section.get(title_field) or "Section")
        try:
            level = max(2, min(int(clean_section.get(level_field) or 2) + 1, 6))
        except (TypeError, ValueError):
            level = 2
        content = str(clean_section.get(content_field) or "").strip()
        lines = ["", f"{'#' * level} {section_title}"]
        if content:
            lines.extend(["", content])
        return lines, report

    def _render_sections(
        self, sections: list[Any], operation: dict[str, Any]
    ) -> tuple[list[str], list[Any]]:
        title_field = str(operation.get("section_title_field") or "title")
        level_field = str(operation.get("section_level_field") or "level")
        content_field = str(operation.get("section_content_field") or "content")
        lines: list[str] = []
        reports: list[Any] = []
        for section in sections[: self.max_sections]:
            section_lines, report = self._render_one_section(
                section,
                title_field=title_field,
                level_field=level_field,
                content_field=content_field,
            )
            if report is not None:
                reports.append(report)
            if section_lines is not None:
                lines.extend(section_lines)
        return lines, reports

    def _render_governance_fields(
        self,
        governance: tuple[
            ExternalAccess,
            DataClassification,
            str | None,
            bool,
            str,
            str,
            str,
        ]
        | None,
    ) -> tuple[ExternalAccess, DataClassification, str | None, bool]:
        if governance is None:
            return self.external_access, DataClassification.INTERNAL, None, False
        return governance[0], governance[1], governance[2], governance[3]

    def _render_body(
        self, record: dict[str, Any], operation: dict[str, Any]
    ) -> tuple[Any, str, str, list[Any]] | None:
        """(raw_id, clean_title, text, reports), or None if there is no body."""
        raw_id = _dig(record, str(operation.get("id_path") or "id"))
        if raw_id in (None, ""):
            return None
        frontmatter = _dig(
            record,
            str(operation.get("frontmatter_path") or "document.frontmatter"),
            None,
        )
        sections = _dig(
            record,
            str(operation.get("sections_path") or "document.sections"),
            [],
        )
        if not isinstance(sections, list):
            sections = []

        clean_title, title_report = self._render_title(record, operation, raw_id)
        body: list[str] = [f"# {clean_title}"]
        reports: list[Any] = [title_report]

        content_lines, content_report = self._render_content_lines(record, operation)
        body.extend(content_lines)
        if content_report is not None:
            reports.append(content_report)

        frontmatter_lines, frontmatter_report = self._render_frontmatter_lines(
            frontmatter
        )
        body.extend(frontmatter_lines)
        if frontmatter_report is not None:
            reports.append(frontmatter_report)

        section_lines, section_reports = self._render_sections(sections, operation)
        body.extend(section_lines)
        reports.extend(section_reports)

        text = "\n".join(body)[: self.max_content_chars].strip()
        if not text:
            return None
        return raw_id, clean_title, text, reports

    def _render_digests(
        self,
        record: dict[str, Any],
        operation: dict[str, Any],
        identity_key: str,
        raw_id: Any,
        text: str,
        reports: list[Any],
    ) -> tuple[str, str, str, list[str], int]:
        document_id = _private_digest(
            identity_key, self.source_alias, self.operation, "document", str(raw_id)
        )
        updated_path = str(operation.get("updated_path") or "")
        raw_updated = _dig(record, updated_path) if updated_path else None
        clean_updated, updated_report = self._privacy.sanitize_text(
            str(raw_updated or "")
        )
        reports.append(updated_report)
        detected = sorted(
            {item for report in reports for item in report.detected_types}
        )
        redactions = sum(report.redactions for report in reports)
        content_digest = _digest(text)
        version_digest = _private_digest(
            identity_key,
            self.source_alias,
            self.operation,
            "document-version",
            str(raw_id),
            clean_updated or content_digest,
        )
        return document_id, content_digest, version_digest, detected, redactions

    def _render_document(
        self,
        record: dict[str, Any],
        operation: dict[str, Any],
        identity_key: str,
        *,
        profile_digest: str = "",
        entity_id: str | None = None,
        governance: tuple[
            ExternalAccess,
            DataClassification,
            str | None,
            bool,
            str,
            str,
            str,
        ]
        | None = None,
    ) -> SourceDocument | None:
        rendered = self._render_body(record, operation)
        if rendered is None:
            return None
        raw_id, clean_title, text, reports = rendered
        document_id, content_digest, version_digest, detected, redactions = (
            self._render_digests(record, operation, identity_key, raw_id, text, reports)
        )
        access, classification, retention, legal_hold = self._render_governance_fields(
            governance
        )
        governed_entity_id = entity_id or (
            f"doc:graphql_document:{hashlib.sha256(document_id.encode('utf-8')).hexdigest()[:24]}"
        )
        return SourceDocument(
            id=document_id,
            source_uri=f"external-source://{self.source_alias}/{document_id}",
            title=clean_title,
            text=text,
            doc_type=_safe_document_type(operation.get("doc_type")),
            metadata={
                "source_alias": self.source_alias,
                "source_kind": "graphql",
                "operation": self.operation,
                "governed_entity_id": governed_entity_id,
                "profile_digest": profile_digest,
                "content_digest": content_digest,
                "classification": classification.value,
                "retention": retention,
                "legal_hold": legal_hold,
                "embedding_handoff": True,
                "privacy": {
                    "redactions": redactions,
                    "detected_types": detected,
                },
            },
            external_access=access,
            updated_at=version_digest,
        )

    @staticmethod
    def _mapping_for(operation: dict[str, Any], kind: str) -> dict[str, Any] | None:
        mappings = operation.get("mappings")
        if not isinstance(mappings, dict):
            return None
        for key in _MAPPING_KEYS[kind]:
            value = mappings.get(key)
            if isinstance(value, dict):
                return value
        return None

    def _mapping_flat_records(
        self,
        seeds: list[Any],
        mapping: dict[str, Any],
        limit: int,
    ) -> tuple[list[tuple[dict[str, Any], Any | None, int]], int]:
        parent_path = str(mapping.get("parent_id_path") or "")
        records: list[tuple[dict[str, Any], Any | None, int]] = []
        truncated = 0
        for seed in seeds:
            if not isinstance(seed, dict):
                continue
            if len(records) >= limit:
                truncated += 1
                continue
            parent = _dig(seed, parent_path) if parent_path else None
            records.append((seed, parent, 0))
        return records, truncated

    def _hierarchy_step(
        self,
        record: dict[str, Any],
        children_path: str,
        id_path: str,
        depth: int,
        max_depth: int,
    ) -> tuple[list[tuple[dict[str, Any], Any, int]], int]:
        """Returns (pushable child stack entries, extra-truncated count)."""
        children = [
            child
            for child in _dig_many(record, children_path)
            if isinstance(child, dict)
        ]
        if not children:
            return [], 0
        if depth + 1 >= max_depth:
            return [], len(children)
        raw_id = _dig(record, id_path)
        pushable = [(child, raw_id, depth + 1) for child in reversed(children)]
        return pushable, 0

    def _mapping_hierarchy_records(
        self,
        seeds: list[Any],
        children_path: str,
        id_path: str,
        limit: int,
        max_depth: int,
    ) -> tuple[list[tuple[dict[str, Any], Any | None, int]], int]:
        records: list[tuple[dict[str, Any], Any | None, int]] = []
        truncated = 0
        stack: list[tuple[dict[str, Any], Any | None, int]] = [
            (seed, None, 0) for seed in reversed(seeds) if isinstance(seed, dict)
        ]
        while stack:
            record, parent, depth = stack.pop()
            if len(records) >= limit:
                truncated += 1
                continue
            records.append((record, parent, depth))
            pushable, extra_truncated = self._hierarchy_step(
                record, children_path, id_path, depth, max_depth
            )
            stack.extend(pushable)
            truncated += extra_truncated
        return records, truncated

    def _mapping_records(
        self,
        roots: list[Any],
        *,
        kind: str,
        mapping: dict[str, Any],
        limit: int,
        max_depth: int,
    ) -> tuple[list[tuple[dict[str, Any], Any | None, int]], int]:
        records_path = str(mapping.get("records_path") or "")
        children_path = str(mapping.get("children_path") or "")
        id_path = str(mapping.get("id_path") or "id")
        records: list[tuple[dict[str, Any], Any | None, int]] = []
        truncated = 0

        for root in roots:
            seeds = _dig_many(root, records_path)
            remaining = limit - len(records)
            if kind != "hierarchy" or not children_path:
                new_records, new_truncated = self._mapping_flat_records(
                    seeds, mapping, remaining
                )
            else:
                new_records, new_truncated = self._mapping_hierarchy_records(
                    seeds, children_path, id_path, remaining, max_depth
                )
            records.extend(new_records)
            truncated += new_truncated
        return records, truncated

    def _entity_node_id(
        self, identity_key: str, *, kind: str, raw_id: Any
    ) -> tuple[str, str]:
        opaque = _private_digest(
            identity_key,
            self.source_alias,
            self.operation,
            kind,
            str(raw_id),
        )
        if kind == "document":
            node_id = (
                "doc:graphql_document:"
                f"{hashlib.sha256(opaque.encode('utf-8')).hexdigest()[:24]}"
            )
        else:
            node_id = f"external:{self.source_alias}:{kind}:{opaque[:32]}"
        return opaque, node_id

    def _selected_properties(
        self, record: dict[str, Any], mapping: dict[str, Any]
    ) -> tuple[dict[str, Any], int, tuple[str, ...]]:
        properties_path = str(mapping.get("properties_path") or "")
        source = _dig(record, properties_path, {}) if properties_path else record
        if not isinstance(source, Mapping):
            source = {}
        selected = {
            str(path): _dig(source, str(path))
            for path in mapping.get("property_allowlist") or []
        }
        clean, report = self._privacy.sanitize(selected)
        if not isinstance(clean, dict):
            clean = {}
        return clean, report.redactions, report.detected_types

    def _target_node_id(
        self,
        identity_key: str,
        *,
        kind: str,
        raw_id: Any,
        known_ids: set[str],
    ) -> str | None:
        if raw_id in (None, ""):
            return None
        _opaque, target = self._entity_node_id(identity_key, kind=kind, raw_id=raw_id)
        return target if target in known_ids else None

    def _entity_links(
        self,
        *,
        identity_key: str,
        item: dict[str, Any],
        known_ids: set[str],
    ) -> list[dict[str, Any]]:
        kind = str(item["kind"])
        mapping = item["mapping"]
        record = item["record"]
        source = str(item["node_id"])
        links: list[dict[str, Any]] = []

        parent_raw = item.get("parent_raw")
        parent_path = str(mapping.get("parent_id_path") or "")
        if parent_raw in (None, "") and parent_path:
            parent_raw = _dig(record, parent_path)
        if parent_raw not in (None, ""):
            parent_kind = str(mapping.get("parent_kind") or "hierarchy").lower()
            if parent_kind not in _ENTITY_KINDS:
                parent_kind = "hierarchy"
            target = self._target_node_id(
                identity_key,
                kind=parent_kind,
                raw_id=parent_raw,
                known_ids=known_ids,
            )
            if target:
                links.append(
                    {
                        "source": source,
                        "target": target,
                        "type": _safe_entity_type(
                            mapping.get("parent_relation"), fallback="PART_OF"
                        ),
                    }
                )

        application_path = str(mapping.get("application_id_path") or "")
        if application_path:
            target = self._target_node_id(
                identity_key,
                kind="application",
                raw_id=_dig(record, application_path),
                known_ids=known_ids,
            )
            if target:
                links.append(
                    {
                        "source": source,
                        "target": target,
                        "type": _safe_entity_type(
                            mapping.get("application_relation"),
                            fallback="DESCRIBES_APPLICATION",
                        ),
                    }
                )

        if kind == "dependency":
            source_path = str(mapping.get("source_id_path") or "")
            target_path = str(mapping.get("target_id_path") or "")
            dependency_source = self._target_node_id(
                identity_key,
                kind="application",
                raw_id=_dig(record, source_path) if source_path else None,
                known_ids=known_ids,
            )
            dependency_target = self._target_node_id(
                identity_key,
                kind="application",
                raw_id=_dig(record, target_path) if target_path else None,
                known_ids=known_ids,
            )
            if dependency_source and dependency_target:
                links.append(
                    {
                        "source": dependency_source,
                        "target": dependency_target,
                        "type": _safe_entity_type(
                            mapping.get("dependency_relation"),
                            fallback="DEPENDS_ON",
                        ),
                        "evidence": source,
                    }
                )
        return links

    def _checkpoint_batch(
        self,
        *,
        documents: list[SourceDocument],
        envelopes: list[ChangeEnvelope],
        versions: dict[str, str],
        checkpoint: ConnectorCheckpoint | None,
        profile_digest: str,
        diagnostics: dict[str, Any],
        governance: tuple[
            ExternalAccess,
            DataClassification,
            str | None,
            bool,
            str,
            str,
            str,
        ],
        snapshot_authoritative: bool,
        allow_empty_snapshot: bool,
    ) -> GraphQLHierarchyBatch:
        prior_state, baseline, prior_sequence = self._load_prior_checkpoint_state(
            checkpoint
        )
        self._validate_snapshot_completeness(
            diagnostics,
            snapshot_authoritative,
            baseline,
            versions,
            allow_empty_snapshot,
        )
        governance_state = self._governance_state_dict(governance)
        changed, changed_documents, changed_envelopes = self._changed_entities(
            documents,
            envelopes,
            versions,
            baseline,
            prior_state,
            profile_digest,
            governance_state,
        )
        tombstones = self._build_tombstones(
            baseline,
            versions,
            prior_state,
            snapshot_authoritative,
            allow_empty_snapshot,
            prior_sequence,
        )
        next_checkpoint = self._build_next_checkpoint(
            prior_state,
            baseline,
            versions,
            profile_digest,
            governance_state,
            snapshot_authoritative,
            allow_empty_snapshot,
            prior_sequence,
        )
        diagnostics = self._finalize_diagnostics(
            diagnostics, versions, documents, changed, changed_documents, tombstones
        )
        return GraphQLHierarchyBatch(
            documents=tuple(changed_documents),
            envelopes=tuple((*changed_envelopes, *tombstones)),
            checkpoint=next_checkpoint,
            diagnostics=diagnostics,
        )

    def _validate_checkpoint_scope(self, prior_state: dict[str, Any]) -> None:
        if prior_state and prior_state.get("checkpoint_format") != _CHECKPOINT_FORMAT:
            raise GraphQLDocumentError("GraphQL checkpoint format is invalid")
        if prior_state and (
            prior_state.get("source_alias") != self.source_alias
            or prior_state.get("operation") != self.operation
        ):
            raise GraphQLDocumentError("GraphQL checkpoint scope is invalid")

    def _validate_checkpoint_baseline(
        self, prior_state: dict[str, Any]
    ) -> dict[str, str]:
        baseline_raw = prior_state.get("versions")
        baseline = baseline_raw if isinstance(baseline_raw, dict) else {}
        if len(baseline) > max(self.max_entities, self.max_documents) or any(
            not isinstance(node_id, str)
            or not isinstance(version, str)
            or not node_id
            or not version
            for node_id, version in baseline.items()
        ):
            raise GraphQLDocumentError("GraphQL checkpoint state is invalid")
        return baseline

    @staticmethod
    def _validate_checkpoint_sequence(prior_state: dict[str, Any]) -> int:
        try:
            prior_sequence = int(prior_state.get("snapshot_sequence") or 0)
        except (TypeError, ValueError):
            raise GraphQLDocumentError("GraphQL checkpoint state is invalid") from None
        if prior_sequence < 0:
            raise GraphQLDocumentError("GraphQL checkpoint state is invalid")
        return prior_sequence

    def _load_prior_checkpoint_state(
        self, checkpoint: ConnectorCheckpoint | None
    ) -> tuple[dict[str, Any], dict[str, str], int]:
        prior_state = checkpoint.state if checkpoint else {}
        self._validate_checkpoint_scope(prior_state)
        baseline = self._validate_checkpoint_baseline(prior_state)
        prior_sequence = self._validate_checkpoint_sequence(prior_state)
        return prior_state, baseline, prior_sequence

    @staticmethod
    def _validate_snapshot_completeness(
        diagnostics: dict[str, Any],
        snapshot_authoritative: bool,
        baseline: dict[str, str],
        versions: dict[str, str],
        allow_empty_snapshot: bool,
    ) -> None:
        incomplete_reasons = sum(
            int(diagnostics.get(key) or 0)
            for key in ("truncated", "partial_errors", "invalid_records")
        )
        if snapshot_authoritative and incomplete_reasons:
            raise GraphQLDocumentError(
                "GraphQL authoritative snapshot did not complete within its bounds"
            )
        if (
            snapshot_authoritative
            and baseline
            and not versions
            and not allow_empty_snapshot
        ):
            raise GraphQLDocumentError(
                "GraphQL empty authoritative snapshot requires explicit approval"
            )

    @staticmethod
    def _governance_state_dict(
        governance: tuple[
            ExternalAccess,
            DataClassification,
            str | None,
            bool,
            str,
            str,
            str,
        ],
    ) -> dict[str, Any]:
        (
            access,
            classification,
            retention,
            legal_hold,
            tenant,
            schema_version,
            mapping_version,
        ) = governance
        return {
            "access": access.model_dump(mode="json"),
            "classification": classification.value,
            "retention": retention,
            "legal_hold": legal_hold,
            "tenant": tenant,
            "schema_version": schema_version,
            "ontology_mapping_version": mapping_version,
        }

    @staticmethod
    def _changed_node_ids(
        versions: dict[str, str],
        baseline: dict[str, str],
        prior_state: dict[str, Any],
        profile_digest: str,
        governance_state: dict[str, Any],
    ) -> set[str]:
        changed = {
            node_id
            for node_id, version in versions.items()
            if str(baseline.get(node_id) or "") != version
        }
        if prior_state and (
            prior_state.get("profile_digest") != profile_digest
            or prior_state.get("governance") != governance_state
        ):
            changed.update(versions)
        return changed

    @staticmethod
    def _changed_entities(
        documents: list[SourceDocument],
        envelopes: list[ChangeEnvelope],
        versions: dict[str, str],
        baseline: dict[str, str],
        prior_state: dict[str, Any],
        profile_digest: str,
        governance_state: dict[str, Any],
    ) -> tuple[set[str], list[SourceDocument], list[ChangeEnvelope]]:
        changed = GraphQLDocumentConnector._changed_node_ids(
            versions, baseline, prior_state, profile_digest, governance_state
        )
        changed_documents = [
            document
            for document in documents
            if str(document.metadata.get("governed_entity_id") or "") in changed
        ]
        changed_envelopes = [
            envelope for envelope in envelopes if envelope.source_object_id in changed
        ]
        return changed, changed_documents, changed_envelopes

    @staticmethod
    def _parse_previous_governance_fields(
        previous_governance: Mapping[str, Any],
    ) -> tuple[ExternalAccess, DataClassification, str | None, bool, str, str, str]:
        try:
            previous_access = _access_from_config(previous_governance.get("access"))
            previous_classification = _classification(
                previous_governance.get("classification")
            )
            previous_retention = (
                str(previous_governance.get("retention") or "").strip() or None
            )
            previous_legal_hold = bool(previous_governance.get("legal_hold", False))
            previous_tenant = str(previous_governance.get("tenant") or "")
            previous_schema = str(previous_governance.get("schema_version") or "1")
            previous_mapping = str(
                previous_governance.get("ontology_mapping_version") or "1"
            )
        except (TypeError, ValueError, GraphQLDocumentError):
            raise GraphQLDocumentError(
                "GraphQL checkpoint governance state is invalid"
            ) from None
        return (
            previous_access,
            previous_classification,
            previous_retention,
            previous_legal_hold,
            previous_tenant,
            previous_schema,
            previous_mapping,
        )

    @classmethod
    def _resolve_previous_governance(
        cls,
        previous_governance: Any,
    ) -> tuple[ExternalAccess, DataClassification, str | None, bool, str, str, str]:
        if not isinstance(previous_governance, Mapping):
            raise GraphQLDocumentError("GraphQL checkpoint governance state is invalid")
        fields = cls._parse_previous_governance_fields(previous_governance)
        previous_retention = fields[2]
        previous_tenant = fields[4]
        if previous_tenant:
            _safe_alias(previous_tenant, label="tenant")
        if previous_retention and not _RETENTION_RE.fullmatch(previous_retention):
            raise GraphQLDocumentError("GraphQL checkpoint governance state is invalid")
        return fields

    def _tombstone_for_node(
        self,
        node_id: str,
        baseline: dict[str, str],
        next_snapshot_digest: str,
        previous_fields: tuple[
            ExternalAccess, DataClassification, str | None, bool, str, str, str
        ],
        prior_state: dict[str, Any],
        versions: dict[str, str],
        allow_empty_snapshot: bool,
    ) -> ChangeEnvelope:
        (
            previous_access,
            previous_classification,
            previous_retention,
            previous_legal_hold,
            previous_tenant,
            previous_schema,
            previous_mapping,
        ) = previous_fields
        delete_version = _digest(
            "graphql-snapshot-delete",
            baseline[node_id],
            next_snapshot_digest,
        )
        return ChangeEnvelope(
            connector="graphql_document",
            operation="delete",
            tenant=previous_tenant,
            source_instance=self.source_alias,
            source_object_id=node_id,
            source_version=delete_version,
            schema_version=previous_schema,
            ontology_mapping_version=previous_mapping,
            source_acl=previous_access,
            classification=previous_classification,
            retention=previous_retention,
            legal_hold=previous_legal_hold,
            provenance={
                "profile_digest": str(prior_state.get("profile_digest") or ""),
                "privacy_gate": True,
                "identity_scheme": "hmac-sha256",
                "snapshot_reconciliation": True,
                "authoritative_empty_approved": bool(
                    not versions and allow_empty_snapshot
                ),
            },
            checkpoint=delete_version,
        )

    def _build_tombstones(
        self,
        baseline: dict[str, str],
        versions: dict[str, str],
        prior_state: dict[str, Any],
        snapshot_authoritative: bool,
        allow_empty_snapshot: bool,
        prior_sequence: int,
    ) -> list[ChangeEnvelope]:
        missing = sorted(set(baseline).difference(versions))
        if not (snapshot_authoritative and missing):
            return []
        previous_governance = prior_state.get("governance")
        previous_fields = self._resolve_previous_governance(previous_governance)
        next_snapshot_digest = _digest(
            self.source_alias,
            self.operation,
            prior_sequence + 1,
            sorted(versions.items()),
        )
        return [
            self._tombstone_for_node(
                node_id,
                baseline,
                next_snapshot_digest,
                previous_fields,
                prior_state,
                versions,
                allow_empty_snapshot,
            )
            for node_id in missing
        ]

    def _build_next_checkpoint(
        self,
        prior_state: dict[str, Any],
        baseline: dict[str, str],
        versions: dict[str, str],
        profile_digest: str,
        governance_state: dict[str, Any],
        snapshot_authoritative: bool,
        allow_empty_snapshot: bool,
        prior_sequence: int,
    ) -> ConnectorCheckpoint:
        checkpoint_changed = not prior_state or any(
            (
                baseline != versions,
                prior_state.get("profile_digest") != profile_digest,
                prior_state.get("governance") != governance_state,
                prior_state.get("snapshot_authoritative") != snapshot_authoritative,
                prior_state.get("allow_empty_snapshot") != allow_empty_snapshot,
            )
        )
        sequence = prior_sequence + int(checkpoint_changed)
        watermark = _digest(
            self.source_alias,
            self.operation,
            sequence,
            sorted(versions.items()),
        )
        return ConnectorCheckpoint(
            has_more=False,
            watermark=watermark,
            seen_ids=sorted(versions)[-self.max_entities :],
            state={
                "checkpoint_format": _CHECKPOINT_FORMAT,
                "source_alias": self.source_alias,
                "operation": self.operation,
                "profile_digest": profile_digest,
                "snapshot_sequence": sequence,
                "snapshot_authoritative": snapshot_authoritative,
                "allow_empty_snapshot": allow_empty_snapshot,
                "governance": governance_state,
                "versions": dict(sorted(versions.items())),
            },
        )

    @staticmethod
    def _finalize_diagnostics(
        diagnostics: dict[str, Any],
        versions: dict[str, str],
        documents: list[SourceDocument],
        changed: set[str],
        changed_documents: list[SourceDocument],
        tombstones: list[ChangeEnvelope],
    ) -> dict[str, Any]:
        return {
            **diagnostics,
            "entities": len(versions),
            "documents": len(documents),
            "changed_entities": len(changed),
            "changed_documents": len(changed_documents),
            "tombstones": len(tombstones),
        }

    def _document_batch(
        self,
        *,
        profile: dict[str, Any],
        operation: dict[str, Any],
        roots: list[Any],
        fetch_diagnostics: dict[str, int],
        checkpoint: ConnectorCheckpoint | None,
    ) -> GraphQLHierarchyBatch:
        identity_key = str(profile["identity_hmac_key"])
        governance = self._governance(profile)
        profile_digest = _digest(operation)
        governance_digest = _digest(
            governance[0].model_dump(mode="json"),
            governance[1].value,
            governance[2:],
        )
        max_documents = self._effective_limit(
            profile,
            "max_documents",
            self.max_documents,
            minimum=1,
            maximum=10_000,
        )
        records: list[dict[str, Any]] = []
        for root in roots:
            values = root if isinstance(root, list) else [root]
            records.extend(value for value in values if isinstance(value, dict))
        documents: list[SourceDocument] = []
        envelopes: list[ChangeEnvelope] = []
        versions: dict[str, str] = {}
        invalid_records = 0
        for record in records[:max_documents]:
            raw_id = _dig(record, str(operation.get("id_path") or "id"))
            if raw_id in (None, ""):
                invalid_records += 1
                continue
            opaque, node_id = self._entity_node_id(
                identity_key, kind="document", raw_id=raw_id
            )
            document = self._render_document(
                record,
                operation,
                identity_key,
                profile_digest=profile_digest,
                entity_id=node_id,
                governance=governance,
            )
            if document is None:
                invalid_records += 1
                continue
            version = _digest(
                str(document.updated_at or _digest(document.text)),
                profile_digest,
                governance_digest,
            )
            document.updated_at = version
            versions[node_id] = version
            payload = {
                "id": node_id,
                "type": "Document",
                "title": document.title,
                "doc_type": document.doc_type,
                "content_digest": document.metadata["content_digest"],
                "source_alias": self.source_alias,
                "source_kind": "graphql",
                "embedding_handoff": True,
            }
            (
                access,
                classification,
                retention,
                legal_hold,
                tenant,
                schema,
                mapping,
            ) = governance
            envelopes.append(
                ChangeEnvelope(
                    connector="graphql_document",
                    tenant=tenant,
                    source_instance=self.source_alias,
                    source_object_id=node_id,
                    source_version=version,
                    schema_version=schema,
                    ontology_mapping_version=mapping,
                    typed_payload=payload,
                    source_acl=access,
                    classification=classification,
                    retention=retention,
                    legal_hold=legal_hold,
                    provenance={
                        "profile_digest": profile_digest,
                        "privacy_gate": True,
                        "identity_scheme": "hmac-sha256",
                    },
                    checkpoint=version,
                )
            )
            documents.append(document)
            del opaque
        return self._checkpoint_batch(
            documents=documents,
            envelopes=envelopes,
            versions=versions,
            checkpoint=checkpoint,
            profile_digest=profile_digest,
            diagnostics={
                **fetch_diagnostics,
                "truncated": max(0, len(records) - max_documents),
                "invalid_records": invalid_records,
            },
            governance=governance,
            snapshot_authoritative=bool(operation.get("snapshot_authoritative", False)),
            allow_empty_snapshot=bool(operation.get("allow_empty_snapshot", False)),
        )

    def _hierarchy_batch(
        self,
        *,
        profile: dict[str, Any],
        operation: dict[str, Any],
        roots: list[Any],
        fetch_diagnostics: dict[str, int],
        checkpoint: ConnectorCheckpoint | None,
    ) -> GraphQLHierarchyBatch:
        identity_key = str(profile["identity_hmac_key"])
        governance = self._governance(profile)
        profile_digest = _digest(operation)
        governance_digest = _digest(
            governance[0].model_dump(mode="json"),
            governance[1].value,
            governance[2:],
        )
        max_entities = self._effective_limit(
            profile,
            "max_entities",
            self.max_entities,
            minimum=1,
            maximum=10_000,
        )
        max_documents = self._effective_limit(
            profile,
            "max_documents",
            self.max_documents,
            minimum=1,
            maximum=10_000,
        )
        max_depth = self._effective_limit(
            profile,
            "max_hierarchy_depth",
            self.max_hierarchy_depth,
            minimum=1,
            maximum=32,
        )
        prepared: list[dict[str, Any]] = []
        known_node_ids: set[str] = set()
        privacy_types: set[str] = set()
        privacy_redactions = 0
        truncated = 0
        invalid_records = 0

        for kind in _ENTITY_KINDS:
            mapping = self._mapping_for(operation, kind)
            if mapping is None:
                continue
            remaining = max(0, max_entities - len(prepared))
            if kind == "document":
                remaining = min(remaining, max_documents)
            if remaining == 0:
                truncated += 1
                continue
            records, mapping_truncated = self._mapping_records(
                roots,
                kind=kind,
                mapping=mapping,
                limit=remaining,
                max_depth=max_depth,
            )
            truncated += mapping_truncated
            for record, parent_raw, depth in records:
                raw_id = _dig(record, str(mapping.get("id_path") or "id"))
                if raw_id in (None, ""):
                    invalid_records += 1
                    continue
                opaque, node_id = self._entity_node_id(
                    identity_key, kind=kind, raw_id=raw_id
                )
                if node_id in known_node_ids:
                    continue
                properties, redactions, detected = self._selected_properties(
                    record, mapping
                )
                privacy_redactions += redactions
                privacy_types.update(detected)
                entity_type = _safe_entity_type(
                    mapping.get("entity_type"), fallback=_DEFAULT_ENTITY_TYPES[kind]
                )
                payload: dict[str, Any] = {
                    "id": node_id,
                    "type": entity_type,
                    "source_alias": self.source_alias,
                    "source_kind": "graphql",
                    "entity_kind": kind,
                    **properties,
                }
                document: SourceDocument | None = None
                if kind == "document":
                    document = self._render_document(
                        record,
                        mapping,
                        identity_key,
                        profile_digest=profile_digest,
                        entity_id=node_id,
                        governance=governance,
                    )
                    if document is None:
                        invalid_records += 1
                        continue
                    payload.update(
                        {
                            "title": document.title,
                            "doc_type": document.doc_type,
                            "content_digest": document.metadata["content_digest"],
                            "embedding_handoff": True,
                        }
                    )
                version_path = str(mapping.get("version_path") or "")
                raw_version = _dig(record, version_path) if version_path else None
                version = _private_digest(
                    identity_key,
                    self.source_alias,
                    self.operation,
                    kind,
                    str(raw_id),
                    str(raw_version or ""),
                    _digest(payload),
                    profile_digest,
                    governance_digest,
                )
                if document is not None:
                    document.updated_at = version
                known_node_ids.add(node_id)
                prepared.append(
                    {
                        "kind": kind,
                        "raw_id": raw_id,
                        "opaque": opaque,
                        "node_id": node_id,
                        "version": version,
                        "record": record,
                        "mapping": mapping,
                        "parent_raw": parent_raw,
                        "depth": depth,
                        "payload": payload,
                        "document": document,
                    }
                )

        documents: list[SourceDocument] = []
        envelopes: list[ChangeEnvelope] = []
        versions: dict[str, str] = {}
        (
            access,
            classification,
            retention,
            legal_hold,
            tenant,
            schema,
            mapping_version,
        ) = governance
        for item in prepared:
            links = self._entity_links(
                identity_key=identity_key, item=item, known_ids=known_node_ids
            )
            payload = dict(item["payload"])
            if links:
                payload["_links"] = links
            node_id = str(item["node_id"])
            version = str(item["version"])
            versions[node_id] = version
            document = item.get("document")
            if isinstance(document, SourceDocument):
                documents.append(document)
            envelopes.append(
                ChangeEnvelope(
                    connector="graphql_document",
                    tenant=tenant,
                    source_instance=self.source_alias,
                    source_object_id=node_id,
                    source_version=version,
                    schema_version=schema,
                    ontology_mapping_version=mapping_version,
                    typed_payload=payload,
                    source_acl=access,
                    classification=classification,
                    retention=retention,
                    legal_hold=legal_hold,
                    provenance={
                        "profile_digest": profile_digest,
                        "privacy_gate": True,
                        "identity_scheme": "hmac-sha256",
                        "pages": fetch_diagnostics.get("pages", 0),
                        "fallbacks": fetch_diagnostics.get("fallbacks", 0),
                        "partial_errors": fetch_diagnostics.get("partial_errors", 0),
                    },
                    checkpoint=version,
                )
            )
        return self._checkpoint_batch(
            documents=documents,
            envelopes=envelopes,
            versions=versions,
            checkpoint=checkpoint,
            profile_digest=profile_digest,
            diagnostics={
                **fetch_diagnostics,
                "truncated": truncated,
                "invalid_records": invalid_records,
                "privacy_redactions": privacy_redactions,
                "privacy_detected_types": sorted(privacy_types),
                "entity_counts": {
                    kind: sum(1 for item in prepared if item["kind"] == kind)
                    for kind in _ENTITY_KINDS
                    if any(item["kind"] == kind for item in prepared)
                },
            },
            governance=governance,
            snapshot_authoritative=bool(operation.get("snapshot_authoritative", False)),
            allow_empty_snapshot=bool(operation.get("allow_empty_snapshot", False)),
        )

    def _build_batch(
        self, checkpoint: ConnectorCheckpoint | None = None
    ) -> GraphQLHierarchyBatch:
        profile = self._resolve_profile()
        operation = profile["operations"][self.operation]
        roots, diagnostics = self._fetch_roots(profile, operation)
        if isinstance(operation.get("mappings"), dict):
            return self._hierarchy_batch(
                profile=profile,
                operation=operation,
                roots=roots,
                fetch_diagnostics=diagnostics,
                checkpoint=checkpoint,
            )
        return self._document_batch(
            profile=profile,
            operation=operation,
            roots=roots,
            fetch_diagnostics=diagnostics,
            checkpoint=checkpoint,
        )

    def _documents(self) -> list[SourceDocument]:
        batch = self._build_batch()
        self.last_envelopes = list(batch.envelopes)
        self.last_checkpoint = batch.checkpoint
        return list(batch.documents)

    def health_check(self) -> bool:
        try:
            self._resolve_profile()
            return True
        except Exception:
            return False

    def load(self) -> Iterator[SourceDocument]:
        if self.dry_run:
            self.plan()
            return
        yield from self._documents()

    def load_envelopes(self) -> Iterator[ChangeEnvelope]:
        """Yield governed mapped entities for the authoritative write boundary."""
        if self.dry_run:
            self.plan()
            return
        batch = self._build_batch()
        self.last_envelopes = list(batch.envelopes)
        self.last_checkpoint = batch.checkpoint
        yield from batch.envelopes

    def _plan_report(self, batch: GraphQLHierarchyBatch) -> dict[str, Any]:
        self.last_envelopes = []
        self.last_checkpoint = batch.checkpoint
        diagnostics = dict(batch.diagnostics)
        report = {
            "status": "planned",
            "dry_run": True,
            "source_alias": self.source_alias,
            "operation": self.operation,
            "profile_digest": batch.checkpoint.state.get("profile_digest"),
            "checkpoint_digest": batch.checkpoint.watermark,
            "counts": {
                "entities": diagnostics.get("entities", 0),
                "documents": diagnostics.get("documents", 0),
                "changed_entities": diagnostics.get("changed_entities", 0),
                "changed_documents": diagnostics.get("changed_documents", 0),
                "pages": diagnostics.get("pages", 0),
                "fallbacks": diagnostics.get("fallbacks", 0),
                "partial_errors": diagnostics.get("partial_errors", 0),
                "truncated": diagnostics.get("truncated", 0),
                "invalid_records": diagnostics.get("invalid_records", 0),
                "tombstones": diagnostics.get("tombstones", 0),
            },
            "entity_counts": diagnostics.get("entity_counts", {}),
            "privacy": {
                "redactions": diagnostics.get("privacy_redactions", 0),
                "detected_types": diagnostics.get("privacy_detected_types", []),
            },
        }
        self.last_plan = report
        return report

    def plan(self, checkpoint: ConnectorCheckpoint | None = None) -> dict[str, Any]:
        """Fetch and map without returning persistable content or identities."""
        return self._plan_report(self._build_batch(checkpoint))

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        batch = self._build_batch(checkpoint)
        self.last_checkpoint = batch.checkpoint
        if self.dry_run:
            self._plan_report(batch)
            self.last_envelopes = []
            documents: list[SourceDocument] = []
        else:
            self.last_envelopes = list(batch.envelopes)
            documents = list(batch.documents)
        return CheckpointedBatch(
            documents=documents,
            checkpoint=batch.checkpoint,
        )

    def poll_envelopes(
        self, checkpoint: ConnectorCheckpoint | None = None
    ) -> GraphQLHierarchyBatch:
        """Return the same delta checkpoint with governed entity envelopes."""
        batch = self._build_batch(checkpoint)
        self.last_checkpoint = batch.checkpoint
        if self.dry_run:
            self._plan_report(batch)
            self.last_envelopes = []
            return GraphQLHierarchyBatch(
                checkpoint=batch.checkpoint, diagnostics=batch.diagnostics
            )
        self.last_envelopes = list(batch.envelopes)
        return batch
