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
    Node,
    OperationDefinitionNode,
    OperationType,
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


def _dig_many(value: Any, path: str) -> list[Any]:
    """Resolve a dotted path while treating lists as bounded fan-out points."""
    current = [value]
    for part in (segment for segment in str(path or "").split(".") if segment):
        resolved: list[Any] = []
        for item in current:
            values = item if isinstance(item, list) else [item]
            for candidate in values:
                if isinstance(candidate, Mapping) and part in candidate:
                    resolved.append(candidate[part])
        current = resolved
    flattened: list[Any] = []
    for item in current:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


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


def _classification(value: Any) -> DataClassification:
    try:
        return DataClassification(str(value or DataClassification.INTERNAL.value))
    except ValueError:
        raise GraphQLDocumentError(
            "GraphQL governance classification is invalid"
        ) from None


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


def _errors_are_allowlisted(
    errors: Any, *, codes: tuple[str, ...], paths: tuple[str, ...]
) -> bool:
    if not isinstance(errors, list) or not errors or not codes or not paths:
        return False
    for error in errors:
        signature = _error_signature(error)
        if signature is None:
            return False
        code, path = signature
        if code not in codes:
            return False
        if not any(
            path == allowed or path.startswith(f"{allowed}.") for allowed in paths
        ):
            return False
    return True


def _validate_query_document(value: Any, *, allow_introspection: bool = False) -> str:
    """Accept exactly one bounded query operation without echoing its text.

    The AST is the authority.  Keyword regexes are not sufficient here: operation
    names, comments, string literals, fragments, and multi-operation documents can
    all make a lexical classifier disagree with what a GraphQL server executes.
    """
    query = str(value or "").strip()
    if len(query.encode("utf-8")) > 200_000:
        raise GraphQLDocumentError("GraphQL query exceeds the configured bound")
    if not query:
        raise GraphQLDocumentError("GraphQL operation must be a read query")
    try:
        document = parse(
            query,
            no_location=True,
            max_tokens=_MAX_GRAPHQL_TOKENS,
            allow_legacy_fragment_variables=False,
        )
    except (GraphQLError, RecursionError, TypeError, ValueError):
        raise GraphQLDocumentError("GraphQL operation is not a valid document") from None

    operations = [
        definition
        for definition in document.definitions
        if isinstance(definition, OperationDefinitionNode)
    ]
    if len(operations) != 1 or operations[0].operation is not OperationType.QUERY:
        raise GraphQLDocumentError("GraphQL operation must be a read query")
    if any(
        not isinstance(
            definition, (OperationDefinitionNode, FragmentDefinitionNode)
        )
        for definition in document.definitions
    ):
        raise GraphQLDocumentError("GraphQL operation contains unsupported definitions")

    selections = [
        selection
        for definition in document.definitions
        for selection in (definition.selection_set.selections if definition.selection_set else ())
    ]
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
        selection_set = getattr(selection, "selection_set", None)
        if selection_set is not None:
            selections.extend(selection_set.selections)
    return query


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
        if (
            isinstance(node, ArgumentNode)
            and node.name.value in {"first", "limit"}
            and isinstance(node.value, VariableNode)
            and node.value.name.value == variable
        ):
            return True
        for key in node.keys:
            child = getattr(node, key, None)
            if isinstance(child, tuple):
                stack.extend(item for item in child if isinstance(item, Node))
            elif isinstance(child, Node):
                stack.append(child)
    return False


def _bounded_transport_payload(response: Any, limit: int) -> tuple[dict[str, Any], int]:
    """Decode a transport response only after enforcing its raw byte bound.

    Custom transports are production-capable injection points, not test-only
    shortcuts.  Requiring bytes (or a bounded byte iterator) prevents ``.json()``
    from allocating an arbitrarily large object before the connector can apply its
    configured response limit.
    """

    raw: bytes
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        raw = content
    elif isinstance(content, bytearray):
        raw = bytes(content)
    else:
        iterator = getattr(response, "iter_bytes", None)
        if not callable(iterator):
            raise GraphQLDocumentError(
                "GraphQL transport must expose a bounded byte response"
            )
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
            raise GraphQLDocumentError(
                "GraphQL transport byte stream is invalid"
            ) from None
        raw = b"".join(chunks)
    if len(raw) > limit:
        raise GraphQLDocumentError("GraphQL response exceeds the configured bound")
    try:
        payload = json.loads(raw, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, RecursionError, UnicodeDecodeError):
        raise GraphQLDocumentError("GraphQL response is not valid JSON") from None
    if not isinstance(payload, dict):
        raise GraphQLDocumentError("GraphQL response is not an object")
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
        max_documents = int(config.get("max_documents") or 100)
        max_sections = int(config.get("max_sections") or 500)
        max_content_chars = int(config.get("max_content_chars") or 2_000_000)
        max_response_bytes = int(config.get("max_response_bytes") or 10_000_000)
        max_total_response_bytes = int(
            config.get("max_total_response_bytes") or 25_000_000
        )
        max_entities = int(config.get("max_entities") or 2_000)
        max_pages = int(config.get("max_pages") or 25)
        page_size = int(config.get("page_size") or 100)
        max_hierarchy_depth = int(config.get("max_hierarchy_depth") or 12)
        max_fallbacks = int(config.get("max_fallbacks") or 2)
        timeout_seconds = float(config.get("timeout_seconds") or 30.0)
        dry_run = bool(config.get("dry_run", False))
        profile = config.get("profile")
        profile_resolver = config.get("profile_resolver")
        transport = config.get("transport")
        privacy_guard = config.get("privacy_guard")

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
        self.max_documents = max(1, min(int(max_documents), 10_000))
        self.max_sections = max(1, min(int(max_sections), 10_000))
        self.max_content_chars = max(1_024, min(int(max_content_chars), 20_000_000))
        self.max_response_bytes = max(1_024, min(int(max_response_bytes), 50_000_000))
        self.max_total_response_bytes = max(
            self.max_response_bytes,
            min(int(max_total_response_bytes), 100_000_000),
        )
        self.max_entities = max(1, min(int(max_entities), 10_000))
        self.max_pages = max(1, min(int(max_pages), 100))
        self.page_size = max(1, min(int(page_size), 1_000))
        self.max_hierarchy_depth = max(1, min(int(max_hierarchy_depth), 32))
        self.max_fallbacks = max(0, min(int(max_fallbacks), 3))
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 120.0))
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
        if self._inline_profile is not None:
            profile = dict(self._inline_profile)
        else:
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
                raise GraphQLDocumentError(
                    "GraphQL profile is not valid JSON"
                ) from None
            if not isinstance(parsed, dict):
                raise GraphQLDocumentError("GraphQL profile must be a JSON object")
            profile = parsed

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
        headers = profile.get("headers") or {}
        operations = profile.get("operations") or {}
        if not isinstance(headers, dict) or not isinstance(operations, dict):
            raise GraphQLDocumentError("GraphQL profile has an invalid shape")
        if len(headers) > 32:
            raise GraphQLDocumentError("GraphQL profile headers are invalid")
        normalized_names: set[str] = set()
        for name, value in headers.items():
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
        op = operations.get(self.operation)
        if not isinstance(op, dict):
            raise GraphQLDocumentError("Requested GraphQL operation is not configured")
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
        identity_key = str(profile.get("identity_hmac_key") or "")
        if len(identity_key) < 32:
            raise GraphQLDocumentError(
                "GraphQL profile requires a 32-character identity HMAC key"
            )
        limits = profile.get("limits") or {}
        if not isinstance(limits, dict):
            raise GraphQLDocumentError("GraphQL profile limits are invalid")

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

        mappings = op.get("mappings")
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
        if mappings is not None:
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
                if not isinstance(mapping, dict):
                    raise GraphQLDocumentError("GraphQL entity mapping is invalid")
                id_path = str(mapping.get("id_path") or "id")
                if not _valid_field_path(id_path):
                    raise GraphQLDocumentError(
                        "GraphQL entity identity mapping is invalid"
                    )
                records_path = str(mapping.get("records_path") or "")
                if records_path and not _valid_field_path(records_path):
                    raise GraphQLDocumentError(
                        "GraphQL entity records mapping is invalid"
                    )
                if any(
                    not _valid_field_path(value)
                    for key, value in mapping.items()
                    if key.endswith("_path") and value not in (None, "")
                ):
                    raise GraphQLDocumentError(
                        "GraphQL entity field path is invalid"
                    )
                allowlist = mapping.get("property_allowlist")
                if (
                    not isinstance(allowlist, list)
                    or len(allowlist) > 256
                ):
                    raise GraphQLDocumentError(
                        "GraphQL entity property allowlist is required"
                    )
                if any(
                    not _valid_field_path(field)
                    for field in allowlist
                ):
                    raise GraphQLDocumentError(
                        "GraphQL entity property allowlist is invalid"
                    )
                if kind == "hierarchy":
                    children_path = str(mapping.get("children_path") or "")
                    if children_path and not _valid_field_path(children_path):
                        raise GraphQLDocumentError(
                            "GraphQL hierarchy children mapping is invalid"
                        )
            if not recognized:
                raise GraphQLDocumentError("GraphQL profile has no entity mappings")

        governance = profile.get("governance") or {}
        if not isinstance(governance, dict):
            raise GraphQLDocumentError("GraphQL governance mapping is invalid")
        _classification(governance.get("classification"))
        if "legal_hold" in governance and not isinstance(
            governance["legal_hold"], bool
        ):
            raise GraphQLDocumentError("GraphQL governance legal hold is invalid")
        retention = str(governance.get("retention") or "").strip()
        if mappings is not None and not retention:
            raise GraphQLDocumentError(
                "GraphQL hierarchy governance requires a retention policy"
            )
        if retention and not _RETENTION_RE.fullmatch(retention):
            raise GraphQLDocumentError("GraphQL governance retention is invalid")
        tenant = str(governance.get("tenant") or "").strip()
        if tenant:
            _safe_alias(tenant, label="tenant")
        for key in ("schema_version", "ontology_mapping_version"):
            if key in governance and not _VERSION_LABEL_RE.fullmatch(
                str(governance[key] or "")
            ):
                raise GraphQLDocumentError("GraphQL governance version is invalid")
        _access_from_config(governance.get("access", profile.get("access")))
        discovery = profile.get("discovery")
        if discovery is not None and not isinstance(discovery, Mapping):
            raise GraphQLDocumentError("GraphQL discovery policy is invalid")
        if isinstance(discovery, Mapping):
            for key in ("enabled", "allow_introspection"):
                if key in discovery and not isinstance(discovery[key], bool):
                    raise GraphQLDocumentError("GraphQL discovery policy is invalid")
        return profile

    def _transport_security(self, profile: dict[str, Any]) -> Any:
        """Resolve a runtime-only TLS profile without exposing it downstream."""
        if self._resolved_tls is not None:
            return self._resolved_tls
        from agent_utilities.core.transport_security import (
            TransportSecurityError,
            resolve_configured_tls_profile,
        )

        configured = profile.get("transport_security", profile.get("tls"))
        profile_name: str | None = None
        profile_ref: str | None = None
        inline: Mapping[str, Any] | None = None
        if isinstance(configured, str):
            profile_name = configured
        elif isinstance(configured, Mapping):
            profile_name = str(
                configured.get("profile_name") or configured.get("profile") or ""
            ).strip() or None
            profile_ref = str(configured.get("profile_ref") or "").strip() or None
            settings = configured.get("settings")
            if isinstance(settings, Mapping):
                inline = settings
            elif not profile_name and not profile_ref:
                inline = configured
        elif configured is not None:
            raise GraphQLDocumentError(
                "GraphQL transport security profile is invalid"
            )
        profile_name = (
            str(profile.get("tls_profile") or "").strip() or profile_name
        )
        profile_ref = (
            str(profile.get("tls_profile_ref") or "").strip() or profile_ref
        )
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

    def _governance(
        self, profile: dict[str, Any]
    ) -> tuple[ExternalAccess, DataClassification, str | None, bool, str, str, str]:
        value = profile.get("governance") or {}
        access = _access_from_config(value.get("access", profile.get("access")))
        if value.get("access") is None and profile.get("access") is None:
            access = self.external_access
        classification = _classification(value.get("classification"))
        if classification == DataClassification.PUBLIC and not access.is_public:
            raise GraphQLDocumentError(
                "GraphQL public classification requires public source access"
            )
        if classification != DataClassification.PUBLIC and access.is_public:
            raise GraphQLDocumentError(
                "GraphQL non-public classification cannot use public source access"
            )
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
            variables = dict(self.variables)
            if isinstance(pagination, dict):
                variables[str(pagination["page_size_variable"])] = page_size
                variables[str(pagination["cursor_variable"])] = cursor
            elif isinstance(read_bound, dict):
                variables[str(read_bound["variable"])] = min(
                    page_size, int(read_bound["maximum"])
                )
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
            has_more = _dig(data, str(pagination["has_more_path"]), False)
            if not isinstance(has_more, bool):
                raise GraphQLDocumentError(
                    "GraphQL pagination continuation flag is not boolean"
                )
            if not has_more:
                break
            next_cursor = _dig(data, str(pagination["next_cursor_path"]))
            if not isinstance(next_cursor, str):
                raise GraphQLDocumentError(
                    "GraphQL pagination returned an invalid continuation"
                )
            next_cursor_text = next_cursor
            if (
                not next_cursor_text
                or next_cursor_text != next_cursor_text.strip()
                or len(next_cursor_text.encode("utf-8")) > 4_096
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in next_cursor_text
                )
                or next_cursor_text in seen_cursors
            ):
                raise GraphQLDocumentError(
                    "GraphQL pagination returned an invalid continuation"
                )
            if page_index + 1 >= max_pages:
                raise GraphQLDocumentError(
                    "GraphQL pagination exceeds the configured page bound"
                )
            seen_cursors.add(next_cursor_text)
            cursor = next_cursor_text

        return roots, {
            "pages": len(roots),
            "response_bytes": total_bytes,
            "fallbacks": fallbacks,
            "partial_errors": partial_errors,
        }

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
        raw_id = _dig(record, str(operation.get("id_path") or "id"))
        if raw_id in (None, ""):
            return None
        raw_title = _dig(record, str(operation.get("title_path") or "title"), raw_id)
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

        clean_title, title_report = self._privacy.sanitize_text(str(raw_title))
        body: list[str] = [f"# {clean_title}"]
        reports = [title_report]
        content_path = str(operation.get("content_path") or "")
        if content_path:
            clean_content, content_report = self._privacy.sanitize_text(
                str(_dig(record, content_path, "") or "")
            )
            reports.append(content_report)
            if clean_content.strip():
                body.extend(["", clean_content.strip()])
        if frontmatter not in (None, "", {}, []):
            clean_frontmatter, report = self._privacy.sanitize(frontmatter)
            reports.append(report)
            if isinstance(clean_frontmatter, str):
                body.extend(["", clean_frontmatter])
            else:
                body.extend(
                    [
                        "",
                        json.dumps(
                            clean_frontmatter,
                            sort_keys=True,
                            ensure_ascii=False,
                            default=str,
                        ),
                    ]
                )

        title_field = str(operation.get("section_title_field") or "title")
        level_field = str(operation.get("section_level_field") or "level")
        content_field = str(operation.get("section_content_field") or "content")
        for section in sections[: self.max_sections]:
            if not isinstance(section, dict):
                continue
            clean_section, report = self._privacy.sanitize(section)
            reports.append(report)
            if not isinstance(clean_section, dict):
                continue
            section_title = str(clean_section.get(title_field) or "Section")
            try:
                level = max(2, min(int(clean_section.get(level_field) or 2) + 1, 6))
            except (TypeError, ValueError):
                level = 2
            content = str(clean_section.get(content_field) or "").strip()
            body.extend(["", f"{'#' * level} {section_title}"])
            if content:
                body.extend(["", content])

        text = "\n".join(body)[: self.max_content_chars].strip()
        if not text:
            return None
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
        access = governance[0] if governance is not None else self.external_access
        classification = (
            governance[1] if governance is not None else DataClassification.INTERNAL
        )
        retention = governance[2] if governance is not None else None
        legal_hold = governance[3] if governance is not None else False
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
            if kind != "hierarchy" or not children_path:
                for seed in seeds:
                    if not isinstance(seed, dict):
                        continue
                    if len(records) >= limit:
                        truncated += 1
                        continue
                    parent_path = str(mapping.get("parent_id_path") or "")
                    parent = _dig(seed, parent_path) if parent_path else None
                    records.append((seed, parent, 0))
                continue

            stack: list[tuple[dict[str, Any], Any | None, int]] = [
                (seed, None, 0) for seed in reversed(seeds) if isinstance(seed, dict)
            ]
            while stack:
                record, parent, depth = stack.pop()
                if len(records) >= limit:
                    truncated += 1
                    continue
                records.append((record, parent, depth))
                children = [
                    child
                    for child in _dig_many(record, children_path)
                    if isinstance(child, dict)
                ]
                if not children:
                    continue
                if depth + 1 >= max_depth:
                    truncated += len(children)
                    continue
                raw_id = _dig(record, id_path)
                stack.extend((child, raw_id, depth + 1) for child in reversed(children))
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
        prior_state = checkpoint.state if checkpoint else {}
        if prior_state and prior_state.get("checkpoint_format") != _CHECKPOINT_FORMAT:
            raise GraphQLDocumentError("GraphQL checkpoint format is invalid")
        if prior_state and (
            prior_state.get("source_alias") != self.source_alias
            or prior_state.get("operation") != self.operation
        ):
            raise GraphQLDocumentError("GraphQL checkpoint scope is invalid")
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
        try:
            prior_sequence = int(prior_state.get("snapshot_sequence") or 0)
        except (TypeError, ValueError):
            raise GraphQLDocumentError("GraphQL checkpoint state is invalid") from None
        if prior_sequence < 0:
            raise GraphQLDocumentError("GraphQL checkpoint state is invalid")

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
        (
            access,
            classification,
            retention,
            legal_hold,
            tenant,
            schema_version,
            mapping_version,
        ) = governance
        governance_state = {
            "access": access.model_dump(mode="json"),
            "classification": classification.value,
            "retention": retention,
            "legal_hold": legal_hold,
            "tenant": tenant,
            "schema_version": schema_version,
            "ontology_mapping_version": mapping_version,
        }
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
        changed_documents = [
            document
            for document in documents
            if str(document.metadata.get("governed_entity_id") or "") in changed
        ]
        changed_envelopes = [
            envelope for envelope in envelopes if envelope.source_object_id in changed
        ]
        tombstones: list[ChangeEnvelope] = []
        missing = sorted(set(baseline).difference(versions))
        previous_governance = prior_state.get("governance")
        if snapshot_authoritative and missing:
            if not isinstance(previous_governance, Mapping):
                raise GraphQLDocumentError(
                    "GraphQL checkpoint governance state is invalid"
                )
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
            if previous_tenant:
                _safe_alias(previous_tenant, label="tenant")
            if previous_retention and not _RETENTION_RE.fullmatch(previous_retention):
                raise GraphQLDocumentError(
                    "GraphQL checkpoint governance state is invalid"
                )
            next_snapshot_digest = _digest(
                self.source_alias,
                self.operation,
                prior_sequence + 1,
                sorted(versions.items()),
            )
            for node_id in missing:
                delete_version = _digest(
                    "graphql-snapshot-delete",
                    baseline[node_id],
                    next_snapshot_digest,
                )
                tombstones.append(
                    ChangeEnvelope(
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
                            "profile_digest": str(
                                prior_state.get("profile_digest") or ""
                            ),
                            "privacy_gate": True,
                            "identity_scheme": "hmac-sha256",
                            "snapshot_reconciliation": True,
                            "authoritative_empty_approved": bool(
                                not versions and allow_empty_snapshot
                            ),
                        },
                        checkpoint=delete_version,
                    )
                )

        checkpoint_changed = not prior_state or any(
            (
                baseline != versions,
                prior_state.get("profile_digest") != profile_digest,
                prior_state.get("governance") != governance_state,
                prior_state.get("snapshot_authoritative")
                != snapshot_authoritative,
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
        next_checkpoint = ConnectorCheckpoint(
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
        diagnostics = {
            **diagnostics,
            "entities": len(versions),
            "documents": len(documents),
            "changed_entities": len(changed),
            "changed_documents": len(changed_documents),
            "tombstones": len(tombstones),
        }
        return GraphQLHierarchyBatch(
            documents=tuple(changed_documents),
            envelopes=tuple((*changed_envelopes, *tombstones)),
            checkpoint=next_checkpoint,
            diagnostics=diagnostics,
        )

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
            access, classification, retention, legal_hold, tenant, schema, mapping = (
                governance
            )
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
