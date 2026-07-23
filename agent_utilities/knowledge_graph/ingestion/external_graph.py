"""Governed native ingestion from registered external graph connections.

Neo4j, Apache AGE, LadybugDB, and other ``GraphBackend`` implementations enter
through the existing named connection registry. This module adds the missing
read-only import path: a secret-backed mapping profile turns bounded query rows
into canonical ``ChangeEnvelope`` objects, applies a zero-PII persistence gate,
and writes them through the same lineage/ACL/idempotency path as other sources.

Only connection and source aliases are persisted. Endpoint URLs, credentials,
query text, variables, local paths, raw external identifiers, and resolved
profile content remain transient.

CONCEPT:AU-KG.ingest.external-graph-federation
CONCEPT:AU-KG.ingest.change-envelope
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
    ingest_envelope,
    read_change_cursor,
)
from agent_utilities.models.company_brain import DataClassification
from agent_utilities.protocols.source_connectors.base import ExternalAccess
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

__all__ = [
    "ExternalGraphIngestionError",
    "ExternalGraphIngestionRequest",
    "ingest_registered_graph",
]

_ALIAS_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_SECRET_REF_RE = re.compile(r"^(?:vault|env|secret)://[A-Za-z0-9_./#-]+$")
_PERSON_ENTITY = re.compile(
    r"(?:^|_)(?:person|user|employee|contact|customer|individual)(?:$|_)",
    re.IGNORECASE,
)
_RETENTION_RE = re.compile(r"^(?:P\d+[DWMY]|[a-z][a-z0-9-]{1,63})$")


class ExternalGraphIngestionError(RuntimeError):
    """A source-safe import error that never includes remote content or config."""


@dataclass(frozen=True)
class ExternalGraphIngestionRequest:
    """Non-secret request options for a bounded registered-graph import."""

    connection: str
    source_alias: str
    profile_ref: str
    variables: dict[str, Any]
    runtime_policy_digest: str = ""
    max_records: int = 1_000
    page_size: int = 500
    max_pages: int = 100
    max_row_bytes: int = 1_048_576
    max_total_bytes: int = 16_777_216
    max_nesting_depth: int = 16
    max_collection_items: int = 10_000
    sync_mode: Literal["auto", "cdc", "snapshot"] = "auto"
    reconcile_deletions: bool = True
    allow_empty_snapshot: bool = False
    classification: DataClassification = DataClassification.CONFIDENTIAL
    retention: str = "P30D"
    legal_hold: bool = False
    tenant: str = ""
    dry_run: bool = False


@dataclass
class _PayloadBudget:
    max_row_bytes: int
    max_total_bytes: int
    max_nesting_depth: int
    max_collection_items: int
    consumed_bytes: int = 0

    def accept(self, row: Mapping[str, Any], *, label: str) -> None:
        try:
            stack: list[tuple[Any, int]] = [(row, 1)]
            visited_containers: set[int] = set()
            while stack:
                value, depth = stack.pop()
                if isinstance(value, Mapping):
                    container_id = id(value)
                    if container_id in visited_containers:
                        raise ExternalGraphIngestionError(
                            f"External graph {label} contains a repeated container"
                        )
                    visited_containers.add(container_id)
                    if depth > self.max_nesting_depth:
                        raise ExternalGraphIngestionError(
                            f"External graph {label} exceeded the nesting-depth bound"
                        )
                    if len(value) > self.max_collection_items:
                        raise ExternalGraphIngestionError(
                            f"External graph {label} exceeded the collection-size bound"
                        )
                    stack.extend((item, depth + 1) for item in value.values())
                elif isinstance(value, list | tuple):
                    container_id = id(value)
                    if container_id in visited_containers:
                        raise ExternalGraphIngestionError(
                            f"External graph {label} contains a repeated container"
                        )
                    visited_containers.add(container_id)
                    if depth > self.max_nesting_depth:
                        raise ExternalGraphIngestionError(
                            f"External graph {label} exceeded the nesting-depth bound"
                        )
                    if len(value) > self.max_collection_items:
                        raise ExternalGraphIngestionError(
                            f"External graph {label} exceeded the collection-size bound"
                        )
                    stack.extend((item, depth + 1) for item in value)
        except ExternalGraphIngestionError:
            raise
        except Exception:
            raise ExternalGraphIngestionError(
                f"External graph {label} is not bounded JSON"
            ) from None
        try:
            row_bytes = len(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError, RecursionError):
            raise ExternalGraphIngestionError(
                f"External graph {label} is not bounded JSON"
            ) from None
        if row_bytes > self.max_row_bytes:
            raise ExternalGraphIngestionError(
                f"External graph {label} exceeded the per-row byte bound"
            )
        self.consumed_bytes += row_bytes
        if self.consumed_bytes > self.max_total_bytes:
            raise ExternalGraphIngestionError(
                "External graph source exceeded the cumulative byte bound"
            )


def _alias(value: str, *, label: str, allow_empty: bool = False) -> str:
    clean = str(value or "").strip().lower()
    if allow_empty and not clean:
        return ""
    if not _ALIAS_RE.fullmatch(clean):
        raise ValueError(f"{label} must be a neutral lowercase alias")
    return clean


def _digest(*parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _private_digest(key: str, *parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hmac.new(
        key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _dig(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in (segment for segment in path.split(".") if segment):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _rows(value: Any, *, max_records: int) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return _rows(json.loads(value), max_records=max_records)
        except (TypeError, ValueError):
            return []
    if isinstance(value, dict):
        for key in ("rows", "result", "data"):
            if key in value:
                return _rows(value[key], max_records=max_records)
        return [value]
    if isinstance(value, Iterable):
        rows: list[dict[str, Any]] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            if len(rows) >= max_records:
                raise ExternalGraphIngestionError(
                    "External graph source exceeded the requested row bound"
                )
            rows.append(row)
        return rows
    return []


def _scan_cypher(query: str, *, label: str) -> list[tuple[str, str]]:
    """Tokenize the executable surface of one strict read statement.

    This is intentionally smaller than a general Cypher parser.  External import
    profiles are allowed a bounded read subset only, so comments, quoted values,
    and backtick identifiers are consumed without classifying their contents and
    every executable bare token must be ASCII. Mutation authority is enforced by
    the selected database's read-only transaction, not by this scanner.
    """

    tokens: list[tuple[str, str]] = []
    index = 0
    size = len(query)
    while index < size:
        char = query[index]
        if char.isspace():
            index += 1
            continue
        if query.startswith("//", index):
            newline = query.find("\n", index + 2)
            index = size if newline < 0 else newline + 1
            continue
        if query.startswith("/*", index):
            end = query.find("*/", index + 2)
            if end < 0:
                raise ExternalGraphIngestionError(f"{label} is not valid Cypher")
            index = end + 2
            continue
        if char in {"'", '"', "`"}:
            delimiter = char
            index += 1
            closed = False
            while index < size:
                current = query[index]
                if current == "\\" and delimiter != "`":
                    index += 2
                    continue
                if current == delimiter:
                    if index + 1 < size and query[index + 1] == delimiter:
                        index += 2
                        continue
                    index += 1
                    closed = True
                    break
                index += 1
            if not closed:
                raise ExternalGraphIngestionError(f"{label} is not valid Cypher")
            continue
        if char == ";":
            raise ExternalGraphIngestionError(
                f"{label} must contain exactly one read statement"
            )
        if char == "$":
            end = index + 1
            while end < size and (
                query[end].isascii() and (query[end].isalnum() or query[end] == "_")
            ):
                end += 1
            if end == index + 1:
                raise ExternalGraphIngestionError(f"{label} is not valid Cypher")
            tokens.append(("parameter", query[index + 1 : end].casefold()))
            index = end
            continue
        if char.isascii() and (char.isalpha() or char == "_"):
            end = index + 1
            while end < size and (
                query[end].isascii() and (query[end].isalnum() or query[end] == "_")
            ):
                end += 1
            tokens.append(("word", query[index:end].upper()))
            index = end
            continue
        if not char.isascii():
            raise ExternalGraphIngestionError(
                f"{label} bare identifiers must be ASCII or backtick-quoted"
            )
        tokens.append(("symbol", char))
        index += 1
    return tokens


def _validate_read_query(query: str, *, label: str) -> None:
    tokens = _scan_cypher(str(query or ""), label=label)
    words = [value for kind, value in tokens if kind == "word"]
    if not words or not (
        words[0] in {"MATCH", "WITH", "UNWIND", "RETURN"}
        or words[:2] == ["OPTIONAL", "MATCH"]
    ):
        raise ExternalGraphIngestionError(f"{label} is not a supported read query")
    if tokens[-2:] != [("word", "LIMIT"), ("parameter", "limit")]:
        raise ExternalGraphIngestionError(
            f"{label} must end with the exact bound LIMIT $limit"
        )
    if [("word", "SKIP"), ("parameter", "offset")] not in [
        tokens[index : index + 2] for index in range(len(tokens) - 1)
    ]:
        raise ExternalGraphIngestionError(
            f"{label} must use the exact page cursor SKIP $offset"
        )
    if [("word", "ORDER"), ("word", "BY")] not in [
        tokens[index : index + 2] for index in range(len(tokens) - 1)
    ]:
        raise ExternalGraphIngestionError(
            f"{label} must define deterministic ORDER BY paging"
        )


def _resolve_profile(
    profile_ref: str,
    *,
    profile: dict[str, Any] | None,
    resolver: Callable[[str], str | None] | None,
) -> dict[str, Any]:
    if profile is not None:
        return dict(profile)
    if not profile_ref:
        raise ExternalGraphIngestionError(
            "External graph ingestion requires a secret-backed profile_ref"
        )
    if not _SECRET_REF_RE.fullmatch(profile_ref):
        raise ExternalGraphIngestionError(
            "External graph profile_ref must use a supported secret-reference scheme"
        )
    if resolver is None:
        from agent_utilities.security.secrets_client import create_secrets_client

        resolver = create_secrets_client().resolve_ref
    try:
        raw = resolver(profile_ref)
    except Exception as exc:
        raise ExternalGraphIngestionError(
            f"External graph profile resolution failed ({type(exc).__name__})"
        ) from None
    if not raw:
        raise ExternalGraphIngestionError(
            "External graph profile could not be resolved"
        )
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        raise ExternalGraphIngestionError(
            "External graph profile is not valid JSON"
        ) from None
    if not isinstance(value, dict):
        raise ExternalGraphIngestionError(
            "External graph profile must be a JSON object"
        )
    return value


def _resolve_identity_key(
    resolved_profile: Mapping[str, Any],
    *,
    connection: str,
    resolver: Callable[[str], str | None] | None,
) -> str:
    """Resolve the approved identity-key ref without hashing or persisting it."""

    from .external_graph_schema import canonical_identity_key_ref

    key_ref = str(resolved_profile.get("identity_hmac_key_ref") or "")
    if key_ref != canonical_identity_key_ref(connection):
        raise ExternalGraphIngestionError(
            "External graph profile has an invalid identity key reference"
        )
    if "identity_hmac_key" in resolved_profile:
        raise ExternalGraphIngestionError(
            "External graph runtime profiles cannot embed identity key material"
        )
    if resolver is None:
        from agent_utilities.security.secrets_client import create_secrets_client

        resolver = create_secrets_client().resolve_ref
    try:
        identity_key = str(resolver(key_ref) or "")
    except Exception as exc:
        raise ExternalGraphIngestionError(
            f"External graph identity key resolution failed ({type(exc).__name__})"
        ) from None
    if len(identity_key) < 32:
        raise ExternalGraphIngestionError(
            "External graph identity key could not be resolved"
        )
    return identity_key


def _read_external(
    engine: Any,
    query: str,
    params: dict[str, Any],
    *,
    max_records: int,
    budget: _PayloadBudget,
    label: str,
) -> list[dict[str, Any]]:
    backend = getattr(engine, "backend", None)
    target = backend if backend is not None else engine
    query_fn = getattr(target, "execute_read", None)
    bounded = False
    if not callable(query_fn) and getattr(target, "read_only", False) is True:
        query_fn = getattr(target, "query_cypher_bounded", None)
        bounded = callable(query_fn)
    if not callable(query_fn):
        raise ExternalGraphIngestionError(
            "Registered graph connection has no enforced read-only surface"
        )
    try:
        result = (
            query_fn(query, params, max_records=max_records)
            if bounded
            else query_fn(query, params)
        )
        rows = _rows(result, max_records=max_records)
        for row in rows:
            budget.accept(row, label=label)
        return rows
    except ExternalGraphIngestionError:
        raise
    except Exception as exc:
        raise ExternalGraphIngestionError(
            f"External graph read failed ({type(exc).__name__})"
        ) from None


def _read_external_pages(
    engine: Any,
    query: str,
    variables: Mapping[str, Any],
    *,
    page_size: int,
    max_pages: int,
    max_records: int,
    budget: _PayloadBudget,
    privacy: PersistencePrivacyGuard,
    label: str,
    required_snapshot_token: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Read a deterministic snapshot fully or fail before materialization."""

    backend = getattr(engine, "backend", None)
    target = backend if backend is not None else engine
    snapshot_reader = getattr(target, "read_snapshot_page", None)
    snapshot_reader = snapshot_reader if callable(snapshot_reader) else None
    if snapshot_reader is None:
        params = dict(variables)
        params.update({"offset": 0, "limit": max_records + 1})
        rows = _read_external(
            engine,
            query,
            params,
            max_records=max_records + 1,
            budget=budget,
            label=label,
        )
        if len(rows) > max_records:
            raise ExternalGraphIngestionError(
                "External graph snapshot exceeded the configured total bound"
            )
        return rows, None
    collected: list[dict[str, Any]] = []
    offset = 0
    snapshot_token = required_snapshot_token
    for _page in range(max_pages):
        remaining = max_records - len(collected)
        if remaining <= 0:
            raise ExternalGraphIngestionError(
                "External graph snapshot exceeded the configured total bound"
            )
        window = min(page_size, remaining)
        params = dict(variables)
        params.update({"offset": offset, "limit": window + 1})
        try:
            result = snapshot_reader(
                query=query,
                params=params,
                max_records=window + 1,
                snapshot_token=snapshot_token,
            )
        except Exception as exc:
            raise ExternalGraphIngestionError(
                f"External graph stable snapshot read failed ({type(exc).__name__})"
            ) from None
        if not isinstance(result, Mapping):
            raise ExternalGraphIngestionError(
                "External graph stable snapshot page has an invalid contract"
            )
        page_token = _safe_resume_token(
            result.get("snapshot_token"), privacy, label="snapshot token"
        )
        if snapshot_token is not None and page_token != snapshot_token:
            raise ExternalGraphIngestionError(
                "External graph stable snapshot token changed during paging"
            )
        snapshot_token = page_token
        rows = _rows(result.get("rows"), max_records=window + 1)
        for row in rows:
            budget.accept(row, label=label)
        has_more = len(rows) > window
        collected.extend(rows[:window])
        if not has_more:
            return collected, snapshot_token
        if len(collected) >= max_records:
            raise ExternalGraphIngestionError(
                "External graph snapshot exceeded the configured total bound"
            )
        offset += window
    raise ExternalGraphIngestionError(
        "External graph snapshot exceeded the configured page bound"
    )


def _change_reader(engine: Any) -> Callable[..., Any] | None:
    """Return the normalized native-CDC surface when a backend advertises it."""

    backend = getattr(engine, "backend", None)
    target = backend if backend is not None else engine
    reader = getattr(target, "read_change_page", None)
    return reader if callable(reader) else None


def _safe_resume_token(
    value: Any, privacy: PersistencePrivacyGuard, *, label: str
) -> str:
    if not isinstance(value, str):
        raise ExternalGraphIngestionError(f"External graph {label} is invalid")
    rendered = value
    if (
        not rendered
        or rendered != rendered.strip()
        or len(rendered.encode("utf-8")) > 4_096
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ExternalGraphIngestionError(f"External graph {label} is invalid")
    clean, report = privacy.sanitize_text(rendered)
    if report.changed or clean != rendered:
        raise ExternalGraphIngestionError(
            f"External graph {label} is not persistence-safe"
        )
    return rendered


def _read_change_pages(
    reader: Callable[..., Any],
    *,
    cursor: str | None,
    page_size: int,
    max_pages: int,
    max_records: int,
    privacy: PersistencePrivacyGuard,
    budget: _PayloadBudget,
) -> tuple[list[dict[str, Any]], str | None]:
    """Drain a normalized source-native CDC feed within hard page/row bounds.

    Backends opt in by exposing ``read_change_page(cursor=..., limit=...)`` and
    returning ``{"events": [...], "next_cursor": str, "has_more": bool}``.
    Events are node-shaped ``upsert`` records or ``delete`` ids. Unsupported or
    malformed feeds fail closed; a declared CDC surface is never silently
    downgraded to a snapshot after it has been selected.
    """

    collected: list[dict[str, Any]] = []
    current = (
        _safe_resume_token(cursor, privacy, label="CDC cursor")
        if cursor not in (None, "")
        else None
    )
    next_cursor = current
    for _page in range(max_pages):
        remaining = max_records - len(collected)
        if remaining <= 0:
            raise ExternalGraphIngestionError(
                "External graph CDC exceeded the configured total bound"
            )
        window = min(page_size, remaining)
        try:
            result = reader(cursor=current, limit=window)
        except Exception as exc:
            raise ExternalGraphIngestionError(
                f"External graph CDC read failed ({type(exc).__name__})"
            ) from None
        if not isinstance(result, Mapping):
            raise ExternalGraphIngestionError(
                "External graph CDC page has an invalid contract"
            )
        raw_events = result.get("events")
        if not isinstance(raw_events, list) or len(raw_events) > window:
            raise ExternalGraphIngestionError(
                "External graph CDC page exceeded its event bound"
            )
        events: list[dict[str, Any]] = []
        for event in raw_events:
            if not isinstance(event, dict):
                raise ExternalGraphIngestionError(
                    "External graph CDC event has an invalid contract"
                )
            budget.accept(event, label="CDC event")
            operation = str(event.get("operation") or "")
            entity = str(event.get("entity") or "node")
            if operation not in {"upsert", "delete"} or entity != "node":
                raise ExternalGraphIngestionError(
                    "External graph CDC event is not a supported node change"
                )
            if operation == "upsert" and not isinstance(event.get("record"), dict):
                raise ExternalGraphIngestionError(
                    "External graph CDC upsert has no record"
                )
            if operation == "delete" and event.get("id") in (None, ""):
                raise ExternalGraphIngestionError(
                    "External graph CDC delete has no identity"
                )
            events.append(event)
        collected.extend(events)
        has_more = result.get("has_more")
        if not isinstance(has_more, bool):
            raise ExternalGraphIngestionError(
                "External graph CDC page has no explicit continuation state"
            )
        raw_next = result.get("next_cursor")
        page_cursor: str | None = None
        if raw_next not in (None, ""):
            page_cursor = _safe_resume_token(raw_next, privacy, label="CDC cursor")
        if events and (not page_cursor or page_cursor == current):
            raise ExternalGraphIngestionError(
                "External graph CDC event cursor did not advance"
            )
        if page_cursor is not None:
            next_cursor = page_cursor
        if not has_more:
            return collected, next_cursor
        if not page_cursor or page_cursor == current:
            raise ExternalGraphIngestionError(
                "External graph CDC continuation cursor did not advance"
            )
        current = page_cursor
    raise ExternalGraphIngestionError(
        "External graph CDC exceeded the configured page bound"
    )


def _mapping(profile: dict[str, Any], key: str) -> dict[str, Any]:
    value = profile.get(key)
    if not isinstance(value, dict):
        raise ExternalGraphIngestionError(f"External graph profile has no {key}")
    return value


def _allowlist(mapping: dict[str, Any], *, label: str) -> tuple[str, ...]:
    value = mapping.get("property_allowlist")
    if not isinstance(value, list) or not value:
        raise ExternalGraphIngestionError(
            f"{label} requires a non-empty property_allowlist"
        )
    fields = tuple(str(item) for item in value if str(item).strip())
    if not fields:
        raise ExternalGraphIngestionError(
            f"{label} requires a non-empty property_allowlist"
        )
    return fields


def _safe_type(
    value: Any, type_map: dict[str, Any], *, fallback: str
) -> tuple[str, str]:
    external = str(value or fallback)
    mapped = str(type_map.get(external) or fallback)
    if not _TYPE_RE.fullmatch(mapped):
        mapped = fallback
    return mapped, external


def _access(profile: dict[str, Any]) -> ExternalAccess:
    raw = profile.get("access")
    access = (
        ExternalAccess.quarantined()
        if raw is None
        else ExternalAccess.model_validate(raw)
    )
    if access.user_emails:
        raise ExternalGraphIngestionError(
            "External graph profiles cannot persist user-email ACLs"
        )
    if not access.is_public and not (access.group_ids or access.markings):
        return ExternalAccess.quarantined()
    return access


def _classification(value: DataClassification | str) -> DataClassification:
    try:
        return DataClassification(str(value))
    except ValueError:
        raise ExternalGraphIngestionError(
            "Invalid external graph classification"
        ) from None


def ingest_registered_graph(
    authority_engine: Any,
    registry: Any,
    request: ExternalGraphIngestionRequest,
    *,
    profile: dict[str, Any] | None = None,
    profile_resolver: Callable[[str], str | None] | None = None,
    privacy_guard: PersistencePrivacyGuard | None = None,
) -> dict[str, Any]:
    """Read a bounded external graph slice and ingest it through envelopes.

    ``profile`` is an offline-test seam. Production entrypoints pass only
    ``profile_ref`` so remote endpoint/query configuration never appears in MCP
    arguments, manifests, checkpoints, or trace metadata.
    """

    connection = _alias(request.connection, label="connection")
    source_alias = _alias(request.source_alias, label="source_alias")
    tenant = _alias(request.tenant, label="tenant", allow_empty=True)
    if any(
        isinstance(value, bool)
        for value in (request.max_records, request.page_size, request.max_pages)
    ):
        raise ExternalGraphIngestionError("External graph page limits must be integers")
    try:
        max_records = max(1, min(int(request.max_records), 10_000))
        page_size = int(request.page_size)
        max_pages = int(request.max_pages)
    except (TypeError, ValueError, OverflowError):
        raise ExternalGraphIngestionError(
            "External graph page limits must be integers"
        ) from None
    if not 1 <= page_size <= 1_000:
        raise ExternalGraphIngestionError("page_size must be between 1 and 1000")
    if not 1 <= max_pages <= 1_000:
        raise ExternalGraphIngestionError("max_pages must be between 1 and 1000")
    structural_bounds = (
        request.max_row_bytes,
        request.max_total_bytes,
        request.max_nesting_depth,
        request.max_collection_items,
    )
    if any(isinstance(value, bool) for value in structural_bounds):
        raise ExternalGraphIngestionError(
            "External graph structural limits must be integers"
        )
    try:
        max_row_bytes = int(request.max_row_bytes)
        max_total_bytes = int(request.max_total_bytes)
        max_nesting_depth = int(request.max_nesting_depth)
        max_collection_items = int(request.max_collection_items)
    except (TypeError, ValueError, OverflowError):
        raise ExternalGraphIngestionError(
            "External graph structural limits must be integers"
        ) from None
    if not 256 <= max_row_bytes <= 8_388_608:
        raise ExternalGraphIngestionError(
            "max_row_bytes must be between 256 and 8388608"
        )
    if not max_row_bytes <= max_total_bytes <= 67_108_864:
        raise ExternalGraphIngestionError(
            "max_total_bytes must cover one row and not exceed 67108864"
        )
    if not 1 <= max_nesting_depth <= 64:
        raise ExternalGraphIngestionError("max_nesting_depth must be between 1 and 64")
    if not 1 <= max_collection_items <= 100_000:
        raise ExternalGraphIngestionError(
            "max_collection_items must be between 1 and 100000"
        )
    if not isinstance(request.reconcile_deletions, bool) or not isinstance(
        request.allow_empty_snapshot, bool
    ):
        raise ExternalGraphIngestionError(
            "External graph reconciliation policy must be boolean"
        )
    sync_mode = str(request.sync_mode or "")
    if sync_mode not in {"auto", "cdc", "snapshot"}:
        raise ExternalGraphIngestionError("sync_mode must be auto, cdc, or snapshot")
    classification = _classification(request.classification)
    retention = str(request.retention or "").strip()
    if not _RETENTION_RE.fullmatch(retention):
        raise ExternalGraphIngestionError(
            "retention must be an ISO duration or a neutral policy alias"
        )
    try:
        from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (
            precheck_source,
        )

        activation = precheck_source("external_graph")
    except Exception:
        raise ExternalGraphIngestionError(
            "External graph connector certification is unavailable"
        ) from None
    if not activation.get("checked") or not activation.get("ok"):
        raise ExternalGraphIngestionError(
            "External graph connector requires a certified capability bundle"
        )
    from .external_graph_schema import canonical_profile_ref

    profile_ref = request.profile_ref or canonical_profile_ref(connection)
    resolved_profile = _resolve_profile(
        profile_ref,
        profile=profile,
        resolver=profile_resolver,
    )
    # A direct ``profile=`` is an isolated test seam. Runtime profiles must be
    # generated by discovery and explicitly approved before any source read.
    if profile is None:
        if resolved_profile.get("profile_format") != "external-graph-profile/v1":
            raise ExternalGraphIngestionError(
                "External graph profile was not generated by schema discovery"
            )
        if resolved_profile.get("approval_status") != "approved":
            raise ExternalGraphIngestionError(
                "External graph mapping profile is not approved"
            )
        if resolved_profile.get("source_alias") != source_alias:
            raise ExternalGraphIngestionError(
                "External graph profile source alias does not match the request"
            )
        from .external_graph_schema import mapping_policy_digest

        if mapping_policy_digest(resolved_profile) != str(
            resolved_profile.get("mapping_digest") or ""
        ):
            raise ExternalGraphIngestionError(
                "External graph mapping profile changed after approval"
            )
        approved_policy_digest = str(
            resolved_profile.get("runtime_policy_digest") or ""
        )
        current_policy_digest = str(request.runtime_policy_digest or "")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", current_policy_digest)
            or current_policy_digest != approved_policy_digest
        ):
            raise ExternalGraphIngestionError(
                "External graph mapping policy drift requires a new proposal"
            )
        approved_sync = resolved_profile.get("sync")
        if not isinstance(approved_sync, Mapping) or dict(approved_sync) != {
            "allow_empty_snapshot": bool(request.allow_empty_snapshot),
            "max_pages": max_pages,
            "max_row_bytes": max_row_bytes,
            "max_total_bytes": max_total_bytes,
            "max_nesting_depth": max_nesting_depth,
            "max_collection_items": max_collection_items,
            "page_size": page_size,
            "reconcile_deletions": bool(request.reconcile_deletions),
            "sync_mode": sync_mode,
        }:
            raise ExternalGraphIngestionError(
                "External graph sync policy drift requires a new proposal"
            )
    identity_key = _resolve_identity_key(
        resolved_profile,
        connection=connection,
        resolver=profile_resolver,
    )
    node_mapping = _mapping(resolved_profile, "node_mapping")
    node_allowlist = _allowlist(node_mapping, label="node_mapping")
    node_query = str(resolved_profile.get("node_query") or "")
    _validate_read_query(node_query, label="node_query")
    edge_query = str(resolved_profile.get("edge_query") or "").strip()
    edge_mapping = resolved_profile.get("edge_mapping")
    edge_allowlist: tuple[str, ...] = ()
    if edge_query or edge_mapping:
        if not edge_query or not isinstance(edge_mapping, dict):
            raise ExternalGraphIngestionError(
                "edge_query and edge_mapping must be configured together"
            )
        _validate_read_query(edge_query, label="edge_query")
        edge_allowlist = _allowlist(edge_mapping, label="edge_mapping")

    access = _access(resolved_profile)
    if classification == DataClassification.PUBLIC and not access.is_public:
        raise ExternalGraphIngestionError(
            "PUBLIC classification requires an explicit public source ACL"
        )
    if classification != DataClassification.PUBLIC and access.is_public:
        raise ExternalGraphIngestionError(
            "A public source ACL requires PUBLIC classification"
        )

    try:
        role = registry.role(connection)
        if role == "mirror":
            raise ExternalGraphIngestionError(
                "Mirror connections cannot be used as ingestion sources"
            )
        external_engine = registry.get_engine(connection)
    except ExternalGraphIngestionError:
        raise
    except Exception as exc:
        raise ExternalGraphIngestionError(
            f"Registered graph connection is unavailable ({type(exc).__name__})"
        ) from None

    expected_schema_digest = str(resolved_profile.get("schema_digest") or "")
    if expected_schema_digest:
        try:
            from .external_graph_schema import discover_external_schema

            backend = (
                registry.backend_kind(connection)
                if callable(getattr(registry, "backend_kind", None))
                else resolved_profile.get("backend_kind")
            )
            discovered, _ = discover_external_schema(
                external_engine,
                backend=backend,
                max_types=int(resolved_profile.get("discovery_max_types") or 200),
            )
        except Exception as exc:
            raise ExternalGraphIngestionError(
                f"External graph schema verification failed ({type(exc).__name__})"
            ) from None
        if discovered.partial:
            raise ExternalGraphIngestionError(
                "External graph schema verification was incomplete; ingestion "
                "requires a new complete proposal"
            )
        if discovered.schema_digest != expected_schema_digest:
            raise ExternalGraphIngestionError(
                "External graph schema drift detected; ingestion requires a new proposal"
            )

    privacy = privacy_guard or PersistencePrivacyGuard()
    payload_budget = _PayloadBudget(
        max_row_bytes=max_row_bytes,
        max_total_bytes=max_total_bytes,
        max_nesting_depth=max_nesting_depth,
        max_collection_items=max_collection_items,
    )
    reader = _change_reader(external_engine)
    use_cdc = sync_mode != "snapshot" and reader is not None
    if sync_mode == "cdc" and reader is None:
        raise ExternalGraphIngestionError(
            "External graph source does not advertise native CDC"
        )
    delete_keys: list[str] = []
    current_cursor: str | None = None
    next_cursor: str | None = None
    if use_cdc:
        try:
            current_cursor = read_change_cursor(
                authority_engine,
                "external-graph",
                source_instance=source_alias,
            )
        except Exception as exc:
            raise ExternalGraphIngestionError(
                f"External graph CDC cursor read failed ({type(exc).__name__})"
            ) from None
        events, next_cursor = _read_change_pages(
            reader,
            cursor=current_cursor,
            page_size=page_size,
            max_pages=max_pages,
            max_records=max_records,
            privacy=privacy,
            budget=payload_budget,
        )
        node_rows = [
            dict(event["record"]) for event in events if event["operation"] == "upsert"
        ]
        delete_keys = [
            str(event["id"]) for event in events if event["operation"] == "delete"
        ]
        edge_rows: list[dict[str, Any]] = []
    else:
        variables = dict(request.variables or {})
        node_rows, snapshot_token = _read_external_pages(
            external_engine,
            node_query,
            variables,
            page_size=page_size,
            max_pages=max_pages,
            max_records=max_records,
            budget=payload_budget,
            privacy=privacy,
            label="node row",
        )
        edge_result = (
            _read_external_pages(
                external_engine,
                edge_query,
                variables,
                page_size=page_size,
                max_pages=max_pages,
                max_records=max_records,
                budget=payload_budget,
                privacy=privacy,
                label="edge row",
                required_snapshot_token=snapshot_token,
            )
            if edge_query
            else ([], snapshot_token)
        )
        edge_rows, _edge_snapshot_token = edge_result
    type_map = resolved_profile.get("type_map")
    if not isinstance(type_map, dict):
        type_map = {}

    id_path = str(node_mapping.get("id_path") or "id")
    type_path = str(node_mapping.get("type_path") or "type")
    props_path = str(node_mapping.get("properties_path") or "properties")
    version_path = str(node_mapping.get("version_path") or "version")
    internal_ids: dict[str, str] = {}
    prepared: list[dict[str, Any]] = []
    privacy_counts: Counter[str] = Counter()
    privacy_redactions = 0
    snapshot_identity_complete = True

    for row in node_rows:
        external_id = _dig(row, id_path)
        if external_id in (None, ""):
            if use_cdc:
                raise ExternalGraphIngestionError(
                    "External graph CDC upsert has no mapped identity"
                )
            snapshot_identity_complete = False
            continue
        external_key = str(external_id)
        if external_key in internal_ids:
            raise ExternalGraphIngestionError(
                "External graph snapshot contains a duplicate identity"
            )
        mapped_type, external_type = _safe_type(
            _dig(row, type_path), type_map, fallback="ExternalEntity"
        )
        if _PERSON_ENTITY.search(mapped_type) or _PERSON_ENTITY.search(external_type):
            privacy_counts["personal_entity"] += 1
            privacy_redactions += 1
            continue
        internal_id = (
            f"external:{source_alias}:"
            f"{_private_digest(identity_key, source_alias, external_key)[:32]}"
        )
        # Publish the mapping only after the entity passes the privacy gate. This
        # also drops every edge whose endpoint was quarantined above.
        internal_ids[external_key] = internal_id
        properties = _dig(row, props_path, {})
        if not isinstance(properties, dict):
            properties = {}
        selected = {field: properties.get(field) for field in node_allowlist}
        clean_properties, report = privacy.sanitize(selected)
        privacy_counts.update({label: 1 for label in report.detected_types})
        privacy_redactions += report.redactions
        clean_external_type, type_report = privacy.sanitize_text(external_type)
        privacy_counts.update({label: 1 for label in type_report.detected_types})
        privacy_redactions += type_report.redactions
        version = _dig(row, version_path, "")
        clean_version, version_report = privacy.sanitize_text(str(version or ""))
        privacy_counts.update({label: 1 for label in version_report.detected_types})
        privacy_redactions += version_report.redactions
        prepared.append(
            {
                "external_key": external_key,
                "internal_id": internal_id,
                "type": mapped_type,
                "external_type": clean_external_type,
                "properties": clean_properties,
                "version": clean_version or _digest(clean_properties),
            }
        )

    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if edge_query and isinstance(edge_mapping, dict):
        source_path = str(edge_mapping.get("source_path") or "source")
        target_path = str(edge_mapping.get("target_path") or "target")
        edge_type_path = str(edge_mapping.get("type_path") or "type")
        edge_props_path = str(edge_mapping.get("properties_path") or "properties")
        edge_type_map = resolved_profile.get("edge_type_map")
        if not isinstance(edge_type_map, dict):
            edge_type_map = {}
        for row in edge_rows:
            source_identity = _dig(row, source_path)
            target_identity = _dig(row, target_path)
            if source_identity in (None, "") or target_identity in (None, ""):
                snapshot_identity_complete = False
                continue
            source_key = str(source_identity)
            target_key = str(target_identity)
            if source_key not in internal_ids or target_key not in internal_ids:
                continue
            properties = _dig(row, edge_props_path, {})
            if not isinstance(properties, dict):
                properties = {}
            selected = {field: properties.get(field) for field in edge_allowlist}
            clean_properties, report = privacy.sanitize(selected)
            privacy_counts.update({label: 1 for label in report.detected_types})
            privacy_redactions += report.redactions
            edge_type, _ = _safe_type(
                _dig(row, edge_type_path), edge_type_map, fallback="EXTERNAL_LINK"
            )
            outgoing[source_key].append(
                {
                    "source": internal_ids[source_key],
                    "target": internal_ids[target_key],
                    "type": edge_type,
                    **clean_properties,
                }
            )

    delete_ids = sorted(
        {
            (
                f"external:{source_alias}:"
                f"{_private_digest(identity_key, source_alias, key)[:32]}"
            )
            for key in delete_keys
        }
    )
    if set(delete_ids).intersection(item["internal_id"] for item in prepared):
        raise ExternalGraphIngestionError(
            "External graph CDC batch contains conflicting changes"
        )
    if (
        not use_cdc
        and snapshot_identity_complete
        and request.reconcile_deletions
        and not prepared
        and not request.allow_empty_snapshot
    ):
        raise ExternalGraphIngestionError(
            "External graph empty snapshot is not approved for reconciliation"
        )

    profile_digest = _digest(
        node_query,
        node_mapping,
        edge_query,
        edge_mapping or {},
        resolved_profile.get("mapping_digest") or "",
        resolved_profile.get("runtime_policy_digest") or "",
        {
            "allow_empty_snapshot": bool(request.allow_empty_snapshot),
            "max_pages": max_pages,
            "max_row_bytes": max_row_bytes,
            "max_total_bytes": max_total_bytes,
            "max_nesting_depth": max_nesting_depth,
            "max_collection_items": max_collection_items,
            "page_size": page_size,
            "reconcile_deletions": bool(request.reconcile_deletions),
            "sync_mode": sync_mode,
        },
    )
    if request.dry_run:
        return {
            "status": "dry_run",
            "source_alias": source_alias,
            "connection": connection,
            "planned_nodes": len(prepared),
            "planned_edges": sum(len(items) for items in outgoing.values()),
            "planned_deletes": len(delete_ids),
            "sync_strategy": "cdc" if use_cdc else "snapshot",
            "snapshot_authoritative": bool(use_cdc or snapshot_identity_complete),
            "profile_digest": profile_digest,
            "privacy": {
                "redactions": privacy_redactions,
                "detected_types": sorted(privacy_counts),
            },
        }

    statuses: Counter[str] = Counter()
    edges = 0
    envelopes: list[ChangeEnvelope] = []
    for item in prepared:
        payload = {
            "id": item["internal_id"],
            "externalToolId": item["internal_id"],
            "type": item["type"],
            "external_type": item["external_type"],
            "external_source_alias": source_alias,
            **item["properties"],
        }
        links = sorted(
            outgoing.get(item["external_key"], []),
            key=lambda value: json.dumps(
                value, sort_keys=True, separators=(",", ":"), default=str
            ),
        )
        if links:
            payload["_links"] = links
            edges += len(links)
        material_version = _private_digest(
            identity_key,
            "material-version",
            source_alias,
            item["internal_id"],
            item["version"],
            payload,
            profile_digest,
        )
        envelopes.append(
            ChangeEnvelope(
                connector="external-graph",
                tenant=tenant,
                source_instance=source_alias,
                source_object_id=item["internal_id"],
                source_version=material_version,
                schema_version=str(resolved_profile.get("adapter_version") or "1"),
                ontology_mapping_version=str(
                    resolved_profile.get("proposal_version") or ""
                ),
                typed_payload=payload,
                source_acl=access,
                classification=classification,
                retention=retention,
                legal_hold=bool(request.legal_hold),
                provenance={
                    "connection_alias": connection,
                    "profile_digest": profile_digest,
                    "privacy_gate": True,
                },
                checkpoint=None,
            )
        )
    for internal_id in delete_ids:
        envelopes.append(
            ChangeEnvelope(
                connector="external-graph",
                operation="delete",
                tenant=tenant,
                source_instance=source_alias,
                source_object_id=internal_id,
                source_version=_private_digest(
                    identity_key,
                    "delete-version",
                    source_alias,
                    internal_id,
                    next_cursor or "",
                    profile_digest,
                ),
                schema_version=str(resolved_profile.get("adapter_version") or "1"),
                ontology_mapping_version=str(
                    resolved_profile.get("proposal_version") or ""
                ),
                classification=classification,
                retention=retention,
                legal_hold=bool(request.legal_hold),
                provenance={
                    "connection_alias": connection,
                    "profile_digest": profile_digest,
                    "privacy_gate": True,
                },
            )
        )

    for envelope in envelopes:
        result = ingest_envelope(authority_engine, envelope)
        statuses[str(result.get("status") or "unknown")] += 1

    incomplete = sum(
        count
        for status, count in statuses.items()
        if status not in {"success", "skipped"}
    )
    checkpoint: str | None
    marker: ChangeEnvelope | None = None
    if use_cdc:
        checkpoint = next_cursor
        if checkpoint and checkpoint != current_cursor and not incomplete:
            marker = ChangeEnvelope.snapshot_complete(
                connector="external-graph",
                tenant=tenant,
                source_instance=source_alias,
                checkpoint=checkpoint,
                live_ids=[],
                fetch_ok=False,
                schema_version=str(resolved_profile.get("adapter_version") or "1"),
                ontology_mapping_version=str(
                    resolved_profile.get("proposal_version") or ""
                ),
                provenance={
                    "connection_alias": connection,
                    "profile_digest": profile_digest,
                    "privacy_gate": True,
                    "sync_strategy": "cdc",
                },
            )
    else:
        checkpoint = _private_digest(
            identity_key,
            "snapshot-checkpoint",
            source_alias,
            profile_digest,
            sorted((item["internal_id"], item["version"]) for item in prepared),
            outgoing,
        )
        if not incomplete:
            marker = ChangeEnvelope.snapshot_complete(
                connector="external-graph",
                tenant=tenant,
                source_instance=source_alias,
                checkpoint=checkpoint,
                live_ids=(
                    [item["internal_id"] for item in prepared]
                    if request.reconcile_deletions and snapshot_identity_complete
                    else []
                ),
                fetch_ok=bool(
                    request.reconcile_deletions and snapshot_identity_complete
                ),
                schema_version=str(resolved_profile.get("adapter_version") or "1"),
                ontology_mapping_version=str(
                    resolved_profile.get("proposal_version") or ""
                ),
                provenance={
                    "authoritative_empty_approved": bool(request.allow_empty_snapshot),
                    "connection_alias": connection,
                    "profile_digest": profile_digest,
                    "privacy_gate": True,
                    "sync_strategy": "snapshot",
                },
            )
    if marker is not None:
        marker_result = ingest_envelope(authority_engine, marker)
        marker_status = str(marker_result.get("status") or "unknown")
        statuses[marker_status] += 1
        if marker_status not in {"success", "skipped"}:
            incomplete += 1
    source_incomplete = bool(not use_cdc and not snapshot_identity_complete)
    return {
        "status": "partial" if incomplete or source_incomplete else "success",
        "source_alias": source_alias,
        "connection": connection,
        "nodes": len(prepared),
        "edges": edges,
        "deletes": len(delete_ids),
        "sync_strategy": "cdc" if use_cdc else "snapshot",
        "snapshot_authoritative": bool(use_cdc or snapshot_identity_complete),
        "results": dict(sorted(statuses.items())),
        "profile_digest": profile_digest,
        "privacy": {
            "redactions": privacy_redactions,
            "detected_types": sorted(privacy_counts),
        },
    }
