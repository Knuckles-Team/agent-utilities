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

    def _check_container_bounds(
        self,
        value: Mapping[Any, Any] | list[Any] | tuple[Any, ...],
        depth: int,
        label: str,
        visited_containers: set[int],
    ) -> None:
        """Shared repeated-container / nesting-depth / collection-size checks.

        Applies identically whether ``value`` is a ``Mapping`` or a ``list``/``tuple``
        (the two branches of the bounded-JSON walk were exact duplicates of this).
        """
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

    def _walk_bounded(self, row: Mapping[str, Any], label: str) -> None:
        """Walk ``row`` depth-first, enforcing nesting/size/repeated-container bounds."""
        stack: list[tuple[Any, int]] = [(row, 1)]
        visited_containers: set[int] = set()
        while stack:
            value, depth = stack.pop()
            if isinstance(value, Mapping):
                self._check_container_bounds(value, depth, label, visited_containers)
                stack.extend((item, depth + 1) for item in value.values())
            elif isinstance(value, list | tuple):
                self._check_container_bounds(value, depth, label, visited_containers)
                stack.extend((item, depth + 1) for item in value)

    def _row_byte_size(self, row: Mapping[str, Any], label: str) -> int:
        """Serialize + size-check one row. Raises on non-JSON or over the per-row byte bound."""
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
        return row_bytes

    def accept(self, row: Mapping[str, Any], *, label: str) -> None:
        try:
            self._walk_bounded(row, label)
        except ExternalGraphIngestionError:
            raise
        except Exception:
            raise ExternalGraphIngestionError(
                f"External graph {label} is not bounded JSON"
            ) from None
        row_bytes = self._row_byte_size(row, label)
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


def _rows_from_iterable(
    value: Iterable[Any], *, max_records: int
) -> list[dict[str, Any]]:
    """The ``Iterable``-of-rows branch of :func:`_rows`."""
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
        return _rows_from_iterable(value, max_records=max_records)
    return []


def _skip_line_comment(query: str, index: int, size: int) -> int:
    """Skip a ``// ...`` line comment starting at ``index``; returns the index after it."""
    newline = query.find("\n", index + 2)
    return size if newline < 0 else newline + 1


def _skip_block_comment(query: str, index: int, label: str) -> int:
    """Skip a ``/* ... */`` block comment starting at ``index``; returns the index after it."""
    end = query.find("*/", index + 2)
    if end < 0:
        raise ExternalGraphIngestionError(f"{label} is not valid Cypher")
    return end + 2


def _scan_quoted_literal(query: str, index: int, size: int, label: str) -> int:
    """Consume a quote-delimited literal starting at ``index`` (the opening quote).

    Returns the index just past the closing (unescaped, undoubled) delimiter.
    """
    delimiter = query[index]
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
    return index


def _scan_parameter_token(
    query: str, index: int, size: int, label: str
) -> tuple[int, tuple[str, str]]:
    """Consume a ``$name`` parameter token starting at ``index`` (the ``$``)."""
    end = index + 1
    while end < size and (
        query[end].isascii() and (query[end].isalnum() or query[end] == "_")
    ):
        end += 1
    if end == index + 1:
        raise ExternalGraphIngestionError(f"{label} is not valid Cypher")
    return end, ("parameter", query[index + 1 : end].casefold())


def _scan_word_token(query: str, index: int, size: int) -> tuple[int, tuple[str, str]]:
    """Consume a bare-word token starting at ``index``."""
    end = index + 1
    while end < size and (
        query[end].isascii() and (query[end].isalnum() or query[end] == "_")
    ):
        end += 1
    return end, ("word", query[index:end].upper())


def _skip_trivia(query: str, index: int, size: int, label: str) -> int | None:
    """Skip whitespace or a comment starting at ``index``.

    Returns the new index, or ``None`` if ``index`` is not trivia (caller should
    scan a token there instead).
    """
    char = query[index]
    if char.isspace():
        return index + 1
    if query.startswith("//", index):
        return _skip_line_comment(query, index, size)
    if query.startswith("/*", index):
        return _skip_block_comment(query, index, label)
    return None


def _is_bare_word_start(char: str) -> bool:
    """Whether ``char`` can start a bare (unquoted) Cypher word token."""
    return char.isascii() and (char.isalpha() or char == "_")


def _scan_token(
    query: str, index: int, size: int, label: str
) -> tuple[int, tuple[str, str] | None]:
    """Scan one non-trivia token starting at ``index``. Returns ``(new_index, token)``.

    ``token`` is ``None`` only for a quoted literal (consumed, but not tokenized).
    """
    char = query[index]
    if char in {"'", '"', "`"}:
        return _scan_quoted_literal(query, index, size, label), None
    if char == ";":
        raise ExternalGraphIngestionError(
            f"{label} must contain exactly one read statement"
        )
    if char == "$":
        return _scan_parameter_token(query, index, size, label)
    if _is_bare_word_start(char):
        return _scan_word_token(query, index, size)
    if not char.isascii():
        raise ExternalGraphIngestionError(
            f"{label} bare identifiers must be ASCII or backtick-quoted"
        )
    return index + 1, ("symbol", char)


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
        skipped = _skip_trivia(query, index, size, label)
        if skipped is not None:
            index = skipped
            continue
        index, token = _scan_token(query, index, size, label)
        if token is not None:
            tokens.append(token)
    return tokens


def _contains_token_pair(
    tokens: list[tuple[str, str]], pair: list[tuple[str, str]]
) -> bool:
    """Whether ``pair`` (2 tokens) appears anywhere as a contiguous subsequence of ``tokens``."""
    return any(tokens[index : index + 2] == pair for index in range(len(tokens) - 1))


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
    if not _contains_token_pair(tokens, [("word", "SKIP"), ("parameter", "offset")]):
        raise ExternalGraphIngestionError(
            f"{label} must use the exact page cursor SKIP $offset"
        )
    if not _contains_token_pair(tokens, [("word", "ORDER"), ("word", "BY")]):
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


def _read_external_snapshot_full(
    engine: Any,
    query: str,
    variables: Mapping[str, Any],
    *,
    max_records: int,
    budget: _PayloadBudget,
    label: str,
) -> list[dict[str, Any]]:
    """Read a full snapshot in one bounded request (no paginated ``read_snapshot_page``)."""
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
    return rows


def _fetch_snapshot_page(
    snapshot_reader: Callable[..., Any],
    query: str,
    params: dict[str, Any],
    *,
    window: int,
    snapshot_token: str | None,
    privacy: PersistencePrivacyGuard,
    label: str,
) -> tuple[Mapping[str, Any], str]:
    """Call ``snapshot_reader`` for one page and validate its snapshot-token contract.

    Returns ``(result, page_token)``.
    """
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
    return result, page_token


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
        rows = _read_external_snapshot_full(
            engine,
            query,
            variables,
            max_records=max_records,
            budget=budget,
            label=label,
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
        result, snapshot_token = _fetch_snapshot_page(
            snapshot_reader,
            query,
            params,
            window=window,
            snapshot_token=snapshot_token,
            privacy=privacy,
            label=label,
        )
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


def _fetch_change_page(
    reader: Callable[..., Any], current: str | None, window: int
) -> Mapping[str, Any]:
    """Call ``reader`` for one CDC page and validate its outer contract."""
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
    return result


def _validate_change_event(event: dict[str, Any]) -> None:
    """Enforce the per-event CDC contract (a supported node upsert/delete shape)."""
    operation = str(event.get("operation") or "")
    entity = str(event.get("entity") or "node")
    if operation not in {"upsert", "delete"} or entity != "node":
        raise ExternalGraphIngestionError(
            "External graph CDC event is not a supported node change"
        )
    if operation == "upsert" and not isinstance(event.get("record"), dict):
        raise ExternalGraphIngestionError("External graph CDC upsert has no record")
    if operation == "delete" and event.get("id") in (None, ""):
        raise ExternalGraphIngestionError("External graph CDC delete has no identity")


def _collect_change_events(
    raw_events: Any, *, window: int, budget: _PayloadBudget
) -> list[dict[str, Any]]:
    """Validate + collect one page's raw CDC events (shape, budget, and per-event contract)."""
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
        _validate_change_event(event)
        events.append(event)
    return events


def _resolve_change_page_cursor(
    result: Mapping[str, Any],
    current: str | None,
    events: list[dict[str, Any]],
    privacy: PersistencePrivacyGuard,
) -> tuple[bool, str | None]:
    """Validate + extract ``(has_more, page_cursor)`` from one CDC page's result."""
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
    return has_more, page_cursor


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
        result = _fetch_change_page(reader, current, window)
        events = _collect_change_events(
            result.get("events"), window=window, budget=budget
        )
        collected.extend(events)
        has_more, page_cursor = _resolve_change_page_cursor(
            result, current, events, privacy
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


@dataclass(frozen=True)
class _ValidatedRequest:
    """Bounds and aliases validated from an ``ExternalGraphIngestionRequest``."""

    connection: str
    source_alias: str
    tenant: str
    max_records: int
    page_size: int
    max_pages: int
    max_row_bytes: int
    max_total_bytes: int
    max_nesting_depth: int
    max_collection_items: int
    sync_mode: str
    classification: DataClassification
    retention: str


def _validate_aliases(
    request: ExternalGraphIngestionRequest,
) -> tuple[str, str, str]:
    connection = _alias(request.connection, label="connection")
    source_alias = _alias(request.source_alias, label="source_alias")
    tenant = _alias(request.tenant, label="tenant", allow_empty=True)
    return connection, source_alias, tenant


def _validate_page_limits(
    request: ExternalGraphIngestionRequest,
) -> tuple[int, int, int]:
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
    return max_records, page_size, max_pages


def _validate_structural_bounds(
    request: ExternalGraphIngestionRequest,
) -> tuple[int, int, int, int]:
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
    return max_row_bytes, max_total_bytes, max_nesting_depth, max_collection_items


def _validate_sync_mode(request: ExternalGraphIngestionRequest) -> str:
    if not isinstance(request.reconcile_deletions, bool) or not isinstance(
        request.allow_empty_snapshot, bool
    ):
        raise ExternalGraphIngestionError(
            "External graph reconciliation policy must be boolean"
        )
    sync_mode = str(request.sync_mode or "")
    if sync_mode not in {"auto", "cdc", "snapshot"}:
        raise ExternalGraphIngestionError("sync_mode must be auto, cdc, or snapshot")
    return sync_mode


def _validate_retention(request: ExternalGraphIngestionRequest) -> str:
    retention = str(request.retention or "").strip()
    if not _RETENTION_RE.fullmatch(retention):
        raise ExternalGraphIngestionError(
            "retention must be an ISO duration or a neutral policy alias"
        )
    return retention


def _validate_request(request: ExternalGraphIngestionRequest) -> _ValidatedRequest:
    """Validate and normalize every non-secret request field, fail closed."""

    connection, source_alias, tenant = _validate_aliases(request)
    max_records, page_size, max_pages = _validate_page_limits(request)
    (
        max_row_bytes,
        max_total_bytes,
        max_nesting_depth,
        max_collection_items,
    ) = _validate_structural_bounds(request)
    sync_mode = _validate_sync_mode(request)
    classification = _classification(request.classification)
    retention = _validate_retention(request)
    return _ValidatedRequest(
        connection=connection,
        source_alias=source_alias,
        tenant=tenant,
        max_records=max_records,
        page_size=page_size,
        max_pages=max_pages,
        max_row_bytes=max_row_bytes,
        max_total_bytes=max_total_bytes,
        max_nesting_depth=max_nesting_depth,
        max_collection_items=max_collection_items,
        sync_mode=sync_mode,
        classification=classification,
        retention=retention,
    )


def _certify_external_graph_connector() -> None:
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


def _resolve_profile_ref(
    request: ExternalGraphIngestionRequest, connection: str
) -> str:
    from .external_graph_schema import canonical_profile_ref

    return request.profile_ref or canonical_profile_ref(connection)


def _validate_profile_approval(
    resolved_profile: dict[str, Any], source_alias: str
) -> None:
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


def _validate_profile_mapping_digest(resolved_profile: dict[str, Any]) -> None:
    from .external_graph_schema import mapping_policy_digest

    if mapping_policy_digest(resolved_profile) != str(
        resolved_profile.get("mapping_digest") or ""
    ):
        raise ExternalGraphIngestionError(
            "External graph mapping profile changed after approval"
        )


def _validate_runtime_policy_digest(
    resolved_profile: dict[str, Any], request: ExternalGraphIngestionRequest
) -> None:
    approved_policy_digest = str(resolved_profile.get("runtime_policy_digest") or "")
    current_policy_digest = str(request.runtime_policy_digest or "")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", current_policy_digest)
        or current_policy_digest != approved_policy_digest
    ):
        raise ExternalGraphIngestionError(
            "External graph mapping policy drift requires a new proposal"
        )


def _validate_sync_policy_digest(
    resolved_profile: dict[str, Any],
    request: ExternalGraphIngestionRequest,
    bounds: _ValidatedRequest,
) -> None:
    approved_sync = resolved_profile.get("sync")
    if not isinstance(approved_sync, Mapping) or dict(approved_sync) != {
        "allow_empty_snapshot": bool(request.allow_empty_snapshot),
        "max_pages": bounds.max_pages,
        "max_row_bytes": bounds.max_row_bytes,
        "max_total_bytes": bounds.max_total_bytes,
        "max_nesting_depth": bounds.max_nesting_depth,
        "max_collection_items": bounds.max_collection_items,
        "page_size": bounds.page_size,
        "reconcile_deletions": bool(request.reconcile_deletions),
        "sync_mode": bounds.sync_mode,
    }:
        raise ExternalGraphIngestionError(
            "External graph sync policy drift requires a new proposal"
        )


def _validate_profile_freshness(
    resolved_profile: dict[str, Any],
    *,
    profile_arg: dict[str, Any] | None,
    source_alias: str,
    request: ExternalGraphIngestionRequest,
    bounds: _ValidatedRequest,
) -> None:
    # A direct ``profile=`` is an isolated test seam. Runtime profiles must be
    # generated by discovery and explicitly approved before any source read.
    if profile_arg is not None:
        return
    _validate_profile_approval(resolved_profile, source_alias)
    _validate_profile_mapping_digest(resolved_profile)
    _validate_runtime_policy_digest(resolved_profile, request)
    _validate_sync_policy_digest(resolved_profile, request, bounds)


@dataclass(frozen=True)
class _Mappings:
    node_mapping: dict[str, Any]
    node_allowlist: tuple[str, ...]
    node_query: str
    edge_query: str
    edge_mapping: Any
    edge_allowlist: tuple[str, ...]


def _resolve_mappings(resolved_profile: dict[str, Any]) -> _Mappings:
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
    return _Mappings(
        node_mapping=node_mapping,
        node_allowlist=node_allowlist,
        node_query=node_query,
        edge_query=edge_query,
        edge_mapping=edge_mapping,
        edge_allowlist=edge_allowlist,
    )


def _validate_access_classification(
    resolved_profile: dict[str, Any], classification: DataClassification
) -> ExternalAccess:
    access = _access(resolved_profile)
    if classification == DataClassification.PUBLIC and not access.is_public:
        raise ExternalGraphIngestionError(
            "PUBLIC classification requires an explicit public source ACL"
        )
    if classification != DataClassification.PUBLIC and access.is_public:
        raise ExternalGraphIngestionError(
            "A public source ACL requires PUBLIC classification"
        )
    return access


def _resolve_external_engine(registry: Any, connection: str) -> Any:
    try:
        role = registry.role(connection)
        if role == "mirror":
            raise ExternalGraphIngestionError(
                "Mirror connections cannot be used as ingestion sources"
            )
        return registry.get_engine(connection)
    except ExternalGraphIngestionError:
        raise
    except Exception as exc:
        raise ExternalGraphIngestionError(
            f"Registered graph connection is unavailable ({type(exc).__name__})"
        ) from None


def _verify_schema_digest(
    resolved_profile: dict[str, Any],
    registry: Any,
    connection: str,
    external_engine: Any,
) -> None:
    expected_schema_digest = str(resolved_profile.get("schema_digest") or "")
    if not expected_schema_digest:
        return
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


def _setup_reader(
    external_engine: Any, sync_mode: str
) -> tuple[Callable[..., Any] | None, bool]:
    reader = _change_reader(external_engine)
    use_cdc = sync_mode != "snapshot" and reader is not None
    if sync_mode == "cdc" and reader is None:
        raise ExternalGraphIngestionError(
            "External graph source does not advertise native CDC"
        )
    return reader, use_cdc


def _read_cdc_rows(
    authority_engine: Any,
    reader: Callable[..., Any] | None,
    *,
    source_alias: str,
    page_size: int,
    max_pages: int,
    max_records: int,
    privacy: PersistencePrivacyGuard,
    budget: _PayloadBudget,
) -> tuple[
    list[dict[str, Any]], list[str], list[dict[str, Any]], str | None, str | None
]:
    assert reader is not None  # guaranteed by use_cdc's `reader is not None` term
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
        budget=budget,
    )
    node_rows = [
        dict(event["record"]) for event in events if event["operation"] == "upsert"
    ]
    delete_keys = [
        str(event["id"]) for event in events if event["operation"] == "delete"
    ]
    edge_rows: list[dict[str, Any]] = []
    return node_rows, delete_keys, edge_rows, current_cursor, next_cursor


def _read_snapshot_rows(
    external_engine: Any,
    *,
    node_query: str,
    edge_query: str,
    variables: dict[str, Any],
    page_size: int,
    max_pages: int,
    max_records: int,
    budget: _PayloadBudget,
    privacy: PersistencePrivacyGuard,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_rows, snapshot_token = _read_external_pages(
        external_engine,
        node_query,
        variables,
        page_size=page_size,
        max_pages=max_pages,
        max_records=max_records,
        budget=budget,
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
            budget=budget,
            privacy=privacy,
            label="edge row",
            required_snapshot_token=snapshot_token,
        )
        if edge_query
        else ([], snapshot_token)
    )
    edge_rows, _edge_snapshot_token = edge_result
    return node_rows, edge_rows


def _sanitize_node_fields(
    *,
    external_type: str,
    properties: dict[str, Any],
    version: Any,
    node_allowlist: tuple[str, ...],
    privacy: PersistencePrivacyGuard,
    privacy_counts: Counter[str],
) -> tuple[dict[str, Any], str, str, int]:
    """Sanitize node properties/type/version; mutates ``privacy_counts`` in place.

    Returns ``(clean_properties, clean_external_type, clean_version, redactions)``.
    """
    selected = {field: properties.get(field) for field in node_allowlist}
    clean_properties, report = privacy.sanitize(selected)
    privacy_counts.update({label: 1 for label in report.detected_types})
    redactions = report.redactions
    clean_external_type, type_report = privacy.sanitize_text(external_type)
    privacy_counts.update({label: 1 for label in type_report.detected_types})
    redactions += type_report.redactions
    clean_version, version_report = privacy.sanitize_text(str(version or ""))
    privacy_counts.update({label: 1 for label in version_report.detected_types})
    redactions += version_report.redactions
    return clean_properties, clean_external_type, clean_version, redactions


def _prepare_node_row(
    row: dict[str, Any],
    *,
    id_path: str,
    type_path: str,
    props_path: str,
    version_path: str,
    type_map: dict[str, Any],
    use_cdc: bool,
    source_alias: str,
    identity_key: str,
    node_allowlist: tuple[str, ...],
    privacy: PersistencePrivacyGuard,
    internal_ids: dict[str, str],
    privacy_counts: Counter[str],
) -> tuple[bool, dict[str, Any] | None, int]:
    """Prepare one node row; returns (identity_present, item_or_None, redactions)."""

    external_id = _dig(row, id_path)
    if external_id in (None, ""):
        if use_cdc:
            raise ExternalGraphIngestionError(
                "External graph CDC upsert has no mapped identity"
            )
        return False, None, 0
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
        return True, None, 1
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
    version = _dig(row, version_path, "")
    clean_properties, clean_external_type, clean_version, redactions = (
        _sanitize_node_fields(
            external_type=external_type,
            properties=properties,
            version=version,
            node_allowlist=node_allowlist,
            privacy=privacy,
            privacy_counts=privacy_counts,
        )
    )
    item = {
        "external_key": external_key,
        "internal_id": internal_id,
        "type": mapped_type,
        "external_type": clean_external_type,
        "properties": clean_properties,
        "version": clean_version or _digest(clean_properties),
    }
    return True, item, redactions


def _prepare_node_rows(
    node_rows: list[dict[str, Any]],
    *,
    node_mapping: dict[str, Any],
    type_map: dict[str, Any],
    use_cdc: bool,
    source_alias: str,
    identity_key: str,
    node_allowlist: tuple[str, ...],
    privacy: PersistencePrivacyGuard,
) -> tuple[dict[str, str], list[dict[str, Any]], Counter[str], int, bool]:
    id_path = str(node_mapping.get("id_path") or "id")
    type_path = str(node_mapping.get("type_path") or "type")
    props_path = str(node_mapping.get("properties_path") or "properties")
    version_path = str(node_mapping.get("version_path") or "version")
    internal_ids: dict[str, str] = {}
    prepared: list[dict[str, Any]] = []
    privacy_counts: Counter[str] = Counter()
    privacy_redactions = 0
    identity_complete = True

    for row in node_rows:
        present, item, redactions = _prepare_node_row(
            row,
            id_path=id_path,
            type_path=type_path,
            props_path=props_path,
            version_path=version_path,
            type_map=type_map,
            use_cdc=use_cdc,
            source_alias=source_alias,
            identity_key=identity_key,
            node_allowlist=node_allowlist,
            privacy=privacy,
            internal_ids=internal_ids,
            privacy_counts=privacy_counts,
        )
        if not present:
            identity_complete = False
            continue
        privacy_redactions += redactions
        if item is not None:
            prepared.append(item)

    return internal_ids, prepared, privacy_counts, privacy_redactions, identity_complete


def _prepare_edge_row(
    row: dict[str, Any],
    *,
    source_path: str,
    target_path: str,
    edge_type_path: str,
    edge_props_path: str,
    edge_type_map: dict[str, Any],
    internal_ids: dict[str, str],
    edge_allowlist: tuple[str, ...],
    privacy: PersistencePrivacyGuard,
    privacy_counts: Counter[str],
) -> tuple[str, tuple[str, dict[str, Any]] | None, int]:
    """Prepare one edge row; returns (state, (source_key, edge) or None, redactions).

    ``state`` is one of ``"missing"`` (no mapped identity on either endpoint),
    ``"dropped"`` (an endpoint was quarantined/absent from ``internal_ids``), or
    ``"ok"``.

    The source key and the edge travel together as ONE optional pair rather
    than as two independently-optional values, because they are never
    independently present: both exist exactly when ``state == "ok"``. Stating
    that in the type is what lets the caller index ``outgoing[source_key]``
    without a `type: ignore` on each of two separate error codes.
    """

    source_identity = _dig(row, source_path)
    target_identity = _dig(row, target_path)
    if source_identity in (None, "") or target_identity in (None, ""):
        return "missing", None, 0
    source_key = str(source_identity)
    target_key = str(target_identity)
    if source_key not in internal_ids or target_key not in internal_ids:
        return "dropped", None, 0
    properties = _dig(row, edge_props_path, {})
    if not isinstance(properties, dict):
        properties = {}
    selected = {field: properties.get(field) for field in edge_allowlist}
    clean_properties, report = privacy.sanitize(selected)
    privacy_counts.update({label: 1 for label in report.detected_types})
    redactions = report.redactions
    edge_type, _ = _safe_type(
        _dig(row, edge_type_path), edge_type_map, fallback="EXTERNAL_LINK"
    )
    edge = {
        "source": internal_ids[source_key],
        "target": internal_ids[target_key],
        "type": edge_type,
        **clean_properties,
    }
    return "ok", (source_key, edge), redactions


def _edge_field_paths(
    edge_mapping: dict[str, Any], resolved_profile: dict[str, Any]
) -> tuple[str, str, str, str, dict[str, Any]]:
    source_path = str(edge_mapping.get("source_path") or "source")
    target_path = str(edge_mapping.get("target_path") or "target")
    edge_type_path = str(edge_mapping.get("type_path") or "type")
    edge_props_path = str(edge_mapping.get("properties_path") or "properties")
    edge_type_map = resolved_profile.get("edge_type_map")
    if not isinstance(edge_type_map, dict):
        edge_type_map = {}
    return source_path, target_path, edge_type_path, edge_props_path, edge_type_map


def _prepare_edges(
    edge_query: str,
    edge_mapping: Any,
    edge_rows: list[dict[str, Any]],
    *,
    internal_ids: dict[str, str],
    edge_allowlist: tuple[str, ...],
    privacy: PersistencePrivacyGuard,
    resolved_profile: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str], int, bool]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    privacy_counts: Counter[str] = Counter()
    privacy_redactions = 0
    identity_complete = True
    if not (edge_query and isinstance(edge_mapping, dict)):
        return outgoing, privacy_counts, privacy_redactions, identity_complete

    source_path, target_path, edge_type_path, edge_props_path, edge_type_map = (
        _edge_field_paths(edge_mapping, resolved_profile)
    )

    for row in edge_rows:
        state, prepared, redactions = _prepare_edge_row(
            row,
            source_path=source_path,
            target_path=target_path,
            edge_type_path=edge_type_path,
            edge_props_path=edge_props_path,
            edge_type_map=edge_type_map,
            internal_ids=internal_ids,
            edge_allowlist=edge_allowlist,
            privacy=privacy,
            privacy_counts=privacy_counts,
        )
        if state == "missing":
            identity_complete = False
            continue
        if prepared is None:  # "dropped" — an endpoint is not in internal_ids
            continue
        source_key, edge = prepared
        privacy_redactions += redactions
        outgoing[source_key].append(edge)

    return outgoing, privacy_counts, privacy_redactions, identity_complete


def _compute_delete_ids(
    delete_keys: list[str],
    *,
    identity_key: str,
    source_alias: str,
    prepared: list[dict[str, Any]],
) -> list[str]:
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
    return delete_ids


def _guard_empty_snapshot(
    *,
    use_cdc: bool,
    snapshot_identity_complete: bool,
    request: ExternalGraphIngestionRequest,
    prepared: list[dict[str, Any]],
) -> None:
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


def _compute_profile_digest(
    *,
    node_query: str,
    node_mapping: dict[str, Any],
    edge_query: str,
    edge_mapping: Any,
    resolved_profile: dict[str, Any],
    request: ExternalGraphIngestionRequest,
    bounds: _ValidatedRequest,
) -> str:
    return _digest(
        node_query,
        node_mapping,
        edge_query,
        edge_mapping or {},
        resolved_profile.get("mapping_digest") or "",
        resolved_profile.get("runtime_policy_digest") or "",
        {
            "allow_empty_snapshot": bool(request.allow_empty_snapshot),
            "max_pages": bounds.max_pages,
            "max_row_bytes": bounds.max_row_bytes,
            "max_total_bytes": bounds.max_total_bytes,
            "max_nesting_depth": bounds.max_nesting_depth,
            "max_collection_items": bounds.max_collection_items,
            "page_size": bounds.page_size,
            "reconcile_deletions": bool(request.reconcile_deletions),
            "sync_mode": bounds.sync_mode,
        },
    )


def _build_dry_run_result(
    *,
    source_alias: str,
    connection: str,
    prepared: list[dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    delete_ids: list[str],
    use_cdc: bool,
    snapshot_identity_complete: bool,
    profile_digest: str,
    privacy_redactions: int,
    privacy_counts: Counter[str],
) -> dict[str, Any]:
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


def _build_node_envelope(
    item: dict[str, Any],
    *,
    outgoing: dict[str, list[dict[str, Any]]],
    source_alias: str,
    identity_key: str,
    profile_digest: str,
    tenant: str,
    resolved_profile: dict[str, Any],
    access: ExternalAccess,
    classification: DataClassification,
    retention: str,
    request: ExternalGraphIngestionRequest,
    connection: str,
) -> tuple[ChangeEnvelope, int]:
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
    edge_count = 0
    if links:
        payload["_links"] = links
        edge_count = len(links)
    material_version = _private_digest(
        identity_key,
        "material-version",
        source_alias,
        item["internal_id"],
        item["version"],
        payload,
        profile_digest,
    )
    envelope = ChangeEnvelope(
        connector="external-graph",
        tenant=tenant,
        source_instance=source_alias,
        source_object_id=item["internal_id"],
        source_version=material_version,
        schema_version=str(resolved_profile.get("adapter_version") or "1"),
        ontology_mapping_version=str(resolved_profile.get("proposal_version") or ""),
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
    return envelope, edge_count


def _build_node_envelopes(
    prepared: list[dict[str, Any]],
    *,
    outgoing: dict[str, list[dict[str, Any]]],
    source_alias: str,
    identity_key: str,
    profile_digest: str,
    tenant: str,
    resolved_profile: dict[str, Any],
    access: ExternalAccess,
    classification: DataClassification,
    retention: str,
    request: ExternalGraphIngestionRequest,
    connection: str,
) -> tuple[list[ChangeEnvelope], int]:
    envelopes: list[ChangeEnvelope] = []
    edges = 0
    for item in prepared:
        envelope, edge_count = _build_node_envelope(
            item,
            outgoing=outgoing,
            source_alias=source_alias,
            identity_key=identity_key,
            profile_digest=profile_digest,
            tenant=tenant,
            resolved_profile=resolved_profile,
            access=access,
            classification=classification,
            retention=retention,
            request=request,
            connection=connection,
        )
        envelopes.append(envelope)
        edges += edge_count
    return envelopes, edges


def _build_delete_envelope(
    internal_id: str,
    *,
    tenant: str,
    source_alias: str,
    identity_key: str,
    next_cursor: str | None,
    resolved_profile: dict[str, Any],
    classification: DataClassification,
    retention: str,
    request: ExternalGraphIngestionRequest,
    connection: str,
    profile_digest: str,
) -> ChangeEnvelope:
    return ChangeEnvelope(
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
        ontology_mapping_version=str(resolved_profile.get("proposal_version") or ""),
        classification=classification,
        retention=retention,
        legal_hold=bool(request.legal_hold),
        provenance={
            "connection_alias": connection,
            "profile_digest": profile_digest,
            "privacy_gate": True,
        },
    )


def _build_delete_envelopes(
    delete_ids: list[str],
    *,
    tenant: str,
    source_alias: str,
    identity_key: str,
    next_cursor: str | None,
    resolved_profile: dict[str, Any],
    classification: DataClassification,
    retention: str,
    request: ExternalGraphIngestionRequest,
    connection: str,
    profile_digest: str,
) -> list[ChangeEnvelope]:
    return [
        _build_delete_envelope(
            internal_id,
            tenant=tenant,
            source_alias=source_alias,
            identity_key=identity_key,
            next_cursor=next_cursor,
            resolved_profile=resolved_profile,
            classification=classification,
            retention=retention,
            request=request,
            connection=connection,
            profile_digest=profile_digest,
        )
        for internal_id in delete_ids
    ]


def _ingest_envelopes(
    authority_engine: Any, envelopes: list[ChangeEnvelope]
) -> Counter[str]:
    statuses: Counter[str] = Counter()
    for envelope in envelopes:
        result = ingest_envelope(authority_engine, envelope)
        statuses[str(result.get("status") or "unknown")] += 1
    return statuses


def _compute_incomplete(statuses: Counter[str]) -> int:
    return sum(
        count
        for status, count in statuses.items()
        if status not in {"success", "skipped"}
    )


def _cdc_marker(
    *,
    next_cursor: str | None,
    current_cursor: str | None,
    incomplete: int,
    tenant: str,
    source_alias: str,
    resolved_profile: dict[str, Any],
    connection: str,
    profile_digest: str,
) -> ChangeEnvelope | None:
    checkpoint = next_cursor
    if not checkpoint or checkpoint == current_cursor or incomplete:
        return None
    return ChangeEnvelope.snapshot_complete(
        connector="external-graph",
        tenant=tenant,
        source_instance=source_alias,
        checkpoint=checkpoint,
        live_ids=[],
        fetch_ok=False,
        schema_version=str(resolved_profile.get("adapter_version") or "1"),
        ontology_mapping_version=str(resolved_profile.get("proposal_version") or ""),
        provenance={
            "connection_alias": connection,
            "profile_digest": profile_digest,
            "privacy_gate": True,
            "sync_strategy": "cdc",
        },
    )


def _snapshot_marker(
    *,
    identity_key: str,
    source_alias: str,
    profile_digest: str,
    prepared: list[dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    incomplete: int,
    request: ExternalGraphIngestionRequest,
    snapshot_identity_complete: bool,
    tenant: str,
    resolved_profile: dict[str, Any],
    connection: str,
) -> ChangeEnvelope | None:
    checkpoint = _private_digest(
        identity_key,
        "snapshot-checkpoint",
        source_alias,
        profile_digest,
        sorted((item["internal_id"], item["version"]) for item in prepared),
        outgoing,
    )
    if incomplete:
        return None
    return ChangeEnvelope.snapshot_complete(
        connector="external-graph",
        tenant=tenant,
        source_instance=source_alias,
        checkpoint=checkpoint,
        live_ids=(
            [item["internal_id"] for item in prepared]
            if request.reconcile_deletions and snapshot_identity_complete
            else []
        ),
        fetch_ok=bool(request.reconcile_deletions and snapshot_identity_complete),
        schema_version=str(resolved_profile.get("adapter_version") or "1"),
        ontology_mapping_version=str(resolved_profile.get("proposal_version") or ""),
        provenance={
            "authoritative_empty_approved": bool(request.allow_empty_snapshot),
            "connection_alias": connection,
            "profile_digest": profile_digest,
            "privacy_gate": True,
            "sync_strategy": "snapshot",
        },
    )


def _build_marker(
    *,
    use_cdc: bool,
    next_cursor: str | None,
    current_cursor: str | None,
    incomplete: int,
    tenant: str,
    source_alias: str,
    resolved_profile: dict[str, Any],
    connection: str,
    profile_digest: str,
    request: ExternalGraphIngestionRequest,
    snapshot_identity_complete: bool,
    identity_key: str,
    prepared: list[dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> ChangeEnvelope | None:
    if use_cdc:
        return _cdc_marker(
            next_cursor=next_cursor,
            current_cursor=current_cursor,
            incomplete=incomplete,
            tenant=tenant,
            source_alias=source_alias,
            resolved_profile=resolved_profile,
            connection=connection,
            profile_digest=profile_digest,
        )
    return _snapshot_marker(
        identity_key=identity_key,
        source_alias=source_alias,
        profile_digest=profile_digest,
        prepared=prepared,
        outgoing=outgoing,
        incomplete=incomplete,
        request=request,
        snapshot_identity_complete=snapshot_identity_complete,
        tenant=tenant,
        resolved_profile=resolved_profile,
        connection=connection,
    )


def _ingest_marker(
    authority_engine: Any, marker: ChangeEnvelope, statuses: Counter[str]
) -> int:
    marker_result = ingest_envelope(authority_engine, marker)
    marker_status = str(marker_result.get("status") or "unknown")
    statuses[marker_status] += 1
    return 1 if marker_status not in {"success", "skipped"} else 0


def _build_result(
    *,
    incomplete: int,
    source_incomplete: bool,
    source_alias: str,
    connection: str,
    prepared: list[dict[str, Any]],
    edges: int,
    delete_ids: list[str],
    use_cdc: bool,
    snapshot_identity_complete: bool,
    statuses: Counter[str],
    profile_digest: str,
    privacy_redactions: int,
    privacy_counts: Counter[str],
) -> dict[str, Any]:
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

    bounds = _validate_request(request)
    _certify_external_graph_connector()

    profile_ref = _resolve_profile_ref(request, bounds.connection)
    resolved_profile = _resolve_profile(
        profile_ref,
        profile=profile,
        resolver=profile_resolver,
    )
    _validate_profile_freshness(
        resolved_profile,
        profile_arg=profile,
        source_alias=bounds.source_alias,
        request=request,
        bounds=bounds,
    )
    identity_key = _resolve_identity_key(
        resolved_profile,
        connection=bounds.connection,
        resolver=profile_resolver,
    )
    mappings = _resolve_mappings(resolved_profile)
    access = _validate_access_classification(resolved_profile, bounds.classification)
    external_engine = _resolve_external_engine(registry, bounds.connection)
    _verify_schema_digest(
        resolved_profile, registry, bounds.connection, external_engine
    )

    privacy = privacy_guard or PersistencePrivacyGuard()
    payload_budget = _PayloadBudget(
        max_row_bytes=bounds.max_row_bytes,
        max_total_bytes=bounds.max_total_bytes,
        max_nesting_depth=bounds.max_nesting_depth,
        max_collection_items=bounds.max_collection_items,
    )
    reader, use_cdc = _setup_reader(external_engine, bounds.sync_mode)

    if use_cdc:
        node_rows, delete_keys, edge_rows, current_cursor, next_cursor = _read_cdc_rows(
            authority_engine,
            reader,
            source_alias=bounds.source_alias,
            page_size=bounds.page_size,
            max_pages=bounds.max_pages,
            max_records=bounds.max_records,
            privacy=privacy,
            budget=payload_budget,
        )
    else:
        node_rows, edge_rows = _read_snapshot_rows(
            external_engine,
            node_query=mappings.node_query,
            edge_query=mappings.edge_query,
            variables=dict(request.variables or {}),
            page_size=bounds.page_size,
            max_pages=bounds.max_pages,
            max_records=bounds.max_records,
            budget=payload_budget,
            privacy=privacy,
        )
        delete_keys = []
        current_cursor = None
        next_cursor = None

    type_map = resolved_profile.get("type_map")
    if not isinstance(type_map, dict):
        type_map = {}

    (
        internal_ids,
        prepared,
        node_privacy_counts,
        node_redactions,
        node_identity_complete,
    ) = _prepare_node_rows(
        node_rows,
        node_mapping=mappings.node_mapping,
        type_map=type_map,
        use_cdc=use_cdc,
        source_alias=bounds.source_alias,
        identity_key=identity_key,
        node_allowlist=mappings.node_allowlist,
        privacy=privacy,
    )
    outgoing, edge_privacy_counts, edge_redactions, edge_identity_complete = (
        _prepare_edges(
            mappings.edge_query,
            mappings.edge_mapping,
            edge_rows,
            internal_ids=internal_ids,
            edge_allowlist=mappings.edge_allowlist,
            privacy=privacy,
            resolved_profile=resolved_profile,
        )
    )
    privacy_counts = node_privacy_counts
    privacy_counts.update(edge_privacy_counts)
    privacy_redactions = node_redactions + edge_redactions
    snapshot_identity_complete = node_identity_complete and edge_identity_complete

    delete_ids = _compute_delete_ids(
        delete_keys,
        identity_key=identity_key,
        source_alias=bounds.source_alias,
        prepared=prepared,
    )
    _guard_empty_snapshot(
        use_cdc=use_cdc,
        snapshot_identity_complete=snapshot_identity_complete,
        request=request,
        prepared=prepared,
    )
    profile_digest = _compute_profile_digest(
        node_query=mappings.node_query,
        node_mapping=mappings.node_mapping,
        edge_query=mappings.edge_query,
        edge_mapping=mappings.edge_mapping,
        resolved_profile=resolved_profile,
        request=request,
        bounds=bounds,
    )
    if request.dry_run:
        return _build_dry_run_result(
            source_alias=bounds.source_alias,
            connection=bounds.connection,
            prepared=prepared,
            outgoing=outgoing,
            delete_ids=delete_ids,
            use_cdc=use_cdc,
            snapshot_identity_complete=snapshot_identity_complete,
            profile_digest=profile_digest,
            privacy_redactions=privacy_redactions,
            privacy_counts=privacy_counts,
        )

    envelopes, edges = _build_node_envelopes(
        prepared,
        outgoing=outgoing,
        source_alias=bounds.source_alias,
        identity_key=identity_key,
        profile_digest=profile_digest,
        tenant=bounds.tenant,
        resolved_profile=resolved_profile,
        access=access,
        classification=bounds.classification,
        retention=bounds.retention,
        request=request,
        connection=bounds.connection,
    )
    envelopes.extend(
        _build_delete_envelopes(
            delete_ids,
            tenant=bounds.tenant,
            source_alias=bounds.source_alias,
            identity_key=identity_key,
            next_cursor=next_cursor,
            resolved_profile=resolved_profile,
            classification=bounds.classification,
            retention=bounds.retention,
            request=request,
            connection=bounds.connection,
            profile_digest=profile_digest,
        )
    )

    statuses = _ingest_envelopes(authority_engine, envelopes)
    incomplete = _compute_incomplete(statuses)

    marker = _build_marker(
        use_cdc=use_cdc,
        next_cursor=next_cursor,
        current_cursor=current_cursor,
        incomplete=incomplete,
        tenant=bounds.tenant,
        source_alias=bounds.source_alias,
        resolved_profile=resolved_profile,
        connection=bounds.connection,
        profile_digest=profile_digest,
        request=request,
        snapshot_identity_complete=snapshot_identity_complete,
        identity_key=identity_key,
        prepared=prepared,
        outgoing=outgoing,
    )
    if marker is not None:
        incomplete += _ingest_marker(authority_engine, marker, statuses)

    source_incomplete = bool(not use_cdc and not snapshot_identity_complete)
    return _build_result(
        incomplete=incomplete,
        source_incomplete=source_incomplete,
        source_alias=bounds.source_alias,
        connection=bounds.connection,
        prepared=prepared,
        edges=edges,
        delete_ids=delete_ids,
        use_cdc=use_cdc,
        snapshot_identity_complete=snapshot_identity_complete,
        statuses=statuses,
        profile_digest=profile_digest,
        privacy_redactions=privacy_redactions,
        privacy_counts=privacy_counts,
    )
