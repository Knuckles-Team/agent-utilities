"""Machine-checked authority contract for AU relational projections.

The engine-native fleet catalog, usage analytics store, and operational state
store are separate domains.  This module is deliberately the executable part
of the authority ADR: it loads the checked-in JSON map, extracts the declared
table schemas from their owning modules, and rejects missing, duplicate, or
conflicting ownership before a caller can treat the map as evidence.

The map describes writes; read models are explicitly non-writable.  No table
in one domain may acquire a second writer in another domain.  SQLite/Postgres
backend differences remain metadata in the map, while this gate checks the
structural facts (domain/table/field ownership and schema drift).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MAP_PATH = Path(__file__).with_name("relational_authority.json")
_DOMAIN_NAMES = frozenset({"engine_fleet_catalog", "usage_store", "state_store"})
_EXPECTED_AUTHORITIES = {
    "engine_fleet_catalog": "epistemic_graph_engine_sql",
    "usage_store": "usage_backend",
    "state_store": "state_backend",
}
_EXPECTED_SCHEMA_SOURCES = {
    "engine_fleet_catalog": "fleet_catalog",
    "usage_store": "usage_store",
    "state_store": "state_store",
}
_EXPECTED_READ_MODEL_FIELDS = {
    "registry_page": {"status", "kind", "items", "count", "next_cursor"},
    "usage_summary": {"session_count", "totals", "cache_hit_rate"},
    "fleet_topology": {"sessions", "workers", "domains", "total"},
}
_EXPECTED_READ_MODEL_NAMES = frozenset(_EXPECTED_READ_MODEL_FIELDS)
_DISCOVERY_BOUND_TABLES = frozenset(
    {"mcp_server_discovery", "mcp_tools", "mcp_prompts", "mcp_resources", "skills"}
)
_DISCOVERY_BOUND_FIELDS = frozenset(
    {
        "discovery_authority_kind",
        "discovery_principal",
        "discovery_grant_digest",
    }
)
_PLACEMENT_STORE_CONTRACT = {
    "postgres_control_plane": {
        "kind": "postgresql",
        "authority": "transactional_control_plane",
        "classes": frozenset(
            {
                "identity",
                "lifecycle",
                "version",
                "release",
                "policy",
                "approval",
                "configuration",
                "quota",
                "idempotency",
                "outbox",
                "audit_reference",
            }
        ),
    },
    "graphos": {
        "kind": "epistemic_graph",
        "authority": "semantic_knowledge",
        "classes": frozenset(
            {
                "semantic_knowledge",
                "semantic_claim",
                "semantic_relationship",
                "semantic_evidence",
            }
        ),
    },
    "native_work_item": {
        "kind": "epistemic_graph_native_work_item",
        "authority": "leases_and_fences",
        "classes": frozenset({"work_lease", "work_fence", "work_retry"}),
    },
    "artifact_store": {
        "kind": "object_artifact_store",
        "authority": "artifact_payload",
        "classes": frozenset({"artifact_payload"}),
    },
    "vector_store": {
        "kind": "tenant_graph_vector_store",
        "authority": "vector_payload",
        "classes": frozenset({"vector_payload"}),
    },
    "observability_store": {
        "kind": "metrics_trace_log_store",
        "authority": "observability_payload",
        "classes": frozenset({"observability_payload"}),
    },
    "secret_provider": {
        "kind": "secret_token_provider",
        "authority": "secret_material",
        "classes": frozenset({"secret_material"}),
    },
}
_PLACEMENT_FIELD_OWNERS = {
    "tenant_identity": "postgres_control_plane",
    "principal_authorization": "postgres_control_plane",
    "registry_lifecycle": "postgres_control_plane",
    "registry_version": "postgres_control_plane",
    "release_activation": "postgres_control_plane",
    "policy_approval": "postgres_control_plane",
    "configuration_reference": "postgres_control_plane",
    "quota_idempotency": "postgres_control_plane",
    "outbox_projection_event": "postgres_control_plane",
    "audit_resolution": "postgres_control_plane",
    "work_item_lease": "native_work_item",
    "work_item_fence": "native_work_item",
    "work_item_retry": "native_work_item",
    "semantic_claim": "graphos",
    "semantic_relationship": "graphos",
    "semantic_evidence": "graphos",
    "artifact_payload": "artifact_store",
    "vector_payload": "vector_store",
    "observability_payload": "observability_store",
    "secret_material": "secret_provider",
}
_PLACEMENT_EVENT_OWNERS = {
    "control_plane_mutation": "postgres_control_plane",
    "work_item_lease": "native_work_item",
    "semantic_observation": "graphos",
    "artifact_committed": "artifact_store",
    "vector_indexed": "vector_store",
    "observability_recorded": "observability_store",
    "secret_reference_rotated": "secret_provider",
}
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "password",
        "passphrase",
        "private_key",
        "private_key_value",
        "client_secret",
        "client_secret_value",
        "access_token",
        "refresh_token",
        "api_key",
        "api_key_value",
        "bearer_token",
        "cookie_value",
        "authorization_header",
        "secret_value",
        "secret_key_value",
        "credential_value",
        "authorization",
        "credentials",
        "credential",
        "token",
        "secret",
        "private_key_material",
        "oauth_token",
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.IGNORECASE,
)


class AuthorityMapError(ValueError):
    """Raised when the authority map cannot prove one-writer ownership."""


def load_authority_map(path: str | Path = _MAP_PATH) -> dict[str, Any]:
    """Load and type-check the durable JSON authority map."""

    candidate = Path(path)
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuthorityMapError("relational authority map is unreadable") from exc
    if not isinstance(data, dict):
        raise AuthorityMapError("relational authority map must be a JSON object")
    return data


def _split_columns(body: str) -> list[str]:
    """Return top-level column/constraint fragments from one CREATE body."""

    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(body):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            parts.append(body[start:index])
            start = index + 1
    parts.append(body[start:])
    return parts


def declared_tables(sql: str) -> dict[str, frozenset[str]]:
    """Extract table columns from a SQL DDL string without executing it.

    This is a small structural parser, not a grep gate: it balances nested
    type expressions and ignores table-level constraints.  It intentionally
    accepts only the DDL shape owned by the three schema modules.
    """

    result: dict[str, frozenset[str]] = {}
    for match in _CREATE_TABLE.finditer(sql):
        table = match.group(1)
        body_start = match.end()
        depth = 1
        quote: str | None = None
        body_end = body_start
        while body_end < len(sql) and depth:
            char = sql[body_end]
            if quote:
                if char == quote:
                    quote = None
            elif char in {"'", '"', "`"}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            body_end += 1
        if depth:
            raise AuthorityMapError(f"unterminated DDL for table {table}")
        columns: list[str] = []
        for fragment in _split_columns(sql[body_start : body_end - 1]):
            fragment = re.sub(r"--[^\n]*", "", fragment).strip()
            if not fragment:
                continue
            first = fragment.split(None, 1)[0].strip('"`')
            if first.upper() in {
                "PRIMARY",
                "FOREIGN",
                "UNIQUE",
                "CHECK",
                "CONSTRAINT",
                "EXCLUDE",
            }:
                continue
            if _IDENTIFIER.fullmatch(first) is None:
                raise AuthorityMapError(
                    f"unsupported column declaration in {table}: {first!r}"
                )
            columns.append(first)
        if table in result:
            raise AuthorityMapError(f"duplicate declared table: {table}")
        result[table] = frozenset(columns)
    return result


def declared_schemas() -> dict[str, dict[str, frozenset[str]]]:
    """Return schemas from the three owning source modules.

    Only authoritative base tables are compared.  Backend-specific FTS/GIN
    indexes and tsvector columns are read projections, not independently
    writable authority domains, and are documented in the ADR/map metadata.
    """

    from agent_utilities.core import sessions
    from agent_utilities.knowledge_graph.core import fleet_catalog_tables
    from agent_utilities.usage import schema as usage_schema

    return {
        "engine_fleet_catalog": {
            table: columns
            for table, ddl in fleet_catalog_tables._DDL.items()
            for table, columns in declared_tables(ddl).items()
        },
        "usage_store": declared_tables(usage_schema._BASE_TABLES),
        "state_store": declared_tables(sessions._SQLITE_DDL),
    }


def _string_list(value: Any, *, path: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{path} must be a list of strings")
        return []
    return value


def _normalise_field_name(value: str) -> str:
    """Normalise a schema/event field for the sensitive-name deny list."""

    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", snake.lower()).strip("_")


def _is_sensitive_field(value: str) -> bool:
    return _normalise_field_name(value) in _SENSITIVE_FIELD_NAMES


def _validate_placement_contract(data: Mapping[str, Any], errors: list[str]) -> None:
    """Validate the cross-system authority/placement section of the map.

    This is intentionally independent from the concrete SQL schema parser:
    placement records cover systems that are not relational tables (GraphOS,
    native WorkItems and the specialized payload stores).  The expected IDs
    below are the closed vocabulary of this contract; adding a new authority
    requires an explicit map and documentation change rather than silently
    creating a second writer.
    """

    raw_contract = data.get("authority_placement")
    if not isinstance(raw_contract, dict):
        errors.append("authority_placement must be an object")
        return
    if raw_contract.get("version") != 1:
        errors.append("unsupported or missing authority-placement version")

    raw_stores = raw_contract.get("stores")
    if not isinstance(raw_stores, list):
        errors.append("authority_placement.stores must be a list")
        return
    stores: dict[str, dict[str, Any]] = {}
    seen_store_classes: dict[str, str] = {}
    all_expected_classes = set().union(
        *(set(contract["classes"]) for contract in _PLACEMENT_STORE_CONTRACT.values())
    )
    for index, raw_store in enumerate(raw_stores):
        path = f"authority_placement.stores[{index}]"
        if not isinstance(raw_store, dict):
            errors.append(f"{path} must be an object")
            continue
        store_id = raw_store.get("id")
        if not isinstance(store_id, str) or store_id not in _PLACEMENT_STORE_CONTRACT:
            errors.append(f"{path}.id is not a supported placement store")
            continue
        if store_id in stores:
            errors.append(f"duplicate placement store: {store_id}")
        stores[store_id] = raw_store
        expected = _PLACEMENT_STORE_CONTRACT[store_id]
        if raw_store.get("kind") != expected["kind"]:
            errors.append(f"placement store kind conflicts: {store_id}")
        if raw_store.get("authority") != expected["authority"]:
            errors.append(f"placement store authority conflicts: {store_id}")
        classes = _string_list(
            raw_store.get("authoritative_classes"),
            path=f"{path}.authoritative_classes",
            errors=errors,
        )
        prohibited = _string_list(
            raw_store.get("prohibited_classes"),
            path=f"{path}.prohibited_classes",
            errors=errors,
        )
        if len(classes) != len(set(classes)):
            errors.append(f"duplicate placement classes: {store_id}")
        if set(classes) & set(prohibited):
            errors.append(f"conflicting placement classes: {store_id}")
        expected_classes = set(expected["classes"])
        if set(classes) != expected_classes:
            errors.append(
                f"placement class drift: {store_id} "
                f"map={sorted(classes)} expected={sorted(expected_classes)}"
            )
        expected_prohibited = all_expected_classes - expected_classes
        if set(prohibited) != expected_prohibited:
            errors.append(
                f"placement prohibition drift: {store_id} "
                f"map={sorted(prohibited)} expected={sorted(expected_prohibited)}"
            )
        for class_name in classes:
            previous = seen_store_classes.get(class_name)
            if previous is not None:
                errors.append(
                    f"duplicate placement authority class: {class_name} "
                    f"({previous}, {store_id})"
                )
            else:
                seen_store_classes[class_name] = store_id

    missing_stores = set(_PLACEMENT_STORE_CONTRACT) - set(stores)
    errors.extend(
        f"missing placement store: {store_id}" for store_id in sorted(missing_stores)
    )

    raw_records = raw_contract.get("records")
    if not isinstance(raw_records, list):
        errors.append("authority_placement.records must be a list")
        raw_records = []
    seen_records: set[str] = set()
    seen_fields: dict[str, str] = {}
    for index, raw_record in enumerate(raw_records):
        path = f"authority_placement.records[{index}]"
        if not isinstance(raw_record, dict):
            errors.append(f"{path} must be an object")
            continue
        record_id = raw_record.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"{path}.id is missing")
        elif record_id in seen_records:
            errors.append(f"duplicate placement record: {record_id}")
        else:
            seen_records.add(record_id)
        store_id = raw_record.get("store")
        if store_id not in _PLACEMENT_STORE_CONTRACT:
            errors.append(f"{path}.store is invalid")
        fields = _string_list(
            raw_record.get("authority_fields"),
            path=f"{path}.authority_fields",
            errors=errors,
        )
        if not fields:
            errors.append(f"{path}.authority_fields must not be empty")
        if len(fields) != len(set(fields)):
            errors.append(f"duplicate authority fields: {record_id!r}")
        for field in fields:
            expected_store = _PLACEMENT_FIELD_OWNERS.get(field)
            if expected_store is None:
                errors.append(f"unknown placement authority field: {field}")
            elif store_id != expected_store:
                errors.append(
                    f"conflicting placement authority: {field} "
                    f"({store_id}, expected {expected_store})"
                )
            previous = seen_fields.get(field)
            if previous is not None:
                errors.append(
                    f"duplicate placement field authority: {field} "
                    f"({previous}, {record_id})"
                )
            else:
                seen_fields[field] = str(record_id)

    missing_fields = set(_PLACEMENT_FIELD_OWNERS) - set(seen_fields)
    errors.extend(
        f"missing placement authority field: {field}"
        for field in sorted(missing_fields)
    )
    unknown_fields = set(seen_fields) - set(_PLACEMENT_FIELD_OWNERS)
    errors.extend(
        f"unknown placement authority field: {field}"
        for field in sorted(unknown_fields)
    )

    raw_events = raw_contract.get("events")
    if not isinstance(raw_events, list):
        errors.append("authority_placement.events must be a list")
        raw_events = []
    seen_events: set[str] = set()
    event_field_owners: dict[str, str] = {}
    for index, raw_event in enumerate(raw_events):
        path = f"authority_placement.events[{index}]"
        if not isinstance(raw_event, dict):
            errors.append(f"{path} must be an object")
            continue
        event_id = raw_event.get("id")
        if not isinstance(event_id, str) or event_id not in _PLACEMENT_EVENT_OWNERS:
            errors.append(f"{path}.id is not a supported placement event")
        elif event_id in seen_events:
            errors.append(f"duplicate placement event: {event_id}")
        else:
            seen_events.add(event_id)
        store_id = raw_event.get("store")
        if store_id not in _PLACEMENT_STORE_CONTRACT:
            errors.append(f"{path}.store is invalid")
        expected_event_store = _PLACEMENT_EVENT_OWNERS.get(str(event_id))
        if expected_event_store is not None and store_id != expected_event_store:
            errors.append(
                f"placement event authority conflicts: {event_id} "
                f"({store_id}, expected {expected_event_store})"
            )
        authority_fields = _string_list(
            raw_event.get("authority_fields"),
            path=f"{path}.authority_fields",
            errors=errors,
        )
        if not authority_fields:
            errors.append(f"{path}.authority_fields must not be empty")
        if len(authority_fields) != len(set(authority_fields)):
            errors.append(f"duplicate event authority fields: {event_id!r}")
        for field in authority_fields:
            expected_field_store = _PLACEMENT_FIELD_OWNERS.get(field)
            if expected_field_store is None:
                errors.append(f"unknown event authority field: {field}")
            elif store_id != expected_field_store:
                errors.append(
                    f"conflicting event writer: {field} "
                    f"({store_id}, expected {expected_field_store})"
                )
            previous = event_field_owners.get(field)
            if previous is not None and previous != str(store_id):
                errors.append(
                    f"duplicate event authority: {field} ({previous}, {store_id})"
                )
            else:
                event_field_owners[field] = str(store_id)
        payload_fields = _string_list(
            raw_event.get("payload_fields"),
            path=f"{path}.payload_fields",
            errors=errors,
        )
        if len(payload_fields) != len(set(payload_fields)):
            errors.append(f"duplicate event payload fields: {event_id!r}")
        for field in payload_fields:
            if _is_sensitive_field(field):
                errors.append(f"secret-bearing event field: {event_id}.{field}")
    missing_event_fields = set(_PLACEMENT_FIELD_OWNERS) - set(event_field_owners)
    errors.extend(
        f"missing event authority field: {field}"
        for field in sorted(missing_event_fields)
    )
    missing_events = set(_PLACEMENT_EVENT_OWNERS) - seen_events
    errors.extend(
        f"missing placement event: {event_id}" for event_id in sorted(missing_events)
    )


def _validate_table_entry(
    raw_table: Any,
    table_path: str,
    *,
    domain_name: str,
    errors: list[str],
    table_names: set[str],
    authority_fields: dict[tuple[str, str, str], str],
) -> None:
    if not isinstance(raw_table, dict):
        errors.append(f"{table_path} must be an object")
        return
    table = raw_table.get("name")
    if not isinstance(table, str) or _IDENTIFIER.fullmatch(table) is None:
        errors.append(f"{table_path}.name is invalid")
        return
    if table in table_names:
        errors.append(f"duplicate authority table: {domain_name}.{table}")
    table_names.add(table)
    authoritative = _string_list(
        raw_table.get("authoritative_fields"),
        path=f"{table_path}.authoritative_fields",
        errors=errors,
    )
    derived = _string_list(
        raw_table.get("derived_fields"),
        path=f"{table_path}.derived_fields",
        errors=errors,
    )
    if set(authoritative) & set(derived):
        errors.append(f"conflicting field roles: {domain_name}.{table}")
    if len(authoritative) != len(set(authoritative)):
        errors.append(f"duplicate authoritative fields: {domain_name}.{table}")
    if len(derived) != len(set(derived)):
        errors.append(f"duplicate derived fields: {domain_name}.{table}")
    prohibited = set(
        _string_list(
            raw_table.get("prohibited_dual_write_domains"),
            path=f"{table_path}.prohibited_dual_write_domains",
            errors=errors,
        )
    )
    expected_prohibited = _DOMAIN_NAMES - {domain_name}
    if prohibited != expected_prohibited:
        errors.append(
            f"{domain_name}.{table} does not prohibit every other write domain"
        )
    for field in authoritative:
        if _is_sensitive_field(field):
            errors.append(
                f"secret-bearing declared column: {domain_name}.{table}.{field}"
            )
        key = (domain_name, table, field)
        if key in authority_fields:
            errors.append(f"duplicate field authority: {'.'.join(key)}")
        authority_fields[key] = "authoritative"
    for field in derived:
        if _is_sensitive_field(field):
            errors.append(
                f"secret-bearing declared column: {domain_name}.{table}.{field}"
            )
        key = (domain_name, table, field)
        if key in authority_fields:
            errors.append(f"duplicate field authority: {'.'.join(key)}")
        authority_fields[key] = "derived"


def _validate_domain_entry(
    raw_domain: Any,
    path: str,
    *,
    errors: list[str],
    seen_domains: set[str],
    seen_authorities: set[str],
    seen_schema_sources: set[str],
    domain_tables: dict[str, set[str]],
    authority_fields: dict[tuple[str, str, str], str],
) -> None:
    if not isinstance(raw_domain, dict):
        errors.append(f"{path} must be an object")
        return
    name = raw_domain.get("name")
    if not isinstance(name, str) or name not in _DOMAIN_NAMES:
        errors.append(f"{path}.name is not a supported authority domain")
        return
    if name in seen_domains:
        errors.append(f"duplicate authority domain: {name}")
    seen_domains.add(name)
    if not isinstance(raw_domain.get("authority"), str) or not raw_domain.get(
        "authority"
    ):
        errors.append(f"{path}.authority is missing")
    elif raw_domain["authority"] != _EXPECTED_AUTHORITIES.get(name):
        errors.append(f"{path}.authority conflicts with the owning domain")
    elif raw_domain["authority"] in seen_authorities:
        errors.append(f"duplicate authority owner: {raw_domain['authority']}")
    else:
        seen_authorities.add(raw_domain["authority"])
    if not isinstance(raw_domain.get("schema_source"), str):
        errors.append(f"{path}.schema_source is missing")
    elif raw_domain["schema_source"] != _EXPECTED_SCHEMA_SOURCES.get(name):
        errors.append(f"{path}.schema_source conflicts with the owning module")
    elif raw_domain["schema_source"] in seen_schema_sources:
        errors.append(f"duplicate schema source: {raw_domain['schema_source']}")
    else:
        seen_schema_sources.add(raw_domain["schema_source"])
    tables = raw_domain.get("tables")
    if not isinstance(tables, list) or not tables:
        errors.append(f"{path}.tables must be a non-empty list")
        return
    table_names: set[str] = set()
    domain_tables[name] = table_names
    for table_index, raw_table in enumerate(tables):
        table_path = f"{path}.tables[{table_index}]"
        _validate_table_entry(
            raw_table,
            table_path,
            domain_name=name,
            errors=errors,
            table_names=table_names,
            authority_fields=authority_fields,
        )


def _validate_domains(
    raw_domains: list[Any], errors: list[str]
) -> tuple[set[str], dict[str, set[str]]]:
    seen_domains: set[str] = set()
    seen_authorities: set[str] = set()
    seen_schema_sources: set[str] = set()
    domain_tables: dict[str, set[str]] = {}
    authority_fields: dict[tuple[str, str, str], str] = {}
    for domain_index, raw_domain in enumerate(raw_domains):
        path = f"domains[{domain_index}]"
        _validate_domain_entry(
            raw_domain,
            path,
            errors=errors,
            seen_domains=seen_domains,
            seen_authorities=seen_authorities,
            seen_schema_sources=seen_schema_sources,
            domain_tables=domain_tables,
            authority_fields=authority_fields,
        )
    return seen_domains, domain_tables


def _validate_schema_drift(
    raw_domains: list[Any],
    domain_tables: dict[str, set[str]],
    schemas: Mapping[str, Mapping[str, frozenset[str]]] | None,
    errors: list[str],
) -> None:
    if schemas is None:
        try:
            schemas = declared_schemas()
        except Exception as exc:  # noqa: BLE001 - gate must fail closed
            errors.append(f"declared schema unavailable: {type(exc).__name__}")
            schemas = {}
    for domain in _DOMAIN_NAMES:
        declared = dict(schemas.get(domain, {}))
        if domain not in domain_tables:
            continue
        mapped_tables = domain_tables[domain]
        for table in sorted(set(declared) - mapped_tables):
            errors.append(f"undeclared table drift: {domain}.{table}")
        for table in sorted(mapped_tables - set(declared)):
            errors.append(f"missing declared table: {domain}.{table}")
        mapped_by_name = {
            table["name"]: table
            for raw_domain in raw_domains
            if isinstance(raw_domain, dict) and raw_domain.get("name") == domain
            for table in raw_domain.get("tables", [])
            if isinstance(table, dict) and isinstance(table.get("name"), str)
        }
        for table, columns in declared.items():
            for column in columns:
                if _is_sensitive_field(column):
                    errors.append(
                        f"secret-bearing schema column: {domain}.{table}.{column}"
                    )
            entry = mapped_by_name.get(table)
            if entry is None:
                continue
            mapped_columns = set(entry.get("authoritative_fields", [])) | set(
                entry.get("derived_fields", [])
            )
            if mapped_columns != set(columns):
                errors.append(
                    f"schema drift: {domain}.{table} map={sorted(mapped_columns)} "
                    f"declared={sorted(columns)}"
                )


def _validate_engine_discovery_bindings(
    raw_domains: list[Any], errors: list[str]
) -> None:
    engine_domain = next(
        (
            raw_domain
            for raw_domain in raw_domains
            if isinstance(raw_domain, dict)
            and raw_domain.get("name") == "engine_fleet_catalog"
        ),
        None,
    )
    if isinstance(engine_domain, dict):
        engine_tables = {
            table.get("name"): table
            for table in engine_domain.get("tables", [])
            if isinstance(table, dict) and isinstance(table.get("name"), str)
        }
        for table_name in sorted(_DISCOVERY_BOUND_TABLES):
            table = engine_tables.get(table_name)
            if table is None:
                continue
            bound_fields = set(table.get("authoritative_fields", [])) | set(
                table.get("derived_fields", [])
            )
            missing_binding = _DISCOVERY_BOUND_FIELDS - bound_fields
            errors.extend(
                f"missing discovery binding field: engine_fleet_catalog.{table_name}.{field}"
                for field in sorted(missing_binding)
            )
        desired = engine_tables.get("mcp_servers")
        if desired is not None:
            desired_fields = set(desired.get("authoritative_fields", [])) | set(
                desired.get("derived_fields", [])
            )
            if desired_fields & _DISCOVERY_BOUND_FIELDS:
                errors.append(
                    "desired mcp_servers must not carry discovery binding fields"
                )


def _validate_read_models(
    data: Mapping[str, Any],
    domain_tables: dict[str, set[str]],
    errors: list[str],
) -> None:
    read_models = data.get("read_models")
    if not isinstance(read_models, list):
        errors.append("read_models must be a list")
        return
    seen_models: set[str] = set()
    for index, model in enumerate(read_models):
        path = f"read_models[{index}]"
        if not isinstance(model, dict):
            errors.append(f"{path} must be an object")
            continue
        name = model.get("name")
        if not isinstance(name, str) or not name or name in seen_models:
            errors.append(f"duplicate or missing read model: {name!r}")
        seen_models.add(str(name))
        owner = model.get("owner_domain")
        if owner not in _DOMAIN_NAMES:
            errors.append(f"{path}.owner_domain is invalid")
        if model.get("write_forbidden") is not True:
            errors.append(f"{path}.write_forbidden must be true")
        fields = _string_list(
            model.get("fields"), path=f"{path}.fields", errors=errors
        )
        if len(fields) != len(set(fields)):
            errors.append(f"duplicate read-model fields: {path}")
        if isinstance(name, str) and set(fields) != _EXPECTED_READ_MODEL_FIELDS.get(
            name, set()
        ):
            errors.append(f"read-model field drift: {name}")
        sources = _string_list(
            model.get("source_tables"),
            path=f"{path}.source_tables",
            errors=errors,
        )
        if not sources:
            errors.append(f"{path}.source_tables must not be empty")
        if owner in domain_tables and any(
            source not in domain_tables[owner] for source in sources
        ):
            errors.append(f"{path} references a table outside its owner domain")
    missing_models = _EXPECTED_READ_MODEL_NAMES - seen_models
    extra_models = seen_models - _EXPECTED_READ_MODEL_NAMES
    errors.extend(f"missing read model: {name}" for name in sorted(missing_models))
    errors.extend(f"unknown read model: {name}" for name in sorted(extra_models))


def validation_errors(
    data: Mapping[str, Any] | None = None,
    *,
    schemas: Mapping[str, Mapping[str, frozenset[str]]] | None = None,
) -> list[str]:
    """Return every structural authority-map violation.

    ``schemas`` is injectable for adversarial tests; production callers use
    :func:`declared_schemas`.  A non-empty result is a security failure and
    must be treated as unavailable rather than as a partial map.
    """

    errors: list[str] = []
    if data is None:
        try:
            data = load_authority_map()
        except AuthorityMapError as exc:
            return [str(exc)]
    if data.get("version") != 1:
        errors.append("unsupported or missing authority-map version")
    if data.get("contract") != "one-writer-per-domain":
        errors.append("authority map does not declare the one-writer contract")
    _validate_placement_contract(data, errors)

    raw_domains = data.get("domains")
    if not isinstance(raw_domains, list):
        return errors + ["domains must be a list"]

    seen_domains, domain_tables = _validate_domains(raw_domains, errors)

    missing_domains = _DOMAIN_NAMES - seen_domains
    errors.extend(
        f"missing authority domain: {name}" for name in sorted(missing_domains)
    )

    _validate_schema_drift(raw_domains, domain_tables, schemas, errors)
    _validate_engine_discovery_bindings(raw_domains, errors)
    _validate_read_models(data, domain_tables, errors)
    return errors


def validate_authority_map(
    data: Mapping[str, Any] | None = None,
    *,
    schemas: Mapping[str, Mapping[str, frozenset[str]]] | None = None,
) -> None:
    """Fail closed unless the durable map and declared schemas agree."""

    errors = validation_errors(data, schemas=schemas)
    if errors:
        raise AuthorityMapError("; ".join(errors))


def main() -> int:
    """CLI entry point for ``scripts/security/check_relational_authority.py``."""

    try:
        validate_authority_map()
    except AuthorityMapError as exc:
        print(f"relational authority gate: FAIL: {exc}")
        return 1
    print("relational authority gate: PASS")
    return 0


__all__ = [
    "AuthorityMapError",
    "declared_schemas",
    "declared_tables",
    "load_authority_map",
    "main",
    "validate_authority_map",
    "validation_errors",
]
