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

    raw_domains = data.get("domains")
    if not isinstance(raw_domains, list):
        return errors + ["domains must be a list"]
    seen_domains: set[str] = set()
    seen_authorities: set[str] = set()
    seen_schema_sources: set[str] = set()
    domain_tables: dict[str, set[str]] = {}
    authority_fields: dict[tuple[str, str, str], str] = {}
    for domain_index, raw_domain in enumerate(raw_domains):
        path = f"domains[{domain_index}]"
        if not isinstance(raw_domain, dict):
            errors.append(f"{path} must be an object")
            continue
        name = raw_domain.get("name")
        if not isinstance(name, str) or name not in _DOMAIN_NAMES:
            errors.append(f"{path}.name is not a supported authority domain")
            continue
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
            continue
        table_names: set[str] = set()
        domain_tables[name] = table_names
        for table_index, raw_table in enumerate(tables):
            table_path = f"{path}.tables[{table_index}]"
            if not isinstance(raw_table, dict):
                errors.append(f"{table_path} must be an object")
                continue
            table = raw_table.get("name")
            if not isinstance(table, str) or _IDENTIFIER.fullmatch(table) is None:
                errors.append(f"{table_path}.name is invalid")
                continue
            if table in table_names:
                errors.append(f"duplicate authority table: {name}.{table}")
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
                errors.append(f"conflicting field roles: {name}.{table}")
            if len(authoritative) != len(set(authoritative)):
                errors.append(f"duplicate authoritative fields: {name}.{table}")
            if len(derived) != len(set(derived)):
                errors.append(f"duplicate derived fields: {name}.{table}")
            prohibited = set(
                _string_list(
                    raw_table.get("prohibited_dual_write_domains"),
                    path=f"{table_path}.prohibited_dual_write_domains",
                    errors=errors,
                )
            )
            expected_prohibited = _DOMAIN_NAMES - {name}
            if prohibited != expected_prohibited:
                errors.append(
                    f"{name}.{table} does not prohibit every other write domain"
                )
            for field in authoritative:
                key = (name, table, field)
                if key in authority_fields:
                    errors.append(f"duplicate field authority: {'.'.join(key)}")
                authority_fields[key] = "authoritative"
            for field in derived:
                key = (name, table, field)
                if key in authority_fields:
                    errors.append(f"duplicate field authority: {'.'.join(key)}")
                authority_fields[key] = "derived"

    missing_domains = _DOMAIN_NAMES - seen_domains
    errors.extend(
        f"missing authority domain: {name}" for name in sorted(missing_domains)
    )

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

    read_models = data.get("read_models")
    if not isinstance(read_models, list):
        errors.append("read_models must be a list")
    else:
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
