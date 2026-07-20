from __future__ import annotations

"""Canonical MCP tool-schema contracts for governed connector ingestion.

Connector presets are configuration; the live MCP ``list_tools`` response is
the served contract.  This module provides a dependency-light, deterministic
projection and fingerprint so ingestion can compare the two *before* the first
source call.  Descriptions and presentation-only JSON-Schema annotations are
excluded from the compatibility projection, while an optional exact digest can
pin the complete normalized input schema for release certification.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "LiveToolContract",
    "ToolSchemaContractError",
    "canonical_input_schema",
    "compatibility_fingerprint",
    "schema_fingerprint",
    "validate_live_tool_contract",
]


class ToolSchemaContractError(RuntimeError):
    """The live MCP tool fleet differs from the signed connector contract."""


@dataclass(frozen=True)
class LiveToolContract:
    """Validated live tool identity and deterministic schema fingerprints."""

    name: str
    schema_sha256: str
    compatibility_sha256: str


_PRESENTATION_KEYS = frozenset(
    {
        "$comment",
        "description",
        "examples",
        "title",
    }
)
_RUNTIME_CONFIGURATION_KEYS = frozenset({"default"})


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _field(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def canonical_input_schema(
    tool: Any, *, include_presentation: bool = True
) -> dict[str, Any]:
    """Return a stable JSON-compatible input schema for one MCP tool object."""

    raw = _field(tool, "inputSchema", "input_schema") or {}
    schema = _jsonable(raw)
    if not isinstance(schema, dict):
        raise ToolSchemaContractError("live MCP tool input schema is not an object")

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalize(value[key])
                for key in sorted(value)
                if key not in _RUNTIME_CONFIGURATION_KEYS
                and (include_presentation or key not in _PRESENTATION_KEYS)
            }
        if isinstance(value, list):
            # JSON-Schema ``required`` and ``enum`` order is not semantic.
            normalized = [normalize(item) for item in value]
            if all(isinstance(item, str) for item in normalized):
                return sorted(normalized)
            return normalized
        return value

    return normalize(schema)


def schema_fingerprint(name: str, schema: Mapping[str, Any]) -> str:
    """Hash an exact tool name/input schema with a versioned domain."""

    payload = json.dumps(
        {"name": str(name), "input_schema": _jsonable(schema)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(
        b"agent-utilities:mcp-tool-schema:v1\x00" + payload
    ).hexdigest()


def compatibility_fingerprint(name: str, schema: Mapping[str, Any]) -> str:
    """Hash the structural tool contract, excluding presentation/runtime values."""

    payload = json.dumps(
        {"name": str(name), "input_schema": _jsonable(schema)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(
        b"agent-utilities:mcp-tool-schema-compat:v1\x00" + payload
    ).hexdigest()


def _tool_list(result: Any) -> list[Any]:
    tools = _field(result, "tools")
    if tools is None:
        tools = result
    if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes, Mapping)):
        return list(tools)
    raise ToolSchemaContractError("MCP list_tools response has no tool list")


def _property_type(schema: Mapping[str, Any], name: str) -> str | None:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    spec = properties.get(name)
    if not isinstance(spec, Mapping):
        return None
    declared = spec.get("type")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list):
        non_null = sorted(str(item) for item in declared if item != "null")
        return "|".join(non_null) if non_null else "null"
    # Pydantic/FastMCP may express Optional[T] through anyOf.
    any_of = spec.get("anyOf")
    if isinstance(any_of, list):
        types = sorted(
            str(item.get("type"))
            for item in any_of
            if isinstance(item, Mapping) and item.get("type") not in (None, "null")
        )
        return "|".join(types) if types else None
    return None


def validate_live_tool_contract(
    list_tools_result: Any,
    *,
    tool_name: str,
    expected_schema_sha256: str = "",
    required_argument_types: Mapping[str, str] | None = None,
) -> LiveToolContract:
    """Validate one live MCP tool and return its exact/compatibility digests.

    ``required_argument_types`` is the signed compatibility contract derived
    from the connector preset (for example ``action: string`` and
    ``params_json: string``).  ``expected_schema_sha256`` optionally pins the
    structural compatibility schema. Descriptions, titles, examples, and
    runtime defaults may evolve without invalidating a release; argument
    names, types, requirements, and constraints remain fail-closed.
    """

    matches = [
        tool
        for tool in _tool_list(list_tools_result)
        if str(_field(tool, "name") or "") == tool_name
    ]
    if not matches:
        raise ToolSchemaContractError(
            f"live MCP server does not expose signed tool {tool_name!r}"
        )
    if len(matches) != 1:
        raise ToolSchemaContractError(
            f"live MCP server exposes duplicate tool name {tool_name!r}"
        )

    tool = matches[0]
    exact_schema = canonical_input_schema(tool, include_presentation=True)
    compatibility_schema = canonical_input_schema(tool, include_presentation=False)
    exact_digest = schema_fingerprint(tool_name, exact_schema)
    compatibility_digest = compatibility_fingerprint(tool_name, compatibility_schema)

    expected = (expected_schema_sha256 or "").strip().lower()
    if expected and compatibility_digest != expected:
        raise ToolSchemaContractError(
            f"live MCP tool schema fingerprint differs for {tool_name!r}"
        )

    for argument, expected_type in sorted((required_argument_types or {}).items()):
        actual_type = _property_type(compatibility_schema, argument)
        if actual_type is None:
            raise ToolSchemaContractError(
                f"live MCP tool {tool_name!r} is missing signed argument {argument!r}"
            )
        accepted = {part for part in actual_type.split("|") if part}
        if expected_type not in accepted:
            raise ToolSchemaContractError(
                f"live MCP tool {tool_name!r} argument {argument!r} has type "
                f"{actual_type!r}, expected {expected_type!r}"
            )

    return LiveToolContract(
        name=tool_name,
        schema_sha256=exact_digest,
        compatibility_sha256=compatibility_digest,
    )
