#!/usr/bin/env python3
"""Generate and verify the current Epistemic Operations Protocol projections.

The packaged JSON Schema catalog is authoritative.  This gate proves:

* exactly twelve current-only schemas exist and reject unknown fields;
* credentials, endpoints, personal fields, and local-path fields are absent;
* every bound JSON object has identical ordered fields in Python and Rust;
* the generated strict Python/Rust client DTOs and manifests match the catalog;
* all cross-file references remain inside the catalog.

Use ``--self-only`` in an isolated agent-utilities checkout.  Workspace and
release validation must supply/discover epistemic-graph and therefore proves
the cross-repository projection too.  ``--write`` is the deterministic
regenerator; generated files must never be edited by hand.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = (
    ROOT / "agent_utilities" / "protocols" / "epistemic_operations" / "schemas" / "v1"
)
CATALOG_PATH = CATALOG_DIR / "catalog.json"
AU_GENERATED = (
    ROOT / "agent_utilities" / "protocols" / "epistemic_operations" / "_generated.py"
)
ENGINE_MANIFEST = Path("protocols/epistemic-operations/v1/manifest.json")
ENGINE_GENERATED = Path("crates/eg-types/src/epistemic_operations_manifest.rs")

REQUIRED_SCHEMAS = (
    "request_context",
    "mutation_batch",
    "change_envelope",
    "work_item",
    "artifact",
    "knowledge_batch",
    "analytics_job",
    "trace_outcome",
    "placement_route",
    "claim_work_item",
    "evidence_bundle",
    "operation_result",
)
REQUIRED_SCHEMA_VERSIONS = {
    "request_context": "2",
    "mutation_batch": "1",
    "change_envelope": "1",
    "work_item": "1",
    "artifact": "1",
    "knowledge_batch": "1",
    "analytics_job": "1",
    "trace_outcome": "1",
    "placement_route": "1",
    "claim_work_item": "1",
    "evidence_bundle": "1",
    "operation_result": "1",
}
FORBIDDEN_PROPERTY_NAMES = {
    "base_url",
    "ca_bundle_path",
    "credential",
    "credentials",
    "display_name",
    "email",
    "endpoint",
    "filesystem_path",
    "local_path",
    "name_of_person",
    "password",
    "secret",
    "token",
    "username",
}
LOCAL_PATH_MARKERS = (
    "c:\\users\\",
    "/home/",
    "/mnt/c/",
    "${workspacefolder}",
)


class ProtocolGateError(RuntimeError):
    """Raised when a protocol contract or generated projection drifts."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolGateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolGateError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolGateError(f"{path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _iter_nodes(value: Any, pointer: str = "#") -> Iterator[tuple[str, Any]]:
    yield pointer, value
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from _iter_nodes(child, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_nodes(child, f"{pointer}/{index}")


def _bound_nodes(schema_name: str, schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    def visit(node: Any, pointer: str) -> Iterator[dict[str, Any]]:
        if not isinstance(node, dict):
            return
        python_model = node.get("x-python-model")
        rust_type = node.get("x-rust-type")
        if bool(python_model) != bool(rust_type):
            raise ProtocolGateError(
                f"{schema_name}{pointer}: Python and Rust bindings must be paired"
            )
        if python_model:
            properties = node.get("properties")
            if not isinstance(properties, dict) or not properties:
                raise ProtocolGateError(
                    f"{schema_name}{pointer}: bound object has no properties"
                )
            yield {
                "schema": schema_name,
                "pointer": pointer,
                "python_model": str(python_model),
                "rust_type": str(rust_type),
                "fields": list(properties),
            }
        definitions = node.get("$defs")
        if isinstance(definitions, dict):
            for key in sorted(definitions):
                yield from visit(definitions[key], f"{pointer}/$defs/{key}")

    yield from visit(schema, "#")


def _validate_schema(
    name: str,
    schema_version: str,
    schema: dict[str, Any],
    schema_files: set[str],
) -> list[dict[str, Any]]:
    expected_id = f"urn:epistemic-operations:v{schema_version}:{name.replace('_', '-')}"
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ProtocolGateError(f"{name}: JSON Schema draft must be 2020-12")
    if schema.get("$id") != expected_id:
        raise ProtocolGateError(f"{name}: expected $id {expected_id!r}")
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        raise ProtocolGateError(f"{name}: root must be a closed object")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or schema.get("required") != list(properties):
        raise ProtocolGateError(
            f"{name}: every root property must be required in declaration order"
        )
    version_property = properties.get("schema_version")
    if (
        not isinstance(version_property, dict)
        or version_property.get("const") != schema_version
    ):
        raise ProtocolGateError(
            f"{name}: schema_version must be constant {schema_version!r}"
        )

    for pointer, node in _iter_nodes(schema):
        if isinstance(node, str):
            lowered = node.lower()
            if any(marker in lowered for marker in LOCAL_PATH_MARKERS):
                raise ProtocolGateError(
                    f"{name}{pointer}: local path marker is forbidden"
                )
            continue
        if not isinstance(node, dict):
            continue
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            target = ref.split("#", 1)[0]
            if target not in schema_files:
                raise ProtocolGateError(
                    f"{name}{pointer}: external reference {ref!r} is outside the catalog"
                )
        node_properties = node.get("properties")
        if isinstance(node_properties, dict):
            forbidden = FORBIDDEN_PROPERTY_NAMES.intersection(node_properties)
            if forbidden:
                raise ProtocolGateError(
                    f"{name}{pointer}: forbidden fields {sorted(forbidden)}"
                )
            if node.get("additionalProperties") is not False:
                raise ProtocolGateError(
                    f"{name}{pointer}: structured object must reject unknown fields"
                )
            if node.get("required") != list(node_properties):
                raise ProtocolGateError(
                    f"{name}{pointer}: every structured field must be required in order"
                )
        node_type = node.get("type")
        is_object = node_type == "object" or (
            isinstance(node_type, list) and "object" in node_type
        )
        if is_object and not isinstance(node_properties, dict):
            if node.get("x-dynamic-map") is not True:
                raise ProtocolGateError(
                    f"{name}{pointer}: open map requires explicit x-dynamic-map"
                )
            if node.get("additionalProperties") is False:
                raise ProtocolGateError(
                    f"{name}{pointer}: dynamic map cannot be closed"
                )

    bindings = list(_bound_nodes(name, schema))
    if not bindings or bindings[0]["pointer"] != "#":
        raise ProtocolGateError(f"{name}: root implementation binding is missing")
    return bindings


def _python_fields(path: Path) -> dict[str, list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ProtocolGateError(
            f"cannot parse Python projection {path}: {exc}"
        ) from exc
    classes: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        classes[node.name] = [
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        ]
    return classes


def _assert_python_closed(path: Path, bindings: list[dict[str, Any]]) -> None:
    source = path.read_text(encoding="utf-8")
    if (
        'model_config = ConfigDict(extra="forbid", frozen=True, strict=True)'
        not in source
    ):
        raise ProtocolGateError(
            "Python ProtocolModel must reject unknown fields and coercion"
        )
    for binding in bindings:
        model = re.escape(str(binding["python_model"]))
        if (
            re.search(rf"^class\s+{model}\(ProtocolModel\):", source, re.MULTILINE)
            is None
        ):
            raise ProtocolGateError(
                f"Python {binding['python_model']} must inherit strict ProtocolModel"
            )


_RUST_STRUCT = re.compile(r"\bpub\s+struct\s+([A-Za-z][A-Za-z0-9_]*)\s*\{")
_RUST_FIELD = re.compile(r"^\s*pub\s+([a-z][A-Za-z0-9_]*)\s*:", re.MULTILINE)
_RUST_OPTION_FIELD = re.compile(
    r"(?P<attributes>(?:\s*#\[[^\]]+\]\s*)*)pub\s+(?P<field>[a-z][A-Za-z0-9_]*)\s*:\s*Option<"
)


def _rust_fields(path: Path) -> dict[str, list[str]]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolGateError(f"cannot read Rust projection {path}: {exc}") from exc
    structs: dict[str, list[str]] = {}
    for match in _RUST_STRUCT.finditer(source):
        depth = 1
        index = match.end()
        while index < len(source) and depth:
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
            index += 1
        if depth:
            raise ProtocolGateError(f"unclosed Rust struct {match.group(1)} in {path}")
        body = source[match.end() : index - 1]
        structs[match.group(1)] = _RUST_FIELD.findall(body)
    return structs


def _assert_rust_closed(path: Path, bindings: list[dict[str, Any]]) -> None:
    source = path.read_text(encoding="utf-8")
    for binding in bindings:
        rust_type = re.escape(str(binding["rust_type"]))
        pattern = (
            rf"#\[serde\(deny_unknown_fields\)\]\s*pub\s+struct\s+{rust_type}\s*\{{"
        )
        if re.search(pattern, source) is None:
            raise ProtocolGateError(
                f"Rust {binding['rust_type']} must deny unknown fields"
            )
    for match in _RUST_OPTION_FIELD.finditer(source):
        if 'deserialize_with = "deserialize_required_option"' not in match.group(
            "attributes"
        ):
            raise ProtocolGateError(
                f"Rust nullable field {match.group('field')} must remain required"
            )


def _assert_projection(
    bindings: list[dict[str, Any]],
    actual: dict[str, list[str]],
    key: str,
    label: str,
) -> None:
    seen: set[str] = set()
    for binding in bindings:
        model = str(binding[key])
        if model in seen:
            raise ProtocolGateError(
                f"{label} binding {model} is declared more than once"
            )
        seen.add(model)
        expected = binding["fields"]
        if actual.get(model) != expected:
            raise ProtocolGateError(
                f"{label} {model} fields drifted: expected {expected}, "
                f"found {actual.get(model)}"
            )


def _pascal_case(value: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _rust_variant(value: str) -> str:
    rendered = _pascal_case(value)
    if not rendered or rendered[0].isdigit():
        rendered = f"V{rendered}"
    return rendered


def _projection_nodes() -> tuple[
    dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]
]:
    """Return bound nodes, external root bindings, and their owning schemas."""

    catalog = _load_json(CATALOG_PATH)
    nodes: dict[str, dict[str, Any]] = {}
    external_roots: dict[str, str] = {}
    owners: dict[str, dict[str, Any]] = {}
    for entry in catalog["schemas"]:
        schema = _load_json(CATALOG_DIR / str(entry["file"]))
        root_model = str(schema["x-python-model"])
        external_roots[str(entry["file"])] = root_model
        for _pointer, node in _iter_nodes(schema):
            if not isinstance(node, dict) or "x-python-model" not in node:
                continue
            model = str(node["x-python-model"])
            if model in nodes:
                raise ProtocolGateError(f"duplicate generated Python model {model}")
            nodes[model] = node
            owners[model] = schema
    return nodes, external_roots, owners


def _ref_model(
    ref: str,
    *,
    owner: dict[str, Any],
    external_roots: dict[str, str],
) -> str | None:
    if ref.startswith("#/$defs/"):
        name = ref.removeprefix("#/$defs/")
        node = (owner.get("$defs") or {}).get(name)
        if isinstance(node, dict) and node.get("x-python-model"):
            return str(node["x-python-model"])
        return None
    filename = ref.split("#", 1)[0]
    return external_roots.get(filename)


def _nonnull_schema(node: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    choices = node.get("oneOf")
    if isinstance(choices, list):
        concrete = [
            choice
            for choice in choices
            if isinstance(choice, dict) and choice.get("type") != "null"
        ]
        nullable = len(concrete) != len(choices)
        if len(concrete) == 1:
            return concrete[0], nullable
    node_type = node.get("type")
    if isinstance(node_type, list) and "null" in node_type:
        concrete_types = [value for value in node_type if value != "null"]
        concrete = dict(node)
        concrete["type"] = (
            concrete_types[0] if len(concrete_types) == 1 else concrete_types
        )
        return concrete, True
    return node, False


def _python_type(
    node: dict[str, Any],
    *,
    model: str,
    field: str,
    owner: dict[str, Any],
    external_roots: dict[str, str],
) -> str:
    concrete, nullable = _nonnull_schema(node)
    ref = concrete.get("$ref")
    if isinstance(ref, str):
        base = _ref_model(ref, owner=owner, external_roots=external_roots) or "Any"
    elif "const" in concrete:
        base = f"Literal[{concrete['const']!r}]"
    elif isinstance(concrete.get("enum"), list):
        values = ", ".join(repr(value) for value in concrete["enum"])
        base = f"Literal[{values}]"
    else:
        node_type = concrete.get("type")
        if node_type == "string":
            constraints: list[str] = []
            if "minLength" in concrete:
                constraints.append(f"min_length={int(concrete['minLength'])}")
            if "maxLength" in concrete:
                constraints.append(f"max_length={int(concrete['maxLength'])}")
            if "pattern" in concrete:
                constraints.append(f"pattern={concrete['pattern']!r}")
            base = (
                f"Annotated[str, Field({', '.join(constraints)})]"
                if constraints
                else "str"
            )
        elif node_type == "integer":
            constraints = []
            if "minimum" in concrete:
                constraints.append(f"ge={int(concrete['minimum'])}")
            if "maximum" in concrete:
                constraints.append(f"le={int(concrete['maximum'])}")
            base = (
                f"Annotated[int, Field({', '.join(constraints)})]"
                if constraints
                else "int"
            )
        elif node_type == "number":
            constraints = []
            if "minimum" in concrete:
                constraints.append(f"ge={concrete['minimum']!r}")
            if "maximum" in concrete:
                constraints.append(f"le={concrete['maximum']!r}")
            base = (
                f"Annotated[float, Field({', '.join(constraints)})]"
                if constraints
                else "float"
            )
        elif node_type == "boolean":
            base = "bool"
        elif node_type == "array":
            item = concrete.get("items")
            item_type = (
                _python_type(
                    item,
                    model=model,
                    field=f"{field}_item",
                    owner=owner,
                    external_roots=external_roots,
                )
                if isinstance(item, dict)
                else "Any"
            )
            base = f"list[{item_type}]"
        elif node_type == "object":
            additional = concrete.get("additionalProperties")
            value_type = (
                _python_type(
                    additional,
                    model=model,
                    field=f"{field}_value",
                    owner=owner,
                    external_roots=external_roots,
                )
                if isinstance(additional, dict)
                else "Any"
            )
            base = f"dict[str, {value_type}]"
        else:
            base = "Any"
    return f"{base} | None" if nullable else base


def _rust_type(
    node: dict[str, Any],
    *,
    model: str,
    field: str,
    owner: dict[str, Any],
    external_roots: dict[str, str],
) -> tuple[str, tuple[str, list[str]] | None, bool]:
    concrete, nullable = _nonnull_schema(node)
    enum: tuple[str, list[str]] | None = None
    ref = concrete.get("$ref")
    if isinstance(ref, str):
        base = _ref_model(ref, owner=owner, external_roots=external_roots) or "Value"
    elif isinstance(concrete.get("const"), str):
        name = f"{model}{_pascal_case(field)}"
        values = [str(concrete["const"])]
        enum = (name, values)
        base = name
    elif isinstance(concrete.get("enum"), list) and all(
        isinstance(value, str) for value in concrete["enum"]
    ):
        name = f"{model}{_pascal_case(field)}"
        values = [str(value) for value in concrete["enum"]]
        enum = (name, values)
        base = name
    else:
        node_type = concrete.get("type")
        if node_type == "string":
            base = "String"
        elif node_type == "integer":
            base = "u64" if int(concrete.get("minimum", -1)) >= 0 else "i64"
        elif node_type == "number":
            base = "f64"
        elif node_type == "boolean" or isinstance(concrete.get("const"), bool):
            base = "bool"
        elif node_type == "array":
            item = concrete.get("items")
            if isinstance(item, dict):
                item_type, item_enum, _ = _rust_type(
                    item,
                    model=model,
                    field=f"{field}_item",
                    owner=owner,
                    external_roots=external_roots,
                )
                if item_enum is not None:
                    enum = item_enum
            else:
                item_type = "Value"
            base = f"Vec<{item_type}>"
        elif node_type == "object":
            additional = concrete.get("additionalProperties")
            if isinstance(additional, dict):
                value_type, value_enum, _ = _rust_type(
                    additional,
                    model=model,
                    field=f"{field}_value",
                    owner=owner,
                    external_roots=external_roots,
                )
                if value_enum is not None:
                    enum = value_enum
            else:
                value_type = "Value"
            base = f"BTreeMap<String, {value_type}>"
        else:
            base = "Value"
    return (f"Option<{base}>" if nullable else base), enum, nullable


def build_manifest() -> tuple[dict[str, Any], Path]:
    catalog = _load_json(CATALOG_PATH)
    if catalog.get("protocol") != "epistemic-operations":
        raise ProtocolGateError("catalog protocol must be epistemic-operations")
    if catalog.get("version") != "1" or catalog.get("status") != "current":
        raise ProtocolGateError("catalog must expose only current version 1")
    if catalog.get("compatibility_policy") != "current-only":
        raise ProtocolGateError("catalog compatibility_policy must be current-only")
    if catalog.get("unknown_field_policy") != "reject":
        raise ProtocolGateError("catalog unknown_field_policy must be reject")

    entries = catalog.get("schemas")
    if not isinstance(entries, list):
        raise ProtocolGateError("catalog schemas must be a list")
    names = tuple(entry.get("name") for entry in entries if isinstance(entry, dict))
    if names != REQUIRED_SCHEMAS:
        raise ProtocolGateError(
            f"catalog must contain exactly {list(REQUIRED_SCHEMAS)} in order"
        )
    versions = {
        str(entry.get("name")): entry.get("version")
        for entry in entries
        if isinstance(entry, dict)
    }
    if versions != REQUIRED_SCHEMA_VERSIONS:
        raise ProtocolGateError(
            "catalog schema versions drifted: "
            f"expected {REQUIRED_SCHEMA_VERSIONS}, found {versions}"
        )
    files = [entry.get("file") for entry in entries]
    if any(not isinstance(file, str) for file in files) or len(set(files)) != len(
        files
    ):
        raise ProtocolGateError("catalog schema filenames must be unique strings")
    schema_files = set(files)
    present_json = {path.name for path in CATALOG_DIR.glob("*.schema.json")}
    if present_json != schema_files:
        raise ProtocolGateError(
            f"schema file set drifted: catalog={sorted(schema_files)}, "
            f"disk={sorted(present_json)}"
        )

    schemas: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry["name"])
        version = str(entry["version"])
        filename = str(entry["file"])
        schema = _load_json(CATALOG_DIR / filename)
        schema_bindings = _validate_schema(name, version, schema, schema_files)
        schemas.append(
            {
                "name": name,
                "version": version,
                "file": filename,
                "id": schema["$id"],
                "title": schema["title"],
                "sha256": _sha256_json(schema),
            }
        )
        bindings.extend(schema_bindings)

    payload = {
        "protocol": catalog["protocol"],
        "version": catalog["version"],
        "compatibility_policy": catalog["compatibility_policy"],
        "unknown_field_policy": catalog["unknown_field_policy"],
        "privacy_contract": catalog["privacy_contract"],
        "schemas": schemas,
        "bindings": bindings,
    }
    manifest = dict(payload)
    manifest["catalog_sha256"] = _sha256_json(payload)
    return manifest, ROOT / str(catalog["python_source"])


def _render_python(manifest: dict[str, Any]) -> str:
    nodes, external_roots, owners = _projection_nodes()
    version_pairs = "\n".join(
        f'    "{entry["name"]}": "{entry["version"]}",' for entry in manifest["schemas"]
    )
    schema_pairs = "\n".join(
        f'    "{entry["name"]}": "{entry["sha256"]}",' for entry in manifest["schemas"]
    )
    binding_pairs = "\n".join(
        "\n".join(
            [f'    "{entry["python_model"]}": (']
            + [f'        "{field}",' for field in entry["fields"]]
            + ["    ),"]
        )
        for entry in manifest["bindings"]
    )
    models: list[str] = []
    for binding in manifest["bindings"]:
        model = str(binding["python_model"])
        node = nodes[model]
        owner = owners[model]
        fields = [
            f"class {model}(ProtocolModel):",
            '    """Schema-generated strict projection."""',
        ]
        for field, field_schema in node["properties"].items():
            fields.append(
                f"    {field}: "
                + _python_type(
                    field_schema,
                    model=model,
                    field=field,
                    owner=owner,
                    external_roots=external_roots,
                )
            )
        models.append("\n".join(fields))
    return (
        '"""Generated strict Epistemic Operations Protocol client projections.\n\n'
        "JSON Schema is authoritative. Regenerate with the protocol gate; do not edit.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Annotated, Any, Literal\n\n"
        "from pydantic import BaseModel, ConfigDict, Field\n\n"
        f'PROTOCOL_NAME = "{manifest["protocol"]}"\n'
        f'PROTOCOL_VERSION = "{manifest["version"]}"\n'
        f'CATALOG_SHA256 = "{manifest["catalog_sha256"]}"\n'
        "SCHEMA_VERSION = {\n"
        f"{version_pairs}\n"
        "}\n"
        "SCHEMA_SHA256 = {\n"
        f"{schema_pairs}\n"
        "}\n\n"
        "\nclass ProtocolModel(BaseModel):\n"
        '    """Fail-closed base for every generated protocol DTO."""\n\n'
        '    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)\n\n\n'
        + "\n\n\n".join(models)
        + "\n\n\n"
        "BINDING_FIELDS = {\n"
        f"{binding_pairs}\n"
        "}\n"
    )


def _render_rust_types(manifest: dict[str, Any]) -> str:
    nodes, external_roots, owners = _projection_nodes()
    enum_order: list[str] = []
    enums: dict[str, list[str]] = {}
    structs: list[str] = []
    for binding in manifest["bindings"]:
        model = str(binding["rust_type"])
        node = nodes[str(binding["python_model"])]
        owner = owners[str(binding["python_model"])]
        lines = [
            "#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]",
            "#[serde(deny_unknown_fields)]",
            f"pub struct {model} {{",
        ]
        for field, field_schema in node["properties"].items():
            rust_type, enum, nullable = _rust_type(
                field_schema,
                model=model,
                field=field,
                owner=owner,
                external_roots=external_roots,
            )
            if enum is not None:
                enum_name, values = enum
                if enum_name not in enums:
                    enum_order.append(enum_name)
                    enums[enum_name] = values
                elif enums[enum_name] != values:
                    raise ProtocolGateError(f"generated Rust enum {enum_name} drifted")
            if nullable:
                lines.append(
                    '    #[serde(deserialize_with = "deserialize_required_option")]'
                )
            lines.append(f"    pub {field}: {rust_type},")
        lines.append("}")
        structs.append("\n".join(lines))

    rendered_enums: list[str] = []
    for name in enum_order:
        lines = [
            "#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]",
            f"pub enum {name} {{",
        ]
        for value in enums[name]:
            lines.append(f'    #[serde(rename = "{value}")]')
            lines.append(f"    {_rust_variant(value)},")
        lines.append("}")
        rendered_enums.append("\n".join(lines))

    return (
        "//! Generated strict Epistemic Operations Protocol serde projections.\n"
        "//!\n"
        "//! JSON Schema is authoritative. Regenerate with the AU protocol gate.\n\n"
        "use std::collections::BTreeMap;\n\n"
        "use serde::{Deserialize, Deserializer, Serialize};\n"
        "use serde_json::Value;\n\n"
        "fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>\n"
        "where\n"
        "    D: Deserializer<'de>,\n"
        "    T: Deserialize<'de>,\n"
        "{\n"
        "    Option::<T>::deserialize(deserializer)\n"
        "}\n\n" + "\n\n".join(rendered_enums) + "\n\n" + "\n\n".join(structs) + "\n"
    )


def _render_rust_manifest(manifest: dict[str, Any]) -> str:
    versions = "\n".join(
        f'    ("{entry["name"]}", "{entry["version"]}"),'
        for entry in manifest["schemas"]
    )
    schemas = "\n".join(
        "\n".join(
            [
                "    (",
                f'        "{entry["name"]}",',
                f'        "{entry["sha256"]}",',
                "    ),",
            ]
        )
        for entry in manifest["schemas"]
    )
    return (
        "//! Auto-generated protocol digests; regenerate from the canonical catalog.\n\n"
        f'pub const PROTOCOL_NAME: &str = "{manifest["protocol"]}";\n'
        f'pub const PROTOCOL_VERSION: &str = "{manifest["version"]}";\n'
        f'pub const CATALOG_SHA256: &str = "{manifest["catalog_sha256"]}";\n'
        "pub const SCHEMA_VERSION: &[(&str, &str)] = &[\n"
        f"{versions}\n"
        "];\n"
        "pub const SCHEMA_SHA256: &[(&str, &str)] = &[\n"
        f"{schemas}\n"
        "];\n"
    )


def _render_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _git_common_sibling() -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    common = Path(result.stdout.strip()).resolve()
    repository_root = common.parent if common.name == ".git" else common
    return repository_root.parent / "epistemic-graph"


def _resolve_engine_root(explicit: Path | None, self_only: bool) -> Path | None:
    if self_only:
        return None
    candidates: Iterable[Path] = (
        [explicit]
        if explicit is not None
        else [ROOT.parent / "epistemic-graph", _git_common_sibling()]
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "crates" / "eg-types").is_dir():
            return candidate.resolve()
    raise ProtocolGateError(
        "epistemic-graph checkout not found; pass --epistemic-graph-root or use "
        "--self-only only for isolated agent-utilities CI"
    )


def _check_or_write(path: Path, expected: str, write: bool) -> None:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
        return
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolGateError(f"missing generated artifact {path}: {exc}") from exc
    if actual != expected:
        raise ProtocolGateError(
            f"generated artifact drifted: {path}; run this gate with --write"
        )


def run(engine_root: Path | None, *, write: bool) -> dict[str, Any]:
    manifest, python_source = build_manifest()
    _check_or_write(AU_GENERATED, _render_python(manifest), write)
    _assert_projection(
        manifest["bindings"],
        _python_fields(python_source),
        "python_model",
        "Python",
    )
    _assert_python_closed(python_source, manifest["bindings"])
    if engine_root is not None:
        catalog = _load_json(CATALOG_PATH)
        rust_source = engine_root / str(catalog["rust_source"])
        _check_or_write(rust_source, _render_rust_types(manifest), write)
        _assert_projection(
            manifest["bindings"],
            _rust_fields(rust_source),
            "rust_type",
            "Rust",
        )
        _assert_rust_closed(rust_source, manifest["bindings"])
        _check_or_write(
            engine_root / ENGINE_MANIFEST,
            _render_manifest(manifest),
            write,
        )
        _check_or_write(
            engine_root / ENGINE_GENERATED,
            _render_rust_manifest(manifest),
            write,
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate projections.")
    parser.add_argument(
        "--self-only",
        action="store_true",
        help="Validate AU schema/model parity without an engine checkout.",
    )
    parser.add_argument(
        "--epistemic-graph-root",
        type=Path,
        help="Explicit epistemic-graph repository root.",
    )
    args = parser.parse_args(argv)
    if args.self_only and args.epistemic_graph_root is not None:
        parser.error("--self-only and --epistemic-graph-root are mutually exclusive")
    try:
        engine_root = _resolve_engine_root(args.epistemic_graph_root, args.self_only)
        manifest = run(engine_root, write=args.write)
    except ProtocolGateError as exc:
        print(f"epistemic-operations protocol gate: FAIL: {exc}", file=sys.stderr)
        return 1
    mode = "regenerated" if args.write else "verified"
    scope = "AU" if engine_root is None else "AU + epistemic-graph"
    print(
        f"epistemic-operations protocol gate: {mode} {len(REQUIRED_SCHEMAS)} "
        f"schemas / {len(manifest['bindings'])} bound objects ({scope}); "
        f"catalog_sha256={manifest['catalog_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
