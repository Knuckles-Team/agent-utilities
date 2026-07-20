#!/usr/bin/env python3
"""Generate typed, deterministic release configuration or migration-plan evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.release import check_compatibility
from scripts.release.generate_release_assembly import write

_INDEX_MIGRATION_SCHEMA = (
    check_compatibility._RELEASE_SCHEMA_ROOT
    / "index-migration-catalog.schema.json"
)


class ReleaseInputError(ValueError):
    """A release input is missing, aliased, or outside the current contract."""


def _matrix(path: Path) -> tuple[dict[str, Any], str]:
    payload = check_compatibility._input_bytes(
        path,
        maximum=check_compatibility._MAX_COMPONENT_SOURCE_BYTES,
    )
    try:
        value = yaml.safe_load(payload)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReleaseInputError("compatibility matrix is invalid") from exc
    if not isinstance(value, dict):
        raise ReleaseInputError("compatibility matrix is invalid")
    check_compatibility.validate_compatibility_matrix(value)
    return value, "sha256:" + hashlib.sha256(payload).hexdigest()


def _index_catalog(path: Path) -> tuple[dict[str, Any], str]:
    payload = check_compatibility._input_bytes(
        path,
        maximum=check_compatibility._MAX_COMPONENT_SOURCE_BYTES,
    )
    value = check_compatibility._json_evidence(payload, "index migration catalog")
    check_compatibility._validate_release_schema(
        value,
        schema_path=_INDEX_MIGRATION_SCHEMA,
        field="index migration catalog",
    )
    if value.get("entryCount") != len(value.get("entries") or ()):
        raise ReleaseInputError("index migration catalog entry count is invalid")
    return value, "sha256:" + hashlib.sha256(payload).hexdigest()


def generate_configuration(*, release_id: str, matrix_path: Path) -> dict[str, Any]:
    matrix, matrix_digest = _matrix(matrix_path)
    value = check_compatibility.release_configuration_document(
        release_id=release_id,
        matrix=matrix,
        matrix_digest=matrix_digest,
    )
    check_compatibility.validate_release_configuration(
        value,
        release_id=release_id,
        matrix=matrix,
        matrix_digest=matrix_digest,
    )
    return value


def generate_migration_plan(
    *,
    release_id: str,
    matrix_path: Path,
    index_migration_catalog_path: Path,
) -> dict[str, Any]:
    matrix, matrix_digest = _matrix(matrix_path)
    catalog, catalog_digest = _index_catalog(index_migration_catalog_path)
    value = check_compatibility.release_migration_plan_document(
        release_id=release_id,
        matrix=matrix,
        matrix_digest=matrix_digest,
        index_migration_catalog_digest=catalog_digest,
        index_migration_count=int(catalog["entryCount"]),
    )
    check_compatibility.validate_release_migration_plan(
        value,
        release_id=release_id,
        matrix=matrix,
        matrix_digest=matrix_digest,
        index_migration_catalog_digest=catalog_digest,
    )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate-graphos-release-input")
    parser.add_argument(
        "kind", choices=("configuration", "migration-plan")
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--index-migration-catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        input_paths = {arguments.matrix.resolve(strict=False)}
        if arguments.index_migration_catalog is not None:
            input_paths.add(
                arguments.index_migration_catalog.resolve(strict=False)
            )
        if arguments.output.resolve(strict=False) in input_paths:
            raise ReleaseInputError("release input output must not alias an input")
        if arguments.kind == "configuration":
            if arguments.index_migration_catalog is not None:
                raise ReleaseInputError(
                    "configuration does not accept an index migration catalog"
                )
            value = generate_configuration(
                release_id=arguments.release_id,
                matrix_path=arguments.matrix,
            )
        else:
            if arguments.index_migration_catalog is None:
                raise ReleaseInputError(
                    "migration plan requires an index migration catalog"
                )
            value = generate_migration_plan(
                release_id=arguments.release_id,
                matrix_path=arguments.matrix,
                index_migration_catalog_path=arguments.index_migration_catalog,
            )
        write(arguments.output, value)
    except Exception as exc:  # noqa: BLE001 - privacy-safe release boundary
        print(json.dumps({"error": type(exc).__name__, "ok": False}, sort_keys=True))
        return 1
    print(json.dumps({"kind": arguments.kind, "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
