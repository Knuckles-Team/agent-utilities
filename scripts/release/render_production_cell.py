#!/usr/bin/python
"""Render the production-cell template from a verified exact release manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


def _load_checker(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("graphos_compatibility_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("compatibility checker could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("release input must be a mapping")
    return value


def _oci_pin(component: dict[str, Any]) -> tuple[str, str]:
    artifact = str(component["artifact"])
    name, separator, digest = artifact.rpartition("@")
    if not separator or not name or digest != component["digest"]:
        raise ValueError("OCI artifact and digest are not an exact pair")
    return name, digest


def render(
    *,
    manifest_path: Path,
    matrix_path: Path,
    template_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    checker = _load_checker(Path(__file__).with_name("check_compatibility.py"))
    manifest = _yaml(manifest_path)
    matrix = _yaml(matrix_path)
    report = checker.verify_release_manifest(
        manifest,
        matrix,
        matrix_path=matrix_path,
        verify_signatures=True,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("render output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in template_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)
    kustomization_path = output_dir / "kustomization.yaml"
    kustomization = _yaml(kustomization_path)
    pins = {
        "graph-os-image": _oci_pin(manifest["components"]["agent-utilities"]),
        "epistemic-graph-image": _oci_pin(manifest["components"]["epistemic-graph"]),
    }
    for image in kustomization.get("images", []):
        logical_name = image.get("name")
        if logical_name not in pins:
            continue
        repository, digest = pins[logical_name]
        image.clear()
        image.update({"name": logical_name, "newName": repository, "digest": digest})
    release_pins = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "graphos-release-pins", "namespace": "graphos-control"},
        "immutable": True,
        "data": {
            "release_id": report["releaseId"],
            "release_digest": report["releaseDigest"],
            "configuration_digest": manifest["configurationDigest"],
            "epistemic_graph_digest": manifest["components"]["epistemic-graph"][
                "digest"
            ],
            "agent_utilities_digest": manifest["components"]["agent-utilities"][
                "digest"
            ],
            "protocol_digest": manifest["components"]["epistemic-operations-protocol"][
                "digest"
            ],
            "connector_catalog_digest": manifest["components"]["connector-bundles"][
                "digest"
            ],
            "skill_catalog_digest": manifest["components"]["prebundled-skills"][
                "digest"
            ],
            "ontology_lock_digest": manifest["components"]["ontology-lock"]["digest"],
            "index_migration_digest": manifest["components"]["index-migrations"][
                "digest"
            ],
        },
    }
    (output_dir / "release-pins.yaml").write_text(
        yaml.safe_dump(release_pins, sort_keys=False), encoding="utf-8"
    )
    kustomization.setdefault("resources", []).append("release-pins.yaml")
    kustomization_path.write_text(
        yaml.safe_dump(kustomization, sort_keys=False), encoding="utf-8"
    )
    return {
        "ok": True,
        "releaseId": report["releaseId"],
        "releaseDigest": report["releaseDigest"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render-graphos-production-cell")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--matrix", type=Path, default=Path("deploy/release/compatibility-matrix.yml")
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("deploy/k8s/production-cell"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = render(
            manifest_path=args.manifest,
            matrix_path=args.matrix,
            template_dir=args.template,
            output_dir=args.output,
        )
    except Exception as exc:  # noqa: BLE001 - privacy-safe CLI boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
