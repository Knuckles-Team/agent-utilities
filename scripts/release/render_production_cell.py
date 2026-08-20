#!/usr/bin/python
"""Render the production-cell template from a verified exact release manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

# The renderer is invoked as ``python scripts/release/render_production_cell.py``
# as well as imported by tests; make the repository root explicit for the
# sibling topology authority in both modes.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.release.production_cell_topology import (
    apply_to_documents,
    canonical_contract,
)
from scripts.release.production_cell_topology import (
    validate as validate_topology,
)


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


def _documents(path: Path) -> list[dict[str, Any]]:
    values = []
    for value in yaml.safe_load_all(path.read_text(encoding="utf-8")):
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object YAML document")
            values.append(value)
    return values


def _oci_parts(image: str) -> tuple[str, str]:
    repository, separator, digest = image.rpartition("@")
    if not separator or not repository or not digest.startswith("sha256:"):
        raise ValueError("topology image is not an OCI digest reference")
    return repository, digest


def _load_topology(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("topology input must be a mapping")
    return value


def _write_documents(path: Path, documents: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump_all(documents, sort_keys=False), encoding="utf-8")


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
    topology_path: Path | None = None,
    engine_identity_contract_path: Path | None = None,
    rollback: bool = False,
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
    topology = None
    if topology_path is not None:
        contract_path = engine_identity_contract_path or (
            template_dir / "engine-identity-contract.v1.json"
        )
        contract = _load_topology(contract_path)
        topology = validate_topology(
            _load_topology(topology_path),
            release_manifest=manifest,
            engine_identity_contract=contract,
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("render output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in template_dir.iterdir():
        if source.is_file() and source.name not in {
            "topology-input.schema.json",
            "production-input.example.json",
            "engine-identity-contract.v1.json",
        }:
            shutil.copy2(source, output_dir / source.name)
    if topology is not None:
        yaml_paths = sorted(output_dir.glob("*.yaml"))
        grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        documents: list[dict[str, Any]] = []
        for path in yaml_paths:
            if path.name == "kustomization.yaml":
                continue
            path_documents = _documents(path)
            grouped[path] = path_documents
            documents.extend(path_documents)
        before_ids = {id(document) for document in documents}
        documents = apply_to_documents(documents, topology, rollback=rollback)
        after_ids = {id(document) for document in documents}
        for path, path_documents in grouped.items():
            _write_documents(
                path,
                [document for document in path_documents if id(document) in after_ids],
            )
        if before_ids == after_ids and not documents:
            raise ValueError("topology renderer produced no Kubernetes documents")
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
        if topology is not None and rollback:
            rollback_image = (
                topology["engine_rollback_image"]
                if logical_name == "epistemic-graph-image"
                else topology["workloads"]["gateway"]["rollback_image"]
            )
            repository, digest = _oci_parts(rollback_image)
        else:
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
    if topology is not None:
        release_pins["data"].update(
            {
                "topology_input_digest": topology["input_digest"],
                "topology_render_mode": "rollback" if rollback else "forward",
                "rollout_evidence_ref": topology["rollout_evidence"],
                "rollback_evidence_ref": topology["rollback_evidence"],
                "current_manifest_ref": topology["current_manifest_ref"],
                "rollback_manifest_ref": topology["rollback_manifest_ref"],
                "engine_current_image": topology["engine_current_image"],
                "engine_rollback_image": topology["engine_rollback_image"],
                "gateway_current_image": topology["workloads"]["gateway"][
                    "current_image"
                ],
                "gateway_rollback_image": topology["workloads"]["gateway"][
                    "rollback_image"
                ],
                "session_store_service": f"{topology['session_service']}.{topology['session_namespace']}.svc.cluster.local",
                "session_store_authority_ref": topology["session_ref"],
                "action_audit_authority_ref": topology["action_audit_ref"],
                "engine_discovery_ref": topology["engine_discovery"],
                "engine_identity_contract_ref": topology["engine_identity_ref"],
                "engine_identity_contract_digest": topology["engine_identity_digest"],
            }
        )
    (output_dir / "release-pins.yaml").write_text(
        yaml.safe_dump(release_pins, sort_keys=False), encoding="utf-8"
    )
    resources = kustomization.setdefault("resources", [])
    if "release-pins.yaml" not in resources:
        resources.append("release-pins.yaml")
    if topology is not None:
        (output_dir / "topology-contract.yaml").write_text(
            yaml.safe_dump(
                canonical_contract(topology, rollback=rollback), sort_keys=False
            ),
            encoding="utf-8",
        )
        if "topology-contract.yaml" not in resources:
            resources.append("topology-contract.yaml")
    kustomization_path.write_text(
        yaml.safe_dump(kustomization, sort_keys=False), encoding="utf-8"
    )
    return {
        "ok": True,
        "releaseId": report["releaseId"],
        "releaseDigest": report["releaseDigest"],
        **(
            {
                "topologyInputDigest": topology["input_digest"],
                "renderMode": "rollback" if rollback else "forward",
            }
            if topology is not None
            else {}
        ),
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
    parser.add_argument(
        "--topology-input",
        type=Path,
        help="measured production-cell topology JSON/YAML",
    )
    parser.add_argument(
        "--engine-identity-contract",
        type=Path,
        help="canonical versioned engine identity contract",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="render the topology's previous immutable image digests",
    )
    args = parser.parse_args(argv)
    try:
        report = render(
            manifest_path=args.manifest,
            matrix_path=args.matrix,
            template_dir=args.template,
            output_dir=args.output,
            topology_path=args.topology_input,
            engine_identity_contract_path=args.engine_identity_contract,
            rollback=args.rollback,
        )
    except Exception as exc:  # noqa: BLE001 - privacy-safe CLI boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
