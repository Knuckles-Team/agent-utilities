#!/usr/bin/env python3
"""Pin every connector manifest's canonical hash into the fleet ``ontology.lock`` (D15).

Sweeps ``agents/*/connector_manifest.yml``, RE-COMPILES each one (never trusts the
on-disk ``provenance.integrity.hash`` blindly — the same compile path
:mod:`scripts.check_connector_manifests` uses), and pins the freshly recomputed
canonical hash into ``agent_utilities/knowledge_graph/ontology.lock`` keyed by the
artifact's path relative to the workspace root (``agents/<pkg>/connector_manifest.yml``)
— the one fleet-wide, byte-stable supply-chain-integrity ledger (X6,
CONCEPT:AU-KG.ontology.supply-chain-integrity). A manifest that fails to compile is
reported and skipped (never silently pinned with a bad hash).

Usage:
  python3 scripts/update_ontology_lock.py --agents-root <path> [--lock-path <path>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.ontology import ontology_integrity  # noqa: E402
from agent_utilities.knowledge_graph.ontology.connector_manifest import (  # noqa: E402
    ConnectorManifest,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (  # noqa: E402
    compile_manifest,
    export_manifest_ttl,
)


def _load(path: Path) -> ConnectorManifest:
    import yaml

    return ConnectorManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents-root", type=Path, required=True)
    ap.add_argument(
        "--lock-path",
        type=Path,
        default=ROOT / "agent_utilities" / "knowledge_graph" / "ontology.lock",
    )
    ap.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="root the artifact keys are stored relative to (default: agents-root's parent)",
    )
    args = ap.parse_args()

    artifact_root = args.artifact_root or args.agents_root.parent
    pinned = 0
    failed: list[str] = []

    for manifest_path in sorted(args.agents_root.glob("*/connector_manifest.yml")):
        try:
            manifest = _load(manifest_path)
            spec = compile_manifest(manifest)
            ttl = export_manifest_ttl(spec, source=manifest.resolved_ontology_source)
            import rdflib

            g = rdflib.Graph()
            g.parse(data=ttl, format="turtle")
            digest, triple_count = ontology_integrity.canonical_hash(g)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{manifest_path}: {exc}")
            continue

        try:
            artifact_key = str(manifest_path.relative_to(artifact_root))
        except ValueError:
            artifact_key = str(manifest_path)

        ontology_integrity.update_lock_entry(
            args.lock_path, artifact_key, digest, triple_count=triple_count
        )
        pinned += 1

    print(f"pinned {pinned} artifact(s) into {args.lock_path}")
    if failed:
        print(f"FAILED to compile {len(failed)} manifest(s):")
        for f in failed:
            print(f"  ✗ {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
