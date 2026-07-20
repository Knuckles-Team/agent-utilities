#!/usr/bin/env python3
"""Connector Ontology Manifest gate (CONCEPT:AU-KG.ontology.connector-manifest-gate).

Drives every ``agents/*/connector_manifest.yml`` (or a single ``--manifest``) to:

  1. **Compiles cleanly** — ``compile_manifest`` + ``export_manifest_ttl`` succeed and
     the result parses as valid Turtle.
  2. **Integrity matches** — the recomputed canonical hash equals
     ``provenance.integrity.hash`` (catches a hand-edited manifest post-signing).
  3. **No un-imported top-level ttl** — the connector's ontology IRI is either already
     ``owl:imports``-ed by the canonical ``ontology.ttl`` or a registered federated
     module (the anti-sprawl invariant ``manifest_compiler.apply_manifest`` enforces).
  4. **Signature/release pin verifies** — the complete manifest (not only its
     compiled ontology graph) must match its trusted signed release pin, or verify
     cryptographically against a configured trusted signer. There is no unsigned
     development bypass on this gate.

Usage:
  python3 scripts/check_connector_manifests.py --agents-root <path>   # sweep the fleet
  python3 scripts/check_connector_manifests.py --manifest <path>      # one manifest

Exit 0 = all manifests compile, hash-match, are wired, and sign-verify.
Exit 1 = one or more violations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,  # noqa: E402
)
from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (  # noqa: E402
    check_manifest_bytes,
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (  # noqa: E402
    is_wired,
)


def _load(path: Path) -> ConnectorManifest:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ConnectorManifest.model_validate(data)


def check_one(path: Path, *, verbose: bool = False) -> list[str]:
    del verbose
    label = f"{path.parent.name}/connector_manifest.yml"
    violations = check_manifest_bytes(path, require_signature=True)

    try:
        manifest = _load(path)
    except Exception as exc:  # noqa: BLE001
        if violations:
            return violations
        return [f"[schema] {label}: does not validate ({type(exc).__name__})"]

    source = manifest.resolved_ontology_source
    if not is_wired(source):
        violations.append(
            f"[anti-sprawl] {label}: <http://knuckles.team/kg/{source}> is "
            "not owl:imports-ed by the canonical ontology.ttl and is not a registered "
            "federated module — add the one owl:imports line before this manifest may "
            "be applied (never introduce an un-imported top-level ttl)."
        )
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="a specific connector_manifest.yml (repeatable)",
    )
    ap.add_argument(
        "--agents-root",
        type=Path,
        help="sweep every agents/*/connector_manifest.yml under this root",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    paths: list[Path] = list(args.manifest or [])
    if args.agents_root:
        paths.extend(sorted(args.agents_root.glob("*/connector_manifest.yml")))
    if not paths:
        print(
            "check_connector_manifests: nothing to check (pass --manifest or --agents-root)"
        )
        return 0

    all_violations: list[str] = []
    for p in paths:
        all_violations.extend(check_one(p, verbose=args.verbose))

    if all_violations:
        print(f"check_connector_manifests: {len(all_violations)} violation(s):")
        for v in all_violations:
            print(f"  ✗ {v}")
        return 1
    print(
        f"check_connector_manifests: OK — {len(paths)} manifest(s) compile, hash-match, wired."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
