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
  4. **Signature verifies** — the ``provenance.signature`` is checked against the
     trusted-signer allowlist (the real ``verify()`` call path is always exercised).
     Fleet-wide signer infra (X6) isn't fully wired yet, so when no signing secret is
     configured at all (``AGENT_UTILITIES_TOKEN_SECRET`` unset) a verification failure
     is reported as a **stub notice**, not a hard violation — once a secret/allowlist
     is configured, the same failure becomes a hard gate violation.

Usage:
  python3 scripts/check_connector_manifests.py --agents-root <path>   # sweep the fleet
  python3 scripts/check_connector_manifests.py --manifest <path>      # one manifest

Exit 0 = all manifests compile, hash-match, are wired, and (when a secret is
configured) sign-verify. Exit 1 = one or more violations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.core._env import setting  # noqa: E402
from agent_utilities.knowledge_graph.ontology import ontology_integrity  # noqa: E402
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,  # noqa: E402
)
from agent_utilities.knowledge_graph.ontology.manifest_compiler import (  # noqa: E402
    compile_manifest,
    export_manifest_ttl,
    is_wired,
)


def _load(path: Path) -> ConnectorManifest:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ConnectorManifest.model_validate(data)


def check_one(path: Path, *, verbose: bool = False) -> list[str]:
    violations: list[str] = []
    notes: list[str] = []

    try:
        manifest = _load(path)
    except Exception as exc:  # noqa: BLE001
        return [
            f"[schema] {path}: does not parse/validate as a ConnectorManifest: {exc}"
        ]

    source = manifest.resolved_ontology_source
    try:
        spec = compile_manifest(manifest)
        ttl = export_manifest_ttl(spec, source=source)
        import rdflib

        g = rdflib.Graph()
        g.parse(data=ttl, format="turtle")
    except Exception as exc:  # noqa: BLE001
        violations.append(f"[compile] {path}: manifest does not compile cleanly: {exc}")
        return violations

    digest, triple_count = ontology_integrity.canonical_hash(g)
    if digest != manifest.provenance.integrity.hash:
        violations.append(
            f"[integrity] {path}: recomputed hash {digest} (n={triple_count}) != "
            f"provenance.integrity.hash {manifest.provenance.integrity.hash} — "
            "regenerate via scripts/generate_connector_manifests.py."
        )

    if not is_wired(source):
        violations.append(
            f"[anti-sprawl] {path}: <http://knuckles.team/kg/{source}> is "
            "not owl:imports-ed by the canonical ontology.ttl and is not a registered "
            "federated module — add the one owl:imports line before this manifest may "
            "be applied (never introduce an un-imported top-level ttl)."
        )

    secret_configured = bool(setting("AGENT_UTILITIES_TOKEN_SECRET", ""))
    verified = ontology_integrity.verify(
        manifest.provenance.integrity.hash,
        manifest.provenance.signature,
        signer_id=manifest.provenance.signer,
    )
    if not verified:
        msg = (
            f"[signature] {path}: provenance.signature did not verify "
            f"(signer={manifest.provenance.signer!r})"
        )
        if secret_configured:
            violations.append(
                msg
                + " — AGENT_UTILITIES_TOKEN_SECRET is configured, this is a hard failure."
            )
        else:
            notes.append(
                msg
                + " — STUB: no AGENT_UTILITIES_TOKEN_SECRET configured, signer infra not yet wired."
            )

    if verbose:
        for n in notes:
            print(f"  · {n}")
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
