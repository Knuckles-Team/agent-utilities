#!/usr/bin/env python3
"""Inject the ``check-ontology-integrity`` pre-commit hook into connector repos (D17).

The per-repo hook template for the C5 Connector Ontology Manifest gate
(CONCEPT:AU-KG.ontology.connector-manifest-gate): it re-compiles the connector's own
``connector_manifest.yml`` and fails closed (non-zero exit) if the recomputed
canonical hash no longer matches the signed ``provenance.integrity.hash`` — the
same fail-closed contract :mod:`scripts.check_connector_manifests` enforces
fleet-wide, wired in per-repo the same way ``env-var-drift`` already is (one
``repo: local`` hook block, triggered only when ontology-relevant files change).

Idempotent (skips a repo whose ``.pre-commit-config.yaml`` already has the hook)
and scoped to connector repos that actually ship a ``connector_manifest.yml`` — a
repo with no manifest yet has nothing to gate, so it is left untouched (never
adds a hook that would always no-op).

Usage:
  python3 scripts/inject_ontology_integrity_hook.py --agents-root <path> [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HOOK_ID = "check-ontology-integrity"

# Matches the fleet's existing ``env-var-drift`` local-hook shape exactly (2-space
# item indent, ``language: system``, ``pass_filenames: false``, a ``files:`` glob
# gating it to only the artifacts that can invalidate the manifest's pinned hash).
HOOK_BLOCK = f"""  - id: {HOOK_ID}
    name: check connector ontology manifest integrity (C5/X6, fail-closed)
    entry: |-
      bash -c '[ -f connector_manifest.yml ] || exit 0; python3 /home/apps/workspace/agent-packages/agent-utilities/scripts/check_connector_manifests.py --manifest connector_manifest.yml'
    language: system
    pass_filenames: false
    always_run: true
    files: ^(connector_manifest\\.yml|.*/ontology/.*\\.ttl|.*/connectors/mcp_source_presets\\.json|a2a\\.json)$
"""


def inject(filepath: Path, *, dry_run: bool = False) -> bool:
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    if HOOK_ID in content:
        return False

    repo_local_idx = content.find("- repo: local")
    if repo_local_idx == -1:
        new_content = content + ("\n" if not content.endswith("\n") else "")
        new_content += "- repo: local\n  hooks:\n" + HOOK_BLOCK
    else:
        hooks_idx = content.find("hooks:", repo_local_idx)
        if hooks_idx == -1:
            return False
        newline_idx = content.find("\n", hooks_idx)
        if newline_idx == -1:
            return False
        new_content = content[: newline_idx + 1] + HOOK_BLOCK + content[newline_idx + 1 :]

    if not dry_run:
        filepath.write_text(new_content, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--agents-root", type=Path, required=True, help="the agents/ fleet root"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    args = ap.parse_args()

    injected: list[str] = []
    skipped_no_manifest: list[str] = []
    skipped_no_precommit: list[str] = []
    skipped_already: list[str] = []

    for connector_dir in sorted(
        d for d in args.agents_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    ):
        if not (connector_dir / "connector_manifest.yml").exists():
            skipped_no_manifest.append(connector_dir.name)
            continue
        precommit = connector_dir / ".pre-commit-config.yaml"
        if not precommit.exists():
            skipped_no_precommit.append(connector_dir.name)
            continue
        if HOOK_ID in precommit.read_text(encoding="utf-8", errors="ignore"):
            skipped_already.append(connector_dir.name)
            continue
        if inject(precommit, dry_run=args.dry_run):
            injected.append(connector_dir.name)

    print(f"injected: {len(injected)} -> {injected}")
    print(f"skipped (already present): {len(skipped_already)}")
    print(f"skipped (no connector_manifest.yml): {len(skipped_no_manifest)}")
    print(f"skipped (no .pre-commit-config.yaml): {len(skipped_no_precommit)} -> {skipped_no_precommit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
