#!/usr/bin/env python3
"""Source integrity gate for connector manifests.

CONCEPT:AU-KG.ontology.connector-manifest-gate.

Validate manifest schemas, compile their declared ontology, and compare its
canonical hash with provenance.integrity.hash using the shared manifest gate.
The optional --check-actions also checks declared mutating tools.

This source-only result does not establish full-document attestation, semantic
admission, or runtime attachment. Publication signatures and runtime release
pins remain enforced by their owning gates. Epistemic Graph owns ontology and
SHACL semantic validation and committed GraphSchema attachment; this command
never substitutes a local registry for that authority.

Usage:
  python3 scripts/check_connector_manifests.py --agents-root <path>
  python3 scripts/check_connector_manifests.py --manifest <path>

Exit 0 = the selected source integrity checks pass (or no manifests selected).
Exit 1 = one or more source integrity violations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (  # noqa: E402
    check_manifest_bytes,
)


def check_one(
    path: Path,
    *,
    verbose: bool = False,
    check_actions: bool = False,
    agents_root: Path | None = None,
) -> list[str]:
    del verbose
    # In-repo artifacts are no longer signature-verified (see the
    # `refactor(release): drop a2a.json and the in-repo signature duplication`
    # commit): a connector manifest never crosses a trust boundary here -- it
    # lives in this git repository, which already supplies content integrity
    # and authorship. What this gate still enforces is the part git does NOT:
    # that the manifest's recorded `provenance.integrity.hash` matches the
    # ontology it actually compiles to. `require_signature` itself is retained
    # (and still proven by
    # `test_connector_manifest_gate.py`) for artifacts that DO leave this
    # repository via `release_signer_for_publication`.
    #
    # Known, accepted gap: the signature covered the whole document, including
    # the `sync` preset/tool-schema block, which the ontology hash does not.
    # Tampering there is now caught by review of the commit, not by this gate.
    #
    # `check_actions` (CA-32/DEC-CA-07, off by default) additionally requires
    # every explicitly mutating-tagged MCP tool in this package to be declared
    # in `actions[]`. Left off the default sweep because 8 of the 72 shipped
    # packages (audio-transcriber, container-manager-mcp, lakekeeper-mcp,
    # microsoft-agent, opensearch-mcp, spark-mcp, systems-manager,
    # tunnel-manager — CA-32-W01 fleet audit) already have this real,
    # pre-existing gap; closing it is a fleet-wide sweep out of THIS lane's
    # scope. `--check-actions` makes the rule runnable today for anyone
    # auditing the fleet (or CA-40..46's own CI, which starts clean).
    return check_manifest_bytes(
        path,
        require_signature=False,
        require_declared_actions=check_actions,
        agents_root=agents_root,
    )


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
    ap.add_argument(
        "--check-actions",
        action="store_true",
        help=(
            "additionally require every explicitly mutating-tagged MCP tool "
            "(tags={'mutating'}, or an annotations={'destructiveHint': True}/"
            "{'readOnlyHint': False}) to be declared in actions[] (CA-32/"
            "DEC-CA-07). Off by default -- 8 shipped packages have this "
            "pre-existing, out-of-lane-scope gap today; see check_one()."
        ),
    )
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
        all_violations.extend(
            check_one(
                p,
                verbose=args.verbose,
                check_actions=args.check_actions,
                agents_root=args.agents_root,
            )
        )

    if all_violations:
        print(f"check_connector_manifests: {len(all_violations)} violation(s):")
        for v in all_violations:
            print(f"  ✗ {v}")
        return 1
    print(
        f"check_connector_manifests: SOURCE INTEGRITY OK — {len(paths)} manifest(s) "
        "compile and hash-match; attestation, semantic admission, and attachment "
        "are not checked by this command."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
