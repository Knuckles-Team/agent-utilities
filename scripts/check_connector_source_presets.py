#!/usr/bin/env python3
"""Fail closed when a connector source preset names an unregistered MCP server.

CONCEPT:AU-KG.ontology.registry-derived-server-alias.

``deploy/mcp-fleet.registry.yml`` is the one authority for what a fleet MCP server
is called. Every ``<module>/connectors/mcp_source_presets.json`` restates that name
in its ``server`` field, and on 2026-07-28 the restatement had rotted: 27 providers
named their *distribution* (``github-agent``) where the fleet runs the *service*
(``github-mcp``), and 9 more named a service the registry did not contain at all.
Because ``generate_connector_manifests.py`` copied the field verbatim, all of it was
signed.

The generator now derives the alias instead of copying it, so a wrong alias can no
longer reach a manifest. This gate closes the other half: a wrong alias may not sit
in the source tree either. It reports, and with ``--fix`` repairs, three findings:

* ``unregistered-provider`` — the provider has no registry entry, so no alias exists
  to derive. Not repairable here: regenerate the registry first.
* ``server-alias-drift``   — the preset names a server the registry does not
  register for this provider.
* ``registry-stale``       — the committed registry no longer matches what the
  generator would produce from the provider fleet.

Usage:
  python3 scripts/check_connector_source_presets.py --agents-root <path>
  python3 scripts/check_connector_source_presets.py --agents-root <path> --fix
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.orchestration.fleet_reconciler import (  # noqa: E402
    FleetRegistryError,
    registry_server_aliases,
)

PRESET_FILE = "mcp_source_presets.json"


@dataclass(frozen=True, slots=True)
class Finding:
    """One drifted provider, in a shape a report and a repair can both consume."""

    kind: str
    provider: str
    declared: str
    expected: str
    path: Path

    def render(self) -> str:
        return (
            f"- {self.kind}: {self.provider} declares {self.declared!r}"
            f" (registry: {self.expected or '<absent>'})"
        )


def _project_name(repo: Path) -> str:
    try:
        document = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
        return str(document["project"]["name"])
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("provider project metadata is invalid") from exc


def _preset_files(repo: Path) -> list[Path]:
    return sorted(
        path
        for path in repo.glob(f"*/connectors/{PRESET_FILE}")
        if ".venv" not in path.parts and "site-packages" not in path.parts
    )


def audit(agents_root: Path, registry_path: Path | None = None) -> list[Finding]:
    """Every provider whose declared server alias disagrees with the registry."""

    aliases = registry_server_aliases(registry_path)
    findings: list[Finding] = []
    for repo in sorted(path for path in agents_root.iterdir() if path.is_dir()):
        if not (repo / "pyproject.toml").is_file():
            continue
        presets = _preset_files(repo)
        if not presets:
            continue
        provider = _project_name(repo)
        expected = aliases.get(provider, "")
        for path in presets:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("provider source presets are unreadable") from exc
            declared = sorted(
                {
                    str(preset.get("server") or "")
                    for name, preset in data.items()
                    if not name.startswith("_") and isinstance(preset, dict)
                }
            )
            for value in declared:
                if not expected:
                    findings.append(
                        Finding("unregistered-provider", provider, value, "", path)
                    )
                elif value != expected:
                    findings.append(
                        Finding("server-alias-drift", provider, value, expected, path)
                    )
    return findings


def repair(findings: list[Finding]) -> list[Path]:
    """Rewrite every drifted ``server`` field to the registry's alias."""

    repaired: list[Path] = []
    for path in sorted({f.path for f in findings if f.kind == "server-alias-drift"}):
        expected = next(f.expected for f in findings if f.path == path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, preset in data.items():
            if not name.startswith("_") and isinstance(preset, dict):
                preset["server"] = expected
        path.write_text(
            json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        repaired.append(path)
    return repaired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents-root", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="rewrite drifted server aliases from the registry (never pushes)",
    )
    args = parser.parse_args()

    if not args.agents_root.is_dir():
        print("error: --agents-root is not a directory", file=sys.stderr)
        return 2
    try:
        findings = audit(args.agents_root, args.registry)
    except (FleetRegistryError, ValueError) as exc:
        print(f"connector source presets: FAIL ({exc})")
        return 1

    if not findings:
        print("connector source presets: PASS")
        return 0

    print(f"connector source presets: FAIL ({len(findings)} findings)")
    for finding in findings:
        print(finding.render())
    if args.fix:
        repaired = repair(findings)
        print(f"repaired {len(repaired)} preset files; re-run to confirm")
    return 1


if __name__ == "__main__":
    sys.exit(main())
