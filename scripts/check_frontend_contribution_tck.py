#!/usr/bin/env python3
"""GOC-24 qualification: run the FrontendContribution.v1 TCK over installed packages.

Discovers every installed ``agent_utilities.frontend_providers`` entry point
(via ``agent_utilities.core.frontend_providers.discover_frontend_contributions``),
prints one line per package with its terminal status/reason -- OK/DEGRADED
packages are visible, BLOCKED/MISSING packages are visible with a reason,
nothing is silently skipped -- and exits non-zero if any package is BLOCKED
(a MISSING descriptor is tolerated during migration per the lane's rollout
policy; a BLOCKED descriptor is not).

The trusted-signer allowlist is read from
``FRONTEND_CONTRIBUTION_TRUSTED_SIGNERS`` (comma-separated signer key ids) via
the sanctioned ``config.setting`` accessor. An unset/empty allowlist fails
every descriptor closed by design (see the module docstring on provenance) --
this is not a bug in the gate, it is the fail-closed default until a real
signer trust store is wired (tracked as an explicit gap, not silently papered
over -- see the GOC-24 handoff).
"""

from __future__ import annotations

import argparse
import json

from agent_utilities.core.config import setting
from agent_utilities.core.frontend_providers import (
    catalog_digest,
    discover_frontend_contributions,
)


def _trusted_signers() -> frozenset[str]:
    raw = setting("FRONTEND_CONTRIBUTION_TRUSTED_SIGNERS", "", cast=str)
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit a JSON report instead of text"
    )
    args = parser.parse_args()

    records = discover_frontend_contributions(trusted_signers=_trusted_signers())
    epoch = catalog_digest(records)

    if args.json:
        print(
            json.dumps(
                {
                    "catalog_epoch": epoch,
                    "packages": [
                        {
                            "package_id": r.package_id,
                            "provider_name": r.provider_name,
                            "status": r.status,
                            "reason": r.reason,
                            "descriptor_digest": r.descriptor_digest,
                        }
                        for r in records
                    ],
                },
                indent=2,
            )
        )
    else:
        for record in records:
            suffix = f" ({record.reason})" if record.reason else ""
            print(f"{record.provider_name}: {record.status}{suffix}")
        counts: dict[str, int] = {}
        for record in records:
            counts[record.status] = counts.get(record.status, 0) + 1
        print(
            f"frontend contribution TCK: {len(records)} package(s) -- "
            + ", ".join(
                f"{status}={counts.get(status, 0)}"
                for status in ("OK", "DEGRADED", "BLOCKED", "MISSING")
            )
            + f" -- catalog_epoch={epoch}"
        )

    blocked = [r for r in records if r.status == "BLOCKED"]
    if blocked:
        print(f"frontend contribution TCK: FAIL ({len(blocked)} blocked package(s))")
        return 1
    print("frontend contribution TCK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
