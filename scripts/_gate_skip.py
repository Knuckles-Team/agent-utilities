"""Exit contract for a gate whose tool or sibling checkout is not present.

A gate that cannot run has not found nothing, but a fresh clone without an
optional native scanner is not a finding either. Locally the gate prints
``SKIPPED (<gate>): <reason>`` and exits 0; under CI (``CI`` set), where the
tool must be provisioned, it prints CANNOT RUN and exits 2.

Use it only for an ABSENT tool or checkout. A tool that is present but fails,
or a malformed configuration, is still a hard failure.
"""

from __future__ import annotations

import os
import sys
from typing import NoReturn


def unavailable(gate: str, reason: str) -> NoReturn:
    if os.environ.get("CI"):
        print(f"{gate}: CANNOT RUN: {reason}", file=sys.stderr)
        raise SystemExit(2)
    print(f"SKIPPED ({gate}): {reason}")
    raise SystemExit(0)
