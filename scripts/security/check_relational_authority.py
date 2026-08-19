#!/usr/bin/env python3
"""Fail-closed gate for the durable relational authority map.

The checked-in map is also the machine-readable authority/placement contract.
The older relational-domain section describes the concrete tables that already
exist in Agent Utilities; the ``authority_placement`` section describes the
cross-system boundary for the remaining control-plane and payload concerns.
Keeping both sections in one source prevents a documentation-only authority
decision from drifting away from the schemas that the gate can inspect.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_utilities.governance.relational_authority import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
