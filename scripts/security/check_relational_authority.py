#!/usr/bin/env python3
"""Fail-closed gate for the durable relational authority map."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_utilities.governance.relational_authority import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
