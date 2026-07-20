#!/usr/bin/env python3
"""Run the distribution-owned bundled-skill contract validator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_utilities.skills.validation import main

if __name__ == "__main__":
    raise SystemExit(main())
