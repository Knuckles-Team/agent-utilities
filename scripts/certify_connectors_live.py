#!/usr/bin/env python3
"""Source-checkout wrapper for the installed connector certification CLI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.integrations.connector_certification_cli import (  # noqa: E402
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
