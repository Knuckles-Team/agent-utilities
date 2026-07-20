#!/usr/bin/env python3
"""Source-tree self-check for the lifecycle-owned certification authority."""

# ruff: noqa: I001 - executable wrapper keeps its import below the module docstring
from agent_utilities.deployment.certification_oidc import main


if __name__ == "__main__":
    raise SystemExit(main())
