"""Fleet-wide skill validation harness.

Discovers every ``SKILL.md`` under a set of repository roots (default:
agent-utilities + epistemic-graph, the two repos that own the graph-os skill
surface — see :mod:`agent_utilities.skills.fleet_harness.cli`) and runs two
independent layers over them:

- **static** (:mod:`.static_checks`): no services required. Frontmatter
  schema, name uniqueness, `skill_type` validity, and structural integrity
  (referenced files/tools actually resolve).
- **functional** (:mod:`.functional_checks`): a live client of a reachable
  graph-os MCP endpoint. Degrades to ``SKIPPED-unreachable`` — never a false
  PASS, never a hang — when no endpoint is reachable within a short timeout.

:mod:`.report` renders both layers to machine-readable JSON and a
human-readable table. :mod:`.cli` is the console-script entry point
(``agent-utilities-validate-skill-fleet``).
"""

from __future__ import annotations

from agent_utilities.skills.fleet_harness.discovery import SkillRecord, discover_skills
from agent_utilities.skills.fleet_harness.static_checks import (
    CheckResult,
    SkillStaticReport,
    run_static_checks,
)

__all__ = [
    "CheckResult",
    "SkillRecord",
    "SkillStaticReport",
    "discover_skills",
    "run_static_checks",
]
