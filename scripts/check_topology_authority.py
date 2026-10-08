#!/usr/bin/env python3
"""One topology authority: EG decides, AU never keeps a second one.

AU-CONTROL-R014/R015 converged competing AU topology authorities into EG's
assembly decision. This static gate keeps them gone, over every
``agent_utilities`` module:

* ``cypher-topology-write`` -- no string literal that writes a
  ``TopologyTemplate`` (``SET``/``MERGE``/``CREATE``/``DELETE``): topology
  vocabulary is a GraphSchema source and templates are published agent-graph
  templates, never AU Cypher;
* ``success-rate-selection`` -- no ``success_rate`` read in a topology
  selection module (learning is EG's calibrated rung, not a local EMA);
* ``pattern-outcome-store`` -- no ``SubagentPatternDecision`` persisted as a
  generic node (the self-reported outcome store is deleted);
* ``second-topology-selector`` -- no ``choose(...)`` call routing on the
  ``au.swarm.topology`` question: the topology is ONE AgentAssemble question.

Usage:
  python3 scripts/check_topology_authority.py [ROOT]

Exit 0 = clean; 1 = each violation printed as ``path:line: rule: detail``.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: Modules whose job is choosing a topology or composing a team.
SELECTION_MODULES = (
    "agent_utilities/graph/topology_engine.py",
    "agent_utilities/graph/team_composer.py",
    "agent_utilities/graph/subagent_patterns.py",
    "agent_utilities/graph/plan_admission.py",
    "agent_utilities/decide/consumers/topology.py",
)
SELECTION_PACKAGES = ("agent_utilities/decide/topology/",)
_WRITE = re.compile(r"\b(SET|MERGE|CREATE|DELETE)\b")


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


def _strings(tree: ast.AST) -> Iterator[ast.Constant]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node


def _cypher_topology_writes(tree: ast.AST, path: str) -> Iterator[Violation]:
    for node in _strings(tree):
        text = str(node.value)
        if "TopologyTemplate" in text and _WRITE.search(text):
            yield Violation(path, node.lineno, "cypher-topology-write", text[:60])


def _success_rate_reads(tree: ast.AST, path: str) -> Iterator[Violation]:
    is_selection_module = path in SELECTION_MODULES or path.startswith(
        SELECTION_PACKAGES
    )
    if not is_selection_module:
        return
    for node in ast.walk(tree):
        named = (isinstance(node, ast.Attribute) and node.attr == "success_rate") or (
            isinstance(node, ast.Constant) and node.value == "success_rate"
        )
        if named:
            line = getattr(node, "lineno", 0)
            yield Violation(path, line, "success-rate-selection", "success_rate read")


def _pattern_outcome_store(tree: ast.AST, path: str) -> Iterator[Violation]:
    for node in _strings(tree):
        if node.value == "SubagentPatternDecision":
            yield Violation(path, node.lineno, "pattern-outcome-store", node.value)


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _second_selector(tree: ast.AST, path: str) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "choose":
            continue
        literals = {a.value for a in node.args if isinstance(a, ast.Constant)}
        if "au.swarm.topology" in literals:
            yield Violation(path, node.lineno, "second-topology-selector", "choose()")


Rule = Callable[[ast.AST, str], Iterator[Violation]]
RULES: tuple[Rule, ...] = (
    _cypher_topology_writes,
    _success_rate_reads,
    _pattern_outcome_store,
    _second_selector,
)


def check_source(source: str, path: str) -> list[Violation]:
    """Every violation in one module's source."""
    tree = ast.parse(source, filename=path)
    return [violation for rule in RULES for violation in rule(tree, path)]


def check_tree(root: Path) -> list[Violation]:
    """Every violation under ``root/agent_utilities``."""
    found: list[Violation] = []
    for file in sorted((root / "agent_utilities").rglob("*.py")):
        relative = file.relative_to(root).as_posix()
        found.extend(check_source(file.read_text(encoding="utf-8"), relative))
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    violations = check_tree(root)
    for violation in violations:
        sys.stderr.write(f"{violation}\n")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
