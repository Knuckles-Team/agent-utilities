#!/usr/bin/env python3
"""One topology authority: EG decides, AU never keeps a second one (ST-7).

SWARM-TOPOLOGY-DECIDE-DESIGN §2 converged three competing AU topology
authorities into EG's assembly decision. This static gate keeps them gone
(the design's §13 deletion proof), over every ``agent_utilities`` module:

* ``cypher-topology-write`` -- no string literal that writes a
  ``TopologyTemplate`` (``SET``/``MERGE``/``CREATE``/``DELETE``): topology
  vocabulary is a GraphSchema source and templates are published agent-graph
  templates, never AU Cypher (invariant T4);
* ``success-rate-selection`` -- no ``success_rate`` read in a topology
  selection module (invariant T5; learning is EG's calibrated rung);
* ``pattern-outcome-store`` -- no ``SubagentPatternDecision`` persisted as a
  generic node (the self-reported outcome store is deleted);
* ``second-topology-selector`` -- no ``choose(...)``/``achoose(...)`` call
  routing on the ``au.swarm.topology`` question (the topology is ONE
  AgentAssemble question), and none on ``au.reasoning.topology`` outside its
  one consumer module (EH-474);
* ``reasoning-outcome-store`` -- no outcome store on a reasoning-graph
  topology: no Cypher string writing a ``ReasoningTopologyVersion`` node and no
  ``reward``/``task_count``/``success_rate`` given to a
  ``ReasoningTopologyVersionNode`` (the self-reported EMA store is deleted;
  EG's ``au.reasoning.topology`` decision chooses, EH-474);
* ``team-success-rate`` -- no success-rate store or threshold on a
  ``TeamConfig``/``TopologyTemplate`` (a declared field, a constructor
  argument, a Cypher string, a ``reuse_threshold`` read): the router's former
  R2 TeamConfig reuse selected on exactly that.

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
    "agent_utilities/decide/consumers/reasoning_topology.py",
)
SELECTION_PACKAGES = (
    "agent_utilities/decide/topology/",
    "agent_utilities/graph/reasoning/",
    "agent_utilities/graph/routing/",
    "agent_utilities/graph/_router_impl.py",
)
#: Records a success rate may never be stored on or selected by.
_TEAM_RECORDS = ("TeamConfig", "TopologyTemplate")
_COUNTERS = frozenset({"success_rate", "usage_count", "reuse_threshold"})
_WRITE = re.compile(r"\b(SET|MERGE|CREATE|DELETE)\b")
#: The reasoning-topology resource and the outcome fields it may never carry.
_REASONING_RECORD = "ReasoningTopologyVersion"
_REASONING_COUNTERS = frozenset({"reward", "task_count", "success_rate"})
#: Decision points with at most one sanctioned call site (``None``: none at all).
_SELECTOR_HOMES: dict[str, str | None] = {
    "au.swarm.topology": None,
    "au.reasoning.topology": "agent_utilities/decide/consumers/reasoning_topology.py",
}
_CHOOSERS = frozenset({"choose", "achoose"})


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


def _is_selection_module(path: str) -> bool:
    return path in SELECTION_MODULES or path.startswith(SELECTION_PACKAGES)


def _success_rate_reads(tree: ast.AST, path: str) -> Iterator[Violation]:
    if not _is_selection_module(path):
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


def _chosen_questions(call: ast.Call) -> set[str]:
    literals = {a.value for a in call.args if isinstance(a, ast.Constant)}
    return {q for q in _SELECTOR_HOMES if q in literals}


def _second_selector(tree: ast.AST, path: str) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in _CHOOSERS:
            continue
        for question in _chosen_questions(node):
            if _SELECTOR_HOMES[question] != path:
                yield Violation(path, node.lineno, "second-topology-selector", question)


def _reasoning_outcome_store(tree: ast.AST, path: str) -> Iterator[Violation]:
    for node in _strings(tree):
        text = str(node.value)
        if _REASONING_RECORD in text and _WRITE.search(text):
            yield Violation(path, node.lineno, "reasoning-outcome-store", text[:60])
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and _call_name(call).startswith(
            _REASONING_RECORD
        ):
            for keyword in call.keywords:
                if keyword.arg in _REASONING_COUNTERS:
                    yield Violation(
                        path, call.lineno, "reasoning-outcome-store", keyword.arg
                    )


def _team_record(name: str) -> bool:
    return name.startswith(_TEAM_RECORDS)


def _counter_fields(tree: ast.AST, path: str) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and _team_record(node.name)):
            continue
        for item in node.body:
            target = getattr(item, "target", None)
            if isinstance(target, ast.Name) and target.id in _COUNTERS:
                yield Violation(path, item.lineno, "team-success-rate", target.id)


def _counter_arguments(tree: ast.AST, path: str) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _team_record(_call_name(node)):
            for keyword in node.keywords:
                if keyword.arg in _COUNTERS:
                    yield Violation(path, node.lineno, "team-success-rate", keyword.arg)


def _counter_queries(tree: ast.AST, path: str) -> Iterator[Violation]:
    for node in _strings(tree):
        text = str(node.value)
        if any(r in text for r in _TEAM_RECORDS) and "success_rate" in text:
            yield Violation(path, node.lineno, "team-success-rate", text[:60])
    for attribute in ast.walk(tree):
        if isinstance(attribute, ast.Attribute) and attribute.attr == "reuse_threshold":
            line = attribute.lineno
            yield Violation(path, line, "team-success-rate", "reuse_threshold")


Rule = Callable[[ast.AST, str], Iterator[Violation]]
RULES: tuple[Rule, ...] = (
    _cypher_topology_writes,
    _success_rate_reads,
    _pattern_outcome_store,
    _second_selector,
    _reasoning_outcome_store,
    _counter_fields,
    _counter_arguments,
    _counter_queries,
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
