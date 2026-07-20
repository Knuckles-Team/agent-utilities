#!/usr/bin/env python3
"""Enforce one engine-native ``WorkItem`` lifecycle authority.

``agent_utilities/orchestration/work_item.py`` is the only module allowed to
define or persist WorkItem lifecycle transitions.  The AgentBus inbox may
construct its initial WorkItem inside the same transaction as the inbox, but it
may not implement a transition.  Passive schemas, protocol objects, and
read-only label references remain valid.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path, PurePosixPath

_PACKAGE = PurePosixPath("agent_utilities")
_AUTHORITY = PurePosixPath("agent_utilities/orchestration/work_item.py")
_BUS_INBOX = PurePosixPath("agent_utilities/messaging/bus_inbox.py")
_PASSIVE_PARTS = frozenset({"models", "protocols", "schemas"})

_LIFECYCLE_CLASSES = frozenset(
    {"WorkItem", "WorkItemStatus", "WorkItemPhase", "OrgPhase"}
)
_EXPLICIT_TRANSITION_FUNCTIONS = frozenset(
    {
        "cancel_work_item",
        "claim_work_item",
        "commit_work_item",
        "defer_work_item",
        "renew_work_item",
        "set_work_item_status",
        "transition_work_item",
    }
)
_TRANSITION_CALLS = _EXPLICIT_TRANSITION_FUNCTIONS | frozenset(
    {
        "cancel_work_item",
        "checkpoint_work_item",
        "claim_next",
        "claim_specific",
        "commit_result",
        "defer_work_item",
        "heartbeat",
        "mark_running",
        "reap_expired_leases",
        "set_work_item_priority",
    }
)
_WRITE_CALLS = frozenset(
    {
        "add_node",
        "add_nodes",
        "apply_change_envelope",
        "compare_and_set_node_fields",
        "create_node",
        "execute_mutation",
        "execute_write",
        "ingest_external_batch",
        "ingest_graph_slice",
        "merge_node",
        "put_node",
        "save_node",
        "set_node_fields",
        "update_node",
        "upsert_node",
        "write_node",
        "write_nodes",
    }
)
_WORK_ITEM_TAG_KEYS = frozenset({"label", "node_type", "nodeType", "type"})
_LIFECYCLE_KEYS = frozenset(
    {
        "attempt",
        "completed_at",
        "deadline_unix",
        "dep_count",
        "depends_on",
        "downstream_ids",
        "error_ref",
        "fairness_group",
        "fencing_token",
        "heartbeat_at",
        "kanbanColumn",
        "lease_epoch",
        "lease_expires_at",
        "lease_owner",
        "max_attempts",
        "next_retry_at",
        "phase",
        "prio_bucket",
        "queue",
        "rework_count",
        "result_ref",
        "state",
        "status",
        "workItemPhase",
    }
)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _contains_work_item_literal(node: ast.AST) -> bool:
    return any(_literal_string(child) == "WorkItem" for child in ast.walk(node))


def _dict_items(node: ast.Dict) -> dict[str, ast.AST]:
    items: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        literal = _literal_string(key) if key is not None else None
        if literal is not None:
            items[literal] = value
    return items


def _is_work_item_lifecycle_payload(node: ast.Dict) -> bool:
    items = _dict_items(node)
    tagged = any(
        key in items and _literal_string(items[key]) == "WorkItem"
        for key in _WORK_ITEM_TAG_KEYS
    )
    return tagged and bool(_LIFECYCLE_KEYS.intersection(items))


def _is_direct_write(name: str) -> bool:
    return name.lstrip("_") in _WRITE_CALLS


def _is_passive_schema(relative: PurePosixPath) -> bool:
    return bool(_PASSIVE_PARTS.intersection(relative.parts))


def _is_orchestration_module(relative: PurePosixPath) -> bool:
    return relative.parts[:2] == ("agent_utilities", "orchestration")


class _BoundaryVisitor(ast.NodeVisitor):
    def __init__(self, relative: PurePosixPath) -> None:
        self.relative = relative
        self.findings: list[str] = []
        self._functions: list[str] = []

    def _add(self, node: ast.AST, message: str) -> None:
        self.findings.append(f"{self.relative}:{node.lineno}: {message}")

    @property
    def _function(self) -> str:
        return self._functions[-1] if self._functions else ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if (
            self.relative != _AUTHORITY
            and not _is_passive_schema(self.relative)
            and node.name in _LIFECYCLE_CLASSES
        ):
            self._add(
                node,
                f"parallel WorkItem lifecycle class {node.name!r}; use "
                "orchestration.work_item",
            )
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if (
            self.relative != _AUTHORITY
            and (_is_orchestration_module(self.relative) or self.relative == _BUS_INBOX)
            and node.name in _EXPLICIT_TRANSITION_FUNCTIONS
        ):
            self._add(
                node,
                f"parallel WorkItem transition {node.name!r}; use "
                "orchestration.work_item",
            )
        self._functions.append(node.name)
        self.generic_visit(node)
        self._functions.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.relative != _AUTHORITY:
            name = _call_name(node.func)
            if _is_direct_write(name) and _contains_work_item_literal(node):
                self._add(
                    node,
                    "direct WorkItem persistence bypasses the native authority",
                )
            if name == "WorkItemNode" and not (
                _is_passive_schema(self.relative)
                or (self.relative == _BUS_INBOX and self._function == "_properties")
            ):
                self._add(
                    node,
                    "WorkItemNode materialization is only permitted in the native "
                    "authority or AgentBus inbox transaction",
                )
            if self.relative == _BUS_INBOX and name in _TRANSITION_CALLS:
                self._add(
                    node,
                    f"AgentBus inbox invokes WorkItem transition {name!r}",
                )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if (
            self.relative not in {_AUTHORITY, _BUS_INBOX}
            and not _is_passive_schema(self.relative)
            and _is_work_item_lifecycle_payload(node)
        ):
            self._add(
                node,
                "WorkItem lifecycle payload is materialized outside the native "
                "authority",
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _literal_string(node.value) == "WorkItem":
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                key = _literal_string(target.slice)
                if key not in _WORK_ITEM_TAG_KEYS:
                    continue
                allowed = self.relative == _AUTHORITY or _is_passive_schema(
                    self.relative
                )
                allowed = allowed or (
                    self.relative == _BUS_INBOX and self._function == "_properties"
                )
                if not allowed:
                    self._add(
                        node,
                        "WorkItem payload tagging is outside an approved "
                        "materialization boundary",
                    )
        self.generic_visit(node)


def violations(path: Path, *, root: Path) -> list[str]:
    """Return deterministic boundary violations for one Python module."""
    relative = PurePosixPath(path.relative_to(root).as_posix())
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
    except (OSError, SyntaxError) as exc:
        return [f"{relative}: cannot inspect WorkItem boundary ({type(exc).__name__})"]
    visitor = _BoundaryVisitor(relative)
    visitor.visit(tree)
    return visitor.findings


def check(root: Path) -> list[str]:
    """Scan production modules and return every WorkItem authority violation."""
    package = root / _PACKAGE
    if not package.is_dir():
        return ["agent_utilities: package root is missing"]
    findings: list[str] = []
    for path in sorted(package.rglob("*.py")):
        findings.extend(violations(path, root=root))
    return findings


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]).resolve() if args else Path(__file__).resolve().parents[1]
    findings = check(root)
    if findings:
        print("Native WorkItem boundary gate failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("Native WorkItem boundary gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
