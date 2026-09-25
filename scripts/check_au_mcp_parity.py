"""Check the legacy AU MCP action inventory against a GraphOS operation registry.

The mapping may carry declaration candidates during migration. A candidate is
never counted as parity. Release mode requires an exact served operation ID or
an explicit reasoned drop for every legacy action.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _ids(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("ops", [])
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected an operation list")
    return {item["id"] if isinstance(item, dict) else item for item in value}


def check(
    inventory: Path,
    mapping: Path,
    registry: Path,
    served_ops: Path | None = None,
    *,
    require_complete: bool = False,
) -> dict[str, int]:
    legacy = _rows(inventory)
    mapped = _rows(mapping)
    names = [row["name"] for row in legacy]
    mapped_names = [row["name"] for row in mapped]
    if len(names) != len(set(names)) or len(mapped_names) != len(set(mapped_names)):
        raise ValueError("duplicate legacy or mapping action name")
    if set(names) != set(mapped_names):
        raise ValueError(
            f"mapping coverage mismatch: missing={sorted(set(names)-set(mapped_names))}, "
            f"extra={sorted(set(mapped_names)-set(names))}"
        )
    legacy_by_name = {row["name"]: row for row in legacy}
    for row in mapped:
        source = legacy_by_name[row["name"]]
        if (row["legacy_tool"], row["legacy_action"]) != (
            source["tool"], source["action"]
        ):
            raise ValueError(f"{row['name']}: legacy tool/action drifted")

    declared = _ids(registry)
    served = _ids(served_ops) if served_ops else set()
    if require_complete and served_ops is None:
        raise ValueError("release check requires served operation evidence")
    if not declared:
        raise ValueError("empty registry cannot establish parity")
    if served - declared:
        raise ValueError("served operation list contains undeclared IDs")

    counts = {"pending": 0, "op": 0, "drop": 0, "candidates": 0}
    for row in mapped:
        name = row["name"]
        disposition = row["disposition"]
        target = row["target"]
        reason = row["reason"].strip()
        candidate = row.get("candidate_op", "")
        if candidate:
            if candidate not in declared:
                raise ValueError(f"{name}: candidate op is absent from registry: {candidate}")
            counts["candidates"] += 1
        if disposition == "pending":
            if target:
                raise ValueError(f"{name}: pending action has a target")
        elif disposition == "op":
            if target not in declared:
                raise ValueError(f"{name}: target op is absent from registry: {target}")
            if target not in served:
                raise ValueError(f"{name}: target op has no served binding proof: {target}")
        elif disposition == "drop":
            if target or not reason:
                raise ValueError(f"{name}: drop needs an empty target and reason")
        else:
            raise ValueError(f"{name}: unknown disposition: {disposition}")
        counts[disposition] += 1

    if require_complete and counts["pending"]:
        raise ValueError(f"{counts['pending']} actions still pending")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("registry", type=Path, help="canonical GraphOS registry JSON")
    parser.add_argument("--served-ops", type=Path, help="IDs with runtime binding proof")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        counts = check(
            args.inventory,
            args.mapping,
            args.registry,
            args.served_ops,
            require_complete=args.require_complete,
        )
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"AU MCP parity: {error}\n")
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
