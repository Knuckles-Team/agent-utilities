#!/usr/bin/env python3
import os
import re
import sys
from collections import defaultdict


def main():
    # If files are provided by pre-commit, search among them for concept_map.md
    files = sys.argv[1:]

    # If no files are provided, default to docs/concept_map.md
    concept_map_file = "docs/concept_map.md"

    if files:
        for f in files:
            if f.endswith("concept_map.md"):
                concept_map_file = f
                break

        # If concept_map.md is not among the changed files, we can just skip or still check it.
        # But for stability, we will just check it if it exists.

    if not os.path.exists(concept_map_file):
        # We only enforce this in repos that have a concept_map.md (i.e. agent-utilities)
        sys.exit(0)

    with open(concept_map_file, encoding="utf-8") as f:
        content = f.read()

    # Match things like `ORCH-1.0` or `KG-2.15`
    # We look for the table definitions or just any bold/code block definitions.
    # Pattern: `([A-Z]+)-(\d+)\.(\d+)`
    pattern = re.compile(r"`(ORCH|KG|AHE|ECO|OS|GBOT|GW)-(\d+)\.(\d+)`")

    pillars = defaultdict(set)
    for match in pattern.finditer(content):
        pillar_name = match.group(1)
        pillar_num = int(match.group(2))
        concept_num = int(match.group(3))
        pillars[f"{pillar_name}-{pillar_num}"].add(concept_num)

    has_errors = False
    for pillar, numbers in pillars.items():
        if not numbers:
            continue
        max_num = max(numbers)
        expected = set(range(max_num + 1))
        missing = expected - numbers
        if missing:
            print(f"ERROR: Pillar {pillar} has missing numbers/gaps: {sorted(missing)}")
            has_errors = True
        if 0 not in numbers:
            print(f"ERROR: Pillar {pillar} is not 0-indexed (missing 0).")
            has_errors = True

    if has_errors:
        sys.exit(1)

    # print("Concept ID validation passed: No gaps found and all are 0-indexed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
