#!/usr/bin/env python3
"""Concept Registry and Documentation Validator.

Robustly parses the canonical concept registry in docs/concept_map.md
and ensures every concept has its dedicated documentation file on disk.
"""

import os
import re
from pathlib import Path


def main():
    root = Path("/home/apps/workspace/agent-packages/agent-utilities")
    concept_map_path = root / "docs/concept_map.md"

    if not concept_map_path.exists():
        print(f"Error: {concept_map_path} does not exist.")
        return

    content = concept_map_path.read_text(encoding="utf-8")

    # Find the table rows of the form:
    # | `CONCEPT-ID` | Canonical Name | ... | [Doc Name](relative_path) |
    rows = re.findall(
        r"\|\s*`([A-Z]+-\d+\.\d+)`\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|",
        content,
    )

    missing_docs = []
    total_docs = 0

    print(
        f"Validating concept documentation from {concept_map_path.relative_to(root)}...\n"
    )

    for row in rows:
        concept_id = row[0].strip()
        concept_name = row[1].strip()
        doc_cell = row[4].strip()

        # Parse markdown link: [Text](path) or just path
        match = re.search(r"\[([^\]]+)\]\(([^)]+)\)", doc_cell)
        if match:
            doc_path_str = match.group(2)
        else:
            doc_path_str = doc_cell

        # Clean anchor tags if any
        doc_path_str = doc_path_str.split("#")[0]

        # Check if the doc page points to a summary rather than a dedicated sub-doc
        if "Pillar Summary" in doc_cell or "Summary" in doc_cell:
            # Resolved to the pillar summary main doc
            full_path = root / "docs" / doc_path_str
        else:
            full_path = root / "docs" / doc_path_str

        total_docs += 1
        if not full_path.exists():
            missing_docs.append((concept_id, concept_name, doc_path_str))

    print(f"Total Concepts Checked: {total_docs}")
    if missing_docs:
        print(f"Missing Doc Files: {len(missing_docs)}")
        for c_id, c_name, path in missing_docs:
            print(f"  ❌ {c_id} - {c_name}: docs/{path} (Not Found)")
        # Exit with error to plug into CI/pre-commit
        os._exit(1)
    else:
        print(
            "  ✅ All concept documentation files exist on disk! 100% concept integrity."
        )


if __name__ == "__main__":
    main()
