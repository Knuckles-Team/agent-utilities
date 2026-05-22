#!/usr/bin/env python3
import re
from pathlib import Path

DOCS_DIR = Path("docs")
CONCEPT_MAP = DOCS_DIR / "concept_map.md"


def load_canonical_concepts():
    """Extract all valid CONCEPT IDs from concept_map.md."""
    valid_concepts = set()
    if not CONCEPT_MAP.exists():
        print(f"Error: {CONCEPT_MAP} not found!")
        return valid_concepts

    with open(CONCEPT_MAP) as f:
        content = f.read()

    # Matches `ORCH-1.0`, `KG-2.12`, etc. inside the markdown table
    matches = re.findall(r"`([A-Z]+-\d+\.\d+)`", content)
    for m in matches:
        valid_concepts.add(m)
    return valid_concepts


def parse_mermaid_nodes(content):
    """Extract nodes with Concept IDs from Mermaid blocks, handles multi-node lines robustly."""
    nodes = []
    in_mermaid = False

    for line in content.split("\n"):
        if line.strip().startswith("```mermaid"):
            in_mermaid = True
            continue
        if in_mermaid and line.strip() == "```":
            in_mermaid = False
            continue

        if in_mermaid:
            line_str = line.strip()
            # Skip empty lines, comments, and structure definitions like 'subgraph' or 'direction'
            if (
                not line_str
                or line_str.startswith("%%")
                or line_str.startswith("subgraph")
                or line_str.startswith("direction")
                or line_str.startswith("style")
                or line_str.startswith("end")
                or line_str.startswith("graph")
                or line_str.startswith("flowchart")
                or line_str.startswith("C4Context")
                or line_str.startswith("C4Container")
                or line_str.startswith("C4Component")
                or line_str.startswith("title")
            ):
                continue

            # Exclude C4 diagram nodes which do not map to canonical concepts
            if any(
                line_str.startswith(x)
                for x in [
                    "Person",
                    "System",
                    "System_Ext",
                    "Container",
                    "Component",
                    "Rel",
                ]
            ):
                continue

            # Split line by arrow connections or transitions to process multi-node lines
            parts = re.split(
                r"\s*(?:--+|-\.-+|==+)(?:>|<|>)?(?:\|[^|]+\|)?\s*", line_str
            )
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Match node definitions like A[Label], A("Label"), etc.
                match = re.match(r"^([A-Za-z0-9_]+)\s*([\[\(\{>]+.*[\]\)\}]+)$", part)
                if match:
                    brackets_label = match.group(2)
                    # Extract text inside the outermost brackets/quotes
                    label_match = re.search(
                        r'[\[\(\{">]+(.*)[\]\)\}"]+', brackets_label
                    )
                    label = label_match.group(1) if label_match else brackets_label
                    label = label.replace('"', "").strip()
                    # Clean up any trailing brackets/quotes
                    label = re.sub(r'^[\[\(\{">]+|[\]\)\}"]+$', "", label).strip()

                    # Exclude non-architectural process nodes
                    if any(
                        x in label.lower()
                        for x in [
                            "phase ",
                            "stage ",
                            "step ",
                            "background research",
                            "synthesis",
                            "feature recommendations",
                            "wiring audit",
                        ]
                    ):
                        continue
                    # Exclude basic shapes and boundaries that represent generic groupings
                    if any(
                        y in label.lower()
                        for y in [
                            "<b>",
                            "<br",
                            "pydantic",
                            "scripts/",
                            "git:",
                            "fastapi",
                            "vite",
                            "react",
                            "textual",
                            "rich",
                            "httpx",
                            "neo4j",
                            "networkx",
                            "database",
                            "sqlite",
                            "postgresql",
                        ]
                    ):
                        continue
                    if any(
                        y == label.lower()
                        for y in [
                            "nx",
                            "val",
                            "exp",
                            "evo",
                            "db",
                            "ui",
                            "api",
                            "cli",
                            "auth",
                            "mcp",
                            "htn",
                            "c4",
                        ]
                    ):
                        continue

                    concept_match = re.search(r"([A-Z]+-\d+\.\d+)", part)
                    if concept_match:
                        nodes.append((concept_match.group(1), part))
                    else:
                        nodes.append((None, part))

    return nodes


def validate_diagrams():
    valid_concepts = load_canonical_concepts()
    print(f"Loaded {len(valid_concepts)} canonical concepts.")

    invalid_usages = []
    missing_ids = []
    total_nodes = 0
    valid_nodes = 0

    for md_file in DOCS_DIR.rglob("*.md"):
        with open(md_file) as f:
            content = f.read()

        nodes = parse_mermaid_nodes(content)
        for concept_id, line in nodes:
            total_nodes += 1
            if concept_id is None:
                missing_ids.append((md_file, line))
            elif concept_id not in valid_concepts:
                invalid_usages.append((md_file, concept_id, line))
            else:
                valid_nodes += 1

    if total_nodes > 0:
        coverage = (valid_nodes / total_nodes) * 100
        print(
            f"Diagram Concept Coverage: {coverage:.2f}% ({valid_nodes}/{total_nodes} nodes mapped)"
        )
    else:
        print("No nodes found.")

    if invalid_usages:
        print(
            f"❌ Found {len(invalid_usages)} invalid concept IDs in Mermaid diagrams:"
        )
        for md_file, concept_id, line in invalid_usages:
            print(
                f"  - {md_file.relative_to(DOCS_DIR)}: '{concept_id}' -> Line: {line}"
            )

    if missing_ids:
        print(f"⚠️ Found {len(missing_ids)} nodes missing Concept IDs:")
        for md_file, line in missing_ids[:20]:  # show first 20
            print(f"  - {md_file.relative_to(DOCS_DIR)}: {line}")
        if len(missing_ids) > 20:
            print(f"  ... and {len(missing_ids) - 20} more.")

    if not invalid_usages and not missing_ids:
        print(
            "✅ 100% Concept Coverage! All diagrams are fully mapped to valid concepts."
        )
        return 0
    return 1


if __name__ == "__main__":
    exit(validate_diagrams())
