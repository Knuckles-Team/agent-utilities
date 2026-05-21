#!/usr/bin/env python3
import os
import re
from pathlib import Path

DOCS_DIR = Path("docs")


def parse_mermaid_nodes(content):
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

            parts = re.split(
                r"\s*(?:--+|-\.-+|==+)(?:>|<|>)?(?:\|[^|]+\|)?\s*", line_str
            )
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                match = re.match(r'^([A-Za-z0-9_]+)\s*([\[\(\{>]+.*[\]\)\}"]+)$', part)
                if match:
                    brackets_label = match.group(2)
                    label_match = re.search(
                        r'[\[\(\{">]+(.*)[\]\)\}"]+', brackets_label
                    )
                    label = label_match.group(1) if label_match else brackets_label
                    label = label.replace('"', "").strip()
                    label = re.sub(r'^[\[\(\{">]+|[\]\)\}"]+$', "", label).strip()

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
                    if not concept_match:
                        nodes.append(label)
    return nodes


def main():
    unmapped = set()
    for md_file in DOCS_DIR.rglob("*.md"):
        with open(md_file, "r") as f:
            content = f.read()
        for label in parse_mermaid_nodes(content):
            unmapped.add(label)

    print(f"Total unique unmapped labels: {len(unmapped)}")
    for lbl in sorted(unmapped)[:100]:
        print(f"  - {lbl}")


if __name__ == "__main__":
    main()
