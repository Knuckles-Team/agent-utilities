#!/usr/bin/env python3
import re
import sys
from pathlib import Path

# Add the parent directory to PYTHONPATH if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_utilities.models.knowledge_pack import (
    KnowledgePackBundle,
    KnowledgePackExporter,
    generate_deterministic_id,
)


def parse_readme(file_path: Path) -> KnowledgePackBundle:
    content = file_path.read_text(encoding="utf-8")

    nodes = []
    edges = []

    # We will create a central node for the pack
    pack_node_id = generate_deterministic_id("ARTICLE", "awesome-deep-trading")
    nodes.append(
        {
            "type": "article",
            "id": pack_node_id,
            "name": "Awesome Deep Trading",
            "description": "List of code, papers, and resources for AI/deep learning/machine learning/neural networks applied to algorithmic trading.",
            "url": "https://github.com/cbailes/awesome-deep-trading",
        }
    )

    current_category = None

    # Simple regex to match markdown links with optional description
    # e.g., * [Title](URL) - Description
    item_pattern = re.compile(r"^\s*[-*]\s+\[(.*?)\]\((.*?)\)(?:\s*-\s*(.*))?")
    category_pattern = re.compile(r"^##\s+(.*)")

    for line in content.splitlines():
        cat_match = category_pattern.match(line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue

        item_match = item_pattern.match(line)
        if item_match:
            title = item_match.group(1).strip()
            url = item_match.group(2).strip()
            description = item_match.group(3).strip() if item_match.group(3) else ""

            is_repo = "github.com" in url
            node_type = "software_project" if is_repo else "article"

            node_id = generate_deterministic_id(node_type, url)

            node = {
                "type": node_type,
                "id": node_id,
                "name": title,
                "url": url,
            }
            if description:
                node["description"] = description
            if current_category:
                node["metadata"] = {"category": current_category}

            nodes.append(node)

            # Link to the main pack
            edges.append(
                {
                    "source": pack_node_id,
                    "target": node_id,
                    "type": "contains",  # Generic edge type, maybe 'mentions' or 'includes'
                    "metadata": {"category": current_category}
                    if current_category
                    else {},
                }
            )

    bundle = KnowledgePackBundle(
        name="awesome-deep-trading",
        domain="finance",
        version="1.0",
        description="A curated list of deep learning applied to algorithmic trading.",
        nodes=nodes,
        edges=edges,
    )
    return bundle


if __name__ == "__main__":
    readme_path = Path(
        "/home/apps/workspace/open-source-libraries/awesome-deep-trading/README.md"
    )
    if not readme_path.exists():
        print(f"Error: {readme_path} not found.")
        sys.exit(1)

    bundle = parse_readme(readme_path)

    out_path = Path(
        "agent_utilities/workflows/presets/finance/awesome-deep-trading-pack.yaml"
    )
    KnowledgePackExporter.to_yaml(bundle, out_path)
    print(
        f"Exported bundle to {out_path} with {len(bundle.nodes)} nodes and {len(bundle.edges)} edges."
    )
