import json
from pathlib import Path
from typing import Dict, Any, List
from .developer_tools import logger
from ..models import AgentDeps


def detect_tech_stack(root: Path) -> Dict[str, Any]:
    """Detects technologies used in the project."""
    stack = {"languages": [], "frameworks": [], "tools": []}

    # Language detection by file presence
    if (root / "package.json").exists():
        stack["languages"].append("TypeScript/JavaScript")
        stack["tools"].append("npm/yarn")
    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
        stack["languages"].append("Python")
    if (root / "go.mod").exists():
        stack["languages"].append("Go")
    if (root / "Cargo.toml").exists():
        stack["languages"].append("Rust")
    if (root / "pom.xml").exists() or (root / "build.gradle").exists():
        stack["languages"].append("Java")

    return stack


def scan_for_entry_points(root: Path) -> List[str]:
    """Identifies potential main entry points."""
    entries = []
    # Common entry patterns
    patterns = [
        "src/main.ts",
        "src/index.ts",
        "src/app.ts",
        "main.py",
        "app.py",
        "run.py",
        "cmd/main.go",
        "src/main.rs",
        "index.js",
    ]
    for p in patterns:
        if (root / p).exists():
            entries.append(p)
    return entries


def bootstrap_project(deps: AgentDeps) -> str:
    """
    Scans the workspace to identify tech stack and entry points,
    then populates AGENTS.md and MEMORY.md with baseline technical metadata.
    """
    root = deps.workspace_path
    logger.info(f"Bootstrapping project at {root}...")

    stack = detect_tech_stack(root)
    entries = scan_for_entry_points(root)

    # Prepare AGENTS.md content
    agents_content = "# Peer Registry (AGENTS.md)\n\n"
    agents_content += "## Known Agents\n"
    agents_content += "- **Self**: {deps.session_id or 'current_session'}\n\n"
    agents_content += "## Project Overview\n"
    agents_content += f"- **Languages**: {', '.join(stack['languages'])}\n"
    agents_content += f"- **Key Entries**: {', '.join(entries)}\n"

    # Prepare MEMORY.md content
    memory_content = "# Project Memory (MEMORY.md)\n\n"
    memory_content += "## Tech Stack Metadata\n"
    memory_content += json.dumps(stack, indent=2) + "\n\n"
    memory_content += "## Architectural Context\n"
    memory_content += f"Initial scan performed on {Path.cwd()}. Detected {len(entries)} entry points.\n"

    # Write files if they don't exist
    agents_path = root / "AGENTS.md"
    memory_path = root / "MEMORY.md"

    written = []
    if not agents_path.exists():
        with open(agents_path, "w") as f:
            f.write(agents_content)
        written.append("AGENTS.md")

    if not memory_path.exists():
        with open(memory_path, "w") as f:
            f.write(memory_content)
        written.append("MEMORY.md")

    if written:
        return f"Successfully bootstrapped {', '.join(written)} based on detected tech stack: {stack}"
    else:
        return f"Project already has metadata. Detected tech stack: {stack}"


# Registration list
onboarding_tools = [bootstrap_project]
