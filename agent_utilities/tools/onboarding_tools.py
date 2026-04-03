import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from pydantic_ai import RunContext

from ..models import (
    AgentDeps,
    IdentityModel,
    UserModel,
    A2ARegistryModel,
    MCPAgentRegistryModel,
)
from ..workspace import (
    CORE_FILES,
    serialize_identity,
    serialize_user_info,
    serialize_a2a_registry,
    serialize_mcp_registry,
)

logger = logging.getLogger(__name__)


def detect_tech_stack(root: Path) -> Dict[str, Any]:
    """Identify languages and frameworks in the repository."""
    stack = {"languages": [], "frameworks": [], "tools": []}

    # Check for specific markers
    markers = {
        "package.json": ("JavaScript/TypeScript", "Node.js"),
        "pyproject.toml": ("Python", "Poetry/Base"),
        "requirements.txt": ("Python", "Pip"),
        "go.mod": ("Go", "Go Modules"),
        "Cargo.toml": ("Rust", "Cargo"),
        "composer.json": ("PHP", "Composer"),
        "Gemfile": ("Ruby", "Bundler"),
        "pom.xml": ("Java", "Maven"),
        "build.gradle": ("Java/Kotlin", "Gradle"),
        "Dockerfile": (None, "Docker"),
        "compose.yaml": (None, "Docker Compose"),
        "docker-compose.yml": (None, "Docker Compose"),
        "next.config.js": (None, "Next.js"),
        "vite.config.ts": (None, "Vite"),
        "tailwind.config.js": (None, "Tailwind CSS"),
    }

    found_files = [f.name for f in root.iterdir() if f.is_file()]

    for filename, (lang, tool) in markers.items():
        if filename in found_files:
            if lang and lang not in stack["languages"]:
                stack["languages"].append(lang)
            if tool and tool not in stack["tools"]:
                stack["tools"].append(tool)

    return stack


def scan_for_entry_points(root: Path) -> List[str]:
    """Find potential main interaction points."""
    entry_patterns = [
        "main.py",
        "app.py",
        "server.py",
        "index.ts",
        "index.js",
        "server.js",
        "main.go",
        "src/main.rs",
        "manage.py",
    ]

    found = []
    for pattern in entry_patterns:
        # Check root and one level deep
        if (root / pattern).exists():
            found.append(pattern)
        else:
            # Check for pattern in subdirs (e.g. src/index.ts)
            for item in root.iterdir():
                if (
                    item.is_dir()
                    and not item.name.startswith(".")
                    and (item / pattern).exists()
                ):
                    found.append(f"{item.name}/{pattern}")

    return found


async def bootstrap_project(ctx: RunContext[AgentDeps]) -> str:
    """
    Scans the workspace to identify tech stack and entry points,
    then populates core metadata files (IDENTITY.md, USER.md, A2A_AGENTS.md, MCP_AGENTS.md, MEMORY.md).
    """
    root = ctx.deps.workspace_path
    logger.info(f"Bootstrapping project at {root}...")

    stack = detect_tech_stack(root)
    entries = scan_for_entry_points(root)

    written = []

    # 1. IDENTITY.md
    identity_path = root / CORE_FILES["IDENTITY"]
    if not identity_path.exists():
        id_model = IdentityModel(
            name="New Agent",
            role="Technical Specialist",
            emoji="🤖",
            vibe="Professional and efficient",
            system_prompt="You are a helpful coding assistant for this project.",
        )
        identity_path.write_text(serialize_identity(id_model), encoding="utf-8")
        written.append(CORE_FILES["IDENTITY"])

    # 2. USER.md
    user_path = root / CORE_FILES["USER"]
    if not user_path.exists():
        u_model = UserModel(name="The Human", emoji="👤")
        user_path.write_text(serialize_user_info(u_model), encoding="utf-8")
        written.append(CORE_FILES["USER"])

    # 3. A2A_AGENTS.md
    a2a_path = root / CORE_FILES["A2A_AGENTS"]
    if not a2a_path.exists():
        a2a_path.write_text(
            serialize_a2a_registry(A2ARegistryModel(peers=[])), encoding="utf-8"
        )
        written.append(CORE_FILES["A2A_AGENTS"])

    # 4. MCP_AGENTS.md
    mcp_path = root / CORE_FILES["MCP_AGENTS"]
    if not mcp_path.exists():
        mcp_path.write_text(
            serialize_mcp_registry(MCPAgentRegistryModel(agents=[], tools=[])),
            encoding="utf-8",
        )
        written.append(CORE_FILES["MCP_AGENTS"])

    # 5. MEMORY.md
    memory_path = root / CORE_FILES["MEMORY"]
    if not memory_path.exists():
        memory_content = "# Project Memory (MEMORY.md)\n\n"
        memory_content += "## Tech Stack Metadata\n"
        memory_content += json.dumps(stack, indent=2) + "\n\n"
        memory_content += "## Architectural Context\n"
        memory_content += f"Initial scan performed on {datetime.now().strftime('%Y-%m-%d')}. Detected {len(entries)} entry points: {', '.join(entries)}.\n"
        memory_path.write_text(memory_content, encoding="utf-8")
        written.append(CORE_FILES["MEMORY"])

    if written:
        return f"Successfully bootstrapped {', '.join(written)} based on detected tech stack: {stack}"
    else:
        return f"Project already has metadata files. Detected tech stack: {stack}"


# Registration list
onboarding_tools = [bootstrap_project]
