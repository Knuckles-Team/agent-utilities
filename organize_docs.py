import os
import re
import shutil
from pathlib import Path

docs_dir = Path("docs")
pillars_dir = docs_dir / "pillars"

# Define the mapping from file to pillar
file_mapping = {
    "AHE_ARCHITECTURE.md": "3_agentic_harness_engineering",
    "agent-os-architecture.md": "5_agent_os_infrastructure",
    "agent-registry.md": "1_graph_orchestration",
    "agents.md": "1_graph_orchestration",
    "agentspec-catalog.md": "3_agentic_harness_engineering",
    "architecture.md": "1_graph_orchestration",
    "building-mcp-servers.md": "4_ecosystem_and_tooling",
    "capabilities.md": "1_graph_orchestration",
    "cognitive-scheduler.md": "5_agent_os_infrastructure",
    "conductor-orchestration.md": "1_graph_orchestration",
    "configuration.md": "5_agent_os_infrastructure",
    "creating-an-agent.md": "4_ecosystem_and_tooling",
    "design-patterns-alignment.md": "3_agentic_harness_engineering",
    "development.md": "4_ecosystem_and_tooling",
    "durable-execution.md": "4_ecosystem_and_tooling",
    "emergent-architecture.md": "2_epistemic_knowledge_graph",
    "features.md": "3_agentic_harness_engineering",
    "first-principles.md": "1_graph_orchestration",
    "hsm.md": "2_epistemic_knowledge_graph",
    "knowledge-graph.md": "2_epistemic_knowledge_graph",
    "mathematical_foundations.md": "2_epistemic_knowledge_graph",
    "models.md": "2_epistemic_knowledge_graph",
    "permissions-kernel.md": "5_agent_os_infrastructure",
    "process-lifecycle.md": "5_agent_os_infrastructure",
    "registry-cache.md": "1_graph_orchestration",
    "rlm.md": "3_agentic_harness_engineering",
    "sdd.md": "1_graph_orchestration",
    "secrets-auth.md": "5_agent_os_infrastructure",
    "secure-sandbox.md": "4_ecosystem_and_tooling",
    "squeeze-evolve-routing.md": "1_graph_orchestration",
    "structured-prompts.md": "1_graph_orchestration",
    "tools.md": "4_ecosystem_and_tooling",
}

# Ensure directories exist
for pillar in set(file_mapping.values()):
    (pillars_dir / pillar).mkdir(parents=True, exist_ok=True)

moved_files = {}

for file_name, pillar in file_mapping.items():
    src = docs_dir / file_name
    if src.exists():
        dst = pillars_dir / pillar / file_name
        shutil.move(src, dst)
        moved_files[file_name] = f"pillars/{pillar}/{file_name}"
        print(f"Moved {file_name} to {moved_files[file_name]}")

# Update links in AGENTS.md, README.md, docs/overview.md, and inside the moved files themselves
files_to_update = [
    Path("AGENTS.md"),
    Path("README.md"),
    docs_dir / "overview.md",
]

# Add all markdown files in docs/pillars/
for root, _, files in os.walk(docs_dir / "pillars"):
    for file in files:
        if file.endswith(".md"):
            files_to_update.append(Path(root) / file)

for fpath in files_to_update:
    if not fpath.exists():
        continue

    with open(fpath) as f:
        content = f.read()

    new_content = content
    for old_name, new_path in moved_files.items():
        # Replace occurrences of old_name
        # Format: ](docs/old_name) -> ](docs/new_path)
        # Format: ](old_name) -> ](new_path) OR ](../new_path) depending on file depth

        if fpath.parent == Path("."):
            # E.g. in AGENTS.md, README.md
            new_content = re.sub(
                rf"\]\(docs/{old_name}\)", f"](docs/{new_path})", new_content
            )
        elif fpath == docs_dir / "overview.md":
            # E.g. in docs/overview.md
            new_content = re.sub(rf"\]\({old_name}\)", f"]({new_path})", new_content)
        else:
            # E.g. in docs/pillars/1_.../some_file.md
            # They need to reference other files in pillars
            # If referencing old_name directly: `](old_name)` -> `](../../pillars/X/old_name)`
            # Let's use absolute relative path from the pillar dir: `../X/old_name`
            pillar_part = new_path.split("/")[1]
            new_content = re.sub(
                rf"\]\({old_name}\)", f"](../{pillar_part}/{old_name})", new_content
            )
            # What if it was referencing `docs/old_name`?
            new_content = re.sub(
                rf"\]\(docs/{old_name}\)",
                f"](../{pillar_part}/{old_name})",
                new_content,
            )

    if new_content != content:
        with open(fpath, "w") as f:
            f.write(new_content)
        print(f"Updated links in {fpath}")

print("Done organizing docs.")
