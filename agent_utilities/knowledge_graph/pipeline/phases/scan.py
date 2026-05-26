import logging
import os
from pathlib import Path

import pathspec

from ..types import (
    PhaseResult,
    PipelineContext,
    PipelinePhase,
)

logger = logging.getLogger(__name__)


def classify_domain(repo_name: str) -> str:
    repo_lower = repo_name.lower()
    if any(
        k in repo_lower
        for k in (
            "hedge",
            "trading",
            "quant",
            "qlib",
            "toad",
            "nadir",
            "kronos",
            "alpha",
            "crypto-trader",
            "trader",
            "crypto",
        )
    ):
        return "quant-trading"
    if any(
        k in repo_lower
        for k in (
            "memory",
            "rag",
            "knowledge",
            "graph",
            "ladybug",
            "blockify",
            "ruvector",
            "context",
            "networkx",
        )
    ):
        return "memory-rag-kg"
    if any(
        k in repo_lower
        for k in (
            "prompt",
            "rules",
            "guide",
            "learn",
            "scratch",
            "harness",
            "intern",
            "awesome",
        )
    ):
        return "prompt-engineering-edu"
    if any(
        k in repo_lower
        for k in (
            "infra",
            "enterprise",
            "worktrunk",
            "procmon",
            "mattermost",
            "twenty",
            "keycloak",
            "caddy",
        )
    ):
        return "enterprise-ai-infra"
    return "agent-frameworks"


async def execute_scan(ctx: PipelineContext, deps: dict[str, PhaseResult]) -> list[str]:
    """Scan the workspace directory and return a list of code files, respecting .gitignore."""
    root = Path(ctx.config.workspace_path).absolute()

    # Load .gitignore if it exists
    spec = None
    gitignore_path = root / ".gitignore"
    if gitignore_path.exists():
        lines = gitignore_path.read_text().splitlines()
        spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)  # type: ignore

    files = []
    if root.exists() and root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune directories in place so os.walk doesn't traverse skipped directories at all
            pruned_dirs = []
            for d in dirnames:
                d_path = Path(dirpath) / d
                d_rel = d_path.relative_to(root)

                # Check if matches gitignore
                if spec and spec.match_file(str(d_rel)):
                    continue

                # Skip patterns from config
                if any(p in str(d_rel) for p in ctx.config.exclude_patterns):
                    continue

                # Skip hidden directories except .specify
                if d.startswith(".") and d != ".specify":
                    continue

                pruned_dirs.append(d)

            dirnames[:] = pruned_dirs

            for f_name in filenames:
                f_path = Path(dirpath) / f_name
                rel_path = f_path.relative_to(root)

                # Skip if file matches gitignore
                if spec and spec.match_file(str(rel_path)):
                    continue

                # Skip patterns from config
                if any(p in str(rel_path) for p in ctx.config.exclude_patterns):
                    continue

                # Also skip hidden/meta dirs/files just in case, except under .specify
                if any(part.startswith(".") and part != "." for part in rel_path.parts):
                    if not any(part == ".specify" for part in rel_path.parts):
                        continue

                # Basic filter for code files
                if f_path.suffix in {
                    ".py",
                    ".ts",
                    ".js",
                    ".tsx",
                    ".jsx",
                    ".go",
                    ".rs",
                    ".cpp",
                    ".c",
                    ".java",
                    ".md",
                }:
                    abs_path = str(f_path.absolute())
                    files.append(abs_path)

                    # Add to graph
                    node_id = f"file:{rel_path}"

                    metadata = {}
                    parts = rel_path.parts
                    if len(parts) >= 2:
                        domain_slug = None
                        repo_name = None

                        if parts[0] == "open-source-libraries" and len(parts) >= 2:
                            known_categories = {
                                "agent-frameworks",
                                "enterprise-ai-infra",
                                "memory-rag-kg",
                                "prompt-engineering-edu",
                                "quant-trading",
                            }
                            if parts[1] in known_categories:
                                if len(parts) >= 3:
                                    repo_name = parts[2]
                                    domain_slug = parts[1]
                                    if len(parts) > 4:
                                        metadata["folder_name"] = "/".join(parts[3:-1])
                                else:
                                    repo_name = parts[1]
                                    domain_slug = parts[1]
                            else:
                                repo_name = parts[1]
                                domain_slug = classify_domain(repo_name)
                                if len(parts) > 3:
                                    metadata["folder_name"] = "/".join(parts[2:-1])

                            SLUG_TO_DOMAIN = {
                                "quant-trading": "Quantitative & Algorithmic Trading",
                                "agent-frameworks": "Agent Frameworks & Core Runtimes",
                                "memory-rag-kg": "Agent Memory, RAG & Knowledge Graphs",
                                "prompt-engineering-edu": "Prompt Engineering, Education & Engineering Guidelines",
                                "enterprise-ai-infra": "Enterprise AI Infrastructure & Domain Integrations",
                            }
                            if domain_slug in SLUG_TO_DOMAIN:
                                metadata["domain"] = SLUG_TO_DOMAIN[domain_slug]
                            else:
                                formatted_domain = " ".join(
                                    word.capitalize()
                                    for word in domain_slug.replace("-", " ")
                                    .replace("_", " ")
                                    .split()
                                )
                                metadata["domain"] = formatted_domain

                        elif parts[0] == "agent-packages":
                            if len(parts) >= 3 and parts[1] in ("agents", "skills"):
                                domain_slug = parts[1]
                                repo_name = parts[2]
                                metadata["domain"] = (
                                    "AI Agents & Multi-Agent Swarms"
                                    if domain_slug == "agents"
                                    else "Agent Skills & Dynamic Workflows"
                                )
                                if len(parts) > 4:
                                    metadata["folder_name"] = "/".join(parts[3:-1])
                            elif len(parts) >= 2:
                                repo_name = parts[1]
                                metadata["domain"] = "Agent Utilities & Shared Packages"
                                if len(parts) > 3:
                                    metadata["folder_name"] = "/".join(parts[2:-1])

                        if domain_slug:
                            metadata["domain_slug"] = domain_slug
                        if repo_name:
                            metadata["repo_name"] = repo_name

                    ctx.nx_graph.add_node(
                        node_id,
                        type="file",
                        name=f_name,
                        file_path=abs_path,
                        metadata=metadata,
                    )

    logger.debug(f"Scan found {len(files)} files")
    return files


scan_phase = PipelinePhase(name="scan", deps=[], execute_fn=execute_scan)
