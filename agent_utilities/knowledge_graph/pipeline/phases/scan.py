import logging
from typing import Dict, List
from pathlib import Path
import pathspec
from ..types import (
    PipelinePhase,
    PipelineContext,
    PhaseResult,
)

logger = logging.getLogger(__name__)


async def execute_scan(ctx: PipelineContext, deps: Dict[str, PhaseResult]) -> List[str]:
    """Scan the workspace directory and return a list of code files, respecting .gitignore."""
    root = Path(ctx.config.workspace_path).absolute()

    # Load .gitignore if it exists
    spec = None
    gitignore_path = root / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", f)

    files = []
    if root.exists() and root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            # Calculate relative path for gitignore matching
            rel_path = path.relative_to(root)

            # Skip if matches gitignore
            if spec and spec.match_file(str(rel_path)):
                continue

            # Skip patterns from config
            if any(p in str(rel_path) for p in ctx.config.exclude_patterns):
                continue

            # Also skip common hidden/meta dirs
            if any(part.startswith(".") and part != "." for part in rel_path.parts):
                if not str(rel_path).startswith("."):
                    continue

            # Basic filter for code files
            if path.suffix in {
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
                abs_path = str(path.absolute())
                files.append(abs_path)

                # Add to graph
                node_id = f"file:{rel_path}"
                ctx.nx_graph.add_node(
                    node_id, type="file", name=path.name, file_path=abs_path
                )

    logger.debug(f"Scan found {len(files)} files")
    return files


scan_phase = PipelinePhase(name="scan", deps=[], execute_fn=execute_scan)
