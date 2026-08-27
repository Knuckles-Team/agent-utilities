import logging
import os
from pathlib import Path
from typing import Any

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
            "rustworkx",
            "epistemic",
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


# Basic filter for code files.
_SCAN_CODE_SUFFIXES = {
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
}

# open-source-libraries/<category>/... : categories that are already domain slugs.
_KNOWN_OSS_CATEGORIES = {
    "agent-frameworks",
    "enterprise-ai-infra",
    "memory-rag-kg",
    "prompt-engineering-edu",
    "quant-trading",
}

_OSS_SLUG_TO_DOMAIN = {
    "quant-trading": "Quantitative & Algorithmic Trading",
    "agent-frameworks": "Agent Frameworks & Core Runtimes",
    "memory-rag-kg": "Agent Memory, RAG & Knowledge Graphs",
    "prompt-engineering-edu": "Prompt Engineering, Education & Engineering Guidelines",
    "enterprise-ai-infra": "Enterprise AI Infrastructure & Domain Integrations",
}


def _load_gitignore_spec(root: Path) -> "pathspec.PathSpec | None":
    """Load ``root/.gitignore`` as a matchable spec, or ``None`` if absent.

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change).
    """
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return None
    lines = gitignore_path.read_text().splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)  # type: ignore


def _should_prune_scan_dir(
    d: str,
    d_rel: Path,
    spec: "pathspec.PathSpec | None",
    exclude_patterns: Any,
) -> bool:
    """True if a walked subdirectory should be pruned from the scan.

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change) -- previously an ``append``-on-fall-through loop body, now the
    inverse ``skip`` predicate driving a list comprehension.
    """
    if spec and spec.match_file(str(d_rel)):
        return True
    if any(p in str(d_rel) for p in exclude_patterns):
        return True
    if d.startswith(".") and d != ".specify":
        return True
    if d in {"node_modules", "__pycache__", "venv"}:
        return True
    return False


def _should_skip_scan_file(
    rel_path: Path,
    spec: "pathspec.PathSpec | None",
    exclude_patterns: Any,
) -> bool:
    """True if a walked file should be skipped (gitignore / exclude / hidden).

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change).
    """
    if spec and spec.match_file(str(rel_path)):
        return True
    if any(p in str(rel_path) for p in exclude_patterns):
        return True
    if any(part.startswith(".") and part != "." for part in rel_path.parts):
        if not any(part == ".specify" for part in rel_path.parts):
            return True
    return False


def _classify_oss_library_path(
    parts: tuple[str, ...], metadata: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Classify a path under ``open-source-libraries/...``; mutates ``metadata`` in place.

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change). Returns ``(domain_slug, repo_name)``.
    """
    if parts[1] in _KNOWN_OSS_CATEGORIES:
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

    if domain_slug in _OSS_SLUG_TO_DOMAIN:
        metadata["domain"] = _OSS_SLUG_TO_DOMAIN[domain_slug]
    else:
        formatted_domain = " ".join(
            word.capitalize()
            for word in domain_slug.replace("-", " ").replace("_", " ").split()
        )
        metadata["domain"] = formatted_domain

    return domain_slug, repo_name


def _classify_agent_packages_path(
    parts: tuple[str, ...], metadata: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Classify a path under ``agent-packages/...``; mutates ``metadata`` in place.

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change). Returns ``(domain_slug, repo_name)``.
    """
    domain_slug: str | None = None
    repo_name: str | None = None
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
    return domain_slug, repo_name


def _build_scan_file_metadata(parts: tuple[str, ...]) -> dict[str, Any]:
    """Compute the graph-node metadata for a scanned file from its relative path parts.

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change).
    """
    metadata: dict[str, Any] = {}
    if len(parts) < 2:
        return metadata

    domain_slug: str | None = None
    repo_name: str | None = None

    if parts[0] == "open-source-libraries":
        domain_slug, repo_name = _classify_oss_library_path(parts, metadata)
    elif parts[0] == "agent-packages":
        domain_slug, repo_name = _classify_agent_packages_path(parts, metadata)

    if domain_slug:
        metadata["domain_slug"] = domain_slug
    if repo_name:
        metadata["repo_name"] = repo_name

    return metadata


def _record_scan_file(
    ctx: PipelineContext, f_path: Path, f_name: str, rel_path: Path
) -> str:
    """Add one accepted scan file to the graph and return its absolute path.

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change).
    """
    abs_path = str(f_path.absolute())
    node_id = f"file:{rel_path}"
    metadata = _build_scan_file_metadata(rel_path.parts)
    ctx.graph.add_node(
        node_id,
        node_type="file",
        name=f_name,
        file_path=abs_path,
        metadata=metadata,
    )
    return abs_path


def _scan_one_directory(
    ctx: PipelineContext,
    root: Path,
    dirpath: str,
    dirnames: list[str],
    filenames: list[str],
    spec: "pathspec.PathSpec | None",
    exclude_patterns: Any,
) -> list[str]:
    """Prune ``dirnames`` in place and return accepted files for one ``os.walk`` step.

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change) -- the per-directory body of the walk loop.
    """
    # Prune directories in place so os.walk doesn't traverse skipped directories at all
    dirnames[:] = [
        d
        for d in dirnames
        if not _should_prune_scan_dir(
            d, (Path(dirpath) / d).relative_to(root), spec, exclude_patterns
        )
    ]

    accepted: list[str] = []
    for f_name in filenames:
        f_path = Path(dirpath) / f_name
        rel_path = f_path.relative_to(root)

        if _should_skip_scan_file(rel_path, spec, exclude_patterns):
            continue
        if f_path.suffix not in _SCAN_CODE_SUFFIXES:
            continue

        accepted.append(_record_scan_file(ctx, f_path, f_name, rel_path))
    return accepted


def _walk_scan_files(
    ctx: PipelineContext,
    root: Path,
    spec: "pathspec.PathSpec | None",
    exclude_patterns: Any,
) -> list[str]:
    """Walk ``root`` and return the accepted code files, respecting ``spec``/excludes.

    Extracted verbatim from ``execute_scan`` (pure extract-method, no behaviour
    change) -- the ``os.walk`` loop itself.
    """
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        files.extend(
            _scan_one_directory(
                ctx, root, dirpath, dirnames, filenames, spec, exclude_patterns
            )
        )
    return files


async def execute_scan(ctx: PipelineContext, deps: dict[str, PhaseResult]) -> list[str]:
    """Scan the workspace directory and return a list of code files, respecting .gitignore."""
    root = Path(ctx.config.workspace_path).absolute()
    spec = _load_gitignore_spec(root)
    exclude_patterns = ctx.config.exclude_patterns

    files: list[str] = []
    if root.exists() and root.is_dir():
        files = _walk_scan_files(ctx, root, spec, exclude_patterns)

    logger.debug(f"Scan found {len(files)} files")
    return files


scan_phase = PipelinePhase(name="scan", deps=[], execute_fn=execute_scan)
