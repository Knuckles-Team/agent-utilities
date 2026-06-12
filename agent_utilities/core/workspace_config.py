#!/usr/bin/python
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from agent_utilities.core.config import setting
from agent_utilities.core.paths import ensure_dirs

logger = logging.getLogger(__name__)


def get_workspace_yml_path() -> Path:
    """Return the path to the global workspace.yml in the XDG config dir."""
    ensure_dirs()
    # E.g. ~/.config/agent-utilities/workspace.yml
    config_dir = (
        Path(setting("XDG_CONFIG_HOME", Path.home() / ".config")) / "agent-utilities"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "workspace.yml"


DEFAULT_WORKSPACE_YML = """# Repository Workspace Configuration
# This file defines the directory structure and repositories for the agent ecosystem.

name: "Agent Packages Workspace"
path: "/home/apps/workspace"
description: "Main development environment for the agent-packages ecosystem."

repositories:
  - url: "https://github.com/Knuckles-Team/pipelines.git"
    description: "GitHub Actions pipelines for the agent ecosystem."
  - url: "http://gitlab.arpa/homelab/pipelines/gitlab-pipelines.git"
    description: "GitLab CI pipelines for the agent ecosystem."

subdirectories:
  agent-packages:
    description: "Core utility packages, shared libraries, and agent implementations."
    repositories:
      - url: "https://github.com/Knuckles-Team/agent-utilities.git"
        description: "Shared utility functions, base classes, and server common code."
      - url: "https://github.com/Knuckles-Team/agent-webui.git"
        description: "Standard web UI component library for agents."
      - url: "https://github.com/Knuckles-Team/agent-terminal-ui.git"
        description: "Standard terminal UI component library for agents."
    subdirectories:
      skills:
        description: "Agent capabilities, skill graphs, and tool definitions."
        repositories:
          - url: "https://github.com/Knuckles-Team/universal-skills.git"
            description: "Central repository for agent skills and capabilities."
          - url: "https://github.com/Knuckles-Team/skill-graphs.git"
            description: "Tool for generating and managing skill graph definitions."
      agents:
        description: "Collection of specialized agent implementations, MCP servers, API Wrappers/Core Python Tools."
        repositories:
          - url: "https://github.com/Knuckles-Team/ansible-tower-mcp.git"
          - url: "https://github.com/Knuckles-Team/archivebox-api.git"
          - url: "https://github.com/Knuckles-Team/arr-mcp.git"
          - url: "https://github.com/Knuckles-Team/atlassian-agent.git"
          - url: "https://github.com/Knuckles-Team/audio-transcriber.git"
          - url: "https://github.com/Knuckles-Team/container-manager-mcp.git"
          - url: "https://github.com/Knuckles-Team/data-science-mcp.git"
          - url: "https://github.com/Knuckles-Team/documentdb-mcp.git"
          - url: "https://github.com/Knuckles-Team/genius-agent.git"
          - url: "https://github.com/Knuckles-Team/github-agent.git"
          - url: "https://github.com/Knuckles-Team/gitlab-api.git"
          - url: "https://github.com/Knuckles-Team/home-assistant-agent.git"
          - url: "https://github.com/Knuckles-Team/jellyfin-mcp.git"
          - url: "https://github.com/Knuckles-Team/langfuse-agent.git"
          - url: "https://github.com/Knuckles-Team/leanix-agent.git"
          - url: "https://github.com/Knuckles-Team/mealie-mcp.git"
          - url: "https://github.com/Knuckles-Team/media-downloader.git"
          - url: "https://github.com/Knuckles-Team/microsoft-agent.git"
          - url: "https://github.com/Knuckles-Team/nextcloud-agent.git"
          - url: "https://github.com/Knuckles-Team/owncast-agent.git"
          - url: "https://github.com/Knuckles-Team/plane-agent.git"
          - url: "https://github.com/Knuckles-Team/portainer-agent.git"
          - url: "https://github.com/Knuckles-Team/postiz-agent.git"
          - url: "https://github.com/Knuckles-Team/qbittorrent-agent.git"
          - url: "https://github.com/Knuckles-Team/repository-manager.git"
          - url: "https://github.com/Knuckles-Team/scholarx.git"
          - url: "https://github.com/Knuckles-Team/searxng-mcp.git"
          - url: "https://github.com/Knuckles-Team/servicenow-api.git"
          - url: "https://github.com/Knuckles-Team/stirlingpdf-agent.git"
          - url: "https://github.com/Knuckles-Team/systems-manager.git"
          - url: "https://github.com/Knuckles-Team/tunnel-manager.git"
          - url: "https://github.com/Knuckles-Team/uptime-kuma-agent.git"
          - url: "https://github.com/Knuckles-Team/vector-mcp.git"
          - url: "https://github.com/Knuckles-Team/wger-agent.git"

maintenance:
  description: "Phased update sequence for the agent ecosystem."
  phases:
    - name: "Phase 1: Core Tools and UIs"
      phase: 1
      projects:
        - "universal-skills"
        - "skill-graphs"
        - "agent-webui"
        - "agent-terminal-ui"
      wait_minutes: 12
    - name: "Phase 2: agent-utilities"
      phase: 2
      projects:
        - "agent-utilities"
      wait_minutes: 3
    - name: "Phase 3: Agents"
      phase: 3
      bulk_bump: True
      bulk_push: True

graph:
  enabled: true
  multimodal: true
  incremental: true
  groups: []
"""


def load_workspace_yml(yml_path: str | None = None) -> dict[str, Any]:
    """Load the workspace.yml file if it exists, otherwise initialize it and return."""
    if yml_path:
        path = Path(yml_path)
    else:
        path = get_workspace_yml_path()

    if not path.exists():
        if not yml_path:
            logger.info(f"workspace.yml not found. Generating default at {path}")
            with open(path, "w") as f:
                f.write(DEFAULT_WORKSPACE_YML)
        else:
            logger.warning(f"Provided workspace.yml not found at {path}")
            return {}

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except Exception as e:
        logger.error(f"Failed to parse workspace.yml at {path}: {e}")
        return {}


def _extract_repositories(
    data: dict[str, Any], current_path: Path
) -> list[tuple[Path, str]]:
    """Recursively extract all repository paths and clone URLs from workspace.yml dictionary."""
    repos = []

    # Check if current level has repositories
    if "repositories" in data and isinstance(data["repositories"], list):
        for repo in data["repositories"]:
            if "url" in repo:
                url = repo["url"]
                # Extract repo name from URL to form the target path
                repo_name = url.split("/")[-1].replace(".git", "")
                target_path = current_path / repo_name
                repos.append((target_path, url))

    # Check subdirectories recursively
    if "subdirectories" in data and isinstance(data["subdirectories"], dict):
        for subdir_name, subdir_data in data["subdirectories"].items():
            subdir_path = current_path / subdir_name
            repos.extend(_extract_repositories(subdir_data, subdir_path))

    return repos


def clone_missing_projects(yml_path: str | None = None) -> list[Path]:
    """Parse workspace.yml, clone missing projects, and return list of all project paths."""
    data = load_workspace_yml(yml_path)
    if not data:
        logger.warning("No workspace.yml found.")
        return []

    base_path = Path(data.get("path", os.getcwd()))
    base_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Bootstrapping workspace at {base_path}...")
    repos = _extract_repositories(data, base_path)

    project_paths = []

    for target_path, url in repos:
        project_paths.append(target_path)
        if not target_path.exists():
            logger.info(f"Cloning missing project: {url} -> {target_path}")
            try:
                import shutil

                git_executable = shutil.which("git")
                if not git_executable:
                    raise RuntimeError("git executable not found in PATH")
                subprocess.run(
                    [git_executable, "clone", url, str(target_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone {url}: {e.stderr}")
        else:
            logger.debug(f"Project already exists: {target_path}")

    return project_paths
