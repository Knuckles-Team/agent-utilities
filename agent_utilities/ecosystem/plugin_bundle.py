#!/usr/bin/python
from __future__ import annotations

"""Plugin Bundle — Unified Skills + Hooks + MCP Config Distribution.

CONCEPT:AU-ECO.toolkit.self-documenting-plugin-bundle — Plugin Bundle Distribution System

A plugin bundles skills, hooks, and MCP configurations into a single
installable package.  Distributed via GitHub and registered in KG.

Bundle format (``plugin.yaml``)::

    name: my-plugin
    version: 1.0.0
    description: A useful plugin bundle
    author: team-platform
    skills:
      - infrastructure-orchestrator
      - container-health-check
    hooks:
      session_start: hooks/start.sh
      session_end: hooks/end.sh
    mcp_configs:
      container-manager: mcp/cm_config.json
    agents_md_overlay: |
      ## Plugin: my-plugin
      This plugin provides infrastructure management tools.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

__all__ = [
    "PluginBundle",
    "PluginBundleManager",
    "install_plugin_from_github",
]


@dataclass
class PluginBundle:
    """A plugin bundle manifest."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    skills: list[str] = field(default_factory=list)
    hooks: dict[str, str] = field(default_factory=dict)
    mcp_configs: dict[str, str] = field(default_factory=dict)
    agents_md_overlay: str = ""
    source_url: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> PluginBundle:
        """Load from plugin.yaml file."""
        import yaml

        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            skills=data.get("skills", []),
            hooks=data.get("hooks", {}),
            mcp_configs=data.get("mcp_configs", {}),
            agents_md_overlay=data.get("agents_md_overlay", ""),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> PluginBundle:
        """Load from plugin.json file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "skills": self.skills,
            "hooks": self.hooks,
            "mcp_configs": self.mcp_configs,
            "agents_md_overlay": self.agents_md_overlay,
            "source_url": self.source_url,
        }

    def to_yaml(self) -> str:
        import yaml

        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


class PluginBundleManager:
    """Manage plugin bundle installation and lifecycle.

    CONCEPT:AU-ECO.toolkit.self-documenting-plugin-bundle — Plugin Bundle Distribution System

    Usage::

        mgr = PluginBundleManager(workspace=runtime_workspace, engine=kg)
        installed = mgr.list_installed()
    """

    PLUGINS_DIR = ".agents/plugins"

    def __init__(
        self,
        workspace: str | Path = ".",
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.plugins_dir = self.workspace / self.PLUGINS_DIR
        self.engine = engine

    def install_from_path(self, bundle_path: str | Path) -> PluginBundle:
        """Reject unsigned local bundles.

        Copying executable hooks and MCP commands from an arbitrary directory is
        not a security boundary. Use the governed Codex/plugin marketplace flow,
        which verifies provenance and requests installation approval.
        """
        del bundle_path
        raise PermissionError("Unsigned plugin bundle installation is disabled")

    def install_from_github(
        self,
        repo: str,
        subpath: str = "",
        branch: str = "main",
    ) -> PluginBundle:
        """Reject unverified network bundle installation."""
        del repo, subpath, branch
        raise PermissionError("Unverified network plugin installation is disabled")

    def uninstall(self, name: str) -> bool:
        """Reject legacy recursive deletion; use the governed plugin manager."""
        del name
        raise PermissionError("Legacy plugin bundle mutation is disabled")

    def list_installed(self) -> list[PluginBundle]:
        """List all installed plugin bundles."""
        bundles: list[PluginBundle] = []
        if not self.plugins_dir.exists():
            return bundles
        for d in sorted(self.plugins_dir.iterdir()):
            if d.is_dir():
                try:
                    bundles.append(self._load_manifest(d))
                except Exception:
                    pass
        return bundles

    def get_installed(self, name: str) -> PluginBundle | None:
        """Get a specific installed plugin."""
        d = self.plugins_dir / name
        if d.exists():
            try:
                return self._load_manifest(d)
            except Exception:
                pass
        return None

    # -- Internal helpers --

    def _load_manifest(self, path: Path) -> PluginBundle:
        for name in ["plugin.yaml", "plugin.yml", "plugin.json"]:
            fp = path / name
            if fp.is_file():
                if name.endswith(".json"):
                    return PluginBundle.from_json(fp)
                return PluginBundle.from_yaml(fp)
        raise FileNotFoundError("Plugin manifest is unavailable")


def install_plugin_from_github(
    repo: str,
    workspace: str | Path = ".",
    subpath: str = "",
    engine: Any = None,
) -> PluginBundle:
    """Convenience: install a plugin from GitHub in one call."""
    mgr = PluginBundleManager(workspace=workspace, engine=engine)
    return mgr.install_from_github(repo, subpath)
