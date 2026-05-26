#!/usr/bin/python
from __future__ import annotations

"""Plugin Bundle — Unified Skills + Hooks + MCP Config Distribution.

CONCEPT:ECO-4.19 — Plugin Bundle Distribution System

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
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

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
            name=data["name"], version=data.get("version", "0.1.0"),
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
            "name": self.name, "version": self.version,
            "description": self.description, "author": self.author,
            "skills": self.skills, "hooks": self.hooks,
            "mcp_configs": self.mcp_configs,
            "agents_md_overlay": self.agents_md_overlay,
            "source_url": self.source_url,
        }

    def to_yaml(self) -> str:
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


class PluginBundleManager:
    """Manage plugin bundle installation and lifecycle.

    CONCEPT:ECO-4.19 — Plugin Bundle Distribution System

    Usage::

        mgr = PluginBundleManager(workspace="/my/project", engine=kg)
        mgr.install_from_path("/path/to/plugin-dir")
        mgr.install_from_github("org/repo", "plugins/my-plugin")
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
        """Install a plugin bundle from a local directory."""
        bp = Path(bundle_path).resolve()
        manifest = self._load_manifest(bp)

        dest = self.plugins_dir / manifest.name
        dest.mkdir(parents=True, exist_ok=True)

        # Copy bundle contents
        if bp.is_dir():
            shutil.copytree(bp, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(bp, dest / bp.name)

        # Install components
        self._install_skills(manifest, dest)
        self._install_hooks(manifest, dest)
        self._install_mcp_configs(manifest, dest)
        self._install_agents_md_overlay(manifest)

        # Register in KG
        self._register_in_kg(manifest)

        logger.info(
            "[ECO-4.4] Installed plugin '%s' v%s (%d skills, %d hooks, %d MCP configs)",
            manifest.name, manifest.version, len(manifest.skills),
            len(manifest.hooks), len(manifest.mcp_configs),
        )
        return manifest

    def install_from_github(
        self, repo: str, subpath: str = "", branch: str = "main",
    ) -> PluginBundle:
        """Clone and install a plugin from GitHub."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            clone_cmd = ["git", "clone", "--depth=1", f"--branch={branch}",
                         f"https://github.com/{repo}.git", tmpdir]
            subprocess.run(clone_cmd, check=True, capture_output=True, timeout=60)

            source = Path(tmpdir) / subpath if subpath else Path(tmpdir)
            bundle = self.install_from_path(source)
            bundle.source_url = f"https://github.com/{repo}"
            return bundle

    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin bundle."""
        dest = self.plugins_dir / name
        if dest.exists():
            shutil.rmtree(dest)
            self._remove_agents_md_overlay(name)
            self._deregister_from_kg(name)
            logger.info("[ECO-4.4] Uninstalled plugin '%s'", name)
            return True
        return False

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
        raise FileNotFoundError(f"No plugin manifest in {path}")

    def _install_skills(self, bundle: PluginBundle, source: Path) -> None:
        for skill_name in bundle.skills:
            logger.debug("[ECO-4.4] Skill '%s' registered from plugin '%s'", skill_name, bundle.name)

    def _install_hooks(self, bundle: PluginBundle, source: Path) -> None:
        for hook_name, hook_path in bundle.hooks.items():
            hp = source / hook_path
            if hp.is_file():
                dest = self.workspace / ".agents" / "hooks" / f"{bundle.name}_{hook_name}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(hp, dest)

    def _install_mcp_configs(self, bundle: PluginBundle, source: Path) -> None:
        for server_name, config_path in bundle.mcp_configs.items():
            cp = source / config_path
            if cp.is_file():
                dest = self.workspace / ".agents" / "mcp" / f"{bundle.name}_{server_name}.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cp, dest)

    def _install_agents_md_overlay(self, bundle: PluginBundle) -> None:
        if not bundle.agents_md_overlay:
            return
        overlay_dir = self.workspace / ".agents" / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        fp = overlay_dir / f"{bundle.name}.md"
        fp.write_text(bundle.agents_md_overlay, encoding="utf-8")

    def _remove_agents_md_overlay(self, name: str) -> None:
        fp = self.workspace / ".agents" / "overlays" / f"{name}.md"
        if fp.exists():
            fp.unlink()

    def _register_in_kg(self, bundle: PluginBundle) -> None:
        if not self.engine:
            return
        try:
            self.engine.add_node(f"plugin_{bundle.name}", "plugin_bundle", {
                "name": bundle.name, "version": bundle.version,
                "description": bundle.description, "author": bundle.author,
                "skill_count": len(bundle.skills),
                "hook_count": len(bundle.hooks),
                "mcp_count": len(bundle.mcp_configs),
                "source_url": bundle.source_url,
                "importance_score": 0.7,
            })
        except Exception as e:
            logger.debug("[ECO-4.4] KG registration failed: %s", e)

    def _deregister_from_kg(self, name: str) -> None:
        if not self.engine:
            return
        try:
            self.engine.remove_node(f"plugin_{name}")
        except Exception:
            pass


def install_plugin_from_github(
    repo: str, workspace: str | Path = ".", subpath: str = "", engine: Any = None,
) -> PluginBundle:
    """Convenience: install a plugin from GitHub in one call."""
    mgr = PluginBundleManager(workspace=workspace, engine=engine)
    return mgr.install_from_github(repo, subpath)
