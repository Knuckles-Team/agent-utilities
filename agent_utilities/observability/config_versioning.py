#!/usr/bin/python
from __future__ import annotations

"""Agent Config Versioning — Database-Driven Config History (CONCEPT:AHE-3.2).

Immutable config snapshots with forward-only rollback. Ported from
MATE's AgentConfigVersion model and update_agent_config() pattern.

Uses SUPERSEDES edges (same pattern as prompt versioning in
engine_registry.py) for KG-native version chain traversal.

OWL: :AgentConfigVersion rdfs:subClassOf :ChangeManifest
"""


import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentConfigSnapshot(BaseModel):
    """Immutable snapshot of an agent's configuration. CONCEPT:AHE-3.2

    Mirrors MATE's AgentConfigVersion model but expressed as a Pydantic
    model for KG persistence via SUPERSEDES edges.
    """

    id: str = ""
    agent_name: str
    model_name: str = ""
    instruction: str = ""
    tools_config: dict[str, Any] = Field(default_factory=dict)
    guardrail_config: dict[str, Any] = Field(default_factory=dict)
    mcp_servers: list[str] = Field(default_factory=list)
    version_number: int = 1
    parent_version_id: str = ""
    created_by: str = "system"
    change_summary: str = ""
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"config_v:{self.agent_name}:v{self.version_number}"


class ConfigDiff(BaseModel):
    """Structured diff between two config versions. CONCEPT:AHE-3.2"""

    agent_name: str
    from_version: int
    to_version: int
    changes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    summary: str = ""


class AgentConfigVersionManager:
    """Manages versioned agent configurations. CONCEPT:AHE-3.2

    Ported from MATE's agent_manager.py update_agent_config() pattern.
    Key design principles:
    1. Forward-only — rollback creates a new version copying target config
    2. Immutable snapshots — no modification of existing versions
    3. SUPERSEDES edges — reuses existing registry pattern from prompts
    """

    def __init__(self, kg_engine: Any = None) -> None:
        self._engine = kg_engine
        self._versions: dict[str, list[AgentConfigSnapshot]] = {}

    def create_version(
        self,
        agent_name: str,
        config: dict[str, Any],
        author: str = "system",
        change_summary: str = "",
    ) -> AgentConfigSnapshot:
        """Create a new config version snapshot.

        Parameters
        ----------
        agent_name : str
            The agent being configured.
        config : dict
            Config dict with model_name, instruction, tools_config, etc.
        author : str
            Who made the change.
        change_summary : str
            Description of what changed.

        Returns
        -------
        AgentConfigSnapshot
            The new version snapshot.
        """
        history = self._versions.setdefault(agent_name, [])
        version_number = len(history) + 1
        parent_id = history[-1].id if history else ""

        snapshot = AgentConfigSnapshot(
            agent_name=agent_name,
            model_name=config.get("model_name", ""),
            instruction=config.get("instruction", ""),
            tools_config=config.get("tools_config", {}),
            guardrail_config=config.get("guardrail_config", {}),
            mcp_servers=config.get("mcp_servers", []),
            version_number=version_number,
            parent_version_id=parent_id,
            created_by=author,
            change_summary=change_summary,
        )

        history.append(snapshot)
        logger.info(
            "Config version %d created for agent '%s' by %s: %s",
            version_number,
            agent_name,
            author,
            change_summary,
        )
        return snapshot

    def get_version_history(
        self,
        agent_name: str,
        limit: int = 20,
    ) -> list[AgentConfigSnapshot]:
        """Get version history for an agent, newest first."""
        history = self._versions.get(agent_name, [])
        return list(reversed(history[-limit:]))

    def get_version(
        self,
        agent_name: str,
        version_number: int,
    ) -> AgentConfigSnapshot | None:
        """Get a specific version by number."""
        history = self._versions.get(agent_name, [])
        for v in history:
            if v.version_number == version_number:
                return v
        return None

    def get_latest(self, agent_name: str) -> AgentConfigSnapshot | None:
        """Get the latest version for an agent."""
        history = self._versions.get(agent_name, [])
        return history[-1] if history else None

    def diff_versions(
        self,
        agent_name: str,
        from_version: int,
        to_version: int,
    ) -> ConfigDiff:
        """Compute structured diff between two versions.

        Parameters
        ----------
        agent_name : str
            The agent.
        from_version : int
            Source version number.
        to_version : int
            Target version number.

        Returns
        -------
        ConfigDiff
            Structured diff with per-field changes.
        """
        v_from = self.get_version(agent_name, from_version)
        v_to = self.get_version(agent_name, to_version)

        if not v_from or not v_to:
            return ConfigDiff(
                agent_name=agent_name,
                from_version=from_version,
                to_version=to_version,
                summary="One or both versions not found",
            )

        changes: dict[str, dict[str, Any]] = {}
        fields_to_compare = [
            "model_name",
            "instruction",
            "tools_config",
            "guardrail_config",
            "mcp_servers",
        ]

        for field_name in fields_to_compare:
            old_val = getattr(v_from, field_name)
            new_val = getattr(v_to, field_name)
            if old_val != new_val:
                changes[field_name] = {"from": old_val, "to": new_val}

        summary_parts = []
        if "model_name" in changes:
            summary_parts.append(
                f"model: {changes['model_name']['from']} → {changes['model_name']['to']}"
            )
        if "instruction" in changes:
            summary_parts.append("instruction updated")
        if "tools_config" in changes:
            summary_parts.append("tools_config changed")
        if "guardrail_config" in changes:
            summary_parts.append("guardrail_config changed")
        if "mcp_servers" in changes:
            summary_parts.append("mcp_servers changed")

        return ConfigDiff(
            agent_name=agent_name,
            from_version=from_version,
            to_version=to_version,
            changes=changes,
            summary="; ".join(summary_parts) if summary_parts else "No changes",
        )

    def rollback_to_version(
        self,
        agent_name: str,
        target_version: int,
        author: str = "rollback",
    ) -> AgentConfigSnapshot:
        """Rollback to a previous version by creating a new version.

        Forward-only: creates a new version that copies the target's
        config. Never destructive — matches MATE and existing SUPERSEDES
        pattern from prompt versioning.

        Parameters
        ----------
        agent_name : str
            The agent.
        target_version : int
            Version number to rollback to.
        author : str
            Who initiated the rollback.

        Returns
        -------
        AgentConfigSnapshot
            The new version (copy of target).

        Raises
        ------
        ValueError
            If target version not found.
        """
        target = self.get_version(agent_name, target_version)
        if not target:
            raise ValueError(
                f"Version {target_version} not found for agent '{agent_name}'"
            )

        return self.create_version(
            agent_name=agent_name,
            config={
                "model_name": target.model_name,
                "instruction": target.instruction,
                "tools_config": target.tools_config,
                "guardrail_config": target.guardrail_config,
                "mcp_servers": target.mcp_servers,
            },
            author=author,
            change_summary=f"Rollback to version {target_version}",
        )

    def summary(self) -> dict[str, Any]:
        """Summary of versioning state."""
        return {
            "agents_tracked": len(self._versions),
            "total_versions": sum(len(v) for v in self._versions.values()),
            "agents": {
                name: len(versions) for name, versions in self._versions.items()
            },
        }
