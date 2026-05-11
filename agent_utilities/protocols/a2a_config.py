from __future__ import annotations

"""A2A agent configuration loader and auto-discovery.

CONCEPT:ECO-4.1 — A2A Config File Loading

Loads ``a2a_config.json``, fetches ``.well-known/agent-card.json`` for each
declared agent, registers them in the Knowledge Graph, and optionally sets
up periodic background refresh to detect capability changes.

The config file mirrors ``mcp_config.json`` in intent — it declares external
A2A agents that should be first-class participants in routing, swarm teams,
and subgraph dispatch.

Example ``a2a_config.json``::

    {
      "agents": {
        "servicenow-agent": {
          "url": "http://10.0.0.18:8001",
          "description": "ServiceNow incident management",
          "auth": "none"
        },
        "gitlab-agent": {
          "url": "http://10.0.0.18:8002",
          "auth": "bearer",
          "auth_token": "secret://a2a/gitlab/token"
        }
      },
      "refresh_interval_seconds": 300
    }
"""


import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_INTERVAL = 300  # 5 minutes


def resolve_a2a_config_path(
    config_ref: str | None = None,
) -> Path | None:
    """Resolve the path to an ``a2a_config.json`` file.

    Resolution order:
    1. Explicit ``config_ref`` argument.
    2. ``A2A_CONFIG`` environment variable.
    3. ``a2a_config.json`` in the current working directory.

    Args:
        config_ref: Optional path or filename to resolve.

    Returns:
        Resolved ``Path`` if the file exists, else ``None``.
    """
    candidates: list[str] = []
    if config_ref:
        candidates.append(config_ref)
    env_val = os.getenv("A2A_CONFIG")
    if env_val:
        candidates.append(env_val)
    candidates.append("a2a_config.json")

    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p
        # Try relative to cwd
        cwd_p = Path.cwd() / c
        if cwd_p.is_file():
            return cwd_p

    return None


def load_a2a_config(config_path: str | Path) -> dict[str, Any]:
    """Load and parse an ``a2a_config.json`` file.

    Args:
        config_path: Absolute or relative path to the config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"A2A config file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _resolve_auth_token(auth_token_ref: str | None) -> str | None:
    """Resolve an auth token reference using the secrets client.

    Supports ``secret://``, ``env://``, ``vault://`` URI schemes
    via ``SecretsClient.resolve_ref()`` (CONCEPT:OS-5.1).

    Args:
        auth_token_ref: Token reference string or ``None``.

    Returns:
        Resolved token value or ``None``.
    """
    if not auth_token_ref:
        return None

    # Fast path: env:// can be resolved without the secrets client
    if auth_token_ref.startswith("env://"):
        var_name = auth_token_ref[len("env://") :]
        return os.environ.get(var_name)

    # Use the secrets client for secret:// and vault://
    try:
        from agent_utilities.security.secrets_client import create_secrets_client

        client = create_secrets_client()
        return client.resolve_ref(auth_token_ref)
    except Exception as e:
        logger.warning(f"Failed to resolve auth token '{auth_token_ref}': {e}")
        return None


async def sync_a2a_agents(
    config_path: str | None = None,
) -> dict[str, Any]:
    """Load ``a2a_config.json`` and register all agents in the Knowledge Graph.

    CONCEPT:ECO-4.1 — A2A Config File Loading

    For each agent declared in the config:
    1. Resolve ``auth_token`` via ``SecretsClient`` if present.
    2. Fetch ``.well-known/agent-card.json`` (soft-fail with warning).
    3. Register as ``AgentNode(agent_type='a2a')`` in the KG.
    4. Ingest full card as ``CallableResource(resource_type='A2A_AGENT')``.
    5. Invalidate the registry cache.

    Args:
        config_path: Optional path to the A2A config file.

    Returns:
        Statistics dict: ``{"registered": N, "skipped": N, "failed": N}``.
    """
    from agent_utilities.protocols.a2a import A2AClient, register_a2a_peer

    stats: dict[str, int] = {"registered": 0, "skipped": 0, "failed": 0}

    resolved_path = resolve_a2a_config_path(config_path)
    if not resolved_path:
        logger.debug("No a2a_config.json found. Skipping A2A agent sync.")
        return stats

    try:
        config = load_a2a_config(resolved_path)
    except Exception as e:
        logger.warning(f"Failed to load A2A config from {resolved_path}: {e}")
        return stats

    agents_config = config.get("agents", {})
    if not agents_config:
        logger.debug("A2A config contains no agents. Nothing to sync.")
        return stats

    client = A2AClient()

    for name, agent_cfg in agents_config.items():
        url = agent_cfg.get("url")
        if not url:
            logger.warning(f"A2A agent '{name}' has no URL — skipping.")
            stats["skipped"] += 1
            continue

        # Resolve auth token if present
        auth_type = agent_cfg.get("auth", "none")
        _auth_token = _resolve_auth_token(agent_cfg.get("auth_token"))  # noqa: F841

        # Fetch agent card (soft-fail)
        card: dict[str, Any] | None = None
        try:
            card = await client.fetch_card(url)
        except Exception as e:
            logger.warning(f"A2A agent '{name}': Failed to fetch card from {url}: {e}")

        if card is None:
            logger.warning(
                f"A2A agent '{name}': .well-known/agent-card.json unreachable "
                f"at {url}. Registering with config-only metadata."
            )

        # Build description from card or config fallback
        description = (
            card.get("description", "")
            if card
            else agent_cfg.get("description", f"External A2A agent: {name}")
        )
        capabilities = (
            ",".join(card.get("capabilities", []))
            if card
            else agent_cfg.get("capabilities", "")
        )

        # Register in KG as AgentNode
        try:
            result = register_a2a_peer(
                name=card.get("name", name) if card else name,
                url=url,
                description=description,
                capabilities=capabilities,
                auth=auth_type,
            )
            logger.info(f"A2A sync: {result}")

            # Ingest full card for richer metadata (skills, embeddings)
            if card:
                try:
                    from agent_utilities.knowledge_graph.core.engine import (
                        IntelligenceGraphEngine,
                    )

                    engine = IntelligenceGraphEngine.get_active()
                    if engine:
                        engine.ingest_a2a_agent_card(url, card)
                        logger.info(
                            f"A2A sync: Ingested full card for '{name}' "
                            f"({len(card.get('capabilities', []))} capabilities)"
                        )
                except Exception as e:
                    logger.debug(f"A2A sync: Card ingestion for '{name}' skipped: {e}")

            stats["registered"] += 1

        except Exception as e:
            logger.error(f"A2A sync: Failed to register '{name}': {e}")
            stats["failed"] += 1

    # Invalidate registry cache after bulk ingestion (CONCEPT:ORCH-1.2)
    try:
        from agent_utilities.graph.config_helpers import invalidate_registry_cache

        invalidate_registry_cache()
        logger.info(
            f"A2A sync complete: {stats['registered']} registered, "
            f"{stats['skipped']} skipped, {stats['failed']} failed."
        )
    except ImportError:
        logger.debug("Registry cache invalidation not available.")

    return stats


async def periodic_a2a_refresh(
    config_path: str,
    interval_seconds: int = DEFAULT_REFRESH_INTERVAL,
) -> None:
    """Background task to periodically re-fetch A2A agent cards.

    CONCEPT:ECO-4.1 — A2A Config File Loading (periodic refresh)

    Runs on a configurable interval to pick up new skills/capabilities
    from remote agents. Should be started as ``asyncio.create_task()``
    at server startup.

    Args:
        config_path: Path to the A2A config file.
        interval_seconds: Refresh interval in seconds (default: 300).
    """
    logger.info(
        f"A2A periodic refresh started (interval: {interval_seconds}s, "
        f"config: {config_path})"
    )
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            stats = await sync_a2a_agents(config_path=config_path)
            logger.info(f"A2A periodic refresh: {stats}")
        except Exception as e:
            logger.warning(f"A2A periodic refresh failed: {e}")
