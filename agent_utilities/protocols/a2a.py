from __future__ import annotations

"""Agent-to-Agent (A2A) Peer Management Module.

This module provides a JSON-RPC client for executing tasks on remote A2A peer agents.
All discovery and registration of peers is now handled via the Knowledge Graph.
"""


import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


from agent_utilities.core.config import *  # noqa: F403
from agent_utilities.models import A2APeerModel, A2ARegistryModel  # noqa: F401
from agent_utilities.protocols.source_connectors.http_safety import (
    configured_source_http_policy,
    safe_get_json,
    safe_get_json_async,
    safe_post_json_async,
)

logger = logging.getLogger(__name__)


class A2AClient:
    """Client for Agent-to-Agent communication using the fastA2A JSON-RPC protocol.

    This client supports fetching agent metadata (cards) and executing
    arbitrary tasks on remote agents with polling for completion.
    """

    def __init__(self, timeout: float = 300.0):
        """Initialize the A2A client.

        Args:
            timeout: Maximum execution timeout for remote tasks in seconds.
        """
        self.timeout = max(1.0, min(float(timeout), 3600.0))

    def fetch_card_sync(self, url: str) -> dict[str, Any] | None:
        """Fetch the agent-card.json from a remote agent (synchronous).

        Args:
            url: The base URL of the remote agent.

        Returns:
            The agent card data as a dictionary, or None if the request fails.

        """
        card_url = f"{url.rstrip('/')}/.well-known/agent-card.json"
        try:
            value = safe_get_json(
                card_url, timeout=5.0, **configured_source_http_policy()
            )
            return value if isinstance(value, dict) else None
        except Exception as e:
            logger.debug("A2A card fetch failed (%s)", type(e).__name__)
        return None

    async def fetch_card(self, url: str) -> dict[str, Any] | None:
        """Fetch the agent-card.json from a remote agent (asynchronous).

        Args:
            url: The base URL of the remote agent.

        Returns:
            The agent card data as a dictionary, or None if the request fails.

        """
        card_url = f"{url.rstrip('/')}/.well-known/agent-card.json"
        try:
            value = await safe_get_json_async(
                card_url, timeout=10.0, **configured_source_http_policy()
            )
            return value if isinstance(value, dict) else None
        except Exception as e:
            logger.debug("A2A card fetch failed (%s)", type(e).__name__)
        return None

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        policy = configured_source_http_policy()
        value = await safe_post_json_async(
            url,
            payload,
            timeout=min(self.timeout, 60.0),
            max_bytes=policy["max_bytes"],
            allowed_private_hosts=policy["allowed_private_hosts"],
            tls_service="a2a",
        )
        if not isinstance(value, dict):
            raise ValueError("A2A response root must be an object")
        return value

    async def execute_task(self, url: str, query: str) -> Any | None:
        """Execute a task on a remote agent via A2A message/send and polling.

        Args:
            url: The A2A endpoint URL.
            query: The natural language task description.

        Returns:
            The final result content from the remote agent, or an error message.

        """
        envelope = await self._execute_result(url, query)
        if envelope["error"]:
            return f"Error: {envelope['error']}"
        return envelope["content"]

    async def _execute_result(self, url: str, query: str) -> dict[str, Any]:
        """Run one bounded A2A exchange and return a privacy-safe envelope."""
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "role": "user",
                    "parts": [{"kind": "text", "text": query}],
                    "messageId": str(uuid.uuid4()),
                }
            },
            "id": 1,
        }

        def failure(message: str) -> dict[str, Any]:
            return {
                "content": "",
                "epistemic": {},
                "metadata": {},
                "error": message,
            }

        try:
            async with asyncio.timeout(self.timeout):
                response = await self._post(url, payload)
                if "error" in response:
                    return failure("A2A peer rejected the request")
                result = response.get("result")
                task_id = result.get("id") if isinstance(result, dict) else None
                if not isinstance(task_id, str) or not task_id:
                    return failure("A2A peer returned no task identifier")

                while True:
                    await asyncio.sleep(2)
                    poll_data = await self._post(
                        url,
                        {
                            "jsonrpc": "2.0",
                            "method": "tasks/get",
                            "params": {"id": task_id},
                            "id": 2,
                        },
                    )
                    if "error" in poll_data:
                        return failure("A2A peer rejected task polling")
                    result = poll_data.get("result")
                    if not isinstance(result, dict):
                        return failure("A2A peer returned an invalid task result")
                    status = result.get("status")
                    state = status.get("state") if isinstance(status, dict) else None
                    if state in {"submitted", "running", "working"}:
                        continue
                    history = result.get("history")
                    if not isinstance(history, list):
                        return failure("A2A task result had no message history")
                    for message in reversed(history):
                        if not isinstance(message, dict) or message.get("role") == "user":
                            continue
                        raw_parts = message.get("parts")
                        parts = raw_parts if isinstance(raw_parts, list) else []
                        content = "".join(
                            str(part.get("text", part.get("content", "")))
                            for part in parts
                            if isinstance(part, dict)
                        )
                        raw_metadata = message.get("metadata")
                        if not isinstance(raw_metadata, dict):
                            raw_metadata = {}
                        from agent_utilities.security.persistence_privacy import (
                            sanitize_for_persistence,
                        )

                        metadata, _privacy = sanitize_for_persistence(raw_metadata)
                        epistemic_keys = {
                            "confidence",
                            "status",
                            "contradiction_count",
                            "policy_labels",
                            "source_refs",
                            "evidence_refs",
                        }
                        epistemic = {
                            key: metadata[key]
                            for key in epistemic_keys
                            if key in metadata
                        }
                        return {
                            "content": content,
                            "epistemic": epistemic,
                            "metadata": metadata,
                            "error": None,
                        }
                    return failure("A2A task result had no assistant message")
        except TimeoutError:
            return failure("A2A execution deadline exceeded")
        except Exception as exc:  # noqa: BLE001 - remote boundary is fail-closed
            logger.debug("A2A execution failed (%s)", type(exc).__name__)
            return failure(f"A2A communication failed ({type(exc).__name__})")

    async def execute_task_with_epistemic(self, url: str, query: str) -> dict[str, Any]:
        """Execute a task and return the RESULT ENVELOPE, including any
        epistemic metadata the peer attached (CONCEPT:AU-KB-CURRENCY A2A
        projection, `04-five-intersections.md` item 1: "epistemic columns
        ... not a default field on ordinary MCP tool-call results").

        :meth:`execute_task` (unchanged, still the primary/default entry
        point every existing caller uses) collapses a result down to its
        text content and drops the A2A Message's own ``metadata`` dict. This
        sibling method is purely additive — same request/poll protocol, but
        also surfaces that ``metadata`` under an ``epistemic`` key, picking
        out the light-epistemic-layer vocabulary
        (``confidence``/``status``/``contradiction_count``/
        ``policy_labels``/``source_refs`` — CONCEPT:AU-KB-CURRENCY /
        CONCEPT:EPI-P3-1) a peer MAY have set on its response message.
        Never fabricates: a peer that sends no metadata (any non-epistemic-
        aware A2A agent, including older versions of this same agent)
        yields ``epistemic: {}``, not guessed values.

        Returns:
            ``{"content": str, "epistemic": dict, "metadata": dict, "error": str | None}``.
            ``metadata`` is the peer metadata after the persistence-privacy
            boundary removes locations, secrets, and personal identifiers;
            ``epistemic`` is the subset of it under the recognized names.
        """
        return await self._execute_result(url, query)

    async def execute_bft_consensus(
        self, urls: list[str], query: str, threshold: float = 0.66
    ) -> Any:
        """CONCEPT:AU-ECO.interop.multi-agent-bft-consensus — Multi-agent BFT consensus for A2A.
        Queries multiple peers simultaneously and requires a consensus threshold to return a validated result.

        Args:
            urls: List of A2A endpoint URLs to query.
            query: The task description.
            threshold: The required consensus ratio (e.g. 0.66 for 2/3 majority).

        Returns:
            The consensus result or an error message.
        """
        import asyncio
        from collections import Counter

        tasks = [self.execute_task(url, query) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for res in results:
            if not isinstance(res, Exception) and not str(res).startswith("Error:"):
                valid_results.append(res)

        if not valid_results:
            return "BFT Error: No valid responses received."

        counts = Counter(str(r).strip() for r in valid_results)
        most_common, count = counts.most_common(1)[0]

        consensus_ratio = count / len(urls)
        if consensus_ratio >= threshold:
            logger.info(
                f"BFT Consensus reached: {consensus_ratio * 100:.1f}% agreement"
            )
            return most_common

        return f"BFT Error: Consensus failed (max agreement {consensus_ratio * 100:.1f}% < {threshold * 100:.1f}%)"

    async def execute_multisig_mutation(
        self, urls: list[str], query: str, signatures: list[str], threshold: int
    ) -> Any:
        """CONCEPT:AU-OS.identity.multisig-cryptographic-mutation — Multi-sig Cryptographic Mutation.
        Executes a mutating task on remote peers, requiring valid cryptographic signatures
        from multiple authorized departments before the mutation is committed.

        Args:
            urls: List of peer A2A URLs.
            query: The mutating task query.
            signatures: List of cryptographic signatures from authorizing agents.
            threshold: Minimum number of signatures required.

        Returns:
            The execution result or error message if authorization fails.
        """
        import asyncio

        if len(signatures) < threshold:
            logger.warning(
                "Multi-sig mutation rejected: %d/%d signatures",
                len(signatures),
                threshold,
            )
            return f"Authorization Error: Insufficient signatures ({len(signatures)} < {threshold})"

        # Serialize signatures into the query context for the remote peers to verify
        sig_context = "\\n## AUTHORIZATION\\nMulti-sig Threshold Met:\\n" + "\\n".join(
            f"- {s}" for s in signatures
        )
        auth_query = f"{query}\\n{sig_context}"

        # Execute across peers
        tasks = [self.execute_task(url, auth_query) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [
            r
            for r in results
            if not isinstance(r, Exception) and not str(r).startswith("Error:")
        ]

        if not valid_results:
            return (
                "Multi-sig Execution Error: All peers failed to execute the mutation."
            )

        return valid_results[0]  # Return the first successful mutation confirmation


# --- Registry Utilities (Graph-Native Fallbacks) ---


def register_a2a_peer(
    name: str,
    url: str,
    description: str = "",
    capabilities: str = "",
    auth: str = "none",
) -> str:
    """Register or update an A2A agent in the Knowledge Graph.

    Args:
        name: The unique identifier for the peer agent.
        url: The connection URL for the peer service.
        description: A brief summary of the peer's purpose.
        capabilities: A comma-separated list of peer specialties.
        auth: Authentication type.

    Returns:
        A confirmation message.
    """
    import time

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.models.knowledge_graph import AgentNode, RegistryNodeType

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "Error: Knowledge Graph engine is not active."

    agent_id = f"a2a:{name.lower().replace(' ', '_')}"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    node = AgentNode(
        id=agent_id,
        name=name,
        type=RegistryNodeType.AGENT,
        agent_type="a2a",
        description=description,
        endpoint_url=url,
        timestamp=ts,
        metadata={"capabilities": capabilities.split(","), "auth": auth},
    )

    try:
        engine.graph.add_node(node.id, **node.model_dump())
        return f"A2A peer '{name}' registered successfully in the Knowledge Graph."
    except Exception as e:
        return f"Error registering A2A peer: {e}"


def delete_a2a_peer(name: str) -> str:
    """Remove an A2A peer agent from the Knowledge Graph."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return "Error: Knowledge Graph engine is not active."

    agent_id = f"a2a:{name.lower().replace(' ', '_')}"

    try:
        if agent_id in engine.graph:
            engine.graph.remove_node(agent_id)
            return f"A2A peer '{name}' removed from the Knowledge Graph."
        else:
            return f"Error: A2A peer '{name}' not found."
    except Exception as e:
        return f"Error removing A2A peer: {e}"


def list_a2a_peers() -> Any:
    """List all known A2A peer agents from the Knowledge Graph."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        return A2ARegistryModel(peers=[])

    query = "MATCH (a:Agent {agent_type: 'a2a'}) RETURN a.name as name, a.description AS description, a.endpoint_url as url, a.metadata as meta"
    results = engine.query_cypher(query)

    peers = {}
    for r in results:
        peers[r["name"]] = A2APeerModel(
            name=r["name"],
            url=r.get("url", ""),
            description=r.get("description", "") or "",
            capabilities=",".join((r.get("meta") or {}).get("capabilities", [])),
        )

    return A2ARegistryModel(peers=list(peers.values()))
