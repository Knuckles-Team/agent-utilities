#!/usr/bin/env python3
from __future__ import annotations

"""CONCEPT:OS-5.5 — Distributed Coordinator with Semantic Sharding.

Coordinates task allocation and priority queue sharding across distributed agent nodes,
routing requests to specialist workers depending on dynamic role mappings.
"""

import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

try:
    import nats
except ImportError:
    nats = None


class DistributedCoordinator:
    """Enterprise-scale Distributed Coordinator using NATS JetStream or local priority queue.

    Dynamically routes tasks based on semantic roles and topologies.
    """

    def __init__(self, nats_url: str | None = None) -> None:
        self.nats_url = nats_url or os.environ.get("NATS_URL", "nats://localhost:4222")
        self._nc: Any = None
        self._js: Any = None
        self._local_queues: dict[str, asyncio.Queue] = {}
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to NATS cluster, fallback to local queues on failure."""
        if nats is None:
            logger.info(
                "NATS client library 'nats-py' not installed. Running in localized fallback mode."
            )
            self._connected = False
            return False

        try:
            self._nc = await nats.connect(self.nats_url, connect_timeout=2.0)
            self._js = self._nc.jetstream()
            # Register streams
            await self._js.add_stream(name="agent_tasks", subjects=["agent.tasks.*"])
            self._connected = True
            logger.info(
                "DistributedCoordinator successfully connected to NATS cluster."
            )
            return True
        except Exception as e:
            logger.warning(
                "Failed to connect to NATS cluster (%s). Falling back to local routing.",
                e,
            )
            self._connected = False
            return False

    async def route_task(self, role: str, task_payload: dict[str, Any]) -> None:
        """Route a task to the appropriate agent shard based on the agent's role."""
        subject = f"agent.tasks.{role.lower().replace(':', '_')}"
        if self._connected and self._js is not None:
            try:
                payload = json.dumps(task_payload).encode("utf-8")
                await self._js.publish(subject, payload)
                logger.info("Routed task to NATS subject: %s", subject)
                return
            except Exception as e:
                logger.error("Failed to publish task to NATS stream: %s", e)

        # Fallback: In-memory queue
        if subject not in self._local_queues:
            self._local_queues[subject] = asyncio.Queue()
        await self._local_queues[subject].put(task_payload)
        logger.info("Routed task locally to queue: %s", subject)

    async def register_listener(
        self, role: str, handler: Callable[[dict[str, Any]], Any]
    ) -> asyncio.Task | None:
        """Register a worker to listen for tasks matching a given agent role."""
        subject = f"agent.tasks.{role.lower().replace(':', '_')}"

        async def _worker_loop():
            while True:
                try:
                    if self._connected and self._js is not None:
                        # Pull subscription
                        sub = await self._js.pull_subscribe(
                            subject, f"durable-{role.replace(':', '_')}"
                        )
                        msgs = await sub.fetch(1, timeout=1.0)
                        if msgs:
                            msg = msgs[0]
                            payload = json.loads(msg.data.decode("utf-8"))
                            await handler(payload)
                            await msg.ack()
                    else:
                        if subject not in self._local_queues:
                            self._local_queues[subject] = asyncio.Queue()
                        payload = await self._local_queues[subject].get()
                        await handler(payload)
                        self._local_queues[subject].task_done()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("Worker subscription loop tick: %s", e)
                    await asyncio.sleep(0.5)

        return asyncio.create_task(_worker_loop())

    async def close(self) -> None:
        """Close external connections."""
        if self._nc is not None:
            await self._nc.close()
            self._connected = False
            logger.info("Closed DistributedCoordinator connection.")
