#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:OS-5.1 — Permissions Kernel (Identity-Based Governance).

Shifts agent security from tool-centric ("is this tool dangerous?") to
identity-centric ("which agent is requesting, and do they have permission?").
Provides signed agent identities, role-based policies, and integration
with the existing Tool Guard and Eunomia authorization layers.

Architecture:
    - **Agent Identity**: HMAC-signed tokens binding ``agent_id`` to a
      ``role`` (admin, operator, specialist, sandbox, guest) and a set
      of capabilities.
    - **Policy Engine**: Loads ``agent_policies.json`` at startup and
      syncs policies to KG ``PolicyNode`` entries.  Each policy maps
      ``role → allowed_tools[], denied_tools[], require_approval_for[]``.
    - **Authorization Flow**: At tool-call time, the kernel checks:
      1. Identity signature validity
      2. Role-based policy match (DENY > REQUIRE_APPROVAL > ALLOW)
      3. Falls back to pattern-based ``tool_guard.py`` for unmatched tools

Integrates with:
    - CONCEPT:OS-5.1 (Secrets & Auth): HMAC key from Secrets Engine
    - CONCEPT:ECO-4.1 (Agent Tool System): Tool Guard pipeline integration
    - CONCEPT:OS-5.2 (Cognitive Scheduler): Priority escalation for CRITICAL roles
    - ``systems-manager``: Eunomia RBAC enforcement

See docs/permissions-kernel.md §CONCEPT:OS-5.2.
"""


import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class AgentRole(StrEnum):
    """Permission roles for agent identities.

    Ordered from most privileged to least privileged.
    """

    ADMIN = "admin"  # Full access, can run destructive ops
    OPERATOR = "operator"  # Can run most tools, approval for destructive
    SPECIALIST = "specialist"  # Limited to domain-specific tools
    SANDBOX = "sandbox"  # Read-only + safe tools only
    GUEST = "guest"  # Read-only, no tool access


class AuthDecision(StrEnum):
    """Authorization decision returned by the Permissions Kernel."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class AgentIdentity(BaseModel):
    """Signed agent identity token.

    Contains the agent's role, granted capabilities, and an HMAC-SHA256
    signature for tamper detection.  Issued by the ``PermissionsKernel``
    when an agent process is spawned.

    Attributes:
        agent_id: Unique agent identifier.
        role: Permission role (admin..guest).
        capabilities: Granted capability identifiers.
        issued_at: Unix timestamp when identity was issued.
        expires_at: Optional expiry timestamp (0 = no expiry).
        signature: HMAC-SHA256 of the identity payload.
    """

    agent_id: str
    role: AgentRole = AgentRole.SPECIALIST
    capabilities: list[str] = Field(default_factory=list)
    issued_at: float = Field(default_factory=time.time)
    expires_at: float = 0.0
    signature: str = ""

    def payload_string(self) -> str:
        """Return the canonical string used for HMAC signing."""
        return f"{self.agent_id}:{self.role}:{','.join(sorted(self.capabilities))}:{self.issued_at}"


class AgentPolicy(BaseModel):
    """Role-based tool access policy.

    Defines which tools a role can access, which are denied, and which
    require human approval.  Glob patterns are supported (e.g. ``delete_*``).

    Attributes:
        role: The role this policy applies to.
        allowed_tools: Glob patterns of allowed tool names (default ["*"]).
        denied_tools: Glob patterns of denied tool names.
        require_approval_for: Glob patterns requiring approval.
        max_token_quota: Maximum per-process token budget for this role.
        description: Human-readable policy description.
    """

    role: AgentRole
    allowed_tools: list[str] = Field(default_factory=lambda: ["*"])
    denied_tools: list[str] = Field(default_factory=list)
    require_approval_for: list[str] = Field(default_factory=list)
    max_token_quota: int = 100_000
    description: str = ""


# Default policies when no agent_policies.json is provided
DEFAULT_POLICIES: list[AgentPolicy] = [
    AgentPolicy(
        role=AgentRole.ADMIN,
        allowed_tools=["*"],
        denied_tools=[],
        require_approval_for=[],
        max_token_quota=500_000,
        description="Full access — can run any tool without approval",
    ),
    AgentPolicy(
        role=AgentRole.OPERATOR,
        allowed_tools=["*"],
        denied_tools=[],
        require_approval_for=[
            "*delete*",
            "*remove*",
            "*drop*",
            "*reboot*",
            "*shutdown*",
        ],
        max_token_quota=200_000,
        description="Broad access — destructive operations require approval",
    ),
    AgentPolicy(
        role=AgentRole.SPECIALIST,
        allowed_tools=["*"],
        denied_tools=["*reboot*", "*shutdown*", "*install*", "*uninstall*"],
        require_approval_for=["*delete*", "*remove*", "*execute*", "*shell*"],
        max_token_quota=100_000,
        description="Domain tools — OS-level operations denied",
    ),
    AgentPolicy(
        role=AgentRole.SANDBOX,
        allowed_tools=["read_*", "list_*", "get_*", "describe_*", "search_*", "view_*"],
        denied_tools=["*"],
        require_approval_for=[],
        max_token_quota=50_000,
        description="Read-only — can only access safe retrieval tools",
    ),
    AgentPolicy(
        role=AgentRole.GUEST,
        allowed_tools=[],
        denied_tools=["*"],
        require_approval_for=[],
        max_token_quota=10_000,
        description="No tool access — can only observe",
    ),
]


class PermissionsKernel:
    """Identity-based permissions kernel for agent governance.

    CONCEPT:OS-5.1 — Permissions Kernel

    Manages the lifecycle of agent identities and enforces role-based
    tool access policies.  Integrates with the existing ``tool_guard.py``
    pipeline as an identity-aware pre-check.

    Args:
        signing_key: HMAC-SHA256 key for identity signing.  If ``None``,
            a random key is generated (suitable for single-process use).
        policies_path: Path to ``agent_policies.json``.  If ``None``,
            uses the built-in ``DEFAULT_POLICIES``.
        engine: Optional KG engine for policy/identity persistence.
    """

    def __init__(
        self,
        signing_key: str | None = None,
        policies_path: str | None = None,
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        self._signing_key = (signing_key or uuid.uuid4().hex).encode()
        self._policies: dict[AgentRole, AgentPolicy] = {}
        self._identities: dict[str, AgentIdentity] = {}
        self.engine = engine

        # Load policies
        if policies_path and os.path.isfile(policies_path):
            self.load_policies(policies_path)
        else:
            self._load_defaults()

        logger.info(
            "PermissionsKernel initialised with %d policies",
            len(self._policies),
        )

    # ── Identity Lifecycle ─────────────────────────────────────────────

    def issue_identity(
        self,
        agent_id: str,
        role: AgentRole = AgentRole.SPECIALIST,
        capabilities: list[str] | None = None,
        ttl_seconds: float = 0.0,
    ) -> AgentIdentity:
        """Create and sign a new agent identity.

        Args:
            agent_id: Unique agent identifier.
            role: Permission role to assign.
            capabilities: Optional list of granted capabilities.
            ttl_seconds: Time-to-live in seconds (0 = no expiry).

        Returns:
            The signed ``AgentIdentity``.
        """
        identity = AgentIdentity(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities or [],
            issued_at=time.time(),
            expires_at=time.time() + ttl_seconds if ttl_seconds > 0 else 0.0,
        )

        # Sign
        identity.signature = self._sign(identity.payload_string())

        # Cache
        self._identities[agent_id] = identity

        # Persist to KG
        self._persist_identity(identity)

        logger.info(
            "Issued identity: agent=%s role=%s capabilities=%s",
            agent_id,
            role,
            capabilities or [],
        )
        return identity

    def verify_identity(self, identity: AgentIdentity) -> bool:
        """Verify the HMAC signature and expiry of an agent identity.

        Args:
            identity: The identity to verify.

        Returns:
            True if the signature is valid and the identity hasn't expired.
        """
        # Check expiry
        if identity.expires_at > 0 and time.time() > identity.expires_at:
            logger.warning(
                "Identity expired: agent=%s (expired_at=%s)",
                identity.agent_id,
                identity.expires_at,
            )
            return False

        # Check signature
        expected = self._sign(identity.payload_string())
        if not hmac.compare_digest(identity.signature, expected):
            logger.warning(
                "Identity signature mismatch: agent=%s",
                identity.agent_id,
            )
            return False

        return True

    def get_identity(self, agent_id: str) -> AgentIdentity | None:
        """Retrieve a cached identity by agent ID.

        Args:
            agent_id: The agent to look up.

        Returns:
            The ``AgentIdentity``, or ``None`` if not found.
        """
        return self._identities.get(agent_id)

    # ── Authorization ──────────────────────────────────────────────────

    def authorize_tool(
        self,
        identity: AgentIdentity,
        tool_name: str,
    ) -> AuthDecision:
        """Determine whether an agent is authorized to call a tool.

        The decision follows a strict precedence:
        1. DENY if identity is invalid or expired
        2. DENY if tool matches ``denied_tools`` patterns
        3. REQUIRE_APPROVAL if tool matches ``require_approval_for`` patterns
        4. ALLOW if tool matches ``allowed_tools`` patterns
        5. DENY otherwise (closed-world assumption)

        Args:
            identity: The calling agent's signed identity.
            tool_name: The tool being requested.

        Returns:
            An ``AuthDecision`` (ALLOW, DENY, or REQUIRE_APPROVAL).
        """
        # Step 1: Verify identity
        if not self.verify_identity(identity):
            return AuthDecision.DENY

        # Step 2: Look up policy for role
        policy = self._policies.get(identity.role)
        if not policy:
            logger.warning(
                "No policy for role=%s, denying agent=%s",
                identity.role,
                identity.agent_id,
            )
            return AuthDecision.DENY

        tool_lower = tool_name.lower()

        # Step 3: Check denied (highest precedence after identity)
        if self._matches_patterns(tool_lower, policy.denied_tools):
            # Deny wins unless an *explicit* (non-wildcard) allowed pattern
            # also matches — a bare "*" in allowed_tools does not override deny.
            explicit_allows = [p for p in policy.allowed_tools if p != "*"]
            if not self._matches_patterns(tool_lower, explicit_allows):
                return AuthDecision.DENY

        # Step 4: Check require_approval
        if self._matches_patterns(tool_lower, policy.require_approval_for):
            return AuthDecision.REQUIRE_APPROVAL

        # Step 5: Check allowed
        if self._matches_patterns(tool_lower, policy.allowed_tools):
            return AuthDecision.ALLOW

        # Default deny (closed world)
        return AuthDecision.DENY

    def get_token_quota_for_role(self, role: AgentRole) -> int:
        """Return the max token quota for a given role.

        Args:
            role: The role to look up.

        Returns:
            Token quota, or 100_000 as default.
        """
        policy = self._policies.get(role)
        return policy.max_token_quota if policy else 100_000

    # ── Policy Management ──────────────────────────────────────────────

    def load_policies(self, path: str) -> None:
        """Load policies from an ``agent_policies.json`` file.

        File format:
        ```json
        {
          "policies": [
            {
              "role": "specialist",
              "allowed_tools": ["*"],
              "denied_tools": ["*reboot*"],
              "require_approval_for": ["*delete*"],
              "max_token_quota": 100000,
              "description": "Domain specialist"
            }
          ]
        }
        ```

        Args:
            path: Path to the JSON policy file.
        """
        try:
            with open(path) as f:
                data = json.load(f)

            policies_data = data.get("policies", [])
            self._policies.clear()

            for pd in policies_data:
                policy = AgentPolicy(**pd)
                self._policies[policy.role] = policy

            logger.info("Loaded %d policies from %s", len(self._policies), path)

        except Exception as e:
            logger.error("Failed to load policies from %s: %s", path, e)
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load the built-in default policies."""
        self._policies.clear()
        for policy in DEFAULT_POLICIES:
            self._policies[policy.role] = policy

    def get_policies(self) -> list[AgentPolicy]:
        """Return all loaded policies.

        Returns:
            List of ``AgentPolicy`` instances.
        """
        return list(self._policies.values())

    # ── KG Synchronization ─────────────────────────────────────────────

    def sync_to_kg(self) -> int:
        """Persist all policies and identities to the Knowledge Graph.

        Policies are stored as ``PolicyNode`` entries; identities as
        ``AgentIdentityNode`` entries.

        Returns:
            Total number of nodes synced.
        """
        if not self.engine:
            return 0

        from ..models.knowledge_graph import (
            AgentIdentityNode,
            PolicyNode,
            RegistryEdgeType,
            RegistryNodeType,
        )

        synced = 0

        # Sync policies
        for role, policy in self._policies.items():
            node_id = f"policy:{role}"
            node = PolicyNode(
                id=node_id,
                type=RegistryNodeType.POLICY,
                name=f"Agent Policy: {role}",
                description=policy.description,
                policy_id=node_id,
                condition=f"role={role}",
                action=f"allowed={len(policy.allowed_tools)}, denied={len(policy.denied_tools)}",
                priority=50,
                applies_to=[str(role)],
                importance_score=0.9,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                metadata={
                    "allowed_tools": policy.allowed_tools,
                    "denied_tools": policy.denied_tools,
                    "require_approval_for": policy.require_approval_for,
                    "max_token_quota": policy.max_token_quota,
                },
            )
            self.engine.graph.add_node(node_id, **node.model_dump())
            synced += 1

        # Sync identities
        for agent_id, identity in self._identities.items():
            node_id = f"identity:{agent_id}"
            node = AgentIdentityNode(  # type: ignore[assignment]
                id=node_id,
                name=f"Identity: {agent_id}",
                description=f"Agent {agent_id} with role {identity.role}",
                role=str(identity.role),
                capabilities=identity.capabilities,
                signature=identity.signature,
                issued_at=identity.issued_at,
                importance_score=0.8,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self.engine.graph.add_node(node_id, **node.model_dump())

            # Link identity to agent node if it exists
            if agent_id in self.engine.graph:
                self.engine.graph.add_edge(
                    agent_id,
                    node_id,
                    type=RegistryEdgeType.HAS_IDENTITY,
                )
            synced += 1

        logger.info("Synced %d nodes to KG", synced)
        return synced

    # ── Private Helpers ────────────────────────────────────────────────

    def _sign(self, payload: str) -> str:
        """Create an HMAC-SHA256 signature of a payload string."""
        return hmac.new(
            self._signing_key,
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _matches_patterns(tool_name: str, patterns: list[str]) -> bool:
        """Check if a tool name matches any glob-like patterns.

        Supports simple glob patterns:
        - ``*`` matches everything
        - ``*prefix`` matches tools ending with prefix
        - ``prefix*`` matches tools starting with prefix
        - ``*middle*`` matches tools containing middle

        Args:
            tool_name: Lowercase tool name.
            patterns: List of glob patterns.

        Returns:
            True if any pattern matches.
        """
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(tool_name, pattern.lower()):
                return True
        return False

    def _persist_identity(self, identity: AgentIdentity) -> None:
        """Persist a single identity to the KG."""
        if not self.engine:
            return

        try:
            from ..models.knowledge_graph import (
                AgentIdentityNode,
                RegistryEdgeType,
            )

            node_id = f"identity:{identity.agent_id}"
            node = AgentIdentityNode(
                id=node_id,
                name=f"Identity: {identity.agent_id}",
                description=f"Agent {identity.agent_id} with role {identity.role}",
                role=str(identity.role),
                capabilities=identity.capabilities,
                signature=identity.signature,
                issued_at=identity.issued_at,
                importance_score=0.8,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self.engine.graph.add_node(node_id, **node.model_dump())

            if identity.agent_id in self.engine.graph:
                self.engine.graph.add_edge(
                    identity.agent_id,
                    node_id,
                    type=RegistryEdgeType.HAS_IDENTITY,
                )
        except Exception as e:
            logger.debug("Failed to persist identity %s: %s", identity.agent_id, e)
