#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-OS.identity.permissions-kernel — Permissions Kernel (Identity-Based Governance).

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
      3. Denies unmatched tools under a closed-world policy

Integrates with:
    - CONCEPT:AU-OS.identity.permissions-kernel (Secrets & Auth): HMAC key from Secrets Engine
    - CONCEPT:AU-ECO.messaging.native-backend-abstraction (Agent Tool System): Tool Guard pipeline integration
    - CONCEPT:AU-OS.state.cognitive-scheduler-preemption (Cognitive Scheduler): Priority escalation for CRITICAL roles
    - ``systems-manager``: Eunomia RBAC enforcement

See docs/pillars/5_agent_os_infrastructure.md §CONCEPT:AU-OS.state.cognitive-scheduler-preemption
"""


import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from ..core.config import AgentConfig
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

_MIN_SIGNING_KEY_BYTES = 32
_MAX_SIGNING_KEY_BYTES = 1_048_576
_MAX_POLICY_FILE_BYTES = 1_048_576
_MAX_POLICIES = 64


class PermissionBootstrapError(RuntimeError):
    """Raised when a verified runtime permission context cannot be created.

    Messages deliberately omit secret references, secret material, and policy
    paths so this exception is safe to surface through startup diagnostics.
    """


class PermissionPolicyError(ValueError):
    """Raised when an explicitly configured policy document is unavailable or invalid."""


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
        """Return the unambiguous canonical JSON used for HMAC signing."""
        return json.dumps(
            {
                "agent_id": self.agent_id,
                "capabilities": sorted(self.capabilities),
                "expires_at": self.expires_at,
                "issued_at": self.issued_at,
                "role": self.role.value,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


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

    model_config = ConfigDict(extra="forbid")

    role: AgentRole
    allowed_tools: list[str] = Field(default_factory=lambda: ["*"], max_length=1_024)
    denied_tools: list[str] = Field(default_factory=list, max_length=1_024)
    require_approval_for: list[str] = Field(default_factory=list, max_length=1_024)
    max_token_quota: int = Field(default=100_000, ge=1, le=10_000_000)
    description: str = Field(default="", max_length=4_096)


@dataclass(frozen=True, slots=True)
class PermissionContext:
    """One verified kernel and signed identity shared by a runtime execution tree."""

    kernel: PermissionsKernel
    identity: AgentIdentity


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

    CONCEPT:AU-OS.identity.permissions-kernel — Permissions Kernel

    Manages the lifecycle of agent identities and enforces role-based
    tool access policies.  Integrates with the existing ``tool_guard.py``
    pipeline as an identity-aware pre-check.

    Args:
        signing_key: Explicit stable HMAC-SHA256 key material for identity
            signing. Runtime callers resolve this from an AgentConfig secret
            reference through :func:`resolve_permission_context`.
        policies_path: Path to ``agent_policies.json``.  If ``None``,
            uses the built-in ``DEFAULT_POLICIES``.
        engine: Optional KG engine for policy/identity persistence.
    """

    def __init__(
        self,
        *,
        signing_key: str | bytes,
        policies_path: str | None = None,
        engine: IntelligenceGraphEngine | None = None,
    ) -> None:
        key_bytes = (
            signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        )
        if not isinstance(key_bytes, bytes):
            raise TypeError("signing_key must be explicit string or bytes material")
        if not (_MIN_SIGNING_KEY_BYTES <= len(key_bytes) <= _MAX_SIGNING_KEY_BYTES):
            raise ValueError("signing_key must contain 32 bytes or more")
        if b"\x00" in key_bytes:
            raise ValueError("signing_key contains invalid material")
        self._signing_key = key_bytes
        self._policies: dict[AgentRole, AgentPolicy] = {}
        self._identities: dict[str, AgentIdentity] = {}
        self.engine = engine

        # An explicitly configured policy is authoritative: missing or malformed
        # input aborts startup instead of silently widening access to defaults.
        if policies_path:
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
            "Issued identity role=%s capability_count=%d",
            role,
            len(capabilities or []),
        )
        return identity

    def derive_agent_id(self, subject: str) -> str:
        """Derive a stable opaque agent ID without retaining the source subject."""

        rendered = str(subject or "").strip()
        encoded = rendered.encode("utf-8")
        if not rendered or len(encoded) > 4_096 or "\x00" in rendered:
            raise ValueError("agent identity subject is invalid")
        digest = hmac.new(
            self._signing_key,
            b"agent-id\x00" + encoded,
            hashlib.sha256,
        ).hexdigest()
        return f"agent:{digest[:32]}"

    def verify_identity(self, identity: AgentIdentity) -> bool:
        """Verify the HMAC signature and expiry of an agent identity.

        Args:
            identity: The identity to verify.

        Returns:
            True if the signature is valid and the identity hasn't expired.
        """
        # Check expiry
        if identity.expires_at > 0 and time.time() > identity.expires_at:
            logger.warning("Identity expired")
            return False

        # Check signature
        expected = self._sign(identity.payload_string())
        if not hmac.compare_digest(identity.signature, expected):
            logger.warning("Identity signature mismatch")
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
        *,
        required_capability: str | None = None,
    ) -> AuthDecision:
        """Determine whether an agent is authorized to call a tool.

        The decision follows a strict precedence:
        1. DENY if identity is invalid or expired
        2. DENY if a non-empty identity capability set does not grant the tool
           name or its declared required capability
        3. DENY if tool matches ``denied_tools`` patterns
        4. REQUIRE_APPROVAL if tool matches ``require_approval_for`` patterns
        5. ALLOW if tool matches ``allowed_tools`` patterns
        6. DENY otherwise (closed-world assumption)

        Args:
            identity: The calling agent's signed identity.
            tool_name: The tool being requested.
            required_capability: Optional semantic capability required by the
                governed action/tool definition.

        Returns:
            An ``AuthDecision`` (ALLOW, DENY, or REQUIRE_APPROVAL).
        """
        # Step 1: Verify identity
        if not self.verify_identity(identity):
            return AuthDecision.DENY

        # Step 2: Look up policy for role
        policy = self._policies.get(identity.role)
        if not policy:
            logger.warning("No policy for role=%s; denying request", identity.role)
            return AuthDecision.DENY

        tool_lower = tool_name.lower()

        # A non-empty identity grant set is an additional closed-world boundary,
        # never an elevation over the role policy. Grants are glob patterns over
        # tool names and may also match the action's explicit semantic capability.
        if identity.capabilities:
            capability_target = str(required_capability or "").strip().lower()
            tool_granted = self._matches_patterns(
                tool_lower, identity.capabilities
            ) or bool(
                capability_target
                and self._matches_patterns(capability_target, identity.capabilities)
            )
            if not tool_granted:
                return AuthDecision.DENY

        # Step 3: Check denied (highest precedence after identity/capability)
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

        Raises:
            PermissionPolicyError: If the configured document is absent,
                oversized, malformed, empty, or contains duplicate roles.
        """
        self._policies.clear()
        try:
            policy_path = Path(path).expanduser()
            if policy_path.is_symlink() or not policy_path.is_file():
                raise PermissionPolicyError(
                    "configured permission policy is unavailable"
                )
            size = policy_path.stat().st_size
            if size <= 0 or size > _MAX_POLICY_FILE_BYTES:
                raise PermissionPolicyError(
                    "configured permission policy has invalid size"
                )

            def reject_constant(_value: str) -> None:
                raise ValueError("non-finite constants are not supported")

            def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
                value: dict[str, object] = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError("duplicate JSON keys are not supported")
                    value[key] = item
                return value

            data = json.loads(
                policy_path.read_text(encoding="utf-8"),
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
            if not isinstance(data, Mapping) or set(data) != {"policies"}:
                raise ValueError("policy document must contain only policies")
            policies_data = data["policies"]
            if (
                not isinstance(policies_data, Sequence)
                or isinstance(policies_data, str | bytes)
                or not 1 <= len(policies_data) <= _MAX_POLICIES
            ):
                raise ValueError("policies must be a bounded non-empty list")

            loaded: dict[AgentRole, AgentPolicy] = {}
            required = {
                "role",
                "allowed_tools",
                "denied_tools",
                "require_approval_for",
                "max_token_quota",
            }
            for raw_policy in policies_data:
                if not isinstance(raw_policy, Mapping) or not required.issubset(
                    raw_policy
                ):
                    raise ValueError("configured policy is incomplete")
                policy = AgentPolicy.model_validate(raw_policy)
                if policy.role in loaded:
                    raise ValueError("configured policy contains duplicate roles")
                loaded[policy.role] = policy
            self._policies = loaded
        except PermissionPolicyError:
            logger.error("Configured permission policy is unavailable")
            raise
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            TypeError,
        ):
            logger.error("Configured permission policy is invalid")
            raise PermissionPolicyError(
                "configured permission policy is invalid"
            ) from None

        logger.info("Loaded %d permission policies", len(self._policies))

    def _load_defaults(self) -> None:
        """Load the built-in default policies."""
        self._policies.clear()
        for policy in DEFAULT_POLICIES:
            current = policy.model_copy(deep=True)
            self._policies[current.role] = current

    def get_policies(self) -> list[AgentPolicy]:
        """Return all loaded policies.

        Returns:
            List of ``AgentPolicy`` instances.
        """
        return [policy.model_copy(deep=True) for policy in self._policies.values()]

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

    def apply_multisig_mutation(
        self, signatures: list[str], threshold: int, mutation_type: str, query: str
    ) -> str:
        """Apply a mutation to the graph using zero-trust consensus signatures."""
        if len(signatures) < threshold:
            raise ValueError(
                f"Insufficient signatures: {len(signatures)} < {threshold}"
            )

        if (
            self.engine
            and hasattr(self.engine.graph, "_client")
            and self.engine.graph._client
        ):
            try:
                return self.engine.graph._client.apply_multisig_mutation(
                    signatures=signatures,
                    threshold=threshold,
                    mutation_type=mutation_type,
                    query=query,
                )
            except Exception as e:
                logger.error(
                    "Multi-sig mutation failed in Rust backend (%s)",
                    type(e).__name__,
                )
                raise

        return "Rust backend unavailable for multi-sig mutation."

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

            # Push to epistemic-graph backend for Zero-Trust Consensus
            if hasattr(self.engine.graph, "_client") and self.engine.graph._client:
                try:
                    self.engine.graph._client.register_identity(
                        agent_id=identity.agent_id,
                        role=str(
                            identity.role
                        ).title(),  # To match AgentRole enum in Rust which is titlecase eg Manager
                        teams=[],
                        signature=identity.signature,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to register identity with Rust backend (%s)",
                        type(e).__name__,
                    )
        except Exception as e:
            logger.debug("Failed to persist identity (%s)", type(e).__name__)


def resolve_permission_context(
    config: AgentConfig,
    *,
    permissions_kernel: PermissionsKernel | None = None,
    agent_identity: AgentIdentity | None = None,
    required: bool = True,
    engine: IntelligenceGraphEngine | None = None,
    agent_subject: str = "graph-runtime",
    role: AgentRole = AgentRole.SPECIALIST,
    capabilities: Sequence[str] = (),
    secret_resolver: Callable[[object], str] | None = None,
) -> PermissionContext | None:
    """Return one verified permission context for a runtime execution tree.

    An explicitly injected kernel and identity must be supplied as a pair and
    must verify against each other. Otherwise, a required context is bootstrapped
    exclusively from ``AgentConfig.permissions_signing_key_ref`` through the
    runtime secret resolver. Raw configuration material and per-process random
    authorities are deliberately unsupported.

    ``secret_resolver`` is dependency injection for bounded tests. Runtime call
    sites omit it and therefore always use the central runtime secret resolver.
    """

    if (permissions_kernel is None) != (agent_identity is None):
        raise PermissionBootstrapError(
            "permission kernel and agent identity must be injected together"
        )
    if permissions_kernel is not None and agent_identity is not None:
        return verify_permission_context(permissions_kernel, agent_identity)
    if not required:
        return None

    signing_key_ref = str(
        getattr(config, "permissions_signing_key_ref", "") or ""
    ).strip()
    if not signing_key_ref:
        raise PermissionBootstrapError(
            "permission signing key reference is required for governed execution"
        )
    if secret_resolver is None:
        from .cli_secrets import resolve_runtime_secret_reference

        secret_resolver = resolve_runtime_secret_reference
    try:
        signing_key = secret_resolver(signing_key_ref)
        kernel = PermissionsKernel(
            signing_key=signing_key,
            policies_path=getattr(config, "agent_policies_path", None),
            engine=engine,
        )
        identity = kernel.issue_identity(
            agent_id=kernel.derive_agent_id(agent_subject),
            role=role,
            capabilities=list(capabilities),
        )
    except Exception:
        raise PermissionBootstrapError("permission context bootstrap failed") from None
    if not kernel.verify_identity(identity):
        raise PermissionBootstrapError("permission context verification failed")
    return PermissionContext(kernel, identity)


def verify_permission_context(
    permissions_kernel: PermissionsKernel,
    agent_identity: AgentIdentity,
) -> PermissionContext:
    """Validate an explicitly injected kernel/identity pair without resolving secrets."""

    try:
        verified = permissions_kernel.verify_identity(agent_identity)
    except Exception:
        verified = False
    if not verified:
        raise PermissionBootstrapError("injected permission context is invalid")
    return PermissionContext(permissions_kernel, agent_identity)
