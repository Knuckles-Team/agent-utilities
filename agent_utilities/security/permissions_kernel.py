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
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from ..core.config import AgentConfig
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

_MIN_SIGNING_KEY_BYTES = 32
_MAX_SIGNING_KEY_BYTES = 1_048_576
_MAX_POLICY_FILE_BYTES = 1_048_576
_MAX_POLICIES = 64

# ── Self-provisioned signing key (CONCEPT:AU-OS.identity.permissions-kernel) ──
#
# When ``permissions_signing_key_ref`` is not configured, governed execution
# provisions its OWN stable HMAC signing authority instead of failing. The key is
# generated with a CSPRNG and then persisted DURABLY, under this one well-known
# name, in the engine's durable secret store via an atomic membership-test-and-
# insert (``set_if_absent``). Every process/replica converges on the SAME stored
# key and reuses it across restarts, so it is a durable *shared* authority — NOT a
# per-process random one (the distinction ``profile_guard`` cares about). A key is
# never returned to a caller unless it is the durably stored value.
#
# One key, one purpose: this authority is DEDICATED to permission-identity signing
# and is never the engine transport HMAC (``graph_service_auth_secret``), the
# durable-identity HMAC (``persistence_identity_hmac_key_ref``), or the store's
# encryption-at-rest key (``EPISTEMIC_GRAPH_ENCRYPTION_KEY``).
WELL_KNOWN_SIGNING_KEY_NAME = "system:permissions-signing-key"

# The stored value is a small VERSIONED document (not a bare key) so a future key
# rotation — new active version signs new identities; older versions still VERIFY
# un-expired identities during a grace window — needs NO data migration. The shape
# already carries N versions.
_SIGNING_KEY_DOC_SCHEMA = "au.permissions-signing-key.v1"
_PROVISIONED_KEY_BYTES = 32

# Version lifecycle: ``active`` signs new identities AND verifies; ``grace`` only
# verifies (a rotated-out key still validating in-flight identities); ``retired``
# is retained for the audit trail but never used to sign or verify.
_VERIFYING_STATUSES = frozenset({"active", "grace"})

# Runtime-issued identities carry a bounded TTL (the signing KEY stays long-lived).
# A long-running governed task never dies at TTL because the boundary re-issues
# on use within the refresh skew — see ``refresh_identity_if_expiring``.
_DEFAULT_IDENTITY_TTL_SECONDS = 3600.0
_DEFAULT_IDENTITY_REFRESH_SKEW_SECONDS = 300.0

# Emit the "no encryption-at-rest configured" hardening advisory at most once.
_ENCRYPTION_POSTURE_WARNED = False


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
        additional_verification_keys: Sequence[str | bytes] = (),
        identity_ttl_seconds: float = 0.0,
        identity_refresh_skew_seconds: float = _DEFAULT_IDENTITY_REFRESH_SKEW_SECONDS,
    ) -> None:
        # Trust model: identities are HMAC-SHA256 signed. In the self-contained,
        # single-process posture the SAME kernel both signs and verifies, so a
        # symmetric secret is the correct, sufficient authority — an identity is
        # never merely "trusted", it is re-verified against this secret on every
        # governed-execution check (see ``authorize_tool`` -> ``verify_identity``).
        # ``signing_key`` is the ACTIVE authority used to SIGN; every key in
        # ``additional_verification_keys`` (rotated-out ``grace`` versions) can
        # still VERIFY an un-expired identity, so a rotation needs no flag-day.
        self._signing_key = self._coerce_key_material(signing_key)
        verification: list[bytes] = [self._signing_key]
        for extra in additional_verification_keys or ():
            material = self._coerce_key_material(extra)
            if material not in verification:
                verification.append(material)
        self._verification_keys: tuple[bytes, ...] = tuple(verification)
        self._identity_ttl_seconds = max(0.0, float(identity_ttl_seconds or 0.0))
        self._identity_refresh_skew_seconds = max(
            0.0, float(identity_refresh_skew_seconds or 0.0)
        )
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

    def reissue_identity(
        self, identity: AgentIdentity, *, ttl_seconds: float | None = None
    ) -> AgentIdentity:
        """Return a freshly-signed copy of ``identity`` with a renewed TTL window.

        The renewal seam: cheap, IN-PROCESS, and with NO external round-trip — it
        re-signs with the stable active key and never touches the durable store.
        Preserves ``agent_id``/``role``/``capabilities`` (so authorization is
        identical) and only refreshes ``issued_at``/``expires_at``. Used at the
        governed-execution boundary so a task that outlives one identity TTL keeps
        running instead of failing verification.
        """
        if ttl_seconds is None:
            span = identity.expires_at - identity.issued_at
            ttl_seconds = span if span > 0 else self._identity_ttl_seconds
        now = time.time()
        fresh = AgentIdentity(
            agent_id=identity.agent_id,
            role=identity.role,
            capabilities=list(identity.capabilities),
            issued_at=now,
            expires_at=(now + ttl_seconds) if ttl_seconds > 0 else 0.0,
        )
        fresh.signature = self._sign(fresh.payload_string())
        self._identities[fresh.agent_id] = fresh
        logger.info("Re-issued identity role=%s (TTL refresh)", fresh.role)
        return fresh

    def refresh_identity_if_expiring(
        self, identity: AgentIdentity, *, now: float | None = None
    ) -> AgentIdentity:
        """Re-issue ``identity`` iff it is within the refresh-skew of expiry.

        Returns the SAME object when there is nothing to do (non-expiring identity,
        or comfortably before the skew window), so the caller can cheaply detect a
        renewal by identity (``is``). This is refresh-on-use — preferred over a
        background daemon: the renewal happens exactly at the governed-execution
        boundary that is about to rely on the identity being valid.
        """
        expires_at = identity.expires_at
        if expires_at <= 0:
            return identity
        current = time.time() if now is None else now
        if current >= expires_at - self._identity_refresh_skew_seconds:
            return self.reissue_identity(identity)
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

        # Check signature against EVERY still-verifying key version (active + any
        # grace versions from a rotation). A constant-time compare per version
        # keeps an old-key-signed but un-expired identity valid across a rotation
        # without ever accepting a tampered or foreign-key signature.
        payload = identity.payload_string()
        for key in self._verification_keys:
            expected = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
            if hmac.compare_digest(identity.signature, expected):
                return True
        logger.warning("Identity signature mismatch")
        return False

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
            self.engine.graph.add_node(node_id, **node.to_graph_properties())
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
            self.engine.graph.add_node(node_id, **node.to_graph_properties())

            # Link identity to agent node if it exists
            if agent_id in self.engine.graph:
                self.engine.graph.add_edge(
                    agent_id,
                    node_id,
                    relationship=RegistryEdgeType.HAS_IDENTITY,
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

    @staticmethod
    def _coerce_key_material(signing_key: str | bytes) -> bytes:
        """Validate and normalize one HMAC key to bounded, NUL-free bytes."""
        key_bytes = (
            signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        )
        if not isinstance(key_bytes, bytes):
            raise TypeError("signing_key must be explicit string or bytes material")
        if not (_MIN_SIGNING_KEY_BYTES <= len(key_bytes) <= _MAX_SIGNING_KEY_BYTES):
            raise ValueError("signing_key must contain 32 bytes or more")
        if b"\x00" in key_bytes:
            raise ValueError("signing_key contains invalid material")
        return key_bytes

    def _sign(self, payload: str) -> str:
        """Create an HMAC-SHA256 signature of a payload string with the active key."""
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
            self.engine.graph.add_node(node_id, **node.to_graph_properties())

            if identity.agent_id in self.engine.graph:
                self.engine.graph.add_edge(
                    identity.agent_id,
                    node_id,
                    relationship=RegistryEdgeType.HAS_IDENTITY,
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
                        "Failed to register identity with Rust backend: %s", e
                    )
        except Exception as e:  # noqa: BLE001 — the KG graph write is a durability/cross-process mirror only; verify_identity/authorize_tool/get_identity all read self._identities (already cached above in issue_identity), not this graph node
            logger.debug("Failed to persist identity: %s", e)


# ── Durable self-provisioning of the signing authority ─────────────────────


@dataclass(frozen=True, slots=True)
class ProvisionedSigningKey:
    """The resolved signing authority: one active signer + every verifying version."""

    active_material: str
    active_key_id: str
    verification_materials: tuple[str, ...]

    @property
    def additional_verification_materials(self) -> tuple[str, ...]:
        """Verifying materials other than the active signer (rotation grace keys)."""
        return tuple(
            m for m in self.verification_materials if m != self.active_material
        )


def _generate_key_material() -> str:
    """CSPRNG key material: 64 ASCII hex chars = 32 bytes of entropy, NUL-free."""
    return secrets.token_hex(_PROVISIONED_KEY_BYTES)


def _new_key_version(status: str = "active") -> dict[str, Any]:
    return {
        "key_id": secrets.token_hex(8),
        "material": _generate_key_material(),
        "created_at": time.time(),
        "status": status,
    }


def _valid_key_material(value: Any) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and _MIN_SIGNING_KEY_BYTES
        <= len(value.encode("utf-8"))
        <= _MAX_SIGNING_KEY_BYTES
        and "\x00" not in value
    )


def _render_key_document(versions: list[dict[str, Any]], active_key_id: str) -> str:
    return json.dumps(
        {
            "schema": _SIGNING_KEY_DOC_SCHEMA,
            "active_key_id": active_key_id,
            "versions": versions,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _provisioned_from_document(value: Any) -> ProvisionedSigningKey | None:
    """Parse a stored versioned key document into a ``ProvisionedSigningKey``.

    Defensive: any structural problem, or an absent/invalid active signer, returns
    ``None`` so the caller (re-)provisions rather than trusting malformed material.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        doc = json.loads(value)
    except (ValueError, TypeError):
        return None
    if not isinstance(doc, Mapping) or doc.get("schema") != _SIGNING_KEY_DOC_SCHEMA:
        return None
    versions = doc.get("versions")
    active_key_id = doc.get("active_key_id")
    if not isinstance(versions, list) or not isinstance(active_key_id, str):
        return None
    active_material: str | None = None
    verifying: list[str] = []
    for entry in versions:
        if not isinstance(entry, Mapping):
            return None
        material = entry.get("material")
        status = entry.get("status")
        key_id = entry.get("key_id")
        if not _valid_key_material(material) or not isinstance(key_id, str):
            return None
        if status in _VERIFYING_STATUSES:
            verifying.append(material)
        if key_id == active_key_id and status == "active":
            active_material = material
    if active_material is None or active_material not in verifying:
        return None
    return ProvisionedSigningKey(
        active_material=active_material,
        active_key_id=active_key_id,
        verification_materials=tuple(dict.fromkeys(verifying)),
    )


def _encryption_at_rest_configured(config: Any) -> bool:
    """Whether the engine's encryption-at-rest is armed for the durable store.

    The au-canonical signal is ``epistemic_graph_encryption_key_ref`` — the launcher
    resolves it and injects ``EPISTEMIC_GRAPH_ENCRYPTION_KEY`` into the engine, which
    then seals the redb durable value blobs with its ChaCha20-Poly1305 value cipher.
    The self-provisioned key is written THROUGH that standard encrypted value path
    (nothing here re-implements or bypasses it), so when this is true the stored
    material is sealed at rest (CONCEPT:AU-OS.identity.encrypted-secret-store). This
    is the same reference ``profile_guard`` requires for a packaged local production
    engine.
    """
    return bool(
        str(getattr(config, "epistemic_graph_encryption_key_ref", "") or "").strip()
    )


def _warn_encryption_posture_once(config: Any) -> None:
    """One-time hardening advisory when no encryption-at-rest key is configured.

    Graduated posture: (1) self-contained default — the durable store's own file
    permissions protect the key; (2) + set ``EPISTEMIC_GRAPH_ENCRYPTION_KEY`` to
    seal it at rest; (3) + set ``PERMISSIONS_SIGNING_KEY_REF`` to an external/KMS
    reference for a rotated, out-of-band authority.
    """
    global _ENCRYPTION_POSTURE_WARNED
    if _ENCRYPTION_POSTURE_WARNED or _encryption_at_rest_configured(config):
        return
    _ENCRYPTION_POSTURE_WARNED = True
    logger.warning(
        "Self-provisioned permission signing key is protected only by the durable "
        "store's file permissions: no encryption-at-rest key is configured. For a "
        "hardened deployment set EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF (seals the "
        "stored key at rest via the engine's value cipher) or "
        "PERMISSIONS_SIGNING_KEY_REF (an external/rotated key). "
        "(CONCEPT:AU-OS.identity.encrypted-secret-store)"
    )


def _audit_signing_key(action: str, key_id: str, detail: str) -> None:
    """Structured, material-free audit line for a signing-key lifecycle event.

    Key MATERIAL is never an argument — only the non-secret ``key_id`` (a random
    public identifier) and counts. The durable ``:Secret`` write itself is
    additionally captured by the engine's hash-chained audit log.
    """
    logger.info(
        "audit permissions-signing-key action=%s key_id=%s %s", action, key_id, detail
    )


def provision_signing_key(
    config: Any, *, secrets_client: Any = None
) -> ProvisionedSigningKey:
    """Resolve — self-provisioning once if needed — the durable signing authority.

    Idempotent across restarts: an already-provisioned document is reused. A first
    provision generates a CSPRNG key and writes a v1 versioned document into the
    engine's durable secret store via an ATOMIC ``set_if_absent`` (a durable
    compare-and-set), so two racing boots converge on ONE key — the loser reads the
    winner's document. The returned material is ALWAYS the durably stored value,
    never a per-process ephemeral one.
    """
    if secrets_client is None:
        from .secrets_client import create_secrets_client

        secrets_client = create_secrets_client()

    name = WELL_KNOWN_SIGNING_KEY_NAME

    # Fast path: already provisioned durably — reuse it (idempotent).
    provisioned = _provisioned_from_document(secrets_client.get(name))
    if provisioned is not None:
        return provisioned

    _warn_encryption_posture_once(config)

    version = _new_key_version()
    document = _render_key_document([version], version["key_id"])
    created = secrets_client.set_if_absent(
        name,
        document,
        purpose="permissions-signing-key",
        provisioned="auto",
        schema=_SIGNING_KEY_DOC_SCHEMA,
    )
    if created:
        _audit_signing_key(
            "provision", version["key_id"], "versions=1 provisioned=auto"
        )

    # Won or lost the race, the authoritative value is now whatever is durably
    # stored — re-read and adopt it so every process agrees on one key.
    provisioned = _provisioned_from_document(secrets_client.get(name))
    if provisioned is not None:
        return provisioned
    raise PermissionBootstrapError(
        "permission signing key could not be durably provisioned"
    )


def rotate_signing_key(
    config: Any, *, secrets_client: Any = None
) -> ProvisionedSigningKey:
    """Rotate the signing authority: new active version, previous active -> grace.

    Rotation-ready by construction — the stored document already carries N
    versions, so this needs NO data migration. The new active version signs new
    identities; the demoted (grace) version still VERIFIES un-expired identities
    until it is later retired. Durable and atomic on the engine backend (a
    compare-and-set on the stored document); the trigger is deliberately manual.
    """
    if secrets_client is None:
        from .secrets_client import create_secrets_client

        secrets_client = create_secrets_client()

    name = WELL_KNOWN_SIGNING_KEY_NAME
    current_value = secrets_client.get(name)
    current = _provisioned_from_document(current_value)
    if current is None:
        # Nothing valid to rotate from — provision a fresh authority instead.
        return provision_signing_key(config, secrets_client=secrets_client)

    doc = json.loads(current_value)
    versions = [dict(v) for v in doc["versions"] if isinstance(v, Mapping)]
    for entry in versions:
        if entry.get("status") == "active":
            entry["status"] = "grace"
    new_version = _new_key_version()
    versions.append(new_version)
    new_document = _render_key_document(versions, new_version["key_id"])

    applied = secrets_client.compare_and_set(name, current_value, new_document)
    if not applied:
        # Lost a concurrent rotation — adopt whatever is now durably stored.
        latest = _provisioned_from_document(secrets_client.get(name))
        if latest is not None:
            return latest
        raise PermissionBootstrapError("permission signing key rotation conflict")
    _audit_signing_key(
        "rotate", new_version["key_id"], f"versions={len(versions)} prior_active=grace"
    )
    resolved = _provisioned_from_document(new_document)
    if resolved is None:  # pragma: no cover - a just-rendered document is always valid
        raise PermissionBootstrapError("permission signing key rotation failed")
    return resolved


def _resolve_identity_ttl(config: Any) -> float:
    raw = getattr(
        config, "permissions_identity_ttl_seconds", _DEFAULT_IDENTITY_TTL_SECONDS
    )
    try:
        ttl = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_IDENTITY_TTL_SECONDS
    return ttl if ttl >= 0 else _DEFAULT_IDENTITY_TTL_SECONDS


def _resolve_identity_refresh_skew(config: Any) -> float:
    raw = getattr(
        config,
        "permissions_identity_refresh_skew_seconds",
        _DEFAULT_IDENTITY_REFRESH_SKEW_SECONDS,
    )
    try:
        skew = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_IDENTITY_REFRESH_SKEW_SECONDS
    return skew if skew >= 0 else _DEFAULT_IDENTITY_REFRESH_SKEW_SECONDS


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
    secrets_client: Any = None,
) -> PermissionContext | None:
    """Return one verified permission context for a runtime execution tree.

    An explicitly injected kernel and identity must be supplied as a pair and must
    verify against each other. Otherwise a required context is bootstrapped from the
    signing authority:

    * If ``AgentConfig.permissions_signing_key_ref`` IS configured, that explicit
      external/rotated key wins and is resolved through the runtime secret resolver
      — unchanged.
    * Otherwise the authority is SELF-PROVISIONED durably
      (:func:`provision_signing_key`): a CSPRNG key stored once, under a well-known
      name, in the engine's durable secret store via an atomic create-if-absent, and
      reused across restarts. This durable *shared* authority is accepted precisely
      because it is not a per-process random one — a per-process ephemeral key is
      never fabricated; if the durable store cannot provide/accept a key, bootstrap
      fails closed with :class:`PermissionBootstrapError`.

    The issued identity carries a bounded ``permissions_identity_ttl_seconds`` TTL
    (transparently re-issued on use at the governed boundary, so a long task never
    dies at TTL) and is least-privilege — the caller's ``role`` (SPECIALIST by
    default, never blanket admin). ``secret_resolver``/``secrets_client`` are
    dependency injection for bounded tests; runtime call sites omit them.
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
    ttl_seconds = _resolve_identity_ttl(config)
    skew_seconds = _resolve_identity_refresh_skew(config)
    try:
        if signing_key_ref:
            # Explicit external/rotated key reference — config override WINS,
            # resolved exactly as before (single active authority).
            resolver = secret_resolver
            if resolver is None:
                from .cli_secrets import resolve_runtime_secret_reference

                resolver = resolve_runtime_secret_reference
            active_material = resolver(signing_key_ref)
            additional_materials: tuple[str, ...] = ()
        else:
            # No explicit reference: durably self-provision one shared authority
            # (idempotent, atomic create-if-absent) rather than failing.
            provisioned = provision_signing_key(config, secrets_client=secrets_client)
            active_material = provisioned.active_material
            additional_materials = provisioned.additional_verification_materials
        kernel = PermissionsKernel(
            signing_key=active_material,
            policies_path=getattr(config, "agent_policies_path", None),
            engine=engine,
            additional_verification_keys=additional_materials,
            identity_ttl_seconds=ttl_seconds,
            identity_refresh_skew_seconds=skew_seconds,
        )
        identity = kernel.issue_identity(
            agent_id=kernel.derive_agent_id(agent_subject),
            role=role,
            capabilities=list(capabilities),
            ttl_seconds=ttl_seconds,
        )
    except PermissionBootstrapError:
        raise
    except Exception as exc:
        raise PermissionBootstrapError("permission context bootstrap failed") from exc
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
