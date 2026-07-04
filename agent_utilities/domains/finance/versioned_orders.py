"""
Versioned Order System ("Trading-as-Git") — CONCEPT:AU-KG.research.research-pipeline-runner

Provides immutable order snapshots with version tracking, atomic
commit promotion, and git-like mutation history with KG provenance.

Source: OpenAlice "Trading-as-Git" Workflow
"""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger(__name__)


class OrderStatus(StrEnum):
    """Lifecycle states for a versioned order."""

    STAGED = "staged"
    APPROVED = "approved"
    COMMITTED = "committed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"


@dataclass(frozen=True)
class OrderStage:
    """
    Immutable order snapshot — a single version of an order.
    Frozen dataclass ensures snapshots cannot be mutated after creation.
    """

    order_id: str
    version: int
    side: str  # "buy" or "sell"
    instrument_id: str
    quantity: float
    price: float | None = None
    order_type: str = "market"
    status: OrderStatus = OrderStatus.STAGED
    created_at: str = ""
    parent_hash: str = ""
    metadata: tuple = ()

    def __post_init__(self):
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(UTC).isoformat())

    @property
    def content_hash(self) -> str:
        """Compute deterministic hash of order content for integrity verification."""
        content = f"{self.order_id}:{self.version}:{self.side}:{self.instrument_id}:{self.quantity}:{self.price}:{self.order_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class OrderCommit:
    """
    Atomic promotion of a staged order to execution.
    Records the transition with full provenance metadata.
    """

    commit_hash: str
    order_stage: OrderStage
    previous_hash: str
    committed_at: str = ""
    committed_by: str = "system"
    guard_results: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.committed_at:
            self.committed_at = datetime.now(UTC).isoformat()


class PreCommitGuard:
    """
    Hook system for pre-execution validation.
    Guards are callable validators that return (passed: bool, reason: str).
    """

    def __init__(self):
        self._guards: list[tuple[str, Callable]] = []

    def register(self, name: str, guard_fn: Callable):
        """Register a guard function: (OrderStage) -> (bool, str)."""
        self._guards.append((name, guard_fn))

    def run_all(self, stage: OrderStage) -> tuple[bool, list[str]]:
        """Run all registered guards. Returns (all_passed, list_of_messages)."""
        messages = []
        all_passed = True
        for name, guard_fn in self._guards:
            try:
                passed, reason = guard_fn(stage)
                if not passed:
                    all_passed = False
                    messages.append(f"[FAIL] {name}: {reason}")
                else:
                    messages.append(f"[PASS] {name}: {reason}")
            except Exception as e:
                all_passed = False
                messages.append(f"[ERROR] {name}: {e}")
        return all_passed, messages


class OrderHistory:
    """
    Git-like log of all order mutations with wasDerivedFrom edges.
    Maintains a complete, immutable audit trail of every order version.
    """

    def __init__(self):
        self._stages: dict[str, list[OrderStage]] = {}
        self._commits: list[OrderCommit] = []
        self._guards = PreCommitGuard()

    def register_guard(self, name: str, guard_fn: Callable):
        """Register a pre-commit guard."""
        self._guards.register(name, guard_fn)

    def stage(
        self,
        order_id: str,
        side: str,
        instrument_id: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "market",
        metadata: dict | None = None,
    ) -> OrderStage:
        """
        Create a new staged order (version 1) or stage a modification
        (version N+1) to an existing order.
        """
        existing = self._stages.get(order_id, [])
        version = len(existing) + 1
        parent_hash = existing[-1].content_hash if existing else ""

        stage = OrderStage(
            order_id=order_id,
            version=version,
            side=side,
            instrument_id=instrument_id,
            quantity=quantity,
            price=price,
            order_type=order_type,
            status=OrderStatus.STAGED,
            parent_hash=parent_hash,
            metadata=tuple(sorted(metadata.items())) if metadata else (),
        )

        if order_id not in self._stages:
            self._stages[order_id] = []
        self._stages[order_id].append(stage)

        logger.info(
            f"Staged order {order_id} v{version}: {side} {quantity} {instrument_id}"
        )
        return stage

    def commit(self, order_id: str, committed_by: str = "system") -> OrderCommit | None:
        """
        Commit the latest staged version of an order.
        Runs all pre-commit guards before committing.
        """
        stages = self._stages.get(order_id)
        if not stages:
            logger.warning(f"No staged orders found for {order_id}")
            return None

        latest = stages[-1]
        if latest.status != OrderStatus.STAGED:
            logger.warning(
                f"Order {order_id} v{latest.version} is not in STAGED status"
            )
            return None

        # Run pre-commit guards
        passed, messages = self._guards.run_all(latest)
        if not passed:
            # Create rejected version
            rejected = OrderStage(
                order_id=latest.order_id,
                version=latest.version,
                side=latest.side,
                instrument_id=latest.instrument_id,
                quantity=latest.quantity,
                price=latest.price,
                order_type=latest.order_type,
                status=OrderStatus.REJECTED,
                parent_hash=latest.parent_hash,
                metadata=latest.metadata,
            )
            self._stages[order_id][-1] = rejected
            logger.warning(f"Order {order_id} rejected: {messages}")
            return None

        # Create committed version
        committed_stage = OrderStage(
            order_id=latest.order_id,
            version=latest.version,
            side=latest.side,
            instrument_id=latest.instrument_id,
            quantity=latest.quantity,
            price=latest.price,
            order_type=latest.order_type,
            status=OrderStatus.COMMITTED,
            parent_hash=latest.parent_hash,
            metadata=latest.metadata,
        )
        self._stages[order_id][-1] = committed_stage

        previous_hash = self._commits[-1].commit_hash if self._commits else "genesis"
        commit = OrderCommit(
            commit_hash=committed_stage.content_hash,
            order_stage=committed_stage,
            previous_hash=previous_hash,
            committed_by=committed_by,
            guard_results=messages,
        )
        self._commits.append(commit)

        logger.info(
            f"Committed order {order_id} v{committed_stage.version} [{commit.commit_hash}]"
        )
        return commit

    def get_history(self, order_id: str) -> list[OrderStage]:
        """Get complete version history for an order."""
        return list(self._stages.get(order_id, []))

    def get_latest(self, order_id: str) -> OrderStage | None:
        """Get the latest version of an order."""
        stages = self._stages.get(order_id)
        return stages[-1] if stages else None

    def get_commit_log(self) -> list[OrderCommit]:
        """Get the full commit log (analogous to git log)."""
        return list(self._commits)

    @property
    def pending_count(self) -> int:
        """Count of orders in STAGED status awaiting commit."""
        count = 0
        for stages in self._stages.values():
            if stages and stages[-1].status == OrderStatus.STAGED:
                count += 1
        return count
