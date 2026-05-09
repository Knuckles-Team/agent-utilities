#!/usr/bin/python
"""Tests for CONCEPT:OS-5.2 — Cognitive Scheduler."""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities.core.cognitive_scheduler import (
    AgentProcess,
    CognitiveScheduler,
    ProcessState,
    SchedulerPriority,
)


@pytest.fixture
def scheduler() -> CognitiveScheduler:
    """Create a scheduler with small limits for testing."""
    return CognitiveScheduler(
        max_concurrent=2,
        default_token_quota=10_000,
        preemption_threshold=0.8,
    )


class TestSchedulerPriority:
    """Test priority ordering."""

    def test_priority_values(self) -> None:
        assert SchedulerPriority.CRITICAL < SchedulerPriority.HIGH
        assert SchedulerPriority.HIGH < SchedulerPriority.NORMAL
        assert SchedulerPriority.NORMAL < SchedulerPriority.LOW

    def test_priority_numeric(self) -> None:
        assert int(SchedulerPriority.CRITICAL) == 0
        assert int(SchedulerPriority.LOW) == 3


class TestAgentProcess:
    """Test AgentProcess model."""

    def test_defaults(self) -> None:
        proc = AgentProcess(agent_id="test-agent")
        assert proc.agent_id == "test-agent"
        assert proc.priority == SchedulerPriority.NORMAL
        assert proc.state == ProcessState.WAITING
        assert proc.token_quota == 100_000
        assert proc.tokens_used == 0
        assert proc.checkpoint_id is None
        assert proc.id.startswith("proc:")

    def test_custom_values(self) -> None:
        proc = AgentProcess(
            agent_id="admin",
            priority=SchedulerPriority.CRITICAL,
            token_quota=500_000,
            task_description="System maintenance",
        )
        assert proc.priority == 0
        assert proc.token_quota == 500_000


class TestCognitiveScheduler:
    """Test the CognitiveScheduler lifecycle."""

    @pytest.mark.asyncio
    async def test_submit_runs_immediately_when_capacity(
        self, scheduler: CognitiveScheduler
    ) -> None:
        proc = await scheduler.submit("agent-a", task="Task A")
        assert proc.state == ProcessState.RUNNING

    @pytest.mark.asyncio
    async def test_submit_queues_when_full(self, scheduler: CognitiveScheduler) -> None:
        await scheduler.submit("agent-a", task="A")
        await scheduler.submit("agent-b", task="B")
        proc_c = await scheduler.submit("agent-c", task="C")
        assert proc_c.state == ProcessState.WAITING
        assert scheduler.get_running_count() == 2
        assert scheduler.get_queue_depth() == 1

    @pytest.mark.asyncio
    async def test_complete_promotes_waiting(
        self, scheduler: CognitiveScheduler
    ) -> None:
        proc_a = await scheduler.submit("agent-a", task="A")
        await scheduler.submit("agent-b", task="B")
        proc_c = await scheduler.submit("agent-c", task="C")
        assert proc_c.state == ProcessState.WAITING

        await scheduler.complete(proc_a.id)
        assert proc_a.state == ProcessState.COMPLETED
        # proc_c should now be running
        assert proc_c.state == ProcessState.RUNNING

    @pytest.mark.asyncio
    async def test_fail_marks_failed(self, scheduler: CognitiveScheduler) -> None:
        proc = await scheduler.submit("agent-a", task="A")
        await scheduler.fail(proc.id, reason="test error")
        assert proc.state == ProcessState.FAILED

    @pytest.mark.asyncio
    async def test_priority_ordering(self, scheduler: CognitiveScheduler) -> None:
        """Higher priority processes should be scheduled first."""
        # Fill capacity
        proc_a = await scheduler.submit(
            "agent-a", priority=SchedulerPriority.NORMAL, task="A"
        )
        await scheduler.submit("agent-b", priority=SchedulerPriority.NORMAL, task="B")

        # Queue two processes with different priorities
        proc_low = await scheduler.submit(
            "agent-low", priority=SchedulerPriority.LOW, task="Low"
        )
        proc_high = await scheduler.submit(
            "agent-high", priority=SchedulerPriority.HIGH, task="High"
        )

        assert proc_low.state == ProcessState.WAITING
        assert proc_high.state == ProcessState.WAITING

        # Complete one running process
        await scheduler.complete(proc_a.id)

        # The HIGH priority process should have been promoted
        assert proc_high.state == ProcessState.RUNNING


class TestTokenQuotas:
    """Test token quota enforcement."""

    @pytest.mark.asyncio
    async def test_within_quota(self, scheduler: CognitiveScheduler) -> None:
        proc = await scheduler.submit("agent-a", task="A")
        assert scheduler.record_tokens(proc.id, 5000) is True

    @pytest.mark.asyncio
    async def test_exceeds_quota(self, scheduler: CognitiveScheduler) -> None:
        proc = await scheduler.submit("agent-a", task="A", token_quota=10_000)
        assert scheduler.record_tokens(proc.id, 10_001) is False

    @pytest.mark.asyncio
    async def test_enforce_quotas_preempts(self, scheduler: CognitiveScheduler) -> None:
        proc = await scheduler.submit("agent-a", task="A", token_quota=10_000)
        scheduler.record_tokens(proc.id, 10_001)
        preempted = await scheduler.enforce_quotas()
        assert proc.id in preempted
        assert proc.state == ProcessState.PAUSED


class TestPreemptionAndResume:
    """Test preemption and context paging."""

    @pytest.mark.asyncio
    async def test_preempt_creates_checkpoint(
        self, scheduler: CognitiveScheduler
    ) -> None:
        proc = await scheduler.submit("agent-a", task="A")
        checkpoint_id = await scheduler.preempt(proc.id, reason="test")
        assert checkpoint_id is not None
        assert checkpoint_id.startswith("ckpt:")
        assert proc.state == ProcessState.PAUSED
        assert proc.checkpoint_id == checkpoint_id
        assert proc.preempted_at is not None

    @pytest.mark.asyncio
    async def test_resume_restores(self, scheduler: CognitiveScheduler) -> None:
        proc = await scheduler.submit("agent-a", task="A")
        await scheduler.preempt(proc.id, reason="test")
        resumed = await scheduler.resume(proc.id)
        assert resumed is True
        assert proc.state == ProcessState.RUNNING

    @pytest.mark.asyncio
    async def test_resume_queues_when_full(self, scheduler: CognitiveScheduler) -> None:
        proc_a = await scheduler.submit("agent-a", task="A")
        await scheduler.submit("agent-b", task="B")
        await scheduler.preempt(proc_a.id)
        # Now agent-b is running, but another fills the slot
        proc_c = await scheduler.submit("agent-c", task="C")
        # Capacity is full (b + c), so resume should queue
        resumed = await scheduler.resume(proc_a.id)
        assert resumed is False
        assert proc_a.state == ProcessState.WAITING


class TestIntrospection:
    """Test scheduler introspection methods."""

    @pytest.mark.asyncio
    async def test_get_process_table(self, scheduler: CognitiveScheduler) -> None:
        await scheduler.submit("agent-a", task="A")
        await scheduler.submit("agent-b", task="B")
        table = scheduler.get_process_table()
        assert len(table) == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, scheduler: CognitiveScheduler) -> None:
        await scheduler.submit("agent-a", task="A")
        stats = scheduler.get_stats()
        assert stats["total_processes"] == 1
        assert stats["max_concurrent"] == 2
        assert stats["capacity_used"] == 1
