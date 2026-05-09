import asyncio
import pytest
from fastapi import HTTPException
from agent_utilities.server.concurrency import AsyncioConcurrencyManager


@pytest.mark.asyncio
async def test_asyncio_concurrency_manager_enqueue():
    cm = AsyncioConcurrencyManager()
    session_id = "test_session_enqueue"

    # Acquire lock for first request
    await cm.acquire(session_id, strategy="enqueue")

    # Try to acquire lock again with enqueue, it should block.
    # We will run it in a background task and see if it completes before release
    acquire_task = asyncio.create_task(cm.acquire(session_id, strategy="enqueue"))

    # Give it some time to block
    await asyncio.sleep(0.1)
    assert not acquire_task.done()

    # Release first lock
    await cm.release(session_id)

    # Now it should acquire
    await asyncio.sleep(0.1)
    assert acquire_task.done()

    # Clean up
    await cm.release(session_id)


@pytest.mark.asyncio
async def test_asyncio_concurrency_manager_reject():
    cm = AsyncioConcurrencyManager()
    session_id = "test_session_reject"

    await cm.acquire(session_id, strategy="reject")

    with pytest.raises(HTTPException) as exc_info:
        await cm.acquire(session_id, strategy="reject")

    assert exc_info.value.status_code == 409
    assert "already active" in exc_info.value.detail

    await cm.release(session_id)


@pytest.mark.asyncio
async def test_asyncio_concurrency_manager_interrupt():
    cm = AsyncioConcurrencyManager()
    session_id = "test_session_interrupt"

    # We need a dummy task that acts as the previous request
    async def dummy_request():
        try:
            await cm.acquire(session_id, strategy="interrupt")
            # Simulate long running request
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        finally:
            await cm.release(session_id)

    task1 = asyncio.create_task(dummy_request())

    # Wait for task1 to acquire lock
    await asyncio.sleep(0.1)

    # Now task2 comes in with interrupt strategy
    await cm.acquire(session_id, strategy="interrupt")

    # task1 should be cancelled
    await asyncio.sleep(0.1)
    assert task1.cancelled() or task1.done()

    await cm.release(session_id)
