"""R050 run-loop consumer of the engine prompt render receipt."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from agent_utilities import host_ports
from agent_utilities.graph import RunPlan, RunRequest, run
from agent_utilities.graph.render_receipt import (
    CALL_SITE,
    CurrentRevisions,
    RenderReceipt,
    RenderReceiptCheck,
    RenderReceiptConsumer,
    RenderReceiptRefused,
    check_render_receipt,
    rendered_bytes_digest,
)

PROMPT = "You are a careful assistant."
DIGEST = "a" * 64


def _receipt(**over) -> RenderReceipt:
    base = {
        "prompt_ref": "prompt:greeting",
        "prompt_revision": "r7",
        "schema_revision": "s3",
        "variables_digest": DIGEST,
        "bytes_digest": rendered_bytes_digest(PROMPT),
        "scope": "tenant-a",
    }
    return RenderReceipt(**{**base, **over})


def _check(receipt=None, *, scope="tenant-a", revisions=None, consumer=...):
    async def reader(prompt_ref: str) -> CurrentRevisions:
        assert prompt_ref == "prompt:greeting"
        return revisions or CurrentRevisions("r7", "s3")

    kwargs = {} if consumer is ... else {"consumer": consumer}
    return RenderReceiptCheck(
        receipt=receipt or _receipt(),
        expected_scope=scope,
        revisions=reader,
        source_snapshot="snap-1",
        **kwargs,
    )


class Transport:
    """Counts the requests that reach the model transport boundary."""

    def __init__(self) -> None:
        self.requests = 0

    def respond(self, messages, info):
        self.requests += 1
        return ModelResponse(parts=[TextPart("answer")])

    async def stream(self, messages, info):
        self.requests += 1
        yield "answer"


def _plan(transport: Transport, check: RenderReceiptCheck | None) -> RunPlan:
    return RunPlan(
        model=FunctionModel(
            function=transport.respond, stream_function=transport.stream
        ),
        instructions=PROMPT,
        steps=("hello",),
        render_receipt=check,
    )


@pytest.mark.spec("MERISTEM-BOUNDARY-R050.1")
def test_consumer_model() -> None:
    consumer = RenderReceiptConsumer()
    assert RenderReceiptConsumer.from_dict(consumer.as_dict()) == consumer
    assert consumer.call_site == CALL_SITE
    with pytest.raises(ValueError, match="unknown dispatch rule"):
        RenderReceiptConsumer.from_dict(
            {"call_site": CALL_SITE, "dispatch_rule": "fan_out"}
        )


@pytest.mark.spec("MERISTEM-BOUNDARY-R050.2")
async def test_dispatch_behavior() -> None:
    transport = Transport()
    result = await run(
        RunRequest(task="hello", run_id="r"), plan=_plan(transport, _check())
    )
    assert result.status == "completed", result.error
    assert transport.requests == 1


@pytest.mark.spec("MERISTEM-BOUNDARY-R050.3")
async def test_producer_wiring() -> None:
    """The receipt arrives on the plan the host decision port returns."""
    transport = Transport()
    seen: list[str] = []

    async def decide(request):
        seen.append(request.run_id)
        return _plan(transport, _check(_receipt(scope="tenant-b")))

    host_ports.register_run_decision_provider(decide)
    result = await run(RunRequest(task="hello", run_id="w"))
    assert seen == ["w"]
    assert result.status == "failed"
    assert "receipt_wrong_scope" in (result.error or "")
    assert transport.requests == 0


@pytest.mark.spec("MERISTEM-BOUNDARY-R050.4")
async def test_partial_refusal() -> None:
    cases = {
        "receipt_invalid": _check(_receipt(variables_digest="not-a-digest")),
        "receipt_stale": _check(revisions=CurrentRevisions("r8", "s3")),
        "receipt_substituted": _check(_receipt(bytes_digest=DIGEST)),
        "receipt_wrong_scope": _check(scope="tenant-z"),
        "consumer_undefined": _check(consumer=None),
    }
    for code, check in cases.items():
        transport = Transport()
        result = await run(
            RunRequest(task="hello", run_id=code), plan=_plan(transport, check)
        )
        assert result.status == "failed", code
        assert code in (result.error or ""), code
        assert transport.requests == 0, code

    async def broken(_ref: str) -> CurrentRevisions:
        raise OSError("engine down")

    check = RenderReceiptCheck(_receipt(), "tenant-a", broken, "snap-1")
    with pytest.raises(RenderReceiptRefused) as refused:
        await check_render_receipt(check, PROMPT)
    assert refused.value.code == "revisions_unavailable"


@pytest.mark.spec("MERISTEM-BOUNDARY-R050.5")
async def test_location_evidence() -> None:
    evidence = await check_render_receipt(_check(), PROMPT)
    assert evidence == {
        "call_site": CALL_SITE,
        "dispatch_rule": "one_request_per_valid_receipt",
        "source_snapshot": "snap-1",
    }
    events = []
    from agent_utilities.graph import iterate_run

    async for event in iterate_run(
        RunRequest(task="hello", run_id="e"), plan=_plan(Transport(), _check())
    ):
        events.append(event)
    checked = [e for e in events if e["type"] == "render_receipt_checked"]
    assert len(checked) == 1
    assert checked[0]["source_snapshot"] == "snap-1"
    assert checked[0]["call_site"] == CALL_SITE
