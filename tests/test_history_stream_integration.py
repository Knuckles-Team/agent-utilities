"""Local qualification of composed history, checkpoint and stream producers."""

import contextlib

from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from agent_utilities import host_ports
from agent_utilities.graph import RunPlan, RunRequest, run


async def test_prior_history_streamed_turn_and_checkpoint_replay_no_duplicate_effect():
    stored = {}
    events = []
    effects = []
    seen = []

    class Checkpoint:
        admission_identity = "trusted-native-admission"

        def __init__(self, run_id):
            self.run_id = run_id

        async def restore(self):
            return stored.get(self.run_id)

        async def save(self, content):
            stored[self.run_id] = content

    class Publisher:
        async def publish(self, kind, content):
            events.append((kind, content))

    @contextlib.asynccontextmanager
    async def checkpoints(request, plan):
        yield Checkpoint(request.run_id)

    @contextlib.asynccontextmanager
    async def publishers(request):
        yield Publisher()

    async def history(request):
        if request.run_id == "first":
            return None
        return {"run_id": "first", "content": stored["first"]}

    tools = FunctionToolset()

    @tools.tool_plain
    def effect() -> str:
        effects.append("effect")
        return "committed observation"

    async def stream(messages, info):
        seen.append(messages)
        last = messages[-1].parts[-1]
        if getattr(last, "part_kind", "") == "tool-return":
            yield "second completed"
        elif getattr(last, "content", "") == "first prompt":
            yield "remember blue"
        else:
            yield {0: DeltaToolCall(name="effect", json_args="{}", tool_call_id="once")}

    host_ports.register_run_checkpoint_factory(checkpoints)
    host_ports.register_run_history_provider(history)
    host_ports.register_durable_run_event_factory(publishers)
    host_ports.register_native_toolset_provider(
        host_ports.NativeToolsetProvider(
            available_tools=lambda: ["effect"], build_toolset=lambda names, key: tools
        )
    )
    model = FunctionModel(stream_function=stream)
    first = await run(
        RunRequest(task="first", run_id="first"),
        plan=RunPlan(model=model, steps=("first prompt",)),
    )
    assert first.status == "completed"
    request = RunRequest(task="second", run_id="second")
    plan = RunPlan(model=model, steps=("second prompt",), tool_names=("effect",))
    second = await run(request, plan=plan)
    assert second.status == "completed" and effects == ["effect"]
    assert any(
        getattr(part, "content", "") == "remember blue"
        for msg in seen[1]
        for part in msg.parts
    )
    assert second.usage["requests"] == 2
    calls, published = len(seen), len(events)
    replay = await run(request, plan=plan)
    assert replay.status == "completed" and replay.output == second.output
    assert effects == ["effect"] and len(seen) == calls and len(events) == published
    assert [kind for kind, _ in events].count("tool_call") == 1
    assert [kind for kind, _ in events].count("tool_result") == 1
