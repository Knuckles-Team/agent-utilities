"""CONCEPT:ORCH-1.33 — Live-path test: UnifiedExecutionEngine dispatches to an adapter.

Asserts the *existing* engine entry point (``run(manifest)``) actually invokes the adapter layer when
a runtime is requested (Wire-First), and preserves prior behaviour when it is not.
"""

from __future__ import annotations

import shutil

import pytest

from agent_utilities.core.execution.adapters import AdapterDefinition, AdapterRegistry, StreamFormat
from agent_utilities.core.execution.engine import UnifiedExecutionEngine
from agent_utilities.models.execution_manifest import AgentSpec, ExecutionManifest

pytestmark = pytest.mark.concept(id="ORCH-1.33")


def _manifest(**meta) -> ExecutionManifest:
    return ExecutionManifest(
        agents=[AgentSpec(agent_id="a", task_template="hello-from-manifest")],
        metadata=meta,
    )


async def test_run_without_runtime_preserves_default():
    eng = UnifiedExecutionEngine()
    res = await eng.run(_manifest())
    assert res.success is True
    assert res.synthesis_output == ""  # default path: no adapter output


async def test_run_with_unregistered_runtime_fails_soft():
    eng = UnifiedExecutionEngine()
    res = await eng.run(_manifest(runtime="no-such-adapter"))
    assert res.success is False


async def test_run_dispatches_to_registered_adapter():
    if not shutil.which("echo"):
        pytest.skip("no echo on PATH")
    reg = AdapterRegistry(load_builtins=False)
    reg.register(
        AdapterDefinition(
            id="echo-adapter",
            bin="echo",
            build_args=lambda model, prompt: [prompt],
            stream_format=StreamFormat.PLAIN,
        )
    )
    eng = UnifiedExecutionEngine(registry=reg)
    res = await eng.run(_manifest(runtime="echo-adapter"))
    assert res.success is True
    assert "hello-from-manifest" in res.synthesis_output
    assert res.telemetry.get("runtime_adapter") == "echo-adapter"


async def test_run_via_model_id_adapter_prefix():
    if not shutil.which("echo"):
        pytest.skip("no echo on PATH")
    reg = AdapterRegistry(load_builtins=False)
    reg.register(
        AdapterDefinition(
            id="echo2", bin="echo", build_args=lambda model, prompt: [prompt],
        )
    )
    eng = UnifiedExecutionEngine(registry=reg)
    m = ExecutionManifest(agents=[AgentSpec(agent_id="a", model_id="adapter:echo2", task_template="x")])
    res = await eng.run(m)
    assert res.success is True
    assert res.telemetry.get("runtime_adapter") == "echo2"
