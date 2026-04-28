import pytest
import asyncio
from agent_utilities.rlm.repl import RLMEnvironment, RecursionLimitError
from agent_utilities.rlm.config import RLMConfig

@pytest.mark.asyncio
async def test_rlm_environment_local_execution():
    config = RLMConfig(enabled=True, use_container=False)
    env = RLMEnvironment(context="Test data", config=config)

    code = """
import json
result = context + " works"
FINAL_VAR('out', result)
print("Debug: ran code")
"""
    vars, stdout = await env.execute(code)

    assert vars['out'] == "Test data works"
    assert "Debug: ran code" in stdout
    assert env.vars['out'] == "Test data works"

@pytest.mark.asyncio
async def test_rlm_environment_async_sub_calls():
    config = RLMConfig(enabled=True, async_enabled=True, use_container=False)
    # We patch run_full_rlm to avoid actual LLM calls
    env = RLMEnvironment(context="Parent", config=config)

    async def mock_run_full_rlm(self_instance, prompt):
        return f"Mocked {prompt} at depth {self_instance.depth}"

    import unittest.mock
    with unittest.mock.patch('agent_utilities.rlm.repl.RLMEnvironment.run_full_rlm', new=mock_run_full_rlm):
        code = """
calls = [
    {"prompt": "A", "context": "Data A"},
    {"prompt": "B", "context": "Data B"}
]
results = await run_parallel_sub_calls(calls)
FINAL_VAR('results', results)
"""
        vars, stdout = await env.execute(code)

        assert "results" in vars
        res = vars["results"]
        assert len(res) == 2
        assert res[0] == "Mocked A at depth 1"
        assert res[1] == "Mocked B at depth 1"

@pytest.mark.asyncio
async def test_rlm_recursion_limit():
    config = RLMConfig(enabled=True, max_depth=1)
    env = RLMEnvironment(context="Parent", config=config, depth=1)

    # Should throw exception if it tries to spawn depth 2 when max is 1
    with pytest.raises(RecursionLimitError):
        await env.rlm_query("Child query")
