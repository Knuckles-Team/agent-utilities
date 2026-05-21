import pytest
from unittest.mock import MagicMock, patch
from agent_utilities.knowledge_graph.memory.memento_compressor import (
    compress_to_memento,
    get_recent_mementos,
    _persist_memento,
)


def test_compress_to_memento_dry_run():
    engine_mock = MagicMock()
    messages = [
        {"role": "user", "content": "Let's calculate the trajectory."},
        {
            "role": "assistant",
            "content": "Executed command: python calc.py. Output: Trajectory=45.2",
        },
    ]

    with patch("pydantic_ai.Agent.run_sync") as mock_run:
        mock_result = MagicMock()
        mock_result.data = "Memento: Trajectory=45.2 calculated via calc.py"
        mock_run.return_value = mock_result

        memento = compress_to_memento(engine_mock, messages, dry_run=True)
        assert memento == "Memento: Trajectory=45.2 calculated via calc.py"
        engine_mock.add_node.assert_not_called()


def test_compress_to_memento_persist():
    engine_mock = MagicMock()
    engine_mock.backend = MagicMock()
    messages = [
        {"role": "user", "content": "Deploy the server"},
    ]

    with patch("pydantic_ai.Agent.run_sync") as mock_run:
        mock_result = MagicMock()
        mock_result.data = "Memento: Server deployed"
        mock_run.return_value = mock_result

        memento = compress_to_memento(
            engine_mock, messages, source="test_agent", dry_run=False
        )
        assert memento == "Memento: Server deployed"
        engine_mock.add_node.assert_called_once()
        args, kwargs = engine_mock.add_node.call_args
        assert args[1] == "Memento"
        assert kwargs["properties"]["content"] == "Memento: Server deployed"
        assert kwargs["properties"]["source"] == "test_agent"
        assert kwargs["properties"]["type"] == "MementoBlock"


def test_get_recent_mementos():
    engine_mock = MagicMock()
    engine_mock.backend = MagicMock()

    engine_mock.backend.execute.return_value = [
        {"content": "Memento 1"},
        {"content": "Memento 2"},
    ]

    mementos = get_recent_mementos(engine_mock, source="test_agent", limit=2)
    assert len(mementos) == 2
    assert mementos[0] == "Memento 1"

    engine_mock.backend.execute.assert_called_once()
    args, kwargs = engine_mock.backend.execute.call_args
    assert "MATCH (m:Memento {source: $source})" in args[0]

    # Params dict is the second positional argument
    params = args[1]
    assert params["source"] == "test_agent"
    assert params["limit"] == 2
