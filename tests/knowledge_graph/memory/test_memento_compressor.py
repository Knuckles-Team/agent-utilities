from unittest.mock import MagicMock, patch

from agent_utilities.knowledge_graph.memory import (
    compress_to_memento,
    get_recent_mementos,
    memento_source_reference,
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
    # Block must be comfortably longer than the memento: the F4 convergence
    # guarantee (_guarantee_shorter) head-truncates any memento that does not
    # actually shrink its source block.
    messages = [
        {
            "role": "user",
            "content": "Deploy the server to the staging swarm and confirm "
            "the health endpoint responds before cutting traffic over.",
        },
    ]

    with patch("pydantic_ai.Agent.run_sync") as mock_run:
        mock_result = MagicMock()
        mock_result.data = "Memento: Server deployed"
        mock_run.return_value = mock_result

        memento = compress_to_memento(
            engine_mock, messages, source="test_agent", dry_run=False
        )
        assert memento == "Memento: Server deployed"
        # Raw transcript retention is disabled by default. Only the privacy-sanitized Memento and
        # opaque source reference cross the persistence boundary.
        calls = {c.args[1]: c.kwargs for c in engine_mock.add_node.call_args_list}
        assert set(calls) == {"Memento"}
        assert calls["Memento"]["properties"]["content"] == "Memento: Server deployed"
        assert calls["Memento"]["properties"]["source"] == memento_source_reference(
            "test_agent"
        )
        assert calls["Memento"]["properties"]["recoverable"] is False
        assert calls["Memento"]["properties"]["type"] == "MementoBlock"
        engine_mock.link_nodes.assert_not_called()


def test_get_recent_mementos():
    engine_mock = MagicMock()
    engine_mock.backend = MagicMock()

    engine_mock.backend.execute.return_value = [
        {"id": "m1", "content": "Memento 1", "timestamp": "1"},
        {"id": "m2", "content": "Memento 2", "timestamp": "2"},
    ]

    mementos = get_recent_mementos(engine_mock, source="test_agent", limit=2)
    assert len(mementos) == 2
    assert mementos[0] == "Memento 1"

    engine_mock.backend.execute.assert_called_once()
    args, kwargs = engine_mock.backend.execute.call_args
    assert "MATCH (m:Memento {source: $source})" in args[0]

    # Params dict is the second positional argument
    params = args[1]
    assert params["source"] == memento_source_reference("test_agent")
    assert params["limit"] == 2
