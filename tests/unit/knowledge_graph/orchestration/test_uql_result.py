"""EG's ``kind``-tagged UQL result: the engine row surface reads rows only."""

import pytest

from agent_utilities.knowledge_graph.orchestration.uql_result import uql_rows


def test_rows_result_yields_its_rows() -> None:
    row = {"id": "n1", "score": 1.0, "channels": {}}
    result = {"kind": "rows", "columns": [], "rows": [row], "warnings": []}
    assert uql_rows(result) == [row]


@pytest.mark.parametrize("kind", ["explain", "profile"])
def test_explain_and_profile_are_refused_not_listed_as_keys(kind: str) -> None:
    with pytest.raises(ValueError, match="not rows"):
        uql_rows({"kind": kind, "plan": {}})


def test_a_bare_row_list_is_refused() -> None:
    with pytest.raises(TypeError, match="not a result dict"):
        uql_rows([{"id": "n1"}])
