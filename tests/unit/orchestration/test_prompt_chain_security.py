from __future__ import annotations

import pytest

from agent_utilities.patterns.prompt_chain import _evaluate_branch_condition


def test_branch_condition_supports_declarative_comparisons() -> None:
    assert _evaluate_branch_condition("output == 'approved'", "approved") is True
    assert _evaluate_branch_condition(
        "output.startswith('ok') and len(output) < 20", "ok: ready"
    ) is True


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "output.__class__.__mro__[1].__subclasses__()",
        "(lambda: 1)()",
        "[item for item in output]",
    ],
)
def test_branch_condition_rejects_python_execution_gadgets(expression: str) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        _evaluate_branch_condition(expression, "untrusted")
