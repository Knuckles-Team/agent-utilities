"""Inductive program synthesis with an Occam/MDL prior (CONCEPT:KG-2.69).

Searches compositions of pure primitives for the shortest program that fits the
examples, ranking with the MDL selection prior so the simplest fitting program wins.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.program_synthesis import (
    ProgramCandidate,
    synthesize,
    synthesize_and_validate,
)
from agent_utilities.harness.selection_operators import select_top_k

pytestmark = pytest.mark.concept("KG-2.69")

PRIMS = {
    "inc": lambda x: x + 1,
    "double": lambda x: x * 2,
    "neg": lambda x: -x,
}


class TestMdlSelection:
    def test_mdl_breaks_ties_toward_shorter(self):
        rows = [
            {"id": "long", "score": 1.0, "length": 4},
            {"id": "short", "score": 1.0, "length": 1},
        ]
        best = select_top_k(
            rows,
            1,
            method="mdl",
            score_key="score",
            length_key="length",
            mdl_weight=0.5,
        )
        assert best[0]["id"] == "short"

    def test_mdl_still_prefers_higher_score(self):
        rows = [
            {"id": "short_worse", "score": 0.4, "length": 1},
            {"id": "long_better", "score": 1.0, "length": 4},
        ]
        best = select_top_k(
            rows,
            1,
            method="mdl",
            score_key="score",
            length_key="length",
            mdl_weight=0.5,
        )
        assert best[0]["id"] == "long_better"  # score dominates a modest length penalty


class TestSynthesis:
    def test_finds_minimal_program(self):
        prog = synthesize(PRIMS, [(1, 2), (3, 6), (5, 10)], max_depth=3)
        assert prog is not None
        assert prog.ops == ("double",)  # not (inc,inc,...) or longer equivalents
        assert prog.length == 1

    def test_prefers_shorter_among_equal_fits(self):
        # (2 -> 4) is fit by both ("double",) and ("inc","inc"); Occam picks the shorter.
        prog = synthesize(PRIMS, [(2, 4)], max_depth=3, mdl_weight=0.5)
        assert prog.ops == ("double",)

    def test_identity_for_passthrough(self):
        prog = synthesize(PRIMS, [(7, 7), (9, 9)], max_depth=2)
        assert prog is not None and prog.ops == () and prog.length == 0

    def test_two_step_composition(self):
        # double then inc: 1->2->3 ; 3->6->7
        prog = synthesize(PRIMS, [(1, 3), (3, 7)], max_depth=3)
        assert prog is not None and prog.run(PRIMS, 5) == 11  # 5*2+1

    def test_no_fit_returns_none(self):
        prog = synthesize(PRIMS, [(1, 1000)], max_depth=2, require_exact=True)
        assert prog is None

    def test_render_is_valid_source(self):
        prog = ProgramCandidate(ops=("double", "inc"))
        src = prog.render()
        ns: dict = {**PRIMS}
        exec(src, ns)  # noqa: S102 — test-only: confirms the rendered source runs
        assert ns["program"](5) == 11


class TestValidation:
    def test_synthesize_and_validate_with_stub_sandbox(self):
        class _Sandbox:
            def validate(self, source: str):
                return ("def program" in source, "ok")

        result = synthesize_and_validate(PRIMS, [(1, 2), (4, 8)], sandbox=_Sandbox())
        assert result is not None
        assert result.program.ops == ("double",)
        assert result.validated is True

    def test_no_program_returns_none(self):
        assert synthesize_and_validate(PRIMS, [(1, 999)], max_depth=2) is None
