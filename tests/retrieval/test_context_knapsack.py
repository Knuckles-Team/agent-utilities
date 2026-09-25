"""EH-399: context sizing as a certified multi-resolution knapsack."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_utilities.knowledge_graph.retrieval import context_knapsack as ck


class _Words:
    identity = "words"

    def count(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in texts]


def _items() -> list[ck.Slice]:
    return [
        ck.Slice("a", "full", "one two three four five six", 0.9),
        ck.Slice("a", "summary", "one two", 0.6),
        ck.Slice("b", "full", "x y z", 0.5),
    ]


def test_the_model_is_one_resolution_per_unit_within_capacity() -> None:
    model = ck.knapsack_model(_items(), [6, 2, 3], 5, 0.0)
    bodies = [c["body"] for c in model["constraints"]]
    assert {"at_most": {"vars": [0, 1], "k": 1}} in bodies
    capacity = bodies[-1]["linear"]
    assert (capacity["relation"], capacity["rhs"]) == ("less_equal", 5)
    first = model["objective"][0]["terms"][0]["coefficient"]
    assert first == {"known": -900000}, (
        "net value is maximised by minimising its negation"
    )


def test_a_certified_solve_is_taken_as_is() -> None:
    requests: list[Mapping[str, Any]] = []

    def solve(request: Mapping[str, Any]) -> Any:
        requests.append(request)
        return {
            "certificate": {
                "status": "optimal",
                "incumbent": {"selected": [False, True, True]},
            },
            "certificate_digest": "sha256:c",
        }

    chosen = ck.select_slices(_items(), _Words(), 5, solve=solve)
    assert chosen.certified and chosen.certificate_digest == "sha256:c"
    assert [(s.group, s.resolution) for s in chosen.chosen] == [
        ("a", "summary"),
        ("b", "full"),
    ]
    assert chosen.tokens == 5 and requests[0]["config"] is None


def test_an_uncertified_or_failed_solve_falls_back_to_the_greedy_fit() -> None:
    gap = {"certificate": {"status": {"feasible_with_gap": {"gap": "1"}}}}
    chosen = ck.select_slices(_items(), _Words(), 5, solve=lambda r: gap)
    assert not chosen.certified and chosen.reason == "not_certified"
    assert sum(1 for s in chosen.chosen if s.group == "a") <= 1
    assert chosen.tokens <= 5

    def boom(request: Mapping[str, Any]) -> Any:
        raise ConnectionError("engine down")

    assert ck.select_slices(_items(), _Words(), 5, solve=boom).reason == "solve_failed"


def test_the_marginal_floor_leaves_unprofitable_items_out() -> None:
    chosen = ck.select_slices(_items(), _Words(), 100, floor_per_token=0.2)
    assert [(s.group, s.resolution) for s in chosen.chosen] == [("a", "summary")]


def test_capacity_is_the_tightest_limit() -> None:
    cap = ck.Capacity(
        window=8000, reserved_output=1000, budget_usd=0.01, usd_per_token=1e-5
    )
    assert cap.tokens() == 1000
    assert cap.tokens(caller_budget=500) == 500
    definition = type(
        "Def",
        (),
        {
            "context_window": 4096,
            "max_output_tokens": 96,
            "cost": type("Cost", (), {"input": 2.0})(),
        },
    )()
    assert ck.capacity_of(definition).tokens() == 4000


def test_the_compiler_fit_uses_the_scoped_sizer_and_keys_bundles_by_it() -> None:
    records = [
        {
            "nid": "a",
            "composite": 0.9,
            "node": {"content": "w " * 50, "summary": "short a"},
        },
        {"nid": "b", "composite": 0.4, "node": {"content": "b body"}},
    ]

    def text_of(record: Mapping[str, Any]) -> str:
        return " ".join(str(v) for v in record["node"].values())

    assert ck.sizing_key("m") == "m"
    sizer = ck.ContextSizer(_Words(), ck.Capacity(window=10))
    with ck.sizing_scope(sizer):
        result = ck.fit_to_budget(records, 10, text_of=text_of)
        assert ck.sizing_key("m") == f"m|{sizer.identity()}"
    assert ck.current_sizer() is None, "the scope ends with the call"
    kept = {r["nid"]: r for r in result.kept}
    assert kept["a"]["node"] == {"summary": "short a"}, "a fits only as its summary"
    assert set(kept) == {"a", "b"} and result.tokens_used <= 10


class _Encoding:
    name = "fake-bpe"

    def encode(self, text: str, disallowed_special: Any = ()) -> list[int]:
        return [ord(c) for c in text if c != " "]


def test_a_tokenizer_counter_counts_with_the_model_encoding() -> None:
    counter = ck.TiktokenCounter(_Encoding(), "tiktoken:fake-bpe")
    assert counter.count(["ab c", ""]) == [3, 0]
    assert ck.tiktoken_counter("no-such-model-anywhere") is None
