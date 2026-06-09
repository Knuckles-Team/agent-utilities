#!/usr/bin/python
"""PreferencePair export + DPO-family refinements (CONCEPT:AHE-3.17).

Covers W1.1 (consolidated export) and W3.1–3.3 (RAPPO reliability filter, TI-DPO
token weights, InSPO reflective conditioning), incl. a live-path test through
``FeedbackService.export_preference_pairs``.
"""

import pytest

from agent_utilities.harness.eval_corpus import EvalCorpus
from agent_utilities.harness.preference_pairs import (
    PreferencePair,
    PreferencePairExporter,
    attach_token_weights,
    reliability_filter,
    with_reflection,
)
from agent_utilities.knowledge_graph.adaptation.feedback import FeedbackService
from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)

pytestmark = pytest.mark.concept("AHE-3.17")


# --- normalization / dedup (export_from) -----------------------------------


def test_export_from_normalizes_all_three_sources():
    exp = PreferencePairExporter()
    pairs = exp.export_from(
        eval_cases=[
            {
                "id": "ec1",
                "query": "2+2?",
                "expected_output": "4",
                "metadata": {"rejected": "5"},
            },
            {"id": "ec2", "query": "no-pair", "expected_output": "x", "metadata": {}},
        ],
        preference_nodes=[
            {"id": "pn1", "prompt": "best sort?", "chosen": "quicksort", "rejected": "bogosort"}
        ],
        corrections=[
            {"id": "co1", "target": "capital?", "corrected_value": "Paris", "original": "Berlin"}
        ],
    )
    by_source = {p.source for p in pairs}
    assert by_source == {"eval_corpus", "distilled", "correction"}
    assert len(pairs) == 3  # the metadata-less eval case yields no pair


def test_export_dedups_identical_content():
    exp = PreferencePairExporter()
    same = {"prompt": "p", "chosen": "c", "rejected": "r"}
    pairs = exp.export_from(
        preference_nodes=[{**same, "id": "a"}, {**same, "id": "b"}]
    )
    assert len(pairs) == 1  # identical (prompt, chosen, rejected) collapse


# --- W3.1 RAPPO reliability filter -----------------------------------------


def test_reliability_filter_drops_ambiguous_and_degenerate():
    pairs = [
        PreferencePair(prompt="p1", chosen="good", rejected="bad", margin=0.9),
        PreferencePair(prompt="p2", chosen="same", rejected="same", margin=1.0),  # degenerate
        PreferencePair(prompt="p3", chosen="a", rejected="b", margin=0.02),  # ambiguous
    ]
    kept, dropped = reliability_filter(pairs, min_margin=0.1)
    assert [p.prompt for p in kept] == ["p1"]
    assert dropped == 2


# --- W3.2 TI-DPO token weights / W3.3 InSPO reflection ----------------------


def test_token_weights_and_reflection_are_attachable():
    p = PreferencePair(prompt="p", chosen="a b c", rejected="x")
    p2 = attach_token_weights(p, [0.1, 0.8, 0.1])
    assert p2.token_weights == [0.1, 0.8, 0.1]
    p3 = with_reflection(p2, alternative="a B c")
    assert p3.alternative == "a B c"
    assert p3.token_weights == [0.1, 0.8, 0.1]  # preserved through the copy


# --- live path: FeedbackService.export_preference_pairs --------------------


def test_feedback_service_exports_pairs_from_live_backend():
    backend = EpistemicGraphBackend()

    # 1) an eval case carrying a rejected output (via the real EvalCorpus path)
    EvalCorpus(backend=backend).add_case(
        "2+2?", "4", metadata={"rejected": "5"}
    )
    # 2) a distilled preference node (as EpisodeToPreferenceRule would write)
    backend.add_node(
        "pref-1", type="preference", prompt="best sort?", chosen="quicksort", rejected="bogosort"
    )
    # 3) a human correction carrying the original (rejected) value
    backend.add_node(
        "corr-1", type="correction", target="capital?", corrected_value="Paris", original="Berlin"
    )
    # 4) a degenerate pair that RAPPO must drop
    backend.add_node(
        "pref-2", type="preference", prompt="noop", chosen="same", rejected="same"
    )

    svc = FeedbackService(backend=backend)
    pairs = svc.export_preference_pairs()

    prompts = {p.prompt for p in pairs}
    assert {"2+2?", "best sort?", "capital?"} <= prompts
    assert "noop" not in prompts  # degenerate pair filtered out by RAPPO
    assert {"eval_corpus", "distilled", "correction"} <= {p.source for p in pairs}
