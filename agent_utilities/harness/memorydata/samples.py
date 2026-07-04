#!/usr/bin/python
from __future__ import annotations

"""Small MemoryData-shaped sample families for sampled bake-off runs (CONCEPT:AU-AHE.harness.when-outcome-names-agent).

The full MemoryData datasets (MemoryAgentBench / LoCoMo / LongBench / MemBench) are *not*
bundled with the upstream repo — they need multi-GB / auth'd downloads (see its README §
"Prepare datasets"). For a kickoff / smoke sweep we instead ship a tiny curated sample per
family that faithfully mirrors each family's *shape* (a context corpus + a recall task whose
gold answer is a literal span of some chunk, so exact-match/containment scoring is
meaningful) and whose ``tag`` matches the router's family priors
(:data:`~agent_utilities.harness.memorydata.router_method.DEFAULT_FAMILY_PRIORS`):

* ``locomo-*``        → context-plane synthesis,
* ``longbench-*``     → latent retrieval,
* ``membench-recall`` → graph rerank,
* ``membench-update`` / ``memoryagentbench-*`` / ``conflict-*`` → bi-temporal as-of.

These are samples, not the published corpora — they prove the harness end-to-end and let the
router compete per family. Swap in the real loaders for a full sweep.
"""

from typing import Any

__all__ = ["sample_families", "FAMILY_TAGS"]

FAMILY_TAGS = (
    "locomo-singlehop",
    "longbench-v2",
    "membench-recall",
    "memoryagentbench-eventqa",
)


def sample_families() -> list[dict[str, Any]]:
    """Return the four sampled families (LoCoMo / LongBench / MemBench / MemoryAgentBench)."""
    return [
        {
            "tag": "locomo-singlehop",
            "name": "LoCoMo (conversational QA, single-hop) — sample",
            "context_chunks": [
                "User: I just adopted a golden retriever puppy named Cooper last weekend.",
                "Assistant: Congratulations! How old is Cooper?",
                "User: He's three months old. I've been taking him to the park near my "
                "apartment in Seattle every morning.",
                "User: My sister Maria flew in from Boston to meet Cooper on Tuesday.",
                "User: Cooper's favorite toy is a squeaky blue elephant.",
            ],
            "tasks": [
                {
                    "task": "qa",
                    "queries": [
                        {
                            "question": "What is the name of the user's puppy?",
                            "answer": "Cooper",
                        },
                        {
                            "question": "What breed is the puppy?",
                            "answer": "golden retriever",
                        },
                        {
                            "question": "Which city does the user live in?",
                            "answer": "Seattle",
                        },
                        {
                            "question": "Who flew in from Boston to meet the puppy?",
                            "answer": "Maria",
                        },
                    ],
                }
            ],
        },
        {
            "tag": "longbench-v2",
            "name": "LongBench (long-context reasoning) — sample",
            "context_chunks": [
                "The Aurelian Compact was a trade agreement signed in 1847 between the "
                "river-states of Veldt and Marran, establishing tariff-free grain exchange.",
                "Under the Compact, the city of Veldt agreed to supply iron ore in return "
                "for Marran's surplus wheat and salt.",
                "A 1853 amendment to the Aurelian Compact added the coastal town of Pell, "
                "which contributed dried fish to the exchange network.",
                "The Compact collapsed in 1861 after a drought devastated Marran's wheat "
                "harvest and Veldt suspended iron shipments.",
            ],
            "tasks": [
                {
                    "task": "mc_reasoning",
                    "queries": [
                        {
                            "question": "In what year was the Aurelian Compact signed?",
                            "answer": "1847",
                        },
                        {
                            "question": "What did Veldt supply under the Compact?",
                            "answer": "iron ore",
                        },
                        {
                            "question": "Which coastal town was added in the 1853 amendment?",
                            "answer": "Pell",
                        },
                        {
                            "question": "What event caused the Compact to collapse?",
                            "answer": "drought",
                        },
                    ],
                }
            ],
        },
        {
            "tag": "membench-recall",
            "name": "MemBench (simple recall) — sample",
            "context_chunks": [
                "The project codename is Nightingale.",
                "The server admin password rotation interval is 90 days.",
                "The primary datacenter is located in Reykjavik.",
                "The on-call engineer this week is Priya Nair.",
                "The build pipeline runs on a cluster named Helios.",
            ],
            "tasks": [
                {
                    "task": "recall",
                    "queries": [
                        {
                            "question": "What is the project codename?",
                            "answer": "Nightingale",
                        },
                        {
                            "question": "Where is the primary datacenter located?",
                            "answer": "Reykjavik",
                        },
                        {
                            "question": "Who is the on-call engineer this week?",
                            "answer": "Priya Nair",
                        },
                        {
                            "question": "What is the build cluster named?",
                            "answer": "Helios",
                        },
                    ],
                }
            ],
        },
        {
            "tag": "memoryagentbench-eventqa",
            "name": "MemoryAgentBench (EventQA / temporal retrieval) — sample",
            "context_chunks": [
                "On January 4th, the team deployed release v2.1 to staging.",
                "On January 11th, a rollback to v2.0 was performed after a memory leak.",
                "On January 18th, release v2.2 shipped to production with the leak fixed.",
                "On February 2nd, the team migrated the cache layer from Redis to KeyDB.",
                "On February 15th, the cache migration was rolled forward to all regions.",
            ],
            "tasks": [
                {
                    "task": "eventqa",
                    "queries": [
                        {
                            "question": "Which release was deployed to staging on January 4th?",
                            "answer": "v2.1",
                        },
                        {
                            "question": "What was performed on January 11th after a memory leak?",
                            "answer": "rollback",
                        },
                        {
                            "question": "What did the team migrate the cache layer to in February?",
                            "answer": "KeyDB",
                        },
                        {
                            "question": "Which release shipped to production with the leak fixed?",
                            "answer": "v2.2",
                        },
                    ],
                }
            ],
        },
    ]
