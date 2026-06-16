"""BrowseComp-Plus: multi-hop QA over a document collection (CONCEPT:AHE-3.32).

Multi-hop retrieval+reasoning across ~1K documents (the paper's BrowseComp-Plus, arXiv:2508.06600,
whose hard-negative/nDCG components already live in ``knowledge_graph/retrieval/``). The synthetic
generator plants a fact chain — person→company→city — across distractor documents and asks a
question that requires hopping all links, so the answer cannot be retrieved by a single lookup.
Prefers a real BrowseComp-Plus export when staged.
"""

from __future__ import annotations

import random

from ..base import LongContextTask, TaskCase, register_task
from ._datasets import load_real_case

_NAMES = ["Mara", "Devon", "Priya", "Tomas", "Lena", "Idris", "Sofia", "Quentin"]
_COMPANIES = ["Helix", "Northwind", "Vertex", "Brightsea", "Cobalt", "Larkspur"]
_CITIES = ["Lisbon", "Tallinn", "Bergen", "Kyoto", "Medellin", "Adelaide"]


class BrowseCompPlusTask(LongContextTask):
    name = "browsecomp_plus"
    complexity = "multi-hop"
    real_dataset = False

    def build(self, scale: int, *, seed: int = 0) -> TaskCase:
        real = load_real_case(self.name, index=seed)
        if real:
            return TaskCase(
                mode="real",
                grader_kind=real.get("grader_kind", "substring"),
                **{k: real[k] for k in ("context", "question", "answer")},
            )

        rng = random.Random(seed)  # nosec B311 — deterministic synthetic benchmark data, not crypto
        person = rng.choice(_NAMES)
        company = rng.choice(_COMPANIES)
        city = rng.choice(_CITIES)
        # The two gold documents that form the hop chain.
        gold = [
            f"DOC employment: {person} is a senior engineer employed at {company}.\n",
            f"DOC headquarters: The company {company} has its headquarters in {city}.\n",
        ]
        docs = list(gold)
        # Distractor documents until we reach ~scale chars.
        idx = 0
        while sum(len(d) for d in docs) < scale:
            dn = rng.choice(_NAMES)
            dc = rng.choice(_COMPANIES)
            dcity = rng.choice(_CITIES)
            docs.append(
                f"DOC note-{idx:05d}: {dn} once consulted for {dc}, now based near {dcity}.\n"
            )
            idx += 1
        rng.shuffle(docs)
        context = "".join(docs)
        return TaskCase(
            context=context,
            question=(
                f"In which city is the headquarters of the company that {person} works at? "
                f"Answer with only the city name."
            ),
            answer=city,
            grader_kind="substring",
            mode="synthetic",
            meta={"person": person, "company": company, "n_docs": len(docs)},
        )


register_task(BrowseCompPlusTask())
