"""EH-397 in production: the loop stage that swaps embedding generations.

The Loop engine's cycle (``LoopController._cycle_insight_stages``) runs
:func:`run_generation_cycle` when ``KG_LOOP_EMBEDDING_GENERATION`` is on. It
composes :class:`~.generation.GenerationSwap` from the process's real parts:

* **trainer** -- the embedding model the gradient substrate PUBLISHED for this
  deployment (``EMBEDDING_GENERATION_MODEL``; fine-tuning itself runs in the
  substrate, EH-347). No published model: the stage stops at ``train``;
* **re-embedder** -- :class:`CorpusReembedder`: every node of the active
  generation copied into the shadow graph with a vector from the new model,
  embedded in batches through the shared model-capacity gate
  (:func:`~agent_utilities.core.model_concurrency.map_concurrent_sync`);
* **capacity lease** -- the work runs as ``BACKGROUND_INGESTION`` priority, so
  the model-capacity gates admit it only on spare capacity and it yields to
  interactive and orchestration calls (ORCH-1.98/1.99);
* **embedders** -- the retriever's own model for the active space, the
  published model (same provider) for the shadow's.

EG then dual-serves the judged runs and activates only with a passing receipt
measured against the generation active now; a model already active is a
no-op. The session's principal must hold EG's ``admin:decision-head``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agent_utilities.decide.learning.generation import (
    Embedder,
    GenerationOutcome,
    GenerationPlan,
    GenerationSwap,
    resolve_generation,
)
from agent_utilities.decide.learning.ops import space_identity
from agent_utilities.decide.learning.runs import unit_text
from agent_utilities.decide.learning.session import LearningSession, current_session

logger = logging.getLogger(__name__)

#: The loop flag and the published model the stage swaps to.
LOOP_FLAG = "KG_LOOP_EMBEDDING_GENERATION"
MODEL_SETTING = "EMBEDDING_GENERATION_MODEL"
#: Nodes copied per property read / embedding batch.
REEMBED_BATCH = 128
_SPACE_PROBE = "embedding space probe"


def shadow_graph_of(logical: str, model: str) -> str:
    """The shadow generation of ``logical`` embedded by ``model``."""
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    return f"{logical}--gen-{digest}"


def _batches(ids: Sequence[str], size: int) -> list[list[str]]:
    return [list(ids[i : i + size]) for i in range(0, len(ids), size)]


@dataclass
class CorpusReembedder:
    """Copy the active generation's nodes into the shadow graph, each with a
    vector from the new model. Stored vectors are never mutated in place."""

    graph: Any
    embedder_for: Callable[[str], Embedder]
    batch: int = REEMBED_BATCH

    async def __call__(self, active: str, shadow: str, model: str) -> int:
        return await asyncio.to_thread(self.copy, active, shadow, model)

    def copy(self, active: str, shadow: str, model: str) -> int:
        from agent_utilities.core.model_concurrency import map_concurrent_sync

        source, target = self.graph.for_graph(active), self.graph.for_graph(shadow)
        embedder = self.embedder_for(model)
        written = 0
        for ids in _batches(source.node_ids(), self.batch):
            props = source._get_node_properties_batch(ids)
            units = [(i, props.get(i) or {}) for i in ids]
            units = [(i, p) for i, p in units if unit_text(p)]
            vectors = map_concurrent_sync(
                [unit_text(p) for _, p in units],
                embedder.get_text_embedding,
                model=model,
            )
            for (node_id, node), vector in zip(units, vectors, strict=True):
                target.add_node(
                    node_id, {k: v for k, v in node.items() if k != "embedding"}
                )
                target.add_embedding(node_id, list(vector))
                written += 1
        return written


def background_lease() -> Any:
    """The capacity lease the swap's training and re-embed run under."""
    from agent_utilities.core.resource_priority import PriorityClass, priority_scope

    return priority_scope(PriorityClass.BACKGROUND_INGESTION)


def embedders_for(base: Embedder) -> Callable[[str], Embedder]:
    """The retriever's own model for its name, the provider's model otherwise
    (one client per model)."""
    made: dict[str, Embedder] = {str(base.model_name): base}

    def embedder_for(model: str) -> Embedder:
        if model not in made:
            from agent_utilities.core.embedding_utilities import (
                create_embedding_model,
            )

            made[model] = create_embedding_model(model=model)
        return made[model]

    return embedder_for


def _space(embedder: Embedder) -> str:
    width = len(embedder.get_text_embedding(_SPACE_PROBE))
    return space_identity(str(embedder.model_name), width)


def generation_plan(
    logical: str, active: str, base: Embedder, shadow: Embedder
) -> GenerationPlan:
    return GenerationPlan(
        logical=logical,
        active_graph=active,
        active_space=_space(base),
        shadow_graph=shadow_graph_of(logical, str(shadow.model_name)),
        shadow_space=_space(shadow),
    )


def _skipped(stage: str, detail: str) -> dict[str, Any]:
    return {"activated": False, "stage": stage, "detail": detail}


def _outcome(done: GenerationOutcome) -> dict[str, Any]:
    return {
        "activated": done.activated,
        "stage": done.stage,
        "detail": done.detail,
        "receipt": dict(done.receipt or {}),
    }


def swap_generation(
    session: LearningSession, graph: Any, base: Embedder, model: str
) -> dict[str, Any]:
    """One governed swap of ``graph``'s embedding generation to ``model``."""
    logical = str(graph.graph_name)
    active = resolve_generation(logical)
    embedder_for = embedders_for(base)
    plan = generation_plan(logical, active, base, embedder_for(model))
    if plan.shadow_graph == active:
        return _skipped("active", f"{model} is already the active generation")
    swap = GenerationSwap(
        session=session,
        train=lambda _base: model,
        reembed=CorpusReembedder(graph, embedder_for),
        lease=background_lease,
        embedder_for=embedder_for,
    )
    return _outcome(session.drive(swap.run(plan, str(base.model_name))))


#: model name -> its embedder, for the generations the vector arm probes.
_EMBEDDERS: dict[str, Embedder] = {}


def generation_embedder(retriever: Any) -> Any:
    """The embedder of the generation the retriever's vector arm probes: the
    published model's once its shadow generation is the active one, the
    retriever's own model otherwise."""
    from agent_utilities.core.config import setting

    base = retriever.embed_model
    model = str(setting(MODEL_SETTING, "") or "")
    name = getattr(getattr(retriever, "engine", None), "graph", None)
    logical = str(getattr(name, "graph_name", "") or "")
    if not model or not logical or model == getattr(base, "model_name", None):
        return base
    if resolve_generation(logical) != shadow_graph_of(logical, model):
        return base
    if model not in _EMBEDDERS:
        _EMBEDDERS[model] = embedders_for(base)(model)
    return _EMBEDDERS[model]


def run_generation_cycle(engine: Any) -> dict[str, Any]:
    """The loop stage: swap the engine graph's embedding generation to the
    published model, governed by EG's receipt."""
    from agent_utilities.core.config import setting

    model = str(setting(MODEL_SETTING, "") or "")
    session = current_session()
    retriever = getattr(engine, "hybrid_retriever", None)
    base = getattr(retriever, "embed_model", None)
    graph = getattr(engine, "graph", None)
    if not model:
        return _skipped("train", f"no published embedding model ({MODEL_SETTING})")
    if session is None or base is None or graph is None:
        return _skipped("setup", "no decision session, embedder or engine graph")
    return swap_generation(session, graph, base, model)


__all__ = [
    "LOOP_FLAG",
    "MODEL_SETTING",
    "CorpusReembedder",
    "background_lease",
    "embedders_for",
    "generation_embedder",
    "generation_plan",
    "run_generation_cycle",
    "shadow_graph_of",
    "swap_generation",
]
