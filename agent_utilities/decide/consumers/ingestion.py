"""EH-030: ingestion lane routing (fast / medium / slow) through EG ``Decide``.

RF-ADR-010 §11's lanes, as the ingestion engine already names them per
document window: ``structured`` is the fast lane (deterministic schema
mapping, no model call), ``mixed`` the medium lane (map the header, extract
the body), ``prose`` the slow lane (open LLM extraction). The structure
classifier's verdict is the deterministic fallback and a declared fact
(``classifier``); ``llm_passes`` states each lane's model cost.
"""

from __future__ import annotations

from collections.abc import Callable

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

#: Lane -> model extraction passes it costs (the fast lane costs none).
LANES: dict[str, float] = {"structured": 0.0, "mixed": 1.0, "prose": 2.0}


def choose_ingestion_lane(source_type: str, classify: Callable[[], str]) -> str:
    """The lane for one window; ``classify()`` (the structure router) is the fallback."""
    verdict = classify()
    options = [
        Option(
            lane, {"classifier": 1.0 if lane == verdict else 0.0, "llm_passes": cost}
        )
        for lane, cost in LANES.items()
    ]
    choice = decide.choose(
        "au.ingestion.lane",
        options,
        lambda: verdict,
        params=[text_param("source_type", source_type or "unknown")],
    )
    return choice.option_id or verdict


__all__ = ["LANES", "choose_ingestion_lane"]
