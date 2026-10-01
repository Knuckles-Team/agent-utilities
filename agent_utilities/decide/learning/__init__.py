"""Retrieval learning over EG's decision log (AU-CONTEXT-R001..AU-CONTEXT-R004, AU half).

EG learns -- deterministically and with provenance -- from what AU's
retrieval runs returned and cited once an independent evaluator judged them:
durable hard negatives, proven retrieval paths, a bounded query-side adapter,
per-class usage, a governed embedding-generation swap. AU attests outcomes on
the live path (:mod:`.runs`), consumes what EG learned (paths as plan options,
the active generation, admission proposals) and moves EG's governed pointers
only with EG's receipts (:mod:`.adapter`, :mod:`.generation`).
"""

from agent_utilities.decide.learning.ops import (
    learn_op,
    outcome_op,
    q16,
    recorded,
    rows_of,
    space_identity,
)
from agent_utilities.decide.learning.session import LearningSession, current_session

__all__ = [
    "LearningSession",
    "current_session",
    "learn_op",
    "outcome_op",
    "q16",
    "recorded",
    "rows_of",
    "space_identity",
]
