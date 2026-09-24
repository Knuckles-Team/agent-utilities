"""Retrieval learning over EG's decision log (EH-394..EH-399, AU half).

EG learns -- deterministically and with provenance -- from what AU's
retrieval runs returned and cited once an independent evaluator judged them:
durable hard negatives, proven retrieval paths, a bounded query-side adapter,
per-class usage, a governed embedding-generation swap. AU attests outcomes on
the live path (:mod:`.runs`), consumes what EG learned (paths as plan options,
the active generation, admission proposals) and moves EG's governed pointers
only with EG's receipts (:mod:`.adapter`, :mod:`.generation`).
"""

from agent_utilities.decide.learning.ops import (
    outcome_op,
    paths_op,
    q16,
    result_of,
    retrieval_op,
    space_identity,
    usage_op,
)
from agent_utilities.decide.learning.session import LearningSession, current_session

__all__ = [
    "LearningSession",
    "current_session",
    "outcome_op",
    "paths_op",
    "q16",
    "result_of",
    "retrieval_op",
    "space_identity",
    "usage_op",
]
