#!/usr/bin/python
from __future__ import annotations

"""The digest-bound inputs of one open-weight policy training job (AU-HARNESS-R002).

Split out of :mod:`agent_utilities.harness.substrate_trainer` so that module
stays under the repository's per-file line cap; :class:`PolicyJobInputs` is a
plain data holder with no dependency on the trainer or the policy-evolution
package, so it lives in its own leaf module and both import it.
"""

from dataclasses import dataclass
from typing import Any

__all__ = ["PolicyJobInputs"]


@dataclass(frozen=True)
class PolicyJobInputs:
    """The digest-bound inputs of one open-weight policy training job (AU-HARNESS-R002).

    Every field names an immutable EG record or digest, so the external trainer
    can verify exactly what it trains on. There is no inline corpus: token
    arrays stay in EG Blob CAS behind the named captures.

    Args:
        capability_id: The attested ``OpenWeightPolicyCapability`` record.
        base_version_id: The ``ModelPolicyVersion`` the adapter is trained on.
        capture_ids: Trainer-eligible ``PolicyCapture`` record ids.
        method: The EG ``TrainingMethod`` wire form (``grpo``/``dpo``/``sft``/
            ``klpo`` with its estimator).
        adapter: The EG ``AdapterSpec`` wire form (LoRA only).
        hyperparameters_digest: SHA-256 hex of the canonical hyperparameters.
        output_destination_ref: Where the new artifact is written; never the
            live one.
    """

    capability_id: str
    base_version_id: str
    capture_ids: tuple[str, ...]
    method: dict[str, Any]
    adapter: dict[str, Any]
    hyperparameters_digest: str
    output_destination_ref: str
