"""Typed cross-layer contracts of the agent stack.

* :mod:`agent_utilities.layers.clients` -- typed L0-L3 and L5 clients over
  epistemic-graph's generated contract.
* :mod:`agent_utilities.layers.harness_port` -- the L4 ``HarnessPort``: one
  typed contract between L3 agent graphs and runtimes, with the ``native``
  (pydantic-ai) and ``claude-code`` (headless CLI) adapters, per-node selection
  (:mod:`~agent_utilities.layers.harness_registry`) and L5 outcome recording
  (:mod:`~agent_utilities.layers.harness_record`).

Capability negotiation and the ``SandboxPort`` remain specified under
``AU-CONTROL-001`` and are not part of this package yet.
"""
