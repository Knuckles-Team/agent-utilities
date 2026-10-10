"""Public trace-identity export for AU application integrations.

Thin re-export of ``agent_utilities.observability.trace_ontology.trace_id`` so
consumers stop importing the internal module. The internal module is itself
slated for deletion by AU-SEMANTIC-R019.2; this export moves with its eventual
owner.
"""

from agent_utilities.observability.trace_ontology import trace_id

__all__ = ["trace_id"]
