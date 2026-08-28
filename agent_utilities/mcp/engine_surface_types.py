"""Typed request value objects for ``engine_surface_tools``'s internal dispatchers.

Leaf module — imports NOTHING from ``agent_utilities`` (CX wD10-R-ARITY / BUG-CX-004 §9b,
same rationale as ``agent_utilities.mcp.bus_types`` / ``agent_utilities.mcp.ontology_types``:
``agent_utilities.mcp.tools`` is the highest-leverage node in au's import SCC because its
``__init__.py`` eagerly imports all 35 tool-registration modules).

``graph_kv_checkpoint`` and ``graph_viz`` keep their full wire signatures unchanged — those
are the published MCP tool schemas FastMCP derives from the function signature, so they are
the wire contract. These dataclasses replace the *internal* over-cap layer immediately behind
that boundary:

* ``_kv_checkpoint_intelligence`` was 20 positional/keyword params for the 5 "intelligence"
  actions (recommend/checkpoint_now/promote/explain/ram_stats) of ``graph_kv_checkpoint`` —
  the per-action handlers underneath it (``_kv_ram_stats_response``, `_kv_recommend_response`,
  `_kv_explain_response`, `_kv_promote_response`, `_kv_checkpoint_now`) were already <=5 params
  and are untouched.
* ``_render_chart`` was 12 params (viz_client + surface/action/spec/dataset + 6 render options)
  for ``graph_viz``'s two chart-producing actions (export_chart/plot_from_query).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class KvCheckpointIntelligenceRequest:
    """Fields the 5 ``graph_kv_checkpoint`` "intelligence" actions draw from — each
    action reads only the 1-8 fields relevant to it."""

    graph: str = ""
    data_b64: str = ""
    model_identity: str = ""
    quantization: str = ""
    serving_engine: str = ""
    engine_version: str = ""
    prefix_digest: str = ""
    tenant: str = ""
    policy_version: str = ""
    run_id: str = ""
    point: str = ""
    checkpoint_id: str = ""
    requesting_tenant: str = ""
    observation_json: str = "{}"
    evidence_bundle_json: str = "{}"
    context_bundle_json: str = "{}"
    sources_json: str = "[]"
    trigger: str = "agent"
    persist: bool = False


@dataclass(frozen=True, slots=True)
class ChartRenderRequest:
    """A ready-to-render chart: the ViewSpec + dataset plus the 6 render options that
    ``graph_viz``'s ``export_chart`` and ``plot_from_query`` actions both feed
    ``client.viz.render`` with."""

    spec: dict[str, Any]
    dataset: dict[str, Any]
    width_px: int = 900
    height_px: int = 560
    format: str = "png"
    max_primitives: int = 200_000
    max_bytes: int = 50_000_000
    dataset_ref: str = "graph_viz"
