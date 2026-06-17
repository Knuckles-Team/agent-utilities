"""Unified KG → external-tool write-back (CONCEPT:KG-2.8/2.9).

One fail-closed, dry-run-first, domain-dispatched write-back core replacing the
three divergent modules (capability/leanix/process). The :data:`WritebackResult`,
the shared federation resolver, and the fail-closed gate live in :mod:`.core`;
each target system is a :class:`~.core.WritebackSink` under :mod:`.sinks`,
registered at import.

Public surface:
* ``run_writeback(target, *, backend, engine, dry_run, **ops)`` — the single core
  both the ``graph_writeback`` MCP tool and its REST twin call.
* ``resolve_writeback_fn(...)`` — the EnrichmentPipeline injection point (capability
  write-back), preserved.
"""

from __future__ import annotations

from .approval import ProposalQueue, approve_proposal
from .core import (
    WritebackResult,
    list_sinks,
    resolve_external_id,
    run_writeback,
)
from .findings import collect_risk_findings, push_findings
from .inventory import collect_inventory_creations, push_inventory

# Import sinks so they self-register (plugin pattern).
from .sinks import ansible as _ansible  # noqa: F401
from .sinks import archimate as _archimate  # noqa: F401
from .sinks import capability as _capability  # noqa: F401
from .sinks import ciso_assistant as _ciso_assistant  # noqa: F401
from .sinks import egeria as _egeria  # noqa: F401
from .sinks import emerald as _emerald  # noqa: F401
from .sinks import erpnext as _erpnext  # noqa: F401
from .sinks import home_assistant as _home_assistant  # noqa: F401
from .sinks import identity as _identity  # noqa: F401
from .sinks import issue_tracker as _issue_tracker  # noqa: F401
from .sinks import leanix as _leanix  # noqa: F401
from .sinks import legal as _legal  # noqa: F401
from .sinks import mealie as _mealie  # noqa: F401
from .sinks import nextcloud as _nextcloud  # noqa: F401
from .sinks import ops as _ops  # noqa: F401
from .sinks import process as _process  # noqa: F401
from .sinks import salesforce as _salesforce  # noqa: F401
from .sinks import servicenow as _servicenow  # noqa: F401
from .sinks import twenty as _twenty  # noqa: F401
from .sinks import wger as _wger  # noqa: F401
from .sinks.capability import resolve_writeback_fn
from .spec_link import link_spec, pull_assigned

__all__ = [
    "WritebackResult",
    "list_sinks",
    "resolve_external_id",
    "run_writeback",
    "resolve_writeback_fn",
    "push_inventory",
    "collect_inventory_creations",
    "push_findings",
    "collect_risk_findings",
    "ProposalQueue",
    "approve_proposal",
    "link_spec",
    "pull_assigned",
]
