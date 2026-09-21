"""Layer rules and findings for the dependency census."""

from __future__ import annotations

from dataclasses import dataclass

LAYERS = ("contracts", "domain", "ports", "application", "adapters", "composition")
_LAYER_INDEX = {name: index for index, name in enumerate(LAYERS)}
UNASSIGNED = "unassigned"

# AU may consume only EG's published Python projection.  Everything else is an
# ownership reversal: GraphOS is the composition owner and the only layer that
# composes the connector SDK.  AU may inspect the SDK package identity, but it
# never imports SDK transport/runner implementation modules.
_EG_PUBLIC_MODULES = frozenset(
    {
        "epistemic_graph.client",
        "epistemic_graph.client_capabilities",
        "epistemic_graph.generated",
        "epistemic_graph.numeric",
        "epistemic_graph.parser",
        "epistemic_graph.pool",
    }
)

# Effective correction to the contradictory canonical placement. Keeping this
# separate makes the exception visible to reports and tests.
CLASSIFICATION_EXCEPTIONS: tuple[tuple[str, str], ...] = (("core.config", "contracts"),)

LAYER_RULES: tuple[tuple[str, str], ...] = (
    ("models", "contracts"),
    ("protocols", "contracts"),
    ("core._env", "contracts"),
    ("security", "domain"),
    ("governance", "domain"),
    ("policies", "domain"),
    ("protocols.source_connectors", "ports"),
    ("core.execution", "ports"),
    ("knowledge_graph", "application"),
    ("orchestration", "application"),
    ("graph", "application"),
    ("harness", "application"),
    ("capabilities", "application"),
    ("domains", "application"),
    ("gateway", "adapters"),
    ("mcp", "adapters"),
    ("server", "adapters"),
    ("messaging", "adapters"),
    ("cli", "adapters"),
    ("ingestion", "adapters"),
    ("deployment", "adapters"),
    ("runtime", "adapters"),
    ("control_plane", "adapters"),
    ("protocols.a2a", "adapters"),
    ("protocols.a2a_client", "adapters"),
    ("protocols.a2a_server", "adapters"),
    ("protocols.acp_adapter", "adapters"),
    ("protocols.agui_emitter", "adapters"),
    ("__main__", "composition"),
    ("server.app", "composition"),
    ("core.config", "composition"),
)


@dataclass(slots=True)
class Violation:
    """One actionable violating import statement."""

    relpath: str
    lineno: int
    src_group: str
    src_layer: str
    tgt_group: str
    tgt_layer: str
    kind: str
    eager: bool

    @property
    def pair(self) -> str:
        return f"{self.src_group} -> {self.tgt_group}"

    def __str__(self) -> str:
        scope = "eager" if self.eager else "deferred"
        return (
            f"  {self.relpath}:{self.lineno}: {self.kind}: "
            f"{self.src_group} ({self.src_layer}) -> "
            f"{self.tgt_group} ({self.tgt_layer}) [{scope}]"
        )


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    """One forbidden dependency across the four product authorities."""

    relpath: str
    lineno: int
    target: str
    reason: str

    def __str__(self) -> str:
        return f"  {self.relpath}:{self.lineno}: {self.reason}: {self.target}"


class ScanIncomplete(RuntimeError):
    """A source file could not be read or parsed."""


def boundary_reason(target: str) -> str | None:
    """Return why an external import reverses the accepted product DAG.

    ``epistemic_graph`` is consumed through its package exports, generated
    contracts, and published client/compute modules.  Private or server-side EG
    modules are never AU dependencies.  The SDK's source-control composition is
    GraphOS-owned; AU may inspect the package root but importing a submodule would
    couple it to connector transport.  GraphOS is outward composition and has no
    valid AU import shape.
    """
    parts = target.split(".")
    root = parts[0]
    if root == "graph_os":
        return "graphos-composition-import"
    if root == "agent_connector_sdk":
        if len(parts) == 1:
            return None
        return "connector-sdk-implementation-import"
    if root != "epistemic_graph" or len(parts) == 1:
        return None
    if any(part.startswith("_") for part in parts[1:]):
        return "epistemic-graph-private-import"
    public_module = ".".join(parts[:2])
    if public_module not in _EG_PUBLIC_MODULES:
        return "epistemic-graph-internal-import"
    return None


def classify(dotted: str) -> tuple[str, str]:
    """Return the longest-matching ``(group, layer)`` for a dotted module."""
    exception = longest_match(dotted, CLASSIFICATION_EXCEPTIONS)
    if exception:
        return exception
    matched = longest_match(dotted, LAYER_RULES)
    if matched:
        return matched
    return (dotted.split(".")[0] if dotted else "<package-root>"), UNASSIGNED


def longest_match(
    dotted: str, rules: tuple[tuple[str, str], ...]
) -> tuple[str, str] | None:
    """Return the longest dotted-prefix match from one rule set."""
    matches = (
        rule for rule in rules if dotted == rule[0] or dotted.startswith(rule[0] + ".")
    )
    return max(matches, key=lambda rule: len(rule[0]), default=None)


def violation_kind(
    *, src_group: str, src_layer: str, tgt_group: str, tgt_layer: str
) -> str | None:
    """Return the broken direction rule, if both endpoints are assigned."""
    if UNASSIGNED in (src_layer, tgt_layer) or src_group == tgt_group:
        return None
    if _LAYER_INDEX[tgt_layer] > _LAYER_INDEX[src_layer]:
        return "outward-import"
    if src_layer == "adapters" and tgt_layer == "adapters":
        return "adapter-crosstalk"
    return None
