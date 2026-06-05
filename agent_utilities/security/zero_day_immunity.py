"""Zero-Day Immunity Middleware.

Combines Topological Analogy Engine (KG-2.7), Vulnerability Scanner (OS-5.11),
and Prompt Injection Scanner (OS-5.4) to auto-detect zero-day jailbreaks via subgraph isomorphism.

Designed to be configurable and disabled by default to save overhead.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ZeroDayImmunity:
    """Detects structural jailbreak patterns without relying on regex."""

    def __init__(self, analogy_engine: Any, enabled: bool = False):
        self.enabled = enabled
        self.analogy_engine = analogy_engine
        # In a real scenario, this loads from the KG:
        self.known_malicious_subgraphs = ["DAN_topology", "roleplay_bypass_topology"]

    def scan_request(self, prompt: str, request_structure: dict[str, Any]) -> bool:
        """Scan a request for structural isomorphism to known attacks."""
        if not self.enabled:
            return True

        logger.debug("Running Zero-Day Immunity structural scan...")

        # 1. Map incoming prompt/request into a structural graph
        incoming_subgraph = self._parse_to_subgraph(prompt, request_structure)

        # 2. Compare against known malicious topological embeddings
        for known_attack in self.known_malicious_subgraphs:
            if hasattr(self.analogy_engine, "find_isomorphism"):
                is_isomorphic = self.analogy_engine.find_isomorphism(
                    incoming_subgraph, known_attack
                )
                if is_isomorphic:
                    logger.error(
                        f"Zero-Day structural match found with {known_attack}!"
                    )
                    return False

        return True

    def _parse_to_subgraph(self, prompt: str, request_structure: dict[str, Any]) -> Any:
        """Convert a text prompt and payload into a structural graph using AST/epistemic-graph dependencies."""
        nodes = []
        edges = []

        # 1. Parse payload structure (request_structure) to nodes/edges
        for k, v in request_structure.items():
            node_id = f"payload:{k}"
            nodes.append(node_id)
            if isinstance(v, str | int | float | bool):
                val_str = str(v)[:30]
                val_node = f"value:{val_str}"
                nodes.append(val_node)
                edges.append((node_id, val_node))

        # 2. Extract code structure from prompt if code is present
        if any(keyword in prompt for keyword in ("class ", "def ", "function ")):
            try:
                from epistemic_graph.parser import RustASTParser

                parser: Any = RustASTParser()
                # Run the local python AST parsing directly as it is fast and synchronous
                parsed = parser.python_ast_parse("prompt.py", prompt)
                for node in parsed.get("nodes", []):
                    nodes.append(node["node_id"])
                for edge in parsed.get("edges", []):
                    edges.append((edge["source"], edge["target"]))
            except Exception as e:
                logger.debug("Failed AST extraction on prompt: %s", e)

        # 3. Simple token/term structural extraction for NL prompts
        else:
            words = [w.strip(".,!?\"'()[]{}").lower() for w in prompt.split()]
            words = [w for w in words if len(w) > 3]
            for idx, word in enumerate(words):
                nodes.append(word)
                if idx > 0:
                    edges.append((words[idx - 1], word))

        # Ensure we always return a valid graph structure
        if not nodes:
            nodes = ["user_input", "override_command"]
            edges = [("user_input", "override_command")]

        return {
            "nodes": list(set(nodes)),
            "edges": list(set(edges)),
        }
