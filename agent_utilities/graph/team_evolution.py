#!/usr/bin/python
"""Graph-Native Team Evolution (CONCEPT:AU-AHE.harness.graph-native-team-evolution).

Enables autonomous agents to evaluate execution traces from the Knowledge Graph
and propose architectural improvements to agent topologies and capabilities.
"""

import logging
from typing import Any

from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class TeamEvolutionEngine:
    """Evaluates agent team performance and evolves topology natively via the KG.

    CONCEPT:AU-AHE.harness.graph-native-team-evolution — Graph-Native Team Evolution
    """

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def evaluate_and_evolve(self, team_id: str) -> dict[str, Any]:
        """Analyze past performance of a team and suggest or apply topological mutations.

        Args:
            team_id: The identifier of the team composition in the KG.

        Returns:
            A dictionary containing the mutation proposals or actions taken.
        """
        logger.info(f"[AHE-3.18] Evaluating team topology for {team_id}")

        mutations = []

        if self.engine.backend:
            # Look up recent episodes/errors for this team
            query = """
            MATCH (t:Team {id: $team_id})-[:HAS_EPISODE]->(e:Episode)
            WHERE e.status = 'failed' OR e.error IS NOT NULL
            RETURN e.id as episode_id, e.error as error
            ORDER BY e.timestamp DESC LIMIT 10
            """

            try:
                results = self.engine.backend.execute(query, {"team_id": team_id})
                if results:
                    # Logic: if repeated errors occur, propose adding a specialized agent
                    # or adding new tools to an existing agent.
                    mutations.append(
                        {
                            "type": "add_specialist",
                            "reason": f"Frequent failures detected in recent episodes for {team_id}.",
                            "proposed_agent": "error_recovery_specialist",
                        }
                    )
                    logger.debug(f"[AHE-3.18] Proposed mutation: {mutations[-1]}")

                    # Store the mutation proposal back into the graph. A single
                    # "MATCH ... MERGE ... SET ... MERGE" statement exceeds the
                    # engine's native Cypher write subset (MERGE supports only a
                    # single bare node, never an edge pattern;
                    # epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184)
                    # -- split into a typed node upsert + ``link_nodes`` (typed
                    # dispatch for a native authority, portable multi-clause
                    # Cypher -- tolerant of the team having been retired
                    # between the read above and this write, matching the
                    # prior MATCH's silent no-op -- for a non-native store).
                    mut_id = f"mut:{team_id}:{len(mutations)}"
                    self.engine.add_node(
                        mut_id,
                        "MutationProposal",
                        {
                            "reason": mutations[-1]["reason"],
                            # "type" is a reserved node-property name on the
                            # typed engine API; persisted as mutation_type.
                            "mutation_type": mutations[-1]["type"],
                            "status": "proposed",
                        },
                    )
                    self.engine.link_nodes(team_id, mut_id, "PROPOSED_MUTATION")
            except Exception as e:
                logger.error(f"[AHE-3.18] Failed to evaluate team {team_id}: {e}")

        return {"mutations_proposed": mutations}
