"""AHE (Agentic Harness Engineering) mixin for IntelligenceGraphEngine.

Extracted from engine.py. Contains self-improvement cycle methods:
outcome recording, self-evaluation, experiments, critique, and prompt optimization.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object


import logging
import time
import uuid

from ..models.knowledge_graph import (
    CritiqueNode,
    ExperimentNode,
    OutcomeEvaluationNode,
    ProposedSkillNode,
    SelfEvaluationNode,
    SpawnedAgentNode,
    SystemPromptNode,
)

logger = logging.getLogger(__name__)


class AHEMixin(_Base):
    """AHE self-improvement capabilities for the KG engine."""

    def spawn_specialized_agent(
        self,
        task_description: str,
        tool_ids: list[str],
        parent_task_id: str | None = None,
    ) -> str:
        """Spawn a specialized sub-agent with a curated toolset and composed prompt."""
        agent_id = f"spawn:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Intelligent prompt composition: Find relevant base prompts in the graph
        base_prompts = self.query_cypher(
            "MATCH (p:SystemPrompt) WHERE p.tags CONTAINS $tag RETURN p.content as content LIMIT 1",
            {"tag": "agent-base"},
        )
        base_text = (
            base_prompts[0]["content"]
            if base_prompts
            else "You are a specialized agent."
        )

        prompt = f"{base_text}\nTask: {task_description}\nAvailable tools: {', '.join(tool_ids)}"

        node = SpawnedAgentNode(
            id=agent_id,
            name=f"Agent {ts}",
            system_prompt=prompt,
            tool_ids=tool_ids,
            created_at=ts,
            parent_task_id=parent_task_id,
            importance_score=0.9,
        )
        self.graph.add_node(node.id, **node.model_dump())
        if self.backend:
            data = self._serialize_node(node, label="SpawnedAgent")
            self._upsert_node("SpawnedAgent", agent_id, data)
            for tid in tool_ids:
                # Use explicit node match for resources
                self.backend.execute(
                    "MATCH (a:SpawnedAgent {id: $aid}), (t:CallableResource {id: $tid}) MERGE (a)-[:USES]->(t)",
                    {"aid": agent_id, "tid": tid},
                )
        return agent_id

    # --- Self-Improvement Tools (Lightning style) ---

    def record_outcome(
        self,
        episode_id: str,
        reward: float,
        feedback: str,
        success_criteria_met: list[str] | None = None,
    ):
        """Record the outcome and reward for an episode (Lightning step 1)."""
        eval_id = f"eval:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = OutcomeEvaluationNode(
            id=eval_id,
            name=f"Eval {episode_id}",
            reward=reward,
            success_criteria_met=success_criteria_met or [],
            feedback_text=feedback,
            timestamp=ts,
        )
        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="OutcomeEvaluation")
            self._upsert_node("OutcomeEvaluation", eval_id, data)
            # Ladybug requires labels for relationship creation
            label = (
                "Episode"
                if episode_id.startswith("ep:") or episode_id.startswith("run:")
                else "ReasoningTrace"
            )
            self.backend.execute(
                f"MATCH (e:{label}), (o:OutcomeEvaluation) WHERE e.id = $eid AND o.id = $oid MERGE (e)-[:PRODUCED_OUTCOME]->(o)",
                {"eid": episode_id, "oid": eval_id},
            )

        # Link in NetworkX as well
        # Note: we don't know the label in NX nodes reliably without checking 'type' property
        if episode_id in self.graph:
            self.graph.add_edge(episode_id, eval_id, type="PRODUCED_OUTCOME")

        return eval_id

    def record_self_evaluation(
        self, episode_id: str, confidence: float, difficulty: float
    ):
        """Record the agent's internal self-evaluation (confidence calibration)."""
        eval_id = f"self_eval:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = SelfEvaluationNode(
            id=eval_id,
            name=f"Self-Eval {episode_id}",
            confidence_calibration=confidence,
            task_difficulty=difficulty,
            timestamp=ts,
        )
        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="SelfEvaluation")
            self._upsert_node("SelfEvaluation", eval_id, data)
            self.backend.execute(
                "MATCH (e:Episode), (s:SelfEvaluation) WHERE e.id = $eid AND s.id = $sid MERGE (e)-[:SELF_REFLECTS_ON]->(s)",
                {"eid": episode_id, "sid": eval_id},
            )

        if episode_id in self.graph:
            self.graph.add_edge(episode_id, eval_id, type="SELF_REFLECTS_ON")

        return eval_id

    def record_experiment(
        self, name: str, variants: list[str], status: str = "running"
    ):
        """Record a new A/B experiment for prompt or tool variants."""
        exp_id = f"exp:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = ExperimentNode(
            id=exp_id,
            name=name,
            status=status,
            timestamp=ts,
        )
        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())
        self.graph.nodes[node.id]["variants"] = variants

        if self.backend:
            data = self._serialize_node(node, label="Experiment")
            data["variants"] = variants
            self._upsert_node("Experiment", exp_id, data)
        return exp_id

    def generate_critique(self, reasoning_trace_id: str, textual_gradient: str) -> str:
        """Generate a critique (textual gradient) for a reasoning trace (Lightning step 2)."""
        crit_id = f"crit:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        node = CritiqueNode(
            id=crit_id,
            name=f"Critique {ts}",
            textual_gradient=textual_gradient,
            timestamp=ts,
        )
        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="Critique")
            self._upsert_node("Critique", crit_id, data)
            self.backend.execute(
                "MATCH (r:ReasoningTrace), (c:Critique) WHERE r.id = $rid AND c.id = $cid MERGE (r)-[:GENERATED_CRITIQUE]->(c)",
                {"rid": reasoning_trace_id, "cid": crit_id},
            )

        if reasoning_trace_id in self.graph:
            self.graph.add_edge(reasoning_trace_id, crit_id, type="GENERATED_CRITIQUE")

        return crit_id

    def optimize_prompt(self, prompt_id: str, critique_id: str) -> str:
        """Create a new optimized version of a system prompt based on a critique (Lightning step 3)."""
        new_id = f"prompt:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.backend:
            old_prompt = self.query_cypher(
                "MATCH (p:SystemPrompt {id: $id}) RETURN p.content as content",
                {"id": prompt_id},
            )
            critique = self.query_cypher(
                "MATCH (c:Critique {id: $id}) RETURN c.textual_gradient as grad",
                {"id": critique_id},
            )

            content = old_prompt[0]["content"] if old_prompt else "Default prompt"
            grad = critique[0]["grad"] if critique else "Improve clarity"

            new_content = f"{content}\n# Optimized based on: {grad}"

            node = SystemPromptNode(
                id=new_id,
                name=f"Optimized {ts}",
                content=new_content,
                version="v-next",
                source="REFINED",
                timestamp=ts,
            )
            data = self._serialize_node(node, label="SystemPrompt")
            self._upsert_node("SystemPrompt", new_id, data)
            self.backend.execute(
                "MATCH (old:SystemPrompt), (new:SystemPrompt) WHERE old.id = $oid AND new.id = $nid MERGE (new)-[:EVOLVED_FROM]->(old)",
                {"oid": prompt_id, "nid": new_id},
            )
            self.backend.execute(
                "MATCH (c:Critique), (p:SystemPrompt) WHERE c.id = $cid AND p.id = $pid MERGE (c)-[:LED_TO]->(p)",
                {"cid": critique_id, "pid": new_id},
            )
        return new_id

    def run_self_improvement_cycle(self):
        """Autonomous loop for background optimization (Lightning trainer)."""
        # 1. Pull recent failures (low reward)
        failures = self.query_cypher(
            "MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) WHERE o.reward < 0.5 RETURN e.id as id, e.description as description LIMIT 5"
        )
        logger.info(f"Self-improvement cycle: found {len(failures)} failures.")
        for fail in failures:
            logger.info(f"Processing failure: {fail['id']}")
            # 2. Generate pseudo-critique if missing
            crit_id = self.generate_critique(
                fail["id"],
                f"Improve the following based on failure: {fail.get('description', '')}",
            )

            # 3. Optimize linked prompt
            # Step-by-step traversal for robustness
            agent_res = self.query_cypher(
                "MATCH (e {id: $eid})-[:EXECUTED_BY]->(a) RETURN a.id as id LIMIT 1",
                {"eid": fail["id"]},
            )
            prompt = None
            if agent_res:
                aid = agent_res[0]["id"]
                prompt_res = self.query_cypher(
                    "MATCH (a {id: $aid})-[:USES]->(p:SystemPrompt) RETURN p.id as id LIMIT 1",
                    {"aid": aid},
                )
                if prompt_res:
                    prompt = prompt_res

            if prompt:
                logger.info(
                    f"Optimizing prompt {prompt[0]['id']} for failure {fail['id']}"
                )
                self.optimize_prompt(prompt[0]["id"], crit_id)
            else:
                logger.warning(
                    f"No prompt linked to failure {fail['id']} via Episode->Agent->Prompt path."
                )

        # 4. Propose new skills
        new_skill_id = self.propose_new_skill_from_experience()
        if new_skill_id:
            logger.info(f"Proposed new skill: {new_skill_id}")

        logger.info("Self-improvement cycle completed.")

    def propose_new_skill_from_experience(self) -> str | None:
        """Analyze successful trajectories and propose a new skill node."""
        # Removed if not self.backend return to allow NX fallback

        # Strategy A: Frequent Tool Sequences
        # Fetch successful episodes and their tool calls in order
        query = """
        MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation)
        WHERE o.reward >= 0.8
        MATCH (e)-[:USED_TOOL]->(t:ToolCall)
        RETURN e.id as ep_id, t.tool_name as tool, t.timestamp as ts
        ORDER BY ep_id, ts
        """
        results = self.query_cypher(query)

        episodes: dict[str, list[str]] = {}
        for row in results:
            ep_id = row["ep_id"]
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(row["tool"])

        # Count sequence frequency
        sequences: dict[tuple[str, ...], int] = {}
        for tools in episodes.values():
            if len(tools) >= 2:
                # Use window of 2-3 tools
                for i in range(len(tools) - 1):
                    seq = tuple(tools[i : i + 2])
                    sequences[seq] = sequences.get(seq, 0) + 1

        # Find most frequent sequence
        if not sequences:
            return None

        best_seq = max(sequences, key=lambda k: sequences[k])
        freq = sequences[best_seq]

        if freq < 3:  # Threshold for "repeated a lot"
            return None

        # Create ProposedSkillNode
        skill_id = f"skill_prop:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        name = f"Sequence: {' -> '.join(best_seq)}"
        node = ProposedSkillNode(
            id=skill_id,
            name=name,
            code_content=f"# Auto-generated skill for repeated sequence: {' -> '.join(best_seq)}\n"
            f"def frequent_sequence_skill(ctx, **kwargs):\n"
            f"    # This skill automates the frequent sequence detected in the KG\n"
            f"    pass",
            frontmatter={
                "name": name.lower().replace(" ", "_").replace(":", ""),
                "description": f"Automated skill for the frequent tool sequence: {', '.join(best_seq)}",
                "tools": list(best_seq),
                "frequency": freq,
            },
            timestamp=ts,
        )

        # Always add to in-memory graph
        self.graph.add_node(node.id, **node.model_dump())
        return node.id
