"""Unit tests for Predict-RLM signatures and GEPA genetic prompt optimization loop.

CONCEPT:ORCH-1.30/31 — RLM GEPA Verification
"""

import pytest
import unittest.mock
from pydantic import BaseModel
from agent_utilities.rlm.predict_rlm import PredictRLM, InputField, OutputField
from agent_utilities.rlm.gepa import (
    Candidate,
    GEPAInstance,
    ParetoCandidatePool,
    ReflectiveMutator,
    GEPAOptimizer,
)
from agent_utilities.rlm.config import RLMConfig


class DummySignature(BaseModel):
    """Summarize some complex project reports."""

    report_text: str = InputField(description="Input raw report text")
    sentiment: str = OutputField(description="Calculated sentiment of the report")
    summary: str = OutputField(description="Concise report summary")


class TestPredictRLM:
    """Tests the structured Predict-RLM Pydantic execution wrapper."""

    def test_signature_parsing(self):
        harness = PredictRLM(DummySignature)
        assert harness.inputs == ["report_text"]
        assert "sentiment" in harness.outputs
        assert "summary" in harness.outputs
        assert len(harness.outputs) == 2

    def test_mount_skill(self):
        harness = PredictRLM(DummySignature)
        dummy_fn = lambda x: x
        harness.mount_skill("test_skill", dummy_fn)
        assert harness.skills["test_skill"] == dummy_fn

    @pytest.mark.asyncio
    async def test_predict_rlm_run(self):
        # We mock RLMEnvironment to return a mock result
        harness = PredictRLM(DummySignature)

        async def mock_run_full_rlm(env_instance, prompt):
            # Simulate what the LLM execution would do: write outputs to vars
            env_instance.vars["sentiment"] = "positive"
            env_instance.vars["summary"] = "A short summary"
            return "Execution complete"

        with unittest.mock.patch(
            "agent_utilities.rlm.repl.RLMEnvironment.run_full_rlm",
            new=mock_run_full_rlm,
        ):
            res = await harness.run(report_text="Project is executing perfectly.")
            assert isinstance(res, DummySignature)
            assert res.sentiment == "positive"
            assert res.summary == "A short summary"
            assert res.report_text == "Project is executing perfectly."


class TestParetoCandidatePool:
    """Tests the multi-objective Pareto dominance sorting and pruning."""

    def test_pareto_dominance(self):
        pool = ParetoCandidatePool(objectives=["accuracy", "efficiency"])

        # Candidate A: high accuracy, low efficiency
        cand_a = Candidate(
            id="A",
            prompt_text="A",
            generation=1,
            scores={"accuracy": 0.9, "efficiency": 0.2},
        )
        # Candidate B: low accuracy, high efficiency
        cand_b = Candidate(
            id="B",
            prompt_text="B",
            generation=1,
            scores={"accuracy": 0.4, "efficiency": 0.8},
        )
        # Candidate C: dominated by A (lower in both)
        cand_c = Candidate(
            id="C",
            prompt_text="C",
            generation=1,
            scores={"accuracy": 0.8, "efficiency": 0.1},
        )

        assert pool.is_dominated(cand_c, cand_a) is True
        assert pool.is_dominated(cand_a, cand_b) is False
        assert pool.is_dominated(cand_b, cand_a) is False

    def test_pool_update_and_frontier(self):
        pool = ParetoCandidatePool(objectives=["accuracy", "efficiency"], max_size=2)

        cand_a = Candidate(
            id="A",
            prompt_text="A",
            generation=1,
            scores={"accuracy": 0.9, "efficiency": 0.2},
        )
        cand_b = Candidate(
            id="B",
            prompt_text="B",
            generation=1,
            scores={"accuracy": 0.4, "efficiency": 0.8},
        )
        cand_c = Candidate(
            id="C",
            prompt_text="C",
            generation=1,
            scores={"accuracy": 0.3, "efficiency": 0.1},  # Dominated by B
        )

        pool.update([cand_a, cand_b, cand_c])

        frontier = pool.get_frontier()
        assert len(frontier) == 2
        assert "A" in [c.id for c in frontier]
        assert "B" in [c.id for c in frontier]
        assert "C" not in [c.id for c in frontier]


class TestReflectiveMutator:
    """Tests the LLM-powered natural language prompt mutation and crossover."""

    @pytest.mark.asyncio
    async def test_mutate(self):
        mutator = ReflectiveMutator()

        class MockResponse:
            output = (
                "```json\n"
                "{\n"
                '  "rationale": "Improved output formatting constraints.",\n'
                '  "mutated_prompt": "Enhanced prompt details."\n'
                "}\n"
                "```"
            )

        async def mock_agent_run(self, prompt, **kwargs):
            return MockResponse()

        with unittest.mock.patch("pydantic_ai.Agent.run", new=mock_agent_run):
            parent = Candidate(
                id="parent_1",
                prompt_text="Base Prompt",
                generation=0,
                scores={"accuracy": 0.5},
            )
            traces = [{"stdout": "Some execution stdout"}]
            feedback = ["Accuracy score too low"]

            child = await mutator.mutate(parent, traces, feedback, generation=1)

            assert child.parent_ids == ["parent_1"]
            assert child.prompt_text == "Enhanced prompt details."
            assert child.rationale == "Improved output formatting constraints."
            assert child.generation == 1

    @pytest.mark.asyncio
    async def test_crossover(self):
        mutator = ReflectiveMutator()

        class MockResponse:
            output = (
                "```json\n"
                "{\n"
                '  "rationale": "Hybrid strategy validation.",\n'
                '  "mutated_prompt": "Merged crossed-over prompt."\n'
                "}\n"
                "```"
            )

        async def mock_agent_run(self, prompt, **kwargs):
            return MockResponse()

        with unittest.mock.patch("pydantic_ai.Agent.run", new=mock_agent_run):
            parent1 = Candidate(id="P1", prompt_text="Prompt 1", generation=1)
            parent2 = Candidate(id="P2", prompt_text="Prompt 2", generation=1)

            child = await mutator.crossover(parent1, parent2, generation=2)

            assert child.parent_ids == ["P1", "P2"]
            assert child.prompt_text == "Merged crossed-over prompt."
            assert child.rationale == "Hybrid strategy validation."


class TestGEPAOptimizer:
    """Tests the full coordinated GEPA prompt optimization loop."""

    @pytest.mark.asyncio
    async def test_optimizer_full_loop(self):
        # Setup dataset
        dataset = [
            GEPAInstance(
                id="inst_1",
                input_data={"report_text": "Good performance overall."},
                reference_output="positive",
            ),
            GEPAInstance(
                id="inst_2",
                input_data={"report_text": "Bad errors in production."},
                reference_output="negative",
            ),
        ]

        # Define an evaluator
        async def mock_evaluator(instance, model_output, execution_trace):
            is_correct = model_output.sentiment == instance.reference_output
            acc = 1.0 if is_correct else 0.0
            return {"accuracy": acc}, f"Evaluated to accuracy {acc}"

        # Initialize Optimizer
        optimizer = GEPAOptimizer(
            signature_class=DummySignature,
            base_prompt="Start.",
            evaluator_fn=mock_evaluator,
            objectives=["accuracy"],
        )

        # Mock the mutant / agent proposed mutations to keep it deterministic and offline
        class MockMutateResponse:
            output = (
                "{\n"
                '  "rationale": "Improved sentiment analysis.",\n'
                '  "mutated_prompt": "Mutated prompt iteration."\n'
                "}"
            )

        async def mock_mutate_run(self_agent, prompt, **kwargs):
            return MockMutateResponse()

        # Mock PredictRLM run so it doesn't call actual LLMs
        async def mock_harness_run(self_harness, **inputs):
            # Return DummySignature with expected reference outputs
            text = str(inputs.get("report_text") or "")
            sentiment = "positive" if "Good" in text else "negative"
            return DummySignature(
                report_text=text,
                sentiment=sentiment,
                summary="Summary of project",
            )

        # Mock create_or_merge_node to ignore graph persistence in test
        async def mock_create_or_merge(node):
            return {"status": "merged"}

        with (
            unittest.mock.patch("pydantic_ai.Agent.run", new=mock_mutate_run),
            unittest.mock.patch(
                "agent_utilities.rlm.predict_rlm.PredictRLM.run", new=mock_harness_run
            ),
            unittest.mock.patch(
                "agent_utilities.rlm.gepa.create_or_merge_node",
                new=mock_create_or_merge,
            ),
        ):
            best = await optimizer.optimize(dataset=dataset, iterations=2, batch_size=2)

            assert best is not None
            assert best.scores["accuracy"] == 1.0
            assert len(optimizer.pool.get_frontier()) > 0

    @pytest.mark.asyncio
    async def test_schema_diversity_sweep(self):
        """Verifies that schema diversity perturbations successfully mutate instance inputs."""
        dataset = [
            GEPAInstance(
                id="inst_1",
                input_data={"report_text": "Good performance overall."},
                reference_output="positive",
            )
        ]

        optimizer = GEPAOptimizer(
            signature_class=DummySignature,
            base_prompt="Start.",
            evaluator_fn=lambda inst, out, trace: ({"accuracy": 1.0}, "Perfect"),
            objectives=["accuracy"],
        )

        class MockMutateResponse:
            output = '{"rationale": "Test", "mutated_prompt": "Mutated"}'

        async def mock_mutate_run(self_agent, prompt, **kwargs):
            return MockMutateResponse()

        captured_inputs = []

        async def mock_harness_run(self_harness, **inputs):
            captured_inputs.append(inputs)
            return DummySignature(
                report_text=str(inputs.get("report_text") or "perturbed"),
                sentiment="positive",
                summary="Summary",
            )

        async def mock_create_or_merge(node):
            return {"status": "merged"}

        with (
            unittest.mock.patch("pydantic_ai.Agent.run", new=mock_mutate_run),
            unittest.mock.patch(
                "agent_utilities.rlm.predict_rlm.PredictRLM.run", new=mock_harness_run
            ),
            unittest.mock.patch(
                "agent_utilities.rlm.gepa.create_or_merge_node",
                new=mock_create_or_merge,
            ),
            unittest.mock.patch("random.random", return_value=1.0),
        ):
            await optimizer.optimize(
                dataset=dataset,
                iterations=1,
                batch_size=1,
                enable_schema_diversity=True,
            )

            assert len(captured_inputs) > 0
            # diverse_batch appends the original then the perturbed one
            perturbed_inputs = [
                inp for inp in captured_inputs if inp != dataset[0].input_data
            ]
            assert len(perturbed_inputs) > 0, "No perturbed inputs were generated"
