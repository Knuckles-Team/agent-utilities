"""Prediction Linkage Layer (ORCH-1.7).

CONCEPT: ORCH-1.7 Prediction Linkage Layer

This microservice fuses confidence score matrices from multiple executing subagents,
translating isolated agent predictions into an ensemble modeling pipeline.
"""

from typing import Any


class PredictionLinkageLayer:
    """Fuses predictions across multiple agents into an ensemble forecast."""

    def __init__(self):
        self.prediction_matrix: dict[str, list[dict[str, Any]]] = {}

    def register_prediction(
        self,
        agent_id: str,
        target: str,
        prediction: float,
        confidence: float,
        timestamp: float,
    ) -> None:
        """Register a single prediction from an agent."""
        if target not in self.prediction_matrix:
            self.prediction_matrix[target] = []

        self.prediction_matrix[target].append(
            {
                "agent_id": agent_id,
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": timestamp,
            }
        )

    def fuse_predictions(self, target: str) -> dict[str, float]:
        """Fuse all registered predictions for a target into a single ensemble forecast using confidence-weighted average."""
        predictions = self.prediction_matrix.get(target, [])
        if not predictions:
            return {"ensemble_prediction": 0.0, "overall_confidence": 0.0}

        total_confidence = sum(p["confidence"] for p in predictions)
        if total_confidence == 0:
            avg_pred = sum(p["prediction"] for p in predictions) / len(predictions)
            return {"ensemble_prediction": avg_pred, "overall_confidence": 0.0}

        weighted_pred = (
            sum(p["prediction"] * p["confidence"] for p in predictions)
            / total_confidence
        )
        return {
            "ensemble_prediction": weighted_pred,
            "overall_confidence": total_confidence / len(predictions),
        }
