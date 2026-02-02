from .drift_detection_layer import DriftDetectionLayer
from .decision_layer import DecisionLayer, DecisionType
from .adaptation_layer import LocalAdaptationLayer


class VisionGuardOrchestrator:
    """
    Coordinates all layers without mixing responsibilities
    """

    def __init__(
        self,
        window_size=5,
        drift_threshold=0.3,
        reliability_threshold=0.7,
        baseline_confidence=0.9
    ):
        self.inference_count = 0

        self.drift_layer = DriftDetectionLayer(
            window_size=window_size,
            drift_threshold=drift_threshold,
            baseline_confidence=baseline_confidence
        )

        self.decision_layer = DecisionLayer(
            drift_threshold=drift_threshold,
            reliability_threshold=reliability_threshold
        )

        self.adaptation_layer = LocalAdaptationLayer()

    def ingest_prediction(self, confidence):
        self.inference_count += 1

        drift_score = self.drift_layer.update(confidence)
        decision = self.decision_layer.decide(
            drift_score,
            self.inference_count
        )

        if decision["decision"] == DecisionType.TRIGGER_ADAPTATION:
            adaptation = self.adaptation_layer.adapt(
                decision["reliability_score"]
            )
            decision["adaptation"] = adaptation

        return decision
