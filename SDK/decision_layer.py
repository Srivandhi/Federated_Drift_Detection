import numpy as np
from enum import Enum
from datetime import datetime


class DecisionType(Enum):
    NO_ACTION = 0
    ALERT_DRIFT = 1
    TRIGGER_ADAPTATION = 2
    PARTICIPATE_FL = 3


class DecisionLayer:
    """
    LAYER 3: Decision Layer
    Converts drift → reliability → action
    """

    def __init__(
        self,
        drift_threshold=0.3,
        reliability_threshold=0.7
    ):
        self.drift_threshold = drift_threshold
        self.reliability_threshold = reliability_threshold

    def compute_reliability(self, drift_score):
        return float(np.clip(1.0 - drift_score, 0.0, 1.0))

    def decide(self, drift_score, inference_count):
        reliability = self.compute_reliability(drift_score)

        action = DecisionType.NO_ACTION
        message = "Model stable"

        if drift_score > self.drift_threshold:
            action = DecisionType.ALERT_DRIFT
            message = "Drift detected"

        if reliability < self.reliability_threshold:
            action = DecisionType.TRIGGER_ADAPTATION
            message = "Low reliability – adapt model"

        elif reliability >= self.reliability_threshold and drift_score <= self.drift_threshold:
            action = DecisionType.PARTICIPATE_FL
            message = "Model reliable – FL participation"

        return {
            "timestamp": datetime.now().isoformat(),
            "inference": inference_count,
            "drift_score": drift_score,
            "reliability_score": reliability,
            "decision": action,
            "message": message
        }
