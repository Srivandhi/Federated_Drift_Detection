import numpy as np
from collections import deque


class DriftDetectionLayer:
    def __init__(self, window_size=5, drift_threshold=0.3):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.confidence_window = []
        self.baseline_confidence = 0.9  # assumed clean baseline

    def ingest_runtime_metrics(self, confidence: float) -> float:
        """
        Ingest runtime confidence and compute drift score.

        Args:
            confidence (float): model confidence at runtime

        Returns:
            drift_score (float)
        """
        self.confidence_window.append(confidence)

        if len(self.confidence_window) < self.window_size:
            return 0.0

        recent_mean = sum(self.confidence_window[-self.window_size:]) / self.window_size
        drift_score = max(0.0, min(1.0, self.baseline_confidence - recent_mean))

        return drift_score

