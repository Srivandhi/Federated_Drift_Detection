import matplotlib.pyplot as plt
import json
from pathlib import Path


class ReviewerVisualizer:
    """
    Generates paper / report ready plots from saved logs
    """

    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.data = self._load_logs()

    def _load_logs(self):
        with open(self.log_file, "r") as f:
            return json.load(f)

    def plot_confidence(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.data["confidence"], label="Confidence")
        plt.title("Confidence over Time")
        plt.xlabel("Inference")
        plt.ylabel("Confidence")
        plt.legend()
        plt.show()

    def plot_drift(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.data["drift_score"], color="orange", label="Drift Score")
        plt.title("Drift Score over Time")
        plt.xlabel("Inference")
        plt.ylabel("Drift Score")
        plt.legend()
        plt.show()

    def plot_reliability(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.data["reliability_score"], color="red", label="Reliability")
        plt.title("Reliability Degradation")
        plt.xlabel("Inference")
        plt.ylabel("Reliability")
        plt.legend()
        plt.show()

    def plot_all(self):
        self.plot_confidence()
        self.plot_drift()
        self.plot_reliability()
