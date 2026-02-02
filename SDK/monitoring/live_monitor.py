import matplotlib.pyplot as plt


class LiveMonitor:
    """
    Live monitoring utility.
    Observes SDK outputs and plots metrics over time.
    """

    def __init__(self):
        self.inference_ids = []
        self.confidences = []
        self.drift_scores = []
        self.reliability_scores = []

        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 8))

    def update(self, inference_id, confidence, decision_output):
        """
        Called after every inference.
        """
        self.inference_ids.append(inference_id)
        self.confidences.append(confidence)
        self.drift_scores.append(decision_output["drift_score"])
        self.reliability_scores.append(decision_output["reliability_score"])

        self._refresh_plot()

    def _refresh_plot(self):
        for axis in self.ax:
            axis.clear()

        self.ax[0].plot(self.inference_ids, self.confidences, label="Confidence")
        self.ax[0].set_ylabel("Confidence")
        self.ax[0].legend()

        self.ax[1].plot(self.inference_ids, self.drift_scores, color="orange", label="Drift Score")
        self.ax[1].set_ylabel("Drift")
        self.ax[1].legend()

        self.ax[2].plot(self.inference_ids, self.reliability_scores, color="red", label="Reliability")
        self.ax[2].set_ylabel("Reliability")
        self.ax[2].set_xlabel("Inference Count")
        self.ax[2].legend()

        plt.tight_layout()
        plt.pause(0.01)
