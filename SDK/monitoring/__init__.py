class LiveMonitor:
    def __init__(self):
        self.confidence = []
        self.drift = []
        self.reliability = []

    def update(self, confidence, decision_output):
        self.confidence.append(confidence)
        self.drift.append(decision_output["drift_score"])
        self.reliability.append(decision_output["reliability_score"])
