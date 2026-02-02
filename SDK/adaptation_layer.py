import numpy as np


class LocalAdaptationLayer:
    """
    LAYER 4: Local Adaptation
    Simulated fine-tuning based on reliability
    """

    def adapt(self, reliability):
        if reliability < 0.3:
            urgency = "CRITICAL"
            epochs = 10
            lr = 5e-4
        elif reliability < 0.5:
            urgency = "HIGH"
            epochs = 7
            lr = 3e-4
        else:
            urgency = "MODERATE"
            epochs = 5
            lr = 1e-4

        improvement = min(0.2, (0.7 - reliability) * 0.7)
        new_reliability = reliability + improvement
        new_drift = 1 - new_reliability

        return {
            "urgency": urgency,
            "epochs": epochs,
            "learning_rate": lr,
            "new_reliability": new_reliability,
            "new_drift": new_drift
        }
