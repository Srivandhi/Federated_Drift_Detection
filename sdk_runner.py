import random
from SDK import DriftDetectionLayer, DecisionLayer

print("\n🔁 Simulating streaming inference with gradual drift\n")

# Initialize layers
decision_layer = DecisionLayer(
    drift_threshold=0.3,
    reliability_threshold=0.7
)

drift_layer = DriftDetectionLayer(
    window_size=5,
    drift_threshold=0.3
)

# Simulated confidence decay
confidence = 0.90

for i in range(1, 21):
    confidence -= random.uniform(0.01, 0.04)
    confidence = max(confidence, 0.45)

    drift_score = drift_layer.ingest_runtime_metrics(confidence)

    decision = decision_layer.decide(
        drift_score=drift_score,
        inference_count=i
    )
    print("RAW decision object:", decision)

    print(
        f"Inference {i:02d} | "
        f"Confidence: {confidence:.3f} | "
        f"Drift: {drift_score:.3f} | "
        f"Decision: {decision['decision'].name}"
    )

