from .client.inference import ClientCVModel
from .models.resnet_classifier import create_model

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ----------------------------
# Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Create model FIRST
# ----------------------------
model = create_model(
    num_classes=2,
    pretrained=False,
    device=device
)

# ----------------------------
# Load trained weights
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "assets" / "best_model.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ----------------------------
# Wrap with client interface
# ----------------------------
client = ClientCVModel(model, device=device)

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

IMAGE_PATH = BASE_DIR / "weather_drift.png"   # ✅ Path object


if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image)

# ----------------------------
# Inference
# ----------------------------
output = client.predict(image_tensor)

print("\nClient CV Model Output:")
print(output)
