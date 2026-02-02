import torch
import torch.nn.functional as F

class ClientCVModel:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, image_tensor):
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)

        probs = F.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        return {
            "prediction": int(pred.item()),
            "confidence": float(confidence.item())
        }
