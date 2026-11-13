"""FastAPI inference service for the existing Next.js Osteon application."""
import base64, io, os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms

from model import HybridQuantumDenseNet

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "models" / "hybrid_quantum_densenet121.pth"
DEVICE = torch.device("cpu")
app = FastAPI(title="Osteon Hybrid Quantum X-ray Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

class PredictRequest(BaseModel):
    imageBase64: str

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
])
model: Optional[HybridQuantumDenseNet] = None

@app.on_event("startup")
def load_model():
    global model
    if CHECKPOINT.exists():
        model = HybridQuantumDenseNet(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
        model.eval()

@app.get("/health")
def health():
    return {"status": "ready" if model is not None else "checkpoint_missing", "checkpoint_loaded": model is not None}

def make_gradcam(image_tensor: torch.Tensor, original: Image.Image) -> str:
    layer = model.backbone.features.denseblock4
    captured = {}
    def hook(_module, _inputs, output):
        captured["features"] = output
        output.retain_grad()
    handle = layer.register_forward_hook(hook)
    model.zero_grad()
    logits = model(image_tensor)
    class_idx = int(logits.argmax(1).item())
    logits[0, class_idx].backward()
    fmap = captured["features"].detach()[0]
    grads = captured["features"].grad.detach()[0]
    weights = grads.mean(dim=(1,2), keepdim=True)
    cam = torch.relu((weights * fmap).sum(0))
    cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(cam[None,None], size=(224,224), mode="bilinear", align_corners=False)[0,0].cpu().numpy()
    handle.remove()
    # Build a simple red/yellow heat overlay without requiring OpenCV.
    base = np.asarray(original.resize((224,224)).convert("RGB"), dtype=np.float32)
    heat = np.zeros_like(base)
    heat[...,0] = np.clip(cam * 2.0, 0, 1) * 255
    heat[...,1] = np.clip((cam - .35) * 1.8, 0, 1) * 210
    alpha = (np.clip(cam, 0, 1) * .42)[...,None]
    overlay = np.clip(base * (1-alpha) + heat * alpha, 0, 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(overlay).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

@app.post("/predict")
def predict(payload: PredictRequest, authorization: Optional[str] = Header(default=None)):
    token = os.getenv("QUANTUM_SERVICE_TOKEN")
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    if model is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not trained or installed yet. Run train.py first.")
    try:
        raw = payload.imageBase64.split(",", 1)[-1]
        image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data URL")
    x = transform(image).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.enable_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        heatmap = make_gradcam(x, image)
    return {
        "prediction": "FRACTURED" if idx == 1 else "NORMAL",
        "confidence": float(probs[idx].detach().item()),
        "probabilities": {"NORMAL": float(probs[0].detach().item()), "FRACTURED": float(probs[1].detach().item())},
        "model": "DenseNet121 + 4-qubit PennyLane QNN",
        "gradcam": heatmap,
        "note": "Experimental research output only. Not a diagnosis; consult a qualified radiologist."
    }
