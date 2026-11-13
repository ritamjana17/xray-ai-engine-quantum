# Osteon Hybrid Quantum Service

This service adds a DenseNet121 + 4-qubit PennyLane classifier to the existing Next.js app. It runs separately from Vercel because PyTorch and PennyLane are Python dependencies.

## 1. Prepare the data

Download the FracAtlas or Bone Fracture Multi-Region dataset, check its license, and curate a consistent binary classification dataset. Place at least 100 unique images in each folder:

```text
quantum-service/data/
  NORMAL/
    normal_001.jpg
    ...
  FRACTURED/
    fractured_001.jpg
    ...
```

Do not put copies or augmented variants of the same source image into different splits. This prototype randomly samples 100 images per class and creates a stratified 70/15/15 split. For formal evaluation, prefer patient-level splitting and a separate untouched external test set.

## 2. Train

Use Python 3.10 or 3.11. A CPU is supported; a GPU is not required for the 4-qubit simulator, though the run can be slow.

```bash
cd quantum-service
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train.py
```

The checkpoint is written to `models/hybrid_quantum_densenet121.pth`. Do not commit private patient images. Treat the checkpoint as a research artifact.

## 3. Run the API

```bash
export QUANTUM_SERVICE_TOKEN='use-a-long-random-secret'
uvicorn api:app --host 0.0.0.0 --port 8000
```

Health check: `GET /health`. Inference: `POST /predict` with JSON `{ "imageBase64": "data:image/jpeg;base64,..." }`.

## 4. Connect the Next.js app

Set these server-side environment variables in `.env.local` and in Vercel:

```env
QUANTUM_SERVICE_URL=https://YOUR-PYTHON-SERVICE/predict
QUANTUM_SERVICE_TOKEN=the-same-long-random-secret
```

Redeploy Vercel. The existing report page will show a new Hybrid Quantum Analysis panel after the Gemini report loads. If the Python service is not configured, the Gemini workflow continues and the panel explains that quantum analysis is unavailable.

## Important limitations

- This is an educational proof of concept, not a medical device.
- 100 images per class are far too few for clinical use or robust generalization.
- A simulated quantum circuit does not demonstrate quantum advantage.
- The Grad-CAM map is an attention visualization, not a fracture segmentation or proof of a fracture.
- The existing Gemini bounding boxes and this model's Grad-CAM are different outputs and should be labeled separately.
