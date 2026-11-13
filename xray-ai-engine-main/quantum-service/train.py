"""Train the 100-per-class research subset.

Expected folders:
  quantum-service/data/NORMAL/*.(jpg|jpeg|png)
  quantum-service/data/FRACTURED/*.(jpg|jpeg|png)

The script selects exactly 100 original images per class and makes a stratified
70/15/15 train/validation/test split. It saves the best validation checkpoint.
"""
import copy, json, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import HybridQuantumDenseNet

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
SEED, PER_CLASS, EPOCHS, BATCH_SIZE = 42, 100, 5, 4
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")  # PennyLane default.qubit simulation
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

class XrayDataset(Dataset):
    def __init__(self, items, training=False):
        self.items = items
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            *([transforms.RandomRotation(5)] if training else []),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        path, label = self.items[index]
        return self.transform(Image.open(path).convert("RGB")), label

normal = sorted(p for p in (DATA / "NORMAL").rglob("*") if p.suffix.lower() in EXTS)
fractured = sorted(p for p in (DATA / "FRACTURED").rglob("*") if p.suffix.lower() in EXTS)
if len(normal) < PER_CLASS or len(fractured) < PER_CLASS:
    raise SystemExit(f"Need {PER_CLASS} images per class. Found NORMAL={len(normal)}, FRACTURED={len(fractured)}")
rng = random.Random(SEED); rng.shuffle(normal); rng.shuffle(fractured)
items = [(str(p), 0) for p in normal[:PER_CLASS]] + [(str(p), 1) for p in fractured[:PER_CLASS]]
paths = np.array([p for p, _ in items]); labels = np.array([y for _, y in items])
train_p, rest_p, train_y, rest_y = train_test_split(paths, labels, test_size=.30, stratify=labels, random_state=SEED)
val_p, test_p, val_y, test_y = train_test_split(rest_p, rest_y, test_size=.50, stratify=rest_y, random_state=SEED)
make = lambda ps, ys: [(str(p), int(y)) for p, y in zip(ps, ys)]
train_loader = DataLoader(XrayDataset(make(train_p, train_y), True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(XrayDataset(make(val_p, val_y)), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(XrayDataset(make(test_p, test_y)), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = HybridQuantumDenseNet(pretrained=True).to(DEVICE)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
best_loss = float("inf"); best_state = None
for epoch in range(EPOCHS):
    model.train(); model.backbone.features.eval(); total = 0.0
    for x, y in train_loader:
        optimizer.zero_grad(); logits = model(x); loss = loss_fn(logits, y)
        loss.backward(); optimizer.step(); total += loss.item() * len(y)
    model.eval(); val_loss = 0.0; correct = 0; count = 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x); val_loss += loss_fn(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item(); count += len(y)
    val_loss /= count
    print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={total/len(train_loader.dataset):.4f}, val_loss={val_loss:.4f}, val_accuracy={correct/count:.3f}")
    if val_loss < best_loss:
        best_loss = val_loss; best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state); model.eval()
y_true=[]; y_pred=[]
with torch.no_grad():
    for x,y in test_loader:
        y_pred.extend(model(x).argmax(1).tolist()); y_true.extend(y.tolist())
print("Held-out test accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["NORMAL", "FRACTURED"], zero_division=0))
torch.save(model.state_dict(), MODEL_DIR / "hybrid_quantum_densenet121.pth")
(MODEL_DIR / "metadata.json").write_text(json.dumps({
    "classes": ["NORMAL", "FRACTURED"], "images_per_class": PER_CLASS,
    "split": {"train": 140, "validation": 30, "test": 30},
    "image_size": [224,224], "qubits": 4, "quantum_layers": 2,
    "seed": SEED, "warning": "Educational prototype; not clinically validated."
}, indent=2))
print("Saved checkpoint to", MODEL_DIR / "hybrid_quantum_densenet121.pth")
