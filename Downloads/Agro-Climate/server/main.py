from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import pickle
import numpy as np
import os

app = FastAPI(title="Agro-Climate DL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Inline Model Definitions (for demo/standalone use) ---
class MLP(nn.Module):
    def __init__(self, input_size=64*64*3, num_classes=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.fc(x)

class CNN(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# --- State ---
MODELS = {}
CLASSES = {}
DEFAULT_CLASSES = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed",
    "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize",
    "Scentless Mayweed", "Shepherd's Purse", "Small-flowered Cranesbill", "Sugar beet"
]

def load_classification_models():
    """Attempt to load saved weights; fall back to random init for demo."""
    global MODELS, CLASSES
    num_classes = 12

    # Try to load class names
    class_path = os.path.join("models", "classification_classes.pkl")
    if os.path.exists(class_path):
        with open(class_path, "rb") as f:
            CLASSES["classification"] = pickle.load(f)
            num_classes = len(CLASSES["classification"])
    else:
        CLASSES["classification"] = DEFAULT_CLASSES

    # Build models
    mlp = MLP(input_size=64*64*3, num_classes=num_classes)
    cnn = CNN(num_classes=num_classes)

    # Try loading saved weights
    mlp_path = os.path.join("models", "mlp_weights.pth")
    cnn_path = os.path.join("models", "cnn_weights.pth")
    if os.path.exists(mlp_path):
        mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))
    if os.path.exists(cnn_path):
        cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))

    mlp.eval()
    cnn.eval()
    MODELS["mlp"] = mlp
    MODELS["cnn"] = cnn
    print(f"[INFO] Classification models loaded  (classes={num_classes})")

@app.on_event("startup")
def startup():
    load_classification_models()

# ------- Routes -------

@app.get("/")
def read_root():
    return {"message": "Agro-Climate DL System API is running"}

@app.post("/predict/classification")
async def predict_classification(
    file: UploadFile = File(...),
    model_type: str = Form("cnn"),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    img_tensor = transform(image).unsqueeze(0)

    model = MODELS.get(model_type, MODELS.get("cnn"))
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)

    class_name = CLASSES["classification"][idx.item()]
    return {
        "class": class_name,
        "confidence": round(conf.item(), 4),
        "model_used": model_type,
    }

@app.post("/predict/yield")
async def predict_yield(
    county_id: int = Form(1),
    gdd: float = Form(1200),
    ppt: float = Form(850),
):
    # Simulated yield prediction — swap for real LSTM/GRU weights later
    base = 150.0 + (gdd / 3000) * 40 + (ppt / 2000) * 20
    predicted_yield = round(base + np.random.normal(0, 2), 2)
    return {
        "predicted_yield": predicted_yield,
        "unit": "bu/acre",
        "county_id": county_id,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
