import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
import os

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "../data/ProjectDataset"  # Update path if needed
MODEL_PATH = "../outputs/checkpoints/best_model.pth"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# -----------------------------
# Dataset & Loader
# -----------------------------
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Calculate MSE
# -----------------------------
y_true = []
y_pred_probs = []

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of "Tumor"
        y_true.extend(labels.cpu().numpy())
        y_pred_probs.extend(probs.cpu().numpy())

# Convert to NumPy arrays
y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

# Compute Mean Squared Error
mse = mean_squared_error(y_true, y_pred_probs)
print(f" Mean Squared Error(MSE): {mse:.6f}")
