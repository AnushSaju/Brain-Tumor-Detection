import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --------------------------
# CONFIGURATION
# --------------------------
DATA_DIR = "../data/ProjectDataset"  # Path to dataset
MODEL_PATH = "../outputs/checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
SAVE_DIR = "../outputs/evaluation_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------
# TRANSFORMS (same as validation)
# --------------------------
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# --------------------------
# LOAD VALIDATION DATASET
# --------------------------
val_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# LOAD MODEL
# --------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: yes/no
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --------------------------
# EVALUATION
# --------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --------------------------
# METRICS
# --------------------------
acc = accuracy_score(all_labels, all_preds) * 100
print(f"\n✅ Model Accuracy: {acc:.2f}%\n")

print("📋 Classification Report:\n", classification_report(all_labels, all_preds, target_names=val_dataset.classes))

# --------------------------
# CONFUSION MATRIX HEATMAP
# --------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_dataset.classes,
            yticklabels=val_dataset.classes)
plt.title("Confusion Matrix (MRI Tumor Detection)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save the image
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.show()

print(f"\n📊 Confusion matrix heatmap saved to: {os.path.join(SAVE_DIR, 'confusion_matrix.png')}")
