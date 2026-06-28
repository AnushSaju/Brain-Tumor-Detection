import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --------------------------
# CONFIGURATION
# --------------------------
BASE_DIR = "../data/ProjectDataset"
MODEL_PATH = "../outputs/checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
SAVE_DIR = "../outputs/performance_matrix"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------
# AUTO-DETECT CORRECT DATA PATH
# --------------------------
if not any(os.scandir(BASE_DIR)):
    nested_dir = os.path.join(BASE_DIR, "brain_tumor_dataset")
    if os.path.exists(nested_dir):
        DATA_DIR = nested_dir
    else:
        raise FileNotFoundError("No valid dataset found in ../data/ProjectDataset/")
else:
    DATA_DIR = BASE_DIR

# --------------------------
# TRANSFORM
# --------------------------
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# --------------------------
# LOAD DATA
# --------------------------
val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = val_dataset.classes

# --------------------------
# LOAD MODEL
# --------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------
# PREDICTIONS
# --------------------------
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# --------------------------
# METRICS
# --------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("Model Performance Summary:")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-score : {f1*100:.2f}%\n")

# Save metrics to CSV
metrics_dict = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score (%)": [accuracy*100, precision*100, recall*100, f1*100]
}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(os.path.join(SAVE_DIR, "model_performance.csv"), index=False)

# --------------------------
# CONFUSION MATRIX
# --------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

# --------------------------
# ROC CURVE
# --------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
plt.close()

# --------------------------
# PRECISION-RECALL CURVE
# --------------------------
prec, rec, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(rec, prec)

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, color='green', lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "precision_recall_curve.png"))
plt.close()

# --------------------------
# BAR CHART OF METRICS
# --------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(7, 5))
sns.barplot(x=metrics, y=values, palette='viridis')
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v*100:.2f}%", ha='center')
plt.savefig(os.path.join(SAVE_DIR, "performance_metrics_bar.png"))
plt.close()

print("All performance plots and metrics saved to:", SAVE_DIR)
