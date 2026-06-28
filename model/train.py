

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt  # <── added for plotting
from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, RandomBrightnessContrast, GaussNoise, GaussianBlur, Normalize
)
from albumentations.pytorch import ToTensorV2

# Debug dataset classes
debug_ds = datasets.ImageFolder("../data/ProjectDataset")
print("DEBUG CLASSES:", debug_ds.classes)
print("DEBUG CLASS-TO-IDX:", debug_ds.class_to_idx)


DATA_DIR = "../data/ProjectDataset"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
OUTPUT_DIR = "../outputs/checkpoints"
PLOT_DIR = "../outputs/training_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img):
        image = np.array(img)
        augmented = self.aug(image=image)
        return augmented["image"]

def get_train_transforms(img_size=224):
    return Compose([
        Resize(img_size, img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        RandomBrightnessContrast(p=0.3),
        GaussNoise(p=0.2),
        GaussianBlur(p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_valid_transforms(img_size=224):
    return Compose([
        Resize(img_size, img_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=AlbumentationsTransform(get_train_transforms(IMG_SIZE))
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = AlbumentationsTransform(get_valid_transforms(IMG_SIZE))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = models.resnet18(weights="IMAGENET1K_V1")
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = train_loss / total
    epoch_acc = correct / total
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_correct / val_total
    print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    # Save best checkpoint
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        best_epoch = epoch + 1
        ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved BEST checkpoint: {ckpt_path}")

    scheduler.step()

print(f"\nBest Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")


epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
plt.plot(epochs_range, val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "accuracy_curve.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "loss_curve.png"))
plt.close()

print(f"Training curves saved to: {os.path.abspath(PLOT_DIR)}")
