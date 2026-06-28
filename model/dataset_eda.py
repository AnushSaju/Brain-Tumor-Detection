# dataset_eda.py
"""
Extended EDA for MRI dataset: saves class distribution, image size distribution,
aspect ratio distribution, and sample grids to a folder for viewing in VS Code.
"""

import os
import random
import matplotlib.pyplot as plt
from torchvision import datasets
from PIL import Image
import numpy as np

DATA_DIR = "../data/ProjectDataset"  # path to your dataset
SAVE_DIR = "../outputs/eda_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load dataset to get classes
ds = datasets.ImageFolder(DATA_DIR)
classes = ds.classes
class_to_idx = ds.class_to_idx
print("Classes:", classes)
print("Class-to-idx:", class_to_idx)

# Count images per class and collect image sizes
counts = {c: 0 for c in classes}
sizes = []
ratios = []

for path, label in ds.samples:
    counts[classes[label]] += 1
    with Image.open(path) as img:
        w, h = img.size
        sizes.append((w, h))
        ratios.append(w / h)

print("Image counts per class:", counts)

# --- Bar chart for counts
plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color=['#1f77b4', '#ff7f0e'])
plt.title("Image count per class")
plt.ylabel("Number of images")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "class_counts.png"))
plt.close()

# --- Histogram of widths & heights
widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(widths, bins=20, color='skyblue', edgecolor='black')
plt.title("Image width distribution")
plt.xlabel("Width (px)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(heights, bins=20, color='salmon', edgecolor='black')
plt.title("Image height distribution")
plt.xlabel("Height (px)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "width_height_distribution.png"))
plt.close()

# --- Histogram of aspect ratios
plt.figure(figsize=(6, 4))
plt.hist(ratios, bins=30, color='purple', edgecolor='black')
plt.title("Aspect ratio (width/height) distribution")
plt.xlabel("Aspect ratio")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "aspect_ratio_distribution.png"))
plt.close()

# --- Grid of random images from each class
for c in classes:
    folder = os.path.join(DATA_DIR, c)
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))]
    sample_paths = random.sample(imgs, min(6, len(imgs)))  # pick up to 6
    cols = 3
    rows = int(np.ceil(len(sample_paths)/cols))
    plt.figure(figsize=(9, 3*rows))
    for i, img_path in enumerate(sample_paths):
        img = Image.open(img_path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(c)
        plt.axis('off')
    plt.suptitle(f"Examples from class '{c}'")
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"examples_{c}.png")
    plt.savefig(out_path)
    plt.close()

print(f"Plots saved to: {os.path.abspath(SAVE_DIR)}")
