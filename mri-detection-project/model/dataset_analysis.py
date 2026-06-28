import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

# -------------------------
# CONFIGURATION
# -------------------------
DATA_DIR = "../data/ProjectDataset"
OUTPUT_DIR = "../outputs/dataset_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# COUNT IMAGES PER CLASS
# -------------------------
dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes
class_counts = {}

for cls in class_names:
    cls_path = os.path.join(DATA_DIR, cls)
    if os.path.exists(cls_path):
        class_counts[cls] = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    else:
        print(f"⚠️ Folder not found: {cls_path}")

print("\n📊 Image count per class:")
for k, v in class_counts.items():
    print(f"{k}: {v} images")

# -------------------------
# BAR CHART OF CLASS DISTRIBUTION
# -------------------------
plt.figure(figsize=(6, 4))
plt.bar(class_counts.keys(), class_counts.values(), color=['#66b3ff', '#ff9999'])
plt.title("Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.grid(axis='y')
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
plt.show()

# -------------------------
# ANALYZE IMAGE PROPERTIES
# -------------------------
widths, heights, aspect_ratios = [], [], []
pixel_means, pixel_stds = [], []

for cls in class_names:
    cls_folder = os.path.join(DATA_DIR, cls)
    if not os.path.exists(cls_folder):
        continue

    for img_name in os.listdir(cls_folder):
        img_path = os.path.join(cls_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        pixel_means.append(np.mean(img))
        pixel_stds.append(np.std(img))

# -------------------------
# PRINT STATISTICS
# -------------------------
print("\n📏 Image Size Statistics:")
print(f"Average Width: {np.mean(widths):.2f}px")
print(f"Average Height: {np.mean(heights):.2f}px")
print(f"Average Aspect Ratio: {np.mean(aspect_ratios):.2f}")

print("\n🎨 Pixel Intensity Statistics:")
print(f"Mean Pixel Value: {np.mean(pixel_means):.2f}")
print(f"Std Dev of Pixel Value: {np.mean(pixel_stds):.2f}")

# -------------------------
# VISUALIZE IMAGE PROPERTY DISTRIBUTIONS
# -------------------------
plt.figure(figsize=(6, 4))
plt.hist(aspect_ratios, bins=20, color='#99cc99', edgecolor='black')
plt.title("Aspect Ratio Distribution")
plt.xlabel("Width / Height")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, "aspect_ratio_distribution.png"))
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(pixel_means, bins=30, color='#ffcc99', edgecolor='black')
plt.title("Mean Pixel Intensity Distribution")
plt.xlabel("Pixel Mean Value")
plt.ylabel("Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, "pixel_intensity_distribution.png"))
plt.show()

print(f"\n✅ All analysis plots saved to: {OUTPUT_DIR}")
