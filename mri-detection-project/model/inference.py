# model/inference.py
"""
Inference script for MRI Tumor Detection with:
- Predictive uncertainty using MC Dropout
- Grad-CAM explainability

Usage:
    python inference.py --image_path path/to/mri.jpg --weights ../outputs/checkpoints/model_epoch10.pth
"""

import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
from custom_cnn import CustomCNN, enable_mc_dropout
from augmentations import get_valid_transforms
from grad_cam import GradCAM, overlay_heatmap_on_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
NUM_CLASSES = 2
ATTENTION = "cbam"
DROPOUT_P = 0.3
MC_SAMPLES = 20  # number of stochastic forward passes for uncertainty

def load_image(image_path: str):
    """Load and transform an image."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = get_valid_transforms(IMG_SIZE)
    transformed = transform(image=image_rgb)
    tensor = transformed["image"].unsqueeze(0)  # [1,C,H,W]
    return tensor, image_bgr

def predict_with_uncertainty(model, image_tensor, mc_samples=20):
    """
    Runs MC Dropout to estimate predictive uncertainty.
    Returns: mean_probs (np.array), std_probs (np.array)
    """
    model.eval()
    enable_mc_dropout(model)  # keep dropout active
    probs = []
    with torch.no_grad():
        for _ in range(mc_samples):
            out = model(image_tensor)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            probs.append(prob)
    probs = np.vstack(probs)
    mean_probs = probs.mean(axis=0)
    std_probs = probs.std(axis=0)
    return mean_probs, std_probs

def main(args):
    # Load model
    model = CustomCNN(num_classes=NUM_CLASSES, attention=ATTENTION, dropout_p=DROPOUT_P)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load image
    image_tensor, original_bgr = load_image(args.image_path)
    image_tensor = image_tensor.to(DEVICE)

    # Predict with uncertainty
    mean_probs, std_probs = predict_with_uncertainty(model, image_tensor, mc_samples=MC_SAMPLES)
    pred_class = np.argmax(mean_probs)
    classes = ["No Tumor", "Tumor"]
    print(f"Predicted: {classes[pred_class]} with prob={mean_probs[pred_class]:.3f} ± {std_probs[pred_class]:.3f}")

    # Grad-CAM
    if args.gradcam:
        cam = GradCAM(model, target_layer=model.block4)
        heatmap = cam.generate(image_tensor, class_idx=pred_class)
        overlay = overlay_heatmap_on_image(heatmap, original_bgr, alpha=0.5)
        out_path = args.out if args.out else "gradcam_result.jpg"
        cv2.imwrite(out_path, overlay)
        print(f"Grad-CAM saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to MRI image")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights .pth")
    parser.add_argument("--gradcam", action="store_true", help="Enable Grad-CAM")
    parser.add_argument("--out", type=str, default="", help="Path to save Grad-CAM overlay")
    args = parser.parse_args()
    main(args)
