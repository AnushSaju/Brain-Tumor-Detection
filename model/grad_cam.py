# model/grad_cam.py
"""
Grad-CAM for explaining CNN predictions.

Example usage:
    from grad_cam import GradCAM
    cam = GradCAM(model, target_layer=model.block4)
    heatmap = cam.generate(input_tensor, class_idx=pred_class)
"""

import cv2
import torch
import numpy as np

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The trained CNN model.
            target_layer: The layer whose activations you want to visualize.
                          For CustomCNN, use model.block4.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Generate Grad-CAM heatmap.
        Args:
            input_tensor: input image tensor of shape [1, C, H, W]
            class_idx: optional class index. If None, uses predicted class.
        Returns:
            heatmap: np.ndarray of shape [H, W] (values 0-255)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0,1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(255 * cam)
        return cam

def overlay_heatmap_on_image(heatmap: np.ndarray, original_image: np.ndarray, alpha: float = 0.5):
    """
    Overlay the heatmap on the original image (for visualization).
    Args:
        heatmap: np.ndarray [H,W] 0-255
        original_image: np.ndarray [H,W,C] BGR or RGB
        alpha: blending factor
    Returns:
        overlay: np.ndarray [H,W,C]
    """
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
