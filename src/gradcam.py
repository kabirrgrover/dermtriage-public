"""
Grad-CAM for visual explainability on MedSigLIP.

Generates heatmaps showing which image regions drive predictions.
Critical for clinical trust and educational value.

Reference: Selvaraju et al., "Grad-CAM" (2017)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Gradient-weighted Class Activation Mapping for MedSigLIP.

    Args:
        model: MedSigLIPClassifier instance.
        target_layer: Layer to compute Grad-CAM on (default: last encoder layer).
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        if target_layer is None:
            self.target_layer = model.vision_model.encoder.layers[-1]
        else:
            self.target_layer = target_layer

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = (output[0] if isinstance(output, tuple) else output).detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = (grad_output[0] if isinstance(grad_output, tuple) else grad_output).detach()

    def generate(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap.

        Bypasses the model's ``torch.no_grad()`` wrapper by calling
        the vision encoder and classifier head directly so that
        gradients flow through the encoder for the backward hooks.

        Returns:
            (heatmap, class_idx) where heatmap is a numpy array in [0, 1].
        """
        self.model.eval()
        # Call encoder + head directly (with gradients enabled)
        input_tensor = input_tensor.requires_grad_(True)
        outputs = self.model.vision_model(pixel_values=input_tensor)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        output = self.model.classifier(features)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=1, keepdim=True)
        cam = (weights * activations).sum(dim=-1)  # (1, num_tokens)

        # Remove CLS token if present, then reshape to 2D grid
        num_tokens = cam.shape[1]
        grid_size = int(np.sqrt(num_tokens))
        if grid_size * grid_size == num_tokens:
            cam = cam.reshape(1, grid_size, grid_size)
        else:
            cam = cam[:, 1:]
            num_patches = cam.shape[1]
            grid_size = int(np.sqrt(num_patches))
            cam = cam[:, :grid_size * grid_size]
            cam = cam.reshape(1, grid_size, grid_size)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = F.interpolate(
            cam.unsqueeze(0),
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return cam.squeeze().cpu().detach().numpy(), class_idx

    def generate_visualization(self, image, input_tensor, class_idx=None, alpha=0.5, colormap="jet"):
        """Generate Grad-CAM overlay on original image.

        Returns:
            (overlay PIL Image, heatmap array, class_idx)
        """
        heatmap, class_idx = self.generate(input_tensor, class_idx)
        img_array = np.array(image.resize((input_tensor.shape[3], input_tensor.shape[2])))

        cmap = cm.get_cmap(colormap)
        heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

        overlay = (1 - alpha) * img_array + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay), heatmap, class_idx


def create_gradcam_figure(image, heatmap, prediction, confidence, class_names, save_path=None):
    """Create a 3-panel Grad-CAM figure (original | heatmap | overlay)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=14)
    axes[1].axis("off")

    img_array = np.array(image.resize((heatmap.shape[1], heatmap.shape[0])))
    cmap = cm.get_cmap("jet")
    heatmap_colored = cmap(heatmap)[:, :, :3] * 255
    overlay = np.clip(0.6 * img_array + 0.4 * heatmap_colored, 0, 255).astype(np.uint8)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {prediction}\nConfidence: {confidence:.1%}", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_gradcam_for_report(model, image_path, transform, class_names, device):
    """Convenience wrapper for clinical report generation."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model)
    heatmap, class_idx = gradcam.generate(input_tensor)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=-1)

    overlay, _, _ = gradcam.generate_visualization(image, input_tensor, class_idx)

    return {
        "original_image": image,
        "heatmap": heatmap,
        "overlay": overlay,
        "prediction": class_names[class_idx],
        "class_idx": class_idx,
        "confidence": probs[0, class_idx].item(),
        "all_probs": {class_names[i]: probs[0, i].item() for i in range(len(class_names))},
    }
