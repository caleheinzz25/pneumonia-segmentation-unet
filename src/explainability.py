"""Grad-CAM explainability for pneumonia segmentation model."""

import argparse
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config, load_config
from src.model import build_model
from src.predict import predict_single
from src.utils import grayscale_to_rgb, set_seed


class GradCAM:
    """Gradient-weighted Class Activation Mapping for segmentation models."""

    def __init__(self, model: torch.nn.Module, target_layer: str | None = None):
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Auto-detect target layer if not specified
        if target_layer is None:
            # Use the last encoder block
            encoder = model.encoder
            layer_names = [name for name, _ in encoder.named_children()]
            target_layer = layer_names[-1] if layer_names else None

        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        if self.target_layer is None:
            return

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._forward_hook)
                module.register_full_backward_hook(self._backward_hook)
                break

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for input.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)

        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1]
        """
        self.model.eval()
        self.gradients = None
        self.activations = None

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Backward on output (sum of positive predictions)
        score = output.sum()
        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")

        # Pool gradients across spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations by pooled gradients
        for i in range(pooled_gradients.shape[0]):
            self.activations[0, i, :, :] *= pooled_gradients[i]

        # Average weighted activations across channels
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap.cpu().numpy()


def apply_colormap(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Apply colormap to heatmap.

    Args:
        heatmap: Normalized heatmap (H, W) in [0, 1]
        colormap: Matplotlib colormap name

    Returns:
        Colored heatmap as RGB image (H, W, 3) in [0, 255]
    """
    cmap = cm.colormaps[colormap]
    colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay heatmap on original image.

    Args:
        image: Original image (H, W) or (H, W, 3)
        heatmap: Heatmap (H, W) in [0, 1]
        alpha: Transparency

    Returns:
        Overlay image (H, W, 3)
    """
    if image.ndim == 2:
        image = grayscale_to_rgb(image)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    colored_heatmap = apply_colormap(heatmap)
    overlay = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)
    return overlay


def generate_gradcam(
    config: Config,
    image_path: str | Path,
    output_dir: Path,
) -> None:
    """Generate Grad-CAM visualization for a single image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config.training.seed)

    # Load model
    model = build_model(config.model, device=device)
    checkpoint = torch.load(config.inference.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Predict
    prob, original_image = predict_single(model, image_path, config, device)

    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer=config.explainability.target_layer)

    # Need to re-preprocess for Grad-CAM
    from src.utils import apply_window, grayscale_to_rgb, read_dicom, resize_image_mask

    image_path = Path(image_path)
    if image_path.suffix.lower() in [".dcm", ".dicom"]:
        image = read_dicom(image_path)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = image.astype(np.float32) / 255.0

    if config.preprocessing.apply_lung_window:
        image = apply_window(image, config.preprocessing.window_level, config.preprocessing.window_width)

    target_w, target_h = config.preprocessing.image_size[1], config.preprocessing.image_size[0]
    image_resized, _ = resize_image_mask(image, None, (target_w, target_h))
    image_rgb = grayscale_to_rgb(image_resized)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_rgb - mean) / std
    input_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float()

    heatmap = gradcam.generate(input_tensor)

    # Resize heatmap to original size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Create overlay
    overlay = overlay_heatmap(image, heatmap_resized, alpha=0.5)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / f"{image_path.stem}_gradcam.png"), overlay)
    cv2.imwrite(
        str(output_dir / f"{image_path.stem}_heatmap.png"),
        apply_colormap(heatmap_resized),
    )

    print(f"Grad-CAM saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="outputs/gradcam")
    args = parser.parse_args()

    config = load_config(args.config)
    generate_gradcam(config, args.input, Path(args.output))
