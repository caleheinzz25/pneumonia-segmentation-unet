"""Inference script for single image or batch prediction."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import Config, load_config
from src.model import build_model
from src.transforms import get_validation_transforms
from src.utils import (
    apply_window,
    grayscale_to_rgb,
    overlay_mask,
    read_dicom,
    resize_image_mask,
)


def predict_single(
    model: torch.nn.Module,
    image_path: str | Path,
    config: Config,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict on a single image.

    Args:
        model: Trained model
        image_path: Path to DICOM or image file
        config: Configuration
        device: Device string

    Returns:
        Tuple of (probability map, input image tensor)
    """
    image_path = Path(image_path)

    # Read image
    if image_path.suffix.lower() in [".dcm", ".dicom"]:
        image = read_dicom(image_path)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = image.astype(np.float32) / 255.0

    h_orig, w_orig = image.shape

    # Apply lung windowing
    if config.preprocessing.apply_lung_window:
        image = apply_window(
            image,
            window_level=config.preprocessing.window_level,
            window_width=config.preprocessing.window_width,
        )

    # Resize
    target_w, target_h = config.preprocessing.image_size[1], config.preprocessing.image_size[0]
    image_resized, _ = resize_image_mask(image, None, (target_w, target_h))
    image_rgb = grayscale_to_rgb(image_resized)

    # Normalize and convert to tensor
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_rgb - mean) / std
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float()

    # Predict
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

    # Resize back to original
    prob_resized = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    return prob_resized, image


def predict_batch(
    model: torch.nn.Module,
    image_paths: list[str | Path],
    config: Config,
    device: str,
) -> list[np.ndarray]:
    """Predict on a batch of images."""
    results = []
    for path in image_paths:
        prob, _ = predict_single(model, path, config, device)
        results.append(prob)
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict Pneumonia Segmentation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_model(config.model, device=device)
    checkpoint = torch.load(config.inference.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path(config.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list(input_path.glob("*.dcm")) + list(input_path.glob("*.png"))

    for img_path in image_paths:
        prob, original_image = predict_single(model, img_path, config, device)
        pred_mask = (prob >= config.inference.threshold).astype(np.float32)

        # Save prediction
        pred_path = output_dir / f"{img_path.stem}_pred.png"
        cv2.imwrite(str(pred_path), (pred_mask * 255).astype(np.uint8))

        # Save overlay
        if config.inference.save_overlay:
            overlay = overlay_mask(
                original_image,
                pred_mask,
                color=tuple(config.inference.overlay_color),
                alpha=config.inference.overlay_alpha,
            )
            overlay_path = output_dir / f"{img_path.stem}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

        print(f"Saved prediction for {img_path.name}")


if __name__ == "__main__":
    main()
