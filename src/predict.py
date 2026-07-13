"""Inference script for single image or batch prediction."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import Config, load_config
from src.lung_segmentation import LungSegmenter
from src.model import build_model
from src.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.utils import (
    apply_window,
    grayscale_to_rgb,
    overlay_mask,
    read_dicom,
    resize_image_mask,
    setup_logging,
)


def predict_single(
    model: torch.nn.Module,
    image_path: str | Path,
    config: Config,
    device: str,
    lung_segmenter: LungSegmenter | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict on a single image.

    Args:
        model: Trained model
        image_path: Path to DICOM or image file
        config: Configuration
        device: Device string

    Returns:
        Tuple of (probability map, input image tensor, lung mask)
    """
    image_path = Path(image_path)

    # Read image
    if image_path.suffix.lower() in [".dcm", ".dicom"]:
        image = read_dicom(image_path)
        if config.preprocessing.normalize and not config.preprocessing.apply_lung_window:
            image = image / 255.0
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

    # Apply lung mask to focus on lung regions (consistent with training)
    lung_mask_applied = False
    lung_mask = None
    lung_mask_dirs = [config.data.test_lung_mask_dir, config.data.lung_mask_dir]
    patient_id = image_path.stem
    for mask_dir in lung_mask_dirs:
        if mask_dir is None:
            continue
        mask_path = Path(mask_dir) / f"{patient_id}.png"
        if mask_path.exists():
            lung_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if lung_mask is not None:
                from src.lung_segmentation import _postprocess_lung_mask
                lung_mask = _postprocess_lung_mask(lung_mask)
                lung_mask = (lung_mask > 127).astype(np.float32)
                lung_mask = cv2.resize(lung_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                image_resized = image_resized * lung_mask
                lung_mask_applied = True
            break

    # Fallback: auto lung segmentation if no precomputed mask found
    if not lung_mask_applied and lung_segmenter is not None:
        lung_mask = lung_segmenter.segment(image_resized, target_h=target_h, target_w=target_w)
        image_resized = image_resized * lung_mask
        lung_mask_applied = True

    image_rgb = grayscale_to_rgb(image_resized)

    # Normalize and convert to tensor
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    image_norm = (image_rgb - mean) / std
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float()

    # Predict
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        if isinstance(logits, list):
            logits = logits[-1]
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

    # Resize back to original
    prob_resized = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    if lung_mask is not None:
        lung_mask_orig = cv2.resize(lung_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    else:
        lung_mask_orig = np.ones((h_orig, w_orig), dtype=np.float32)

    return prob_resized, image, lung_mask_orig


def predict_batch(
    model: torch.nn.Module,
    image_paths: list[str | Path],
    config: Config,
    device: str,
) -> list[np.ndarray]:
    """Predict on a batch of images."""
    results = []
    for path in image_paths:
        prob, _, _ = predict_single(model, path, config, device)
        results.append(prob)
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict Pneumonia Segmentation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(logs_dir=config.output.logs_dir, run_name="predict")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_model(config.model, device=device)
    checkpoint = torch.load(config.inference.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load lung segmenter for auto lung masking
    lung_segmenter = LungSegmenter(device=device)

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path(config.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = list(input_path.glob("*.dcm")) + list(input_path.glob("*.png"))

    for img_path in image_paths:
        prob, original_image, lung_mask = predict_single(model, img_path, config, device, lung_segmenter=lung_segmenter)
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
