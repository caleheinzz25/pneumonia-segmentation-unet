"""
Precompute Lung Masks for New Image Data
=========================================
GPU-accelerated batch lung segmentation using torchxrayvision PSPNet.
Based on the Lungmask project (~/Projects/github/Lungmask).

Generates binary lung mask PNGs compatible with the training pipeline.
Supports DICOM and standard image formats with batch processing,
resume support, and optional visualization.

Usage:
    # Precompute masks for new training data (saves to default lung_segmentation dir)
    uv run python -m scripts.precompute_lungmask --input data/new_train_images/

    # Precompute masks for test data
    uv run python -m scripts.precompute_lungmask --input data/test_images/ --output data/lung_masks/test/

    # With batch processing and visualization
    uv run python -m scripts.precompute_lungmask --input data/new_images/ --batch-size 16 --visualize
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ──────────────────────────────────────────────────────────────
# DICOM Reading (from Lungmask project)
# ──────────────────────────────────────────────────────────────

def read_dicom_image(dcm_path: str) -> np.ndarray:
    """Read a DICOM file and return the pixel array as uint8.

    Handles photometric interpretation (MONOCHROME1 inversion)
    and DICOM windowing automatically.
    """
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Handle photometric interpretation (some CXR are inverted)
    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array

    # Apply DICOM windowing if available
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        else:
            ww = float(ww)
        img_min = wc - ww / 2
        img_max = wc + ww / 2
        pixel_array = np.clip(pixel_array, img_min, img_max)

    # Normalize to 0-255
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)

    return pixel_array.astype(np.uint8)


def load_image(image_path: Path) -> np.ndarray:
    """Load an image as uint8 grayscale.

    Supports DICOM and standard image formats (PNG, JPG, TIFF).
    """
    suffix = image_path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        return read_dicom_image(str(image_path))
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def save_mask(mask: np.ndarray, output_path: str):
    """Save a binary mask as a PNG file."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    img = Image.fromarray(mask, mode="L")
    img.save(output_path)


def find_images(input_dir: Path) -> list[Path]:
    """Find all supported image files in a directory."""
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def create_visualization(
    image: np.ndarray,
    lung_mask: np.ndarray,
    output_path: str,
    patient_id: str,
):
    """Create a side-by-side visualization with lung mask overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))

    # Original
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Lung mask overlay
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(lung_mask, cmap="Reds", alpha=0.35, vmin=0, vmax=255)
    axes[1].set_title("Lung Mask (PSPNet)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Masked image
    masked = image.copy()
    masked[lung_mask == 0] = 0
    axes[2].imshow(masked, cmap="gray")
    axes[2].set_title("Masked Image", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(patient_id, fontsize=9, color="gray")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precompute lung segmentation masks using PSPNet (GPU-accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New training data → default lung_segmentation folder
  uv run python -m scripts.precompute_lungmask --input data/new_train_images/

  # Test data → test lung mask folder
  uv run python -m scripts.precompute_lungmask --input data/test_images/ --output data/lung_masks/test/

  # With batch processing and visualization
  uv run python -m scripts.precompute_lungmask --input data/new_images/ --batch-size 16 --visualize

  # Process only first 100 images (for testing)
  uv run python -m scripts.precompute_lungmask --input data/images/ --limit 100
        """,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory containing chest X-ray images (DICOM, PNG, JPG)",
    )
    parser.add_argument(
        "--output", type=str, default="data/lung_masks/lung_segmentation",
        help="Output directory for lung mask PNGs (default: data/lung_masks/lung_segmentation)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="GPU batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only N images (useful for testing)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate overlay visualizations",
    )
    parser.add_argument(
        "--visualize-count", type=int, default=30,
        help="Number of visualizations to generate (default: 30)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        viz_dir = output_dir.parent / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    all_images = find_images(input_dir)
    if not all_images:
        print(f"[ERROR] No supported images found in {input_dir}")
        print(f"  Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    if args.limit is not None:
        all_images = all_images[:args.limit]

    # Filter out already-processed (resume support)
    images_to_process = [
        p for p in all_images
        if not (output_dir / f"{p.stem}.png").exists()
    ]
    already_done = len(all_images) - len(images_to_process)

    # System info
    print(f"{'=' * 60}")
    print(f"  PRECOMPUTE LUNG SEGMENTATION MASKS")
    print(f"{'=' * 60}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  GPU        : Not available (using CPU)")
    print(f"  Input      : {input_dir}")
    print(f"  Output     : {output_dir}")
    print(f"  Images     : {len(all_images)} total")
    if already_done > 0:
        print(f"  Skipping   : {already_done} already processed")
    print(f"  To process : {len(images_to_process)}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Visualize  : {args.visualize}")
    print(f"{'=' * 60}")
    print()

    if not images_to_process:
        print("  All images already processed!")
        return

    # Initialize model
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from src.lung_segmentation import LungSegmentationModel
    seg_model = LungSegmentationModel(device=device)
    print()

    # Process in batches
    errors = []
    success = 0
    vis_count = 0
    batch_images = []
    batch_paths = []

    def flush_batch():
        """Process accumulated batch through the segmentation model."""
        nonlocal success, vis_count
        if not batch_images:
            return

        try:
            masks = seg_model.predict_batch(batch_images)
        except Exception:
            # Fall back to single image processing
            masks = []
            for img in batch_images:
                try:
                    masks.append(seg_model.predict(img))
                except Exception as e2:
                    masks.append(np.zeros_like(img, dtype=np.uint8))
                    errors.append((batch_paths[len(masks) - 1].stem, str(e2)))

        for img, mask, img_path in zip(batch_images, masks, batch_paths):
            pid = img_path.stem
            out_path = output_dir / f"{pid}.png"
            if not out_path.exists():
                save_mask(mask, str(out_path))
                success += 1

            # Visualization
            if args.visualize and vis_count < args.visualize_count:
                viz_path = viz_dir / f"{pid}.png"
                if not viz_path.exists():
                    create_visualization(img, mask, str(viz_path), pid)
                    vis_count += 1

    pbar = tqdm(images_to_process, desc="Processing", dynamic_ncols=True)
    for img_path in pbar:
        try:
            image = load_image(img_path)
            batch_images.append(image)
            batch_paths.append(img_path)

            if len(batch_images) >= args.batch_size:
                flush_batch()
                batch_images.clear()
                batch_paths.clear()

            pbar.set_postfix_str(f"ok={success} fail={len(errors)}")

        except Exception as e:
            errors.append((img_path.stem, str(e)))
            if len(errors) <= 5:
                tqdm.write(f"  [WARN] {img_path.name}: {e}")

    # Flush remaining batch
    if batch_images:
        flush_batch()

    # Summary
    print()
    print(f"{'=' * 60}")
    print(f"  COMPLETE")
    print(f"  Processed   : {success}/{len(images_to_process)} images")
    if errors:
        print(f"  Errors      : {len(errors)}")
    total_masks = len(list(output_dir.glob("*.png")))
    print(f"  Total masks : {total_masks}")
    if args.visualize:
        total_viz = len(list(viz_dir.glob("*.png")))
        print(f"  Visualizations: {total_viz}")
    print(f"  Saved to    : {output_dir}")
    print(f"{'=' * 60}")

    if errors:
        print(f"\nErrors:")
        for pid, err in errors[:10]:
            print(f"  {pid}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
