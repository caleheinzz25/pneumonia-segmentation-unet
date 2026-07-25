"""
Precompute All Masks
====================
Generates BOTH required mask types for the training pipeline in a single run:

  1. Lung Segmentation Masks  → data/lung_masks/lung_segmentation/
     Generated via PSPNet (torchxrayvision), GPU-accelerated batch inference.

  2. Pneumonia Ground Truth Masks → data/lung_masks/pneumonia_ground_truth/
     Generated from bounding box annotations in stage_2_train_labels.csv.
     Positive samples only (Target=1).

  3. Combined Visualization    → data/lung_masks/combined/  (optional --visualize)

Usage:
    # Full pipeline (recommended — all 26K training images)
    uv run python -m scripts.precompute_all_masks

    # Only GT masks (fast, no GPU needed)
    uv run python -m scripts.precompute_all_masks --skip-lung

    # Only lung masks (GPU)
    uv run python -m scripts.precompute_all_masks --skip-gt

    # Test with 100 images
    uv run python -m scripts.precompute_all_masks --limit 100

    # Test images lung masks
    uv run python -m scripts.precompute_all_masks --skip-gt --input-test
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "rsna-pneumonia-detection-challenge"
TRAIN_DICOM_DIR = DATA_ROOT / "stage_2_train_images"
TEST_DICOM_DIR = DATA_ROOT / "stage_2_test_images"
TRAIN_LABELS_CSV = DATA_ROOT / "stage_2_train_labels.csv"

OUT_LUNG_SEG = PROJECT_ROOT / "data" / "lung_masks" / "lung_segmentation"
OUT_LUNG_TEST = PROJECT_ROOT / "data" / "lung_masks" / "test"
OUT_GT = PROJECT_ROOT / "data" / "lung_masks" / "pneumonia_ground_truth"
OUT_COMBINED = PROJECT_ROOT / "data" / "lung_masks" / "combined"
OUT_VIZ = PROJECT_ROOT / "data" / "lung_masks" / "visualizations"


# ──────────────────────────────────────────────────────────────────
# DICOM reading
# ──────────────────────────────────────────────────────────────────

def read_dicom_uint8(dcm_path: Path) -> np.ndarray:
    """Read DICOM → uint8 grayscale, handling MONOCHROME1 inversion."""
    ds = pydicom.dcmread(str(dcm_path))
    pixels = ds.pixel_array.astype(np.float32)

    # Photometric inversion
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    # DICOM windowing (if available)
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if hasattr(wc, "__iter__"):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if hasattr(ww, "__iter__"):
            ww = float(ww[0])
        else:
            ww = float(ww)
        pixels = np.clip(pixels, wc - ww / 2, wc + ww / 2)

    pmin, pmax = pixels.min(), pixels.max()
    if pmax > pmin:
        pixels = (pixels - pmin) / (pmax - pmin) * 255.0
    else:
        pixels = np.zeros_like(pixels)

    return pixels.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────
# Ground Truth Mask Generation (from Bounding Boxes)
# ──────────────────────────────────────────────────────────────────

def build_gt_masks(
    labels_csv: Path,
    dicom_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    verbose: bool = True,
) -> tuple[int, int, list[str]]:
    """Generate pneumonia ground truth masks from bounding box annotations.

    For each positive patient (Target=1), reads their DICOM to get image
    dimensions then fills all bounding boxes into a binary mask PNG.
    Negative patients (Target=0) are skipped — their GT is implicitly all-zeros.

    Returns:
        (n_success, n_skip, errors)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)
    df["patientId"] = df["patientId"].astype(str)

    # Only positive patients
    pos_df = df[df["Target"] == 1].copy()
    patient_groups = pos_df.groupby("patientId")
    all_pos_pids = sorted(patient_groups.groups.keys())

    if limit is not None:
        all_pos_pids = all_pos_pids[:limit]

    # Filter out already-processed
    to_process = [
        pid for pid in all_pos_pids
        if not (output_dir / f"{pid}.png").exists()
    ]
    already_done = len(all_pos_pids) - len(to_process)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  STEP 1 — PNEUMONIA GROUND TRUTH MASKS")
        print(f"{'='*60}")
        print(f"  CSV        : {labels_csv}")
        print(f"  Output     : {output_dir}")
        print(f"  Positive patients: {len(all_pos_pids)}")
        if already_done > 0:
            print(f"  Already done: {already_done} (skipped)")
        print(f"  To process: {len(to_process)}")
        print()

    if not to_process:
        print("  All GT masks already exist. Skipping.")
        return 0, already_done, []

    success = 0
    errors = []

    pbar = tqdm(to_process, desc="GT masks", dynamic_ncols=True)
    for pid in pbar:
        try:
            dcm_path = dicom_dir / f"{pid}.dcm"

            # Get image size from DICOM header (fast — no pixel decoding needed)
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            h = int(ds.Rows)
            w = int(ds.Columns)

            # Build mask from all bounding boxes for this patient
            mask = np.zeros((h, w), dtype=np.uint8)
            rows = patient_groups.get_group(pid)
            for _, row in rows.iterrows():
                if pd.notna(row.get("x")) and pd.notna(row.get("width")):
                    x = max(0, int(float(row["x"])))
                    y = max(0, int(float(row["y"])))
                    bw = int(float(row["width"]))
                    bh = int(float(row["height"]))
                    x2 = min(w, x + bw)
                    y2 = min(h, y + bh)
                    if x2 > x and y2 > y:
                        mask[y:y2, x:x2] = 255

            # Save
            out_path = output_dir / f"{pid}.png"
            Image.fromarray(mask, mode="L").save(out_path)
            success += 1
            pbar.set_postfix_str(f"ok={success} fail={len(errors)}")

        except Exception as e:
            errors.append(f"{pid}: {e}")
            if len(errors) <= 5:
                tqdm.write(f"  [WARN] GT {pid}: {e}")

    return success, already_done, errors


# ──────────────────────────────────────────────────────────────────
# Lung Segmentation Mask Generation (PSPNet)
# ──────────────────────────────────────────────────────────────────

def build_lung_masks(
    dicom_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 8,
    limit: int | None = None,
    verbose: bool = True,
    label: str = "lung",
) -> tuple[int, int, list[str]]:
    """Generate lung segmentation masks using torchxrayvision PSPNet.

    Args:
        dicom_dir: Directory with DICOM files.
        output_dir: Where to save binary mask PNGs.
        device: 'cuda' or 'cpu'.
        batch_size: GPU batch size.
        limit: Max number to process (for testing).
        verbose: Print progress header.
        label: Label for tqdm description.

    Returns:
        (n_success, n_skip, errors)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dcm = sorted(dicom_dir.glob("*.dcm"))
    if limit is not None:
        all_dcm = all_dcm[:limit]

    # Resume: skip already-processed
    to_process = [p for p in all_dcm if not (output_dir / f"{p.stem}.png").exists()]
    already_done = len(all_dcm) - len(to_process)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  STEP 2 — LUNG SEGMENTATION MASKS ({label.upper()})")
        print(f"{'='*60}")
        print(f"  Input      : {dicom_dir}")
        print(f"  Output     : {output_dir}")
        print(f"  Device     : {device}")
        if torch.cuda.is_available() and device == "cuda":
            print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  Batch size : {batch_size}")
        print(f"  Total DICOMs: {len(all_dcm)}")
        if already_done > 0:
            print(f"  Already done: {already_done} (skipped)")
        print(f"  To process: {len(to_process)}")
        print()

    if not to_process:
        print(f"  All lung masks ({label}) already exist. Skipping.")
        return 0, already_done, []

    # Load model
    from src.lung_segmentation import LungSegmentationModel
    seg_model = LungSegmentationModel(device=device)
    print()

    success = 0
    errors = []
    batch_images: list[np.ndarray] = []
    batch_paths: list[Path] = []

    def flush_batch() -> None:
        nonlocal success
        if not batch_images:
            return
        try:
            masks = seg_model.predict_batch(batch_images)
        except Exception:
            # Fall back to single inference
            masks = []
            for img in batch_images:
                try:
                    masks.append(seg_model.predict(img))
                except Exception as e2:
                    masks.append(np.zeros_like(img, dtype=np.uint8))
                    errors.append(f"{batch_paths[len(masks)-1].stem}: {e2}")

        for img_arr, mask, p in zip(batch_images, masks, batch_paths):
            out_path = output_dir / f"{p.stem}.png"
            if not out_path.exists():
                Image.fromarray(mask, mode="L").save(out_path)
                nonlocal success  # noqa: F821
                success += 1

    # Workaround: can't use nonlocal in nested function with += in Python <3.12 cleanly
    # Using a list as a counter to allow mutation in flush_batch
    _counter = [0]

    def flush_batch_v2() -> None:
        if not batch_images:
            return
        try:
            masks = seg_model.predict_batch(batch_images)
        except Exception:
            masks = []
            for img in batch_images:
                try:
                    masks.append(seg_model.predict(img))
                except Exception as e2:
                    masks.append(np.zeros_like(img, dtype=np.uint8))
                    errors.append(f"{batch_paths[len(masks)-1].stem}: {e2}")

        for mask, p in zip(masks, batch_paths):
            out_path = output_dir / f"{p.stem}.png"
            if not out_path.exists():
                Image.fromarray(mask, mode="L").save(out_path)
                _counter[0] += 1

    pbar = tqdm(to_process, desc=f"Lung ({label})", dynamic_ncols=True)
    for dcm_path in pbar:
        try:
            image = read_dicom_uint8(dcm_path)
            batch_images.append(image)
            batch_paths.append(dcm_path)

            if len(batch_images) >= batch_size:
                flush_batch_v2()
                batch_images.clear()
                batch_paths.clear()
                pbar.set_postfix_str(f"ok={_counter[0]} fail={len(errors)}")

        except Exception as e:
            errors.append(f"{dcm_path.stem}: {e}")
            if len(errors) <= 5:
                tqdm.write(f"  [WARN] {dcm_path.name}: {e}")

    # Flush remaining
    if batch_images:
        flush_batch_v2()

    return _counter[0], already_done, errors


# ──────────────────────────────────────────────────────────────────
# Combined Visualization
# ──────────────────────────────────────────────────────────────────

def build_combined_visualizations(
    dicom_dir: Path,
    lung_mask_dir: Path,
    gt_mask_dir: Path,
    output_dir: Path,
    count: int = 50,
) -> None:
    """Create side-by-side overlay images (lung mask + GT mask) for inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find patients that have both masks
    lung_pids = {p.stem for p in lung_mask_dir.glob("*.png")}
    gt_pids = {p.stem for p in gt_mask_dir.glob("*.png")}
    # Prefer positive cases for visualization
    positive_pids = sorted(gt_pids & lung_pids)[:count]
    remaining_slots = count - len(positive_pids)
    # Fill up with negative cases (have lung mask only)
    neg_pids = sorted(lung_pids - gt_pids)[:remaining_slots]
    all_pids = positive_pids + neg_pids

    existing = {p.stem for p in output_dir.glob("*.png")}
    to_viz = [pid for pid in all_pids if pid not in existing]

    if not to_viz:
        print(f"  All {count} visualizations already exist. Skipping.")
        return

    print(f"\n  Generating {len(to_viz)} combined visualizations...")

    for pid in tqdm(to_viz, desc="Visualizations", dynamic_ncols=True):
        try:
            dcm_path = dicom_dir / f"{pid}.dcm"
            if not dcm_path.exists():
                continue

            image = read_dicom_uint8(dcm_path)
            lung_path = lung_mask_dir / f"{pid}.png"
            gt_path = gt_mask_dir / f"{pid}.png"

            lung_mask = np.array(Image.open(lung_path)) if lung_path.exists() else np.zeros_like(image)
            gt_mask = np.array(Image.open(gt_path)) if gt_path.exists() else np.zeros_like(image)

            has_pneumonia = gt_path.exists()
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # Original
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Original CXR", fontweight="bold")
            axes[0].axis("off")

            # Lung mask
            axes[1].imshow(image, cmap="gray")
            axes[1].imshow(lung_mask, cmap="Blues", alpha=0.35, vmin=0, vmax=255)
            axes[1].set_title("Lung Mask (PSPNet)", fontweight="bold")
            axes[1].axis("off")

            # GT pneumonia mask
            axes[2].imshow(image, cmap="gray")
            if has_pneumonia:
                axes[2].imshow(gt_mask, cmap="Reds", alpha=0.45, vmin=0, vmax=255)
            axes[2].set_title(f"GT Pneumonia {'(Positive)' if has_pneumonia else '(Negative)'}", fontweight="bold")
            axes[2].axis("off")

            # Both overlaid
            axes[3].imshow(image, cmap="gray")
            axes[3].imshow(lung_mask, cmap="Blues", alpha=0.25, vmin=0, vmax=255)
            if has_pneumonia:
                axes[3].imshow(gt_mask, cmap="Reds", alpha=0.40, vmin=0, vmax=255)
            axes[3].set_title("Combined", fontweight="bold")
            axes[3].axis("off")

            label = "POSITIVE" if has_pneumonia else "NEGATIVE"
            fig.suptitle(f"{pid}  [{label}]", fontsize=9, color="dimgray")
            plt.tight_layout()
            plt.savefig(str(output_dir / f"{pid}.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            tqdm.write(f"  [WARN] Viz {pid}: {e}")


# ──────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precompute lung segmentation + pneumonia GT masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-lung", action="store_true",
                        help="Skip lung segmentation mask generation (PSPNet)")
    parser.add_argument("--skip-gt", action="store_true",
                        help="Skip pneumonia ground truth mask generation")
    parser.add_argument("--input-test", action="store_true",
                        help="Also generate lung masks for test images (data/lung_masks/test/)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="GPU batch size for PSPNet inference (default: 8)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images per step (for testing)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate combined lung+GT overlay visualizations")
    parser.add_argument("--visualize-count", type=int, default=50,
                        help="Number of combined visualizations to generate (default: 50)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  PRECOMPUTE ALL MASKS")
    print(f"{'='*60}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Device      : {device}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Skip lung   : {args.skip_lung}")
    print(f"  Skip GT     : {args.skip_gt}")
    print(f"  Test masks  : {args.input_test}")
    print(f"  Visualize   : {args.visualize}")
    print()

    # ── Verify paths ──────────────────────────────────────────────
    if not TRAIN_DICOM_DIR.exists():
        print(f"[ERROR] Training DICOM dir not found: {TRAIN_DICOM_DIR}")
        sys.exit(1)
    if not TRAIN_LABELS_CSV.exists():
        print(f"[ERROR] Labels CSV not found: {TRAIN_LABELS_CSV}")
        sys.exit(1)

    all_errors = []

    # ── Step 1: Pneumonia GT masks ────────────────────────────────
    if not args.skip_gt:
        ok, skip, errs = build_gt_masks(
            labels_csv=TRAIN_LABELS_CSV,
            dicom_dir=TRAIN_DICOM_DIR,
            output_dir=OUT_GT,
            limit=args.limit,
        )
        all_errors.extend(errs)
        print(f"\n  GT masks: {ok} new + {skip} already done")
        if errs:
            print(f"  GT errors: {len(errs)}")

    # ── Step 2: Lung masks (train images) ─────────────────────────
    if not args.skip_lung:
        ok, skip, errs = build_lung_masks(
            dicom_dir=TRAIN_DICOM_DIR,
            output_dir=OUT_LUNG_SEG,
            device=device,
            batch_size=args.batch_size,
            limit=args.limit,
            label="train",
        )
        all_errors.extend(errs)
        print(f"\n  Lung masks (train): {ok} new + {skip} already done")
        if errs:
            print(f"  Lung errors: {len(errs)}")

    # ── Step 3: Lung masks (test images) ──────────────────────────
    if args.input_test and not args.skip_lung:
        if TEST_DICOM_DIR.exists():
            ok, skip, errs = build_lung_masks(
                dicom_dir=TEST_DICOM_DIR,
                output_dir=OUT_LUNG_TEST,
                device=device,
                batch_size=args.batch_size,
                limit=args.limit,
                label="test",
                verbose=True,
            )
            all_errors.extend(errs)
            print(f"\n  Lung masks (test): {ok} new + {skip} already done")
        else:
            print(f"\n  [SKIP] Test DICOM dir not found: {TEST_DICOM_DIR}")

    # ── Step 4: Combined visualizations ───────────────────────────
    if args.visualize and OUT_LUNG_SEG.exists() and OUT_GT.exists():
        build_combined_visualizations(
            dicom_dir=TRAIN_DICOM_DIR,
            lung_mask_dir=OUT_LUNG_SEG,
            gt_mask_dir=OUT_GT,
            output_dir=OUT_COMBINED,
            count=args.visualize_count,
        )

    # ── Final summary ─────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE  ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Count outputs
    if OUT_GT.exists():
        print(f"  GT masks        : {len(list(OUT_GT.glob('*.png'))):,} files → {OUT_GT}")
    if OUT_LUNG_SEG.exists():
        print(f"  Lung masks (train): {len(list(OUT_LUNG_SEG.glob('*.png'))):,} files → {OUT_LUNG_SEG}")
    if OUT_LUNG_TEST.exists():
        print(f"  Lung masks (test) : {len(list(OUT_LUNG_TEST.glob('*.png'))):,} files → {OUT_LUNG_TEST}")
    if OUT_COMBINED.exists():
        print(f"  Visualizations  : {len(list(OUT_COMBINED.glob('*.png'))):,} files → {OUT_COMBINED}")

    if all_errors:
        print(f"\n  Total errors: {len(all_errors)}")
        for e in all_errors[:10]:
            print(f"    {e}")
        if len(all_errors) > 10:
            print(f"    ... and {len(all_errors)-10} more")
    print()


if __name__ == "__main__":
    main()
