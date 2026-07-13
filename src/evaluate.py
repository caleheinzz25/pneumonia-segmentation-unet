"""Evaluation script: compute metrics and generate visualizations with TTA support."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, load_config
from src.dataset import RSNADataset, get_train_val_split
from src.metrics import SegmentationMetrics
from src.model import build_model
from src.transforms import get_validation_transforms
from src.utils import overlay_mask, set_seed, setup_logging


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    tta_transforms: list[str] | None = None,
) -> np.ndarray:
    """Predict with test-time augmentation.

    Applies specified transforms, averages predictions for more robust results.

    Args:
        model: Trained model
        images: Input batch (B, C, H, W) on device
        tta_transforms: List of TTA transform names. Supported: "hflip"

    Returns:
        Probability maps as numpy array (B, H, W)
    """
    if not tta_transforms:
        logits = model(images)
        if isinstance(logits, list):
            logits = logits[-1]
        return torch.sigmoid(logits).cpu().numpy()

    probs_list = []

    # Original prediction
    logits = model(images)
    if isinstance(logits, list):
        logits = logits[-1]
    probs = torch.sigmoid(logits)
    probs_list.append(probs)

    for transform in tta_transforms:
        if transform == "hflip":
            flipped = torch.flip(images, dims=[-1])
            logits_f = model(flipped)
            if isinstance(logits_f, list):
                logits_f = logits_f[-1]
            probs_f = torch.sigmoid(logits_f)
            probs_f = torch.flip(probs_f, dims=[-1])
            probs_list.append(probs_f)

    # Average all predictions
    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    return avg_probs.cpu().numpy()


@torch.no_grad()
def evaluate(config: Config) -> dict[str, float]:
    """Run full evaluation on validation set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config.training.seed)

    # Load model
    model = build_model(config.model, device=device)
    checkpoint = torch.load(config.inference.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {config.inference.model_path}")

    # TTA config
    use_tta = config.inference.use_tta
    tta_transforms = config.inference.tta_transforms if use_tta else None
    if use_tta:
        print(f"TTA enabled: {tta_transforms}")

    # Validation data
    _, val_ids = get_train_val_split(
        data_config=config.data,
        val_split=config.training.val_split,
        seed=config.training.seed,
        stratified=config.training.stratified_split,
    )

    val_transform = get_validation_transforms(config.preprocessing)
    val_dataset = RSNADataset(
        data_config=config.data,
        prep_config=config.preprocessing,
        patient_ids=val_ids,
        transform=val_transform,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    metrics = SegmentationMetrics(metrics=config.evaluation.metrics)
    output_dir = Path(config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store only lightweight data for visualization (not full images)
    viz_samples: list[dict] = []
    viz_indices = set()
    total_samples = len(val_dataset)
    num_viz = min(config.evaluation.num_visualization_samples, total_samples)
    if total_samples > 0:
        viz_indices = set(np.random.choice(total_samples, num_viz, replace=False))

    sample_idx = 0
    pbar = tqdm(
        val_loader,
        desc="Evaluating",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].cpu().numpy()
        patient_ids = batch["patient_id"]

        # Predict with TTA
        probs = predict_with_tta(model, images, tta_transforms)

        for i in range(probs.shape[0]):
            sample_metrics = metrics.update(probs[i, 0], masks[i, 0], threshold=config.inference.threshold)

            # Only store samples needed for visualization
            if sample_idx in viz_indices:
                d_val = sample_metrics.get("dice")
                if d_val is None:
                    from src.metrics import dice_score
                    pred_binary = (probs[i, 0] >= config.inference.threshold).astype(np.float32)
                    d_val = dice_score(pred_binary, masks[i, 0])
                viz_samples.append({
                    "image": images[i].cpu().numpy(),
                    "prob": probs[i, 0],
                    "mask": masks[i, 0],
                    "patient_id": patient_ids[i],
                    "dice": d_val
                })
            sample_idx += 1

    results = metrics.compute()
    print("\n" + metrics.get_summary())

    # Save metrics to JSON
    metrics_path = Path(config.evaluation.save_metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save per-class breakdown
    per_class = _compute_per_class_metrics(val_loader, model, device, config, tta_transforms)
    if per_class:
        per_class_path = output_dir / "per_class_metrics.json"
        with open(per_class_path, "w") as f:
            json.dump(per_class, f, indent=2)
        print(f"Per-class metrics saved to {per_class_path}")

    # Visualize random samples
    if viz_samples:
        fig, axes = plt.subplots(len(viz_samples), 3, figsize=(12, 4 * len(viz_samples)))
        if len(viz_samples) == 1:
            axes = axes.reshape(1, -1)

        for idx, sample in enumerate(viz_samples):
            image = sample["image"]
            prob = sample["prob"]
            mask = sample["mask"]
            patient_id = sample["patient_id"]

            # Denormalize image for visualization
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_viz = (image * std + mean).transpose(1, 2, 0)
            img_viz = np.clip(img_viz, 0, 1)

            pred_binary = (prob >= config.inference.threshold).astype(np.float32)
            overlay = overlay_mask(
                img_viz,
                pred_binary,
                color=tuple(config.inference.overlay_color),
                alpha=config.inference.overlay_alpha,
            )
            gt_overlay = overlay_mask(
                img_viz,
                mask,
                color=(0, 255, 0),
                alpha=0.4,
            )

            axes[idx, 0].imshow(img_viz)
            axes[idx, 0].set_title(f"Input: {patient_id}")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(gt_overlay)
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis("off")

            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title(f"Prediction (Dice: {sample['dice']:.3f})")
            axes[idx, 2].axis("off")

        plt.tight_layout()
        viz_path = output_dir / "evaluation_samples.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visualizations saved to {viz_path}")

    return results


@torch.no_grad()
def _compute_per_class_metrics(
    val_loader: DataLoader,
    model: torch.nn.Module,
    device: str,
    config: Config,
    tta_transforms: list[str] | None,
) -> dict:
    """Compute metrics separately for positive (pneumonia) and negative (normal) patients."""
    from src.metrics import dice_score, iou_score

    pos_dices = []
    neg_dices = []
    pos_ious = []
    neg_ious = []

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].cpu().numpy()
        probs = predict_with_tta(model, images, tta_transforms)

        for i in range(probs.shape[0]):
            pred_binary = (probs[i, 0] >= config.inference.threshold).astype(np.float32)
            mask = masks[i, 0]

            if mask.sum() > 0:
                pos_dices.append(dice_score(pred_binary, mask))
                pos_ious.append(iou_score(pred_binary, mask))
            else:
                neg_dices.append(dice_score(pred_binary, mask))
                neg_ious.append(iou_score(pred_binary, mask))

    result = {}
    if pos_dices:
        result["positive"] = {
            "count": len(pos_dices),
            "dice": float(np.mean(pos_dices)),
            "iou": float(np.mean(pos_ious)),
        }
    if neg_dices:
        result["negative"] = {
            "count": len(neg_dices),
            "dice": float(np.mean(neg_dices)),
            "iou": float(np.mean(neg_ious)),
        }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pneumonia Segmentation Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--tta", action="store_true", default=None,
        help="Enable test-time augmentation (overrides config)"
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable test-time augmentation (overrides config)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.tta:
        config.inference.use_tta = True
    elif args.no_tta:
        config.inference.use_tta = False

    setup_logging(logs_dir=config.output.logs_dir, run_name="evaluate")
    evaluate(config)
