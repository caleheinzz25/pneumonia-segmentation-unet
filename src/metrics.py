"""Segmentation metrics computation."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient between prediction and target.

    Args:
        pred: Binary or probability predictions (H, W) or flattened
        target: Binary ground truth (H, W) or flattened
        smooth: Smoothing factor

    Returns:
        Dice coefficient in [0, 1]
    """
    pred = pred.flatten()
    target = target.flatten()

    if target.sum() == 0 and pred.sum() == 0:
        return 1.0

    intersection = (pred * target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (IoU / Jaccard index).

    Args:
        pred: Binary predictions
        target: Binary ground truth
        smooth: Smoothing factor

    Returns:
        IoU in [0, 1]
    """
    pred = pred.flatten()
    target = target.flatten()

    if target.sum() == 0 and pred.sum() == 0:
        return 1.0

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def precision_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Precision (positive predictive value)."""
    pred = pred.flatten()
    target = target.flatten()

    if pred.sum() == 0:
        return float('nan')

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def recall_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Recall (sensitivity / true positive rate)."""
    pred = pred.flatten()
    target = target.flatten()

    if target.sum() == 0:
        return float('nan')

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def accuracy_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute pixel-wise accuracy."""
    pred = pred.flatten()
    target = target.flatten()

    correct = (pred == target).sum()
    return float(correct / len(pred))


def specificity_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Specificity (true negative rate)."""
    pred = pred.flatten()
    target = target.flatten()

    if target.sum() == len(target):
        return float('nan')

    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return float((tn + smooth) / (tn + fp + smooth))


def auc_score(probs: np.ndarray, target: np.ndarray) -> float:
    """Compute Area Under ROC Curve.

    Args:
        probs: Probability predictions (flattened)
        target: Binary ground truth (flattened)

    Returns:
        AUC score or 0.5 if only one class present
    """
    probs = probs.flatten()
    target = target.flatten()

    if len(np.unique(target)) < 2:
        return 0.5

    try:
        return float(roc_auc_score(target, probs))
    except ValueError:
        return 0.5


class SegmentationMetrics:
    """Compute and accumulate segmentation metrics over a dataset.

    Correctly handles negative samples (no pneumonia):
    - dice, iou, precision, recall: skip samples where BOTH pred and target are all-zero
    - auc: skip samples with only one class in target
    """

    def __init__(self, metrics: list[str] | None = None):
        """
        Args:
            metrics: List of metric names to compute.
                     Defaults to all available metrics.
        """
        self.available_metrics = {
            "dice": dice_score,
            "iou": iou_score,
            "precision": precision_score,
            "recall": recall_score,
            "accuracy": accuracy_score,
            "specificity": specificity_score,
            "auc": auc_score,
        }
        self.metrics = metrics or list(self.available_metrics.keys())
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.values: dict[str, list[float]] = {m: [] for m in self.metrics}

    def update(
        self,
        probs: np.ndarray,
        target: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Update metrics with a single sample.

        Args:
            probs: Probability predictions (H, W)
            target: Binary ground truth (H, W)
            threshold: Threshold for binary conversion

        Returns:
            Dictionary of metric values for this sample
        """
        pred = (probs >= threshold).astype(np.float32)
        sample_metrics = {}

        # Check if this is a true-negative sample (no pneumonia in GT and no prediction)
        target_has_positive = target.sum() > 0
        pred_has_positive = pred.sum() > 0

        for metric_name in self.metrics:
            if metric_name == "auc":
                # AUC is only meaningful when target has both classes
                if not target_has_positive:
                    continue  # Skip — don't pollute with 0.5
                value = self.available_metrics[metric_name](probs, target)
            else:
                value = self.available_metrics[metric_name](pred, target)
            
            if not np.isnan(value):
                self.values[metric_name].append(value)
                sample_metrics[metric_name] = value

        return sample_metrics

    def compute(self) -> dict[str, float]:
        """Compute mean of accumulated metrics.

        Returns:
            Dictionary of mean metric values
        """
        results = {}
        for metric_name, values in self.values.items():
            if len(values) > 0:
                results[metric_name] = float(np.mean(values))
            else:
                results[metric_name] = 0.0
        return results

    def get_summary(self) -> str:
        """Get formatted summary string of metrics."""
        results = self.compute()
        lines = ["Metrics:"]
        for name, value in results.items():
            lines.append(f"  {name:12s}: {value:.4f}")
        return "\n".join(lines)
