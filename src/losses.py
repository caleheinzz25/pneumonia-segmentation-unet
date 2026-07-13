"""Loss functions for binary segmentation with class imbalance handling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Per-sample Dice loss for binary segmentation.

    Skips true-negative samples (both prediction and target are all-zero)
    so the loss signal is not diluted by the majority of normal samples.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # (B, C, H, W) -> (B, C*H*W)
        p = probs.view(probs.shape[0], -1)
        t = targets.view(targets.shape[0], -1)

        intersection = (p * t).sum(dim=1)
        p_sum = p.sum(dim=1)
        t_sum = t.sum(dim=1)

        dice_per_sample = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)

        non_tn_mask = (p_sum >= self.smooth) | (t_sum >= self.smooth)
        if non_tn_mask.sum() == 0:
            return logits.sum() * 0.0

        return 1.0 - dice_per_sample[non_tn_mask].mean()


class TverskyLoss(nn.Module):
    """Per-sample Tversky loss with configurable alpha (FN) and beta (FP) weights."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Per-sample computation to avoid batch-size-dependent scaling
        p = probs.view(probs.shape[0], -1)
        t = targets.view(targets.shape[0], -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)

        tversky_per_sample = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        # Skip true-negative samples
        non_tn_mask = (p.sum(dim=1) >= self.smooth) | (t.sum(dim=1) >= self.smooth)
        if non_tn_mask.sum() == 0:
            return logits.sum() * 0.0

        return 1.0 - tversky_per_sample[non_tn_mask].mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss: (1 - Tversky)^gamma for focusing on hard examples."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky_loss = self.tversky(logits, targets)
        return torch.pow(tversky_loss, self.gamma)


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation — focuses on hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class UnifiedFocalLoss(nn.Module):
    """Unified Focal Loss: combines Focal loss + Focal Tversky loss.

    Designed for severe class imbalance in medical image segmentation.
    Reference: "Unified Focal Loss: Generalising Dice and cross entropy-based
    losses to handle class imbalanced medical image segmentation" (Yeung et al.)

    Uses per-sample Tversky computation for stable gradients regardless of
    batch composition.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        tversky_alpha: float = 0.6,
        tversky_beta: float = 0.4,
        tversky_gamma: float = 0.75,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.focal_tversky = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            smooth=smooth,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ft_loss = self.focal_tversky(logits, targets)
        focal_loss = self.focal(logits, targets)
        return (1.0 - self.focal_weight) * ft_loss + self.focal_weight * focal_loss


class BCEWithLogitsLossWeighted(nn.Module):
    """BCE with logits loss with optional positive weighting."""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight_value = pos_weight
        self.register_buffer("_pos_weight", torch.tensor([pos_weight]), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight_value != 1.0:
            if self._pos_weight.device != logits.device:
                self._pos_weight = self._pos_weight.to(
                    device=logits.device, dtype=logits.dtype
                )
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self._pos_weight
            )
        return F.binary_cross_entropy_with_logits(logits, targets)


class IoULoss(nn.Module):
    """Per-sample IoU (Jaccard) loss for binary segmentation.

    Skips true-negative samples so the loss is not dominated by normal images.
    Directly optimizes the IoU metric used for evaluation.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = probs.view(probs.shape[0], -1)
        t = targets.view(targets.shape[0], -1)

        intersection = (p * t).sum(dim=1)
        p_sum = p.sum(dim=1)
        t_sum = t.sum(dim=1)

        iou_per_sample = (intersection + self.smooth) / (
            p_sum + t_sum - intersection + self.smooth
        )

        non_tn_mask = (p_sum >= self.smooth) | (t_sum >= self.smooth)
        if non_tn_mask.sum() == 0:
            return logits.sum() * 0.0

        return 1.0 - iou_per_sample[non_tn_mask].mean()


class ComboLoss(nn.Module):
    """Combined loss: weighted sum of multiple losses.

    Supports: dice_bce, dice_bce_iou, focal_tversky, unified_focal, dice, bce
    """

    def __init__(
        self,
        loss_type: str = "unified_focal",
        bce_weight: float = 0.5,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 0.75,
        pos_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.bce_weight = bce_weight

        if loss_type == "dice_bce":
            self.dice = DiceLoss()
            self.bce = BCEWithLogitsLossWeighted(pos_weight=pos_weight)
        elif loss_type == "dice_bce_iou":
            self.dice = DiceLoss()
            self.iou = IoULoss()
            self.bce = BCEWithLogitsLossWeighted(pos_weight=pos_weight)
        elif loss_type == "focal_tversky":
            self.focal_tversky = FocalTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=tversky_gamma,
            )
            self.bce = BCEWithLogitsLossWeighted(pos_weight=pos_weight)
        elif loss_type == "unified_focal":
            self.unified_focal = UnifiedFocalLoss(
                focal_weight=bce_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_gamma=tversky_gamma,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
        elif loss_type == "dice":
            self.dice = DiceLoss()
        elif loss_type == "bce":
            self.bce = BCEWithLogitsLossWeighted(pos_weight=pos_weight)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "dice_bce":
            return (1 - self.bce_weight) * self.dice(logits, targets) + \
                   self.bce_weight * self.bce(logits, targets)
        elif self.loss_type == "dice_bce_iou":
            # Split remaining weight 60% Dice / 40% IoU after BCE portion
            other_weight = 1.0 - self.bce_weight
            dice_w = 0.6 * other_weight
            iou_w = 0.4 * other_weight
            return dice_w * self.dice(logits, targets) + \
                   iou_w * self.iou(logits, targets) + \
                   self.bce_weight * self.bce(logits, targets)
        elif self.loss_type == "focal_tversky":
            return (1 - self.bce_weight) * self.focal_tversky(logits, targets) + \
                   self.bce_weight * self.bce(logits, targets)
        elif self.loss_type == "unified_focal":
            return self.unified_focal(logits, targets)
        elif self.loss_type == "dice":
            return self.dice(logits, targets)
        elif self.loss_type == "bce":
            return self.bce(logits, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")



