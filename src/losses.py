"""Loss functions for binary segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation.

    Skips true-negative samples (both prediction and target are all-zero)
    so the loss signal is not diluted by the majority of normal samples.
    False positives on normal images are penalized through the BCE component.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        batch_size = probs.shape[0]

        # Per-sample dice, skipping true negatives to avoid diluting the loss
        dice_sum = 0.0
        count = 0
        for i in range(batch_size):
            p = probs[i].view(-1)
            t = targets[i].view(-1)
            p_sum = p.sum()
            t_sum = t.sum()

            # Skip true-negative samples — both pred and target are empty
            # These are handled by BCE loss instead
            if p_sum < self.smooth and t_sum < self.smooth:
                continue

            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)
            dice_sum += dice
            count += 1

        if count == 0:
            # All samples in batch are true negatives → no dice loss to compute
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return 1.0 - dice_sum / count


class TverskyLoss(nn.Module):
    """Tversky loss with configurable alpha (false negative) and beta (false positive) weights."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky


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


class BCEWithLogitsLossWeighted(nn.Module):
    """BCE with logits loss with optional positive weighting."""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight_value = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight_value != 1.0:
            pos_weight = torch.tensor(
                [self.pos_weight_value],
                device=logits.device,
                dtype=logits.dtype,
            )
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        return F.binary_cross_entropy_with_logits(logits, targets)


class ComboLoss(nn.Module):
    """Combined loss: weighted sum of multiple losses.

    Supports combinations of:
    - BCEWithLogitsLoss
    - DiceLoss
    - FocalTverskyLoss
    """

    def __init__(
        self,
        loss_type: str = "focal_tversky",
        bce_weight: float = 0.5,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 0.75,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.bce_weight = bce_weight

        if loss_type == "dice_bce":
            self.dice = DiceLoss()
            self.bce = BCEWithLogitsLossWeighted(pos_weight=pos_weight)
        elif loss_type == "focal_tversky":
            self.focal_tversky = FocalTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=tversky_gamma,
            )
            self.bce = BCEWithLogitsLossWeighted(pos_weight=pos_weight)
        elif loss_type == "dice":
            self.dice = DiceLoss()
        elif loss_type == "bce":
            self.bce = BCEWithLogitsLossWeighted(pos_weight=pos_weight)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "dice_bce":
            dice_loss = self.dice(logits, targets)
            bce_loss = self.bce(logits, targets)
            return (1 - self.bce_weight) * dice_loss + self.bce_weight * bce_loss
        elif self.loss_type == "focal_tversky":
            ft_loss = self.focal_tversky(logits, targets)
            bce_loss = self.bce(logits, targets)
            return (1 - self.bce_weight) * ft_loss + self.bce_weight * bce_loss
        elif self.loss_type == "dice":
            return self.dice(logits, targets)
        elif self.loss_type == "bce":
            return self.bce(logits, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
