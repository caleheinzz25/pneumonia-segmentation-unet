"""Training script with AMP, gradient accumulation, early stopping, and rich logging."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import Config, load_config
from src.dataset import RSNADataset, get_train_val_split
from src.losses import ComboLoss
from src.metrics import SegmentationMetrics
from src.model import build_model
from src.transforms import get_training_transforms, get_validation_transforms
from src.utils import overlay_mask, set_seed


def get_gpu_memory() -> str:
    """Get GPU memory usage string."""
    if not torch.cuda.is_available():
        return "N/A"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"{allocated:.1f}G/{reserved:.1f}G"


def get_lr(optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def log_sample_predictions(
    writer: SummaryWriter,
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    num_samples: int = 4,
) -> None:
    """Log sample predictions to TensorBoard."""
    model.eval()
    samples_logged = 0
    images_list = []
    masks_list = []
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"]

            logits = model(images)
            probs = torch.sigmoid(logits).cpu()

            for i in range(images.shape[0]):
                if samples_logged >= num_samples:
                    break
                # Denormalize image for visualization
                img = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = (img * std + mean).clamp(0, 1)

                images_list.append(img)
                masks_list.append(masks[i])
                preds_list.append(probs[i])
                samples_logged += 1

            if samples_logged >= num_samples:
                break

    # Create overlay visualization
    for i in range(len(images_list)):
        img = images_list[i].permute(1, 2, 0).numpy()
        mask = masks_list[i][0].numpy()
        pred = preds_list[i][0].numpy()
        pred_binary = (pred > 0.5).astype(np.float32)

        # GT overlay (green)
        gt_overlay = overlay_mask(img, mask, color=(0, 255, 0), alpha=0.4)
        # Pred overlay (red)
        pred_overlay = overlay_mask(img, pred_binary, color=(0, 0, 255), alpha=0.4)

        # Stack: original | GT | prediction | pred overlay
        combined = np.concatenate([
            (img * 255).astype(np.uint8),
            gt_overlay,
            (pred_overlay).astype(np.uint8),
        ], axis=1)

        writer.add_image(f"Samples/sample_{i}", combined.transpose(2, 0, 1) / 255.0, epoch)


def build_optimizer(model: nn.Module, config: Config):
    """Build optimizer based on config."""
    params = [p for p in model.parameters() if p.requires_grad]

    if config.optimizer.type == "adamw":
        return AdamW(params, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    elif config.optimizer.type == "adam":
        from torch.optim import Adam

        return Adam(params, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    elif config.optimizer.type == "sgd":
        return SGD(
            params, lr=config.optimizer.lr, momentum=0.9, weight_decay=config.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.type}")


def build_scheduler(optimizer, config: Config):
    """Build learning rate scheduler."""
    if config.scheduler.type == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.cosine_t_max,
            eta_min=config.scheduler.cosine_eta_min,
        )
    elif config.scheduler.type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.scheduler.reduce_factor,
            patience=config.scheduler.reduce_patience,
            min_lr=config.scheduler.reduce_min_lr,
            verbose=True,
        )
    else:
        return None


def _batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[float, int, int]:
    """Compute dice score, predicted positive pixels, and target positive pixels."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)
    return dice.item(), int(preds.sum().item()), int(targets.sum().item())


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    device: str,
    accumulation_steps: int,
    epoch: int,
    writer: SummaryWriter | None,
) -> tuple[float, float]:
    """Train for one epoch with gradient accumulation and AMP."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = len(dataloader)
    optimizer.zero_grad()

    pbar = tqdm(
        dataloader,
        desc=f"E{epoch:02d}[T]",
        dynamic_ncols=True,
        miniters=10,
    )
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, masks)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_loss = loss.item() * accumulation_steps
        total_loss += batch_loss

        # Compute running dice + pos pixel counts (no grad for speed)
        with torch.no_grad():
            batch_dice, pred_pos, target_pos = _batch_metrics(logits, masks)
        total_dice += batch_dice

        running_loss = total_loss / (batch_idx + 1)
        running_dice = total_dice / (batch_idx + 1)
        pbar.set_postfix_str(
            f"loss={running_loss:.4f} dice={running_dice:.4f} "
            f"pred_pos={pred_pos} target_pos={target_pos} lr={get_lr(optimizer):.2e}"
        )

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    if writer:
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Dice/train", avg_dice, epoch)
        writer.add_scalar("LR", get_lr(optimizer), epoch)

    return avg_loss, avg_dice


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    writer: SummaryWriter | None,
) -> tuple[float, dict[str, float]]:
    """Run validation and compute metrics."""
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics()

    pbar = tqdm(
        dataloader,
        desc=f"E{epoch:02d}[V]",
        dynamic_ncols=True,
        miniters=10,
    )
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = masks.cpu().numpy()

        for i in range(probs.shape[0]):
            metrics.update(probs[i, 0], targets[i, 0])

        current_results = metrics.compute()
        pbar.set_postfix_str(
            f"loss={total_loss / (pbar.n + 1):.4f} "
            f"dice={current_results['dice']:.4f} "
            f"iou={current_results['iou']:.4f}"
        )

    avg_loss = total_loss / len(dataloader)
    metric_results = metrics.compute()

    if writer:
        writer.add_scalar("Loss/val", avg_loss, epoch)
        for name, value in metric_results.items():
            writer.add_scalar(f"Metrics/{name}", value, epoch)

    return avg_loss, metric_results


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_dice: float,
    val_loss: float,
    val_metrics: dict[str, float],
    elapsed: float,
    lr: float,
    best_dice: float,
    epochs_no_improve: int,
    patience: int,
) -> None:
    """Print formatted epoch summary."""
    print()
    print("=" * 80)
    print(f"  Epoch {epoch}/{total_epochs} Summary")
    print("-" * 80)
    print(f"  Time       : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  LR         : {lr:.2e}")
    print(f"  GPU Memory : {get_gpu_memory()}")
    print()
    print(f"  {'Loss':<12s}  Train: {train_loss:.6f}  |  Val: {val_loss:.6f}")
    print(f"  {'Dice':<12s}  Train: {train_dice:.6f}  |  Val: {val_metrics['dice']:.6f}")
    print()
    print(f"  {'Metric':<12s}  {'Value':>10s}")
    print(f"  {'-'*12}  {'-'*10}")
    for name, value in val_metrics.items():
        print(f"  {name:<12s}  {value:>10.6f}")
    print()
    print(f"  Best Dice  : {best_dice:.6f}  (no improve: {epochs_no_improve}/{patience})")
    print("=" * 80)
    print()


def train(config: Config) -> None:
    """Main training function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{'=' * 80}")
    print(f"  PNEUMONIA SEGMENTATION - ATTENTION UNET++ TRAINING")
    print(f"{'=' * 80}")
    print(f"  Device     : {device}")
    if device == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Seed       : {config.training.seed}")
    print(f"  AMP        : {config.training.use_amp}")
    print(f"  Grad Accum : {config.training.accumulation_steps}")
    print(f"{'=' * 80}")
    print()

    set_seed(config.training.seed)

    # Prepare data splits
    train_ids, val_ids = get_train_val_split(
        data_config=config.data,
        val_split=config.training.val_split,
        seed=config.training.seed,
        stratified=config.training.stratified_split,
    )
    print(f"  Dataset Split: {len(train_ids):,} train / {len(val_ids):,} val patients")
    print()

    # Datasets and dataloaders
    train_transform = get_training_transforms(config.augmentation, config.preprocessing)
    val_transform = get_validation_transforms(config.preprocessing)

    train_dataset = RSNADataset(
        data_config=config.data,
        prep_config=config.preprocessing,
        patient_ids=train_ids,
        transform=train_transform,
        is_train=True,
    )
    val_dataset = RSNADataset(
        data_config=config.data,
        prep_config=config.preprocessing,
        patient_ids=val_ids,
        transform=val_transform,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    print(f"  Train batches: {len(train_loader):,} (batch_size={config.training.batch_size})")
    print(f"  Val batches  : {len(val_loader):,}")
    print()

    # Model
    model = build_model(config.model, device=device)

    # Freeze encoder if configured
    if config.training.encoder_freeze_epochs > 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(f"  Encoder frozen for first {config.training.encoder_freeze_epochs} epochs")
        print()

    # Loss, optimizer, scheduler
    criterion = ComboLoss(
        loss_type=config.loss.type,
        bce_weight=config.loss.bce_weight,
        tversky_alpha=config.loss.tversky_alpha,
        tversky_beta=config.loss.tversky_beta,
        tversky_gamma=config.loss.tversky_gamma,
        pos_weight=config.loss.pos_weight,
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.GradScaler(device=device, enabled=config.training.use_amp and device == "cuda")

    # Logging
    writer = SummaryWriter(log_dir=config.output.tensorboard_dir)
    best_dice = 0.0
    epochs_no_improve = 0

    print(f"  Starting training for up to {config.training.epochs} epochs...")
    print()

    # Training loop
    for epoch in range(1, config.training.epochs + 1):
        start_time = time.time()

        # Unfreeze encoder after warmup
        if epoch == config.training.encoder_freeze_epochs + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True
            print("  >>> Encoder unfrozen <<<")
            optimizer = build_optimizer(model, config)
            if scheduler:
                scheduler = build_scheduler(optimizer, config)

        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            config.training.accumulation_steps, epoch, writer,
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch, writer)

        val_dice = val_metrics["dice"]
        elapsed = time.time() - start_time
        current_lr = get_lr(optimizer)

        # Print detailed epoch summary
        print_epoch_summary(
            epoch, config.training.epochs, train_loss, train_dice, val_loss,
            val_metrics, elapsed, current_lr, best_dice, epochs_no_improve,
            config.training.early_stopping_patience,
        )

        # Log sample predictions to TensorBoard
        if writer and epoch % 5 == 1:  # Log every 5 epochs
            log_sample_predictions(writer, model, val_loader, device, epoch, num_samples=4)

        # Scheduler step
        if scheduler:
            if config.scheduler.type == "reduce_on_plateau":
                scheduler.step(val_dice)
            else:
                scheduler.step()

        # Checkpoint best model
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            checkpoint_path = Path(config.output.checkpoints_dir) / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": config,
            }, checkpoint_path)
            print(f"  [CHECKPOINT] Best model saved! (Dice: {best_dice:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  [INFO] No improvement ({epochs_no_improve}/{config.training.early_stopping_patience})")

        # Early stopping
        if epochs_no_improve >= config.training.early_stopping_patience:
            print()
            print(f"  [STOP] Early stopping triggered after {epoch} epochs")
            break

        # Save latest checkpoint
        latest_path = Path(config.output.checkpoints_dir) / "latest_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
        }, latest_path)
        print()

    writer.close()
    print()
    print(f"{'=' * 80}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best validation Dice: {best_dice:.6f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pneumonia Segmentation Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
