"""Training script with AMP, gradient accumulation, early stopping, and rich logging."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import Config, load_config
from src.dataset import RSNADataset, get_train_val_split
from src.losses import ComboLoss
from src.metrics import SegmentationMetrics
from src.model import build_model
from src.transforms import get_training_transforms, get_validation_transforms
from src.utils import overlay_mask, set_seed, setup_logging


def get_gpu_memory() -> str:
    """Get GPU memory usage string."""
    if not torch.cuda.is_available():
        return "N/A"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"{allocated:.1f}G/{reserved:.1f}G"


def get_lr(optimizer) -> float:
    """Get current learning rate from optimizer."""
    if optimizer.param_groups:
        return optimizer.param_groups[-1]["lr"]
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
    """Build optimizer with differential LR: lower LR for pretrained encoder, higher for decoder."""
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    # Include both param groups regardless of requires_grad, so optimizer tracks them.
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config.optimizer.lr * 0.1})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": config.optimizer.lr})

    if not param_groups:
        raise ValueError("No parameters found to optimize.")

    if config.optimizer.type == "adamw":
        return AdamW(param_groups, weight_decay=config.optimizer.weight_decay)
    elif config.optimizer.type == "adam":
        from torch.optim import Adam

        return Adam(param_groups, weight_decay=config.optimizer.weight_decay)
    elif config.optimizer.type == "sgd":
        return SGD(param_groups, momentum=0.9, weight_decay=config.optimizer.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.type}")


def build_scheduler(optimizer, config: Config, steps_per_epoch: int = 1):
    """Build learning rate scheduler."""
    if config.scheduler.type == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.cosine_t_max,
            eta_min=config.scheduler.cosine_eta_min,
        )
    elif config.scheduler.type == "one_cycle":
        # steps_per_epoch = actual optimizer steps, not batch count
        # With grad accumulation, optimizer steps = ceil(num_batches / accum_steps)
        max_lrs = [config.optimizer.lr * 0.1, config.optimizer.lr]
        # If only one param group (e.g. encoder frozen), use single max_lr
        if len(optimizer.param_groups) == 1:
            max_lrs = [optimizer.param_groups[0]["lr"]]
        return OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            epochs=config.training.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 10% warmup
            anneal_strategy="cos",
            div_factor=10.0,  # initial_lr = max_lr / 10
            final_div_factor=100.0,  # final_lr = initial_lr / 100
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
    """Compute per-sample dice score (excluding true negatives), pred/target positive pixel counts."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    batch_size = logits.shape[0]
    total_pred_pos = int(preds.sum().item())
    total_target_pos = int(targets.sum().item())

    # Per-sample dice, skipping true negatives (both pred and target all-zero)
    dice_scores = []
    for i in range(batch_size):
        p = preds[i].view(-1)
        t = targets[i].view(-1)
        p_sum = p.sum()
        t_sum = t.sum()
        # True-negative samples get a perfect dice score of 1.0
        if p_sum == 0 and t_sum == 0:
            dice_scores.append(1.0)
            continue
        intersection = (p * t).sum()
        dice = (2.0 * intersection + 1e-6) / (p_sum + t_sum + 1e-6)
        dice_scores.append(dice.item())

    avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    return avg_dice, total_pred_pos, total_target_pos


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
    scheduler=None,
    scheduler_type: str = "",
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

            # Step OneCycleLR per optimizer step (not per epoch)
            if scheduler and scheduler_type == "one_cycle":
                scheduler.step()

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


def train(config: Config, resume_path: str | None = None) -> None:
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
        persistent_workers=True if config.training.num_workers > 0 else False,
        prefetch_factor=4 if config.training.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=True if config.training.num_workers > 0 else False,
        prefetch_factor=4 if config.training.num_workers > 0 else None,
    )

    print(f"  Train batches: {len(train_loader):,} (batch_size={config.training.batch_size})")
    print(f"  Val batches  : {len(val_loader):,}")
    print()

    # Model
    model = build_model(config.model, device=device)

    # Loss function (independent of optimizer)
    criterion = ComboLoss(
        loss_type=config.loss.type,
        bce_weight=config.loss.bce_weight,
        tversky_alpha=config.loss.tversky_alpha,
        tversky_beta=config.loss.tversky_beta,
        tversky_gamma=config.loss.tversky_gamma,
        pos_weight=config.loss.pos_weight,
    )

    # Logging
    best_dice = 0.0
    epochs_no_improve = 0
    start_epoch = 1

    # Resume from checkpoint — load model weights and determine start_epoch FIRST
    # so we know the correct encoder freeze state before building optimizer
    checkpoint = None
    if resume_path:
        checkpoint_file = Path(resume_path)
        if checkpoint_file.exists():
            print(f"  [RESUME] Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_dice = checkpoint.get("best_dice", 0.0)
            epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
            print(f"  [RESUME] Resuming from epoch {start_epoch} (best dice: {best_dice:.6f})")
        else:
            print(f"  [WARNING] Checkpoint not found: {checkpoint_file}, starting from scratch")
            print()

    # Set encoder freeze state based on where we are in training
    if start_epoch <= config.training.encoder_freeze_epochs:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(f"  Encoder frozen for epochs 1-{config.training.encoder_freeze_epochs}")
    else:
        for param in model.encoder.parameters():
            param.requires_grad = True
        if checkpoint is not None:
            print(f"  [RESUME] Encoder unfrozen (epoch {start_epoch} > {config.training.encoder_freeze_epochs})")
    print()

    # Build optimizer and scheduler AFTER correct freeze state is set
    # This ensures param group count matches what the checkpoint expects
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / config.training.accumulation_steps)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=optimizer_steps_per_epoch)
    scaler = torch.GradScaler(device=device, enabled=config.training.use_amp and device == "cuda")

    # Now load optimizer/scheduler/scaler state from checkpoint
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("  [RESUME] Optimizer state restored successfully")
        except (ValueError, KeyError) as e:
            print(f"  [WARNING] Could not load optimizer state dict: {e}")
            print(f"  [WARNING] Proceeding with freshly initialized optimizer.")
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("  [RESUME] Scheduler state restored successfully")
            except (ValueError, KeyError):
                print("  [WARNING] Could not load scheduler state dict, using fresh scheduler.")
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print()

    # TensorBoard writer — use purge_step on resume so graphs continue seamlessly
    writer = SummaryWriter(
        log_dir=config.output.tensorboard_dir,
        purge_step=start_epoch if resume_path else None,
    )

    print(f"  Starting training for up to {config.training.epochs} epochs...")
    print()

    # Training loop
    for epoch in range(start_epoch, config.training.epochs + 1):
        start_time = time.time()

        # Unfreeze encoder after warmup
        if epoch == config.training.encoder_freeze_epochs + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True
            print("  >>> Encoder unfrozen <<<")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            config.training.accumulation_steps, epoch, writer,
            scheduler=scheduler, scheduler_type=config.scheduler.type,
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

        # Scheduler step (OneCycleLR steps per-batch inside train_one_epoch)
        if scheduler and config.scheduler.type != "one_cycle":
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
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict(),
                "best_dice": best_dice,
                "epochs_no_improve": epochs_no_improve,
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

        # Save latest checkpoint (full state for resume)
        latest_path = Path(config.output.checkpoints_dir) / "latest_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict(),
            "best_dice": best_dice,
            "epochs_no_improve": epochs_no_improve,
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
    parser.add_argument(
        "--resume", type=str, default=None, nargs="?", const="outputs/checkpoints/latest_model.pth",
        help="Resume training from checkpoint. Default: outputs/checkpoints/latest_model.pth",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    is_resume = args.resume is not None
    setup_logging(logs_dir=config.output.logs_dir, run_name="train", resume=is_resume)
    train(config, resume_path=args.resume)
