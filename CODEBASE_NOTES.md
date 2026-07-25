# Codebase Learning Notes — Pneumonia Segmentation U-Net

> **Project**: Undergraduate thesis (Skripsi) — Pneumonia detection & localization from Chest X-Rays using Attention U-Net with EfficientNet-B3 backbone.
> **Target metric**: Validation Dice ≥ 0.70
> **Dataset**: RSNA Pneumonia Detection Challenge (DICOM format)

---

## 📁 Project Structure

```
pneumonia-segmentation-unet/
├── config.yaml              # Master config (all hyperparams in one place)
├── app.py                   # Gradio app entrypoint (thin wrapper)
├── pyproject.toml           # Python deps & tool config (uv-managed)
├── Dockerfile               # Container: python:3.10-slim, exposes :7860
├── requirements.txt         # Pip fallback deps
├── PROJECT.md               # Thesis milestone tracker
├── src/                     # Core ML library
│   ├── config.py            # Typed dataclass config loader
│   ├── model.py             # SegmentationModel wrapper (SMP)
│   ├── dataset.py           # RSNADataset + train/val split logic
│   ├── transforms.py        # Albumentations pipelines
│   ├── losses.py            # All loss functions
│   ├── metrics.py           # SegmentationMetrics accumulator
│   ├── train.py             # Full training loop (AMP, grad accum, EMA, TensorBoard)
│   ├── evaluate.py          # Evaluation script
│   ├── predict.py           # Inference / TTA
│   ├── explainability.py    # Grad-CAM
│   ├── lung_segmentation.py # Lung mask precomputation (TorchXRayVision)
│   └── utils.py             # DICOM reading, mask utils, logging
├── app/
│   └── gradio_app.py        # Full Gradio UI (30KB, feature-rich)
├── scripts/
│   ├── run_train.sh         # Main train launcher (uv + tee log)
│   ├── run_resume_train.sh  # Resume from latest checkpoint
│   ├── run_train_fresh.sh   # Fresh train (deletes checkpoints first)
│   ├── precompute_lungmask.py   # Batch lung mask generation
│   └── precompute_all_masks.py  # Batch pneumonia mask generation
├── data/                    # Dataset root (gitignored)
├── outputs/                 # Model checkpoints, logs, TensorBoard
└── doc/                     # LaTeX thesis documents
```

---

## ⚙️ Configuration System (`config.yaml` + `src/config.py`)

The entire project is driven by a single `config.yaml`. All parameters are typed via Python dataclasses — no raw dicts escape into the rest of the code.

### Config Loading Flow

```
config.yaml (YAML)
    └─► yaml.safe_load() → raw dict
    └─► _to_dataclass(Config, raw) → recursive dict-to-dataclass conversion
    └─► Path validation (train dir, labels CSV must exist)
    └─► Output dir creation (outputs/, checkpoints/, logs/, tensorboard/)
    └─► returns Config
```

`_to_dataclass()` in `config.py` is a generic recursive converter — it walks a dataclass's `__dataclass_fields__` and maps YAML keys into the correct typed fields. Nested sections (e.g., `data:`, `model:`) become nested dataclass instances automatically.

### Config Sections

| Section | Dataclass | Key Purpose |
|---|---|---|
| `data` | `DataConfig` | All file/dir paths + `negative_ratio` sampling |
| `preprocessing` | `PreprocessingConfig` | Image size, CLAHE, windowing, normalization |
| `model` | `ModelConfig` | SMP architecture, encoder, decoder, attention |
| `training` | `TrainingConfig` | Epochs, batch, AMP, EMA, early stopping, freeze |
| `optimizer` | `OptimizerConfig` | Type, LR, weight_decay, `encoder_lr_factor` |
| `scheduler` | `SchedulerConfig` | Cosine / OneCycle / ReduceLROnPlateau params |
| `loss` | `LossConfig` | Loss type + all Focal/Tversky hyperparams |
| `augmentation` | `AugmentationConfig` | All Albumentations probabilities & limits |
| `inference` | `InferenceConfig` | Threshold, TTA transforms, overlay options |
| `evaluation` | `EvaluationConfig` | Output dir, metrics list, visualization count |
| `explainability` | `ExplainabilityConfig` | Grad-CAM layer, colormap |
| `app` | `AppConfig` | Gradio host/port/title |
| `output` | `OutputConfig` | All output dir paths |

---

## 🧠 Model (`src/model.py`)

- **Architecture**: U-Net via `segmentation-models-pytorch` (SMP)
- **Encoder**: `timm-efficientnet-b3`, pretrained on ImageNet
- **Decoder attention**: `scse` (Squeeze-and-Excitation: Spatial + Channel)
- **Decoder channels**: `[256, 128, 64, 32, 16]` (5 upsampling stages)
- **Output**: single-channel logit map (H, W) — sigmoid applied externally

### `SegmentationModel` class

```python
class SegmentationModel(nn.Module):
    self.model = smp.create_model(...)   # full UNet
    self.encoder = self.model.encoder    # exposed for Grad-CAM
```

- `forward(x)` → raw logits `(B, 1, H, W)`
- `get_encoder_features(x)` → list of intermediate feature maps
- `num_parameters` property → counts only trainable params

### Encoder Freeze Strategy

The encoder starts **frozen** for `encoder_freeze_epochs=12` epochs so the decoder warms up without disturbing pretrained ImageNet features. After epoch 12, the encoder is unfrozen and fine-tuned with `encoder_lr_factor=0.02` (50× slower than the decoder LR).

---

## 📊 Dataset (`src/dataset.py`)

### `RSNADataset`

Reads RSNA Pneumonia Challenge data (DICOM files + CSV labels).

**Item loading pipeline** (`__getitem__`):
1. Read DICOM → grayscale numpy array
2. Optionally apply HU windowing (lung window)
3. Build bbox mask from label CSV (fallback)
4. Resize image + mask to `[512, 512]`
5. Load **precomputed pneumonia mask** (PNG) — overrides bbox mask if exists
6. Load **lung segmentation mask** → zero-out non-lung image regions + mask GT to lung
7. Convert grayscale → RGB (3-channel duplicate for pretrained encoder)
8. Apply Albumentations transform (aug + normalize + ToTensorV2)

**Mask priority**: precomputed PNG > bbox-derived mask

### Train/Val Split (`get_train_val_split`)

- Patient-level split (no data leakage — one patient never in both sets)
- `stratified=True` → preserves positive/negative ratio via `sklearn.train_test_split`
- `negative_ratio=0.10` → keeps only 10% negatives: `N = P * (0.10 / 0.90)`
  - Focuses training on positive (pneumonia) cases

---

## 🔄 Transforms (`src/transforms.py`)

Uses **Albumentations** library. ImageNet mean/std are hardcoded constants.

### Training Pipeline

| Transform | Purpose |
|---|---|
| `HorizontalFlip(p=0.5)` | Standard augmentation |
| `VerticalFlip(p=0.05)` | Rare but valid for CXR |
| `Affine(translate, scale, rotate)` | Geometric shift/scale/rotation |
| `ElasticTransform(p=0.0)` | Anatomical deformation (disabled) |
| `RandomBrightnessContrast(p=0.4)` | X-ray exposure variation |
| `RandomGamma(p=0.3)` | Gamma correction variation |
| `CLAHE(p=0.3)` | Local contrast enhancement |
| `GaussNoise(p=0.2)` | Sensor noise simulation |
| `GridDropout(p=0.3)` | Forces learning local features |
| `CoarseDropout(p=0.2)` | Occlusion robustness (cutout) |
| `Normalize(ImageNet)` | Normalize to pretrained encoder stats |
| `ToTensorV2()` | numpy → torch tensor |

### Validation Pipeline

Only `Normalize` + `ToTensorV2` — no augmentation.

---

## 📉 Loss Functions (`src/losses.py`)

All losses are implemented **per-sample** (not per-batch) to avoid scaling issues with mixed positive/negative batches. True-negative samples (both pred and target all-zero) are **skipped** to avoid diluting the gradient signal.

### Available Loss Types

| Class | Formula | Use case |
|---|---|---|
| `DiceLoss` | `1 - 2*TP / (P_sum + T_sum)` | Overlap-based, class-imbalance robust |
| `IoULoss` | `1 - TP / (P + T - TP)` | Direct Jaccard optimization |
| `TverskyLoss` | `1 - TP / (TP + α*FN + β*FP)` | Configurable FP/FN penalty |
| `FocalTverskyLoss` | `TverskyLoss ^ γ` | Focus on hard examples |
| `FocalLoss` | `-α(1-p_t)^γ log(p_t)` | Suppresses easy background pixels |
| `UnifiedFocalLoss` | `(1-w)*FocalTversky + w*Focal` | **Currently used** — best for imbalance |
| `BCEWithLogitsLossWeighted` | BCE with pos_weight | Baseline |
| `ComboLoss` | Dispatcher for all above | Used in `train.py` |

### Current Loss Config (`unified_focal`)

```
UnifiedFocalLoss(
    focal_weight    = 0.5   # equal mix of Focal and FocalTversky
    tversky_alpha   = 0.3   # FN weight (lower = less recall penalty)
    tversky_beta    = 0.7   # FP weight (higher = strong precision pressure)
    tversky_gamma   = 0.75  # focal exponent for hard examples
    focal_alpha     = 0.25  # class balance factor in FocalLoss
    focal_gamma     = 2.0   # focusing factor
)
```

**Design reasoning**: Precision was the bottleneck (0.57 vs 0.70 recall), so `tversky_beta=0.7` penalizes FP 2.3× more than FN. Focal Loss additionally suppresses easy-to-classify background pixels.

---

## 🏋️ Training Loop (`src/train.py`)

### Key Features

- **Mixed Precision (AMP)**: `torch.amp.autocast` + `torch.GradScaler`
- **Gradient Accumulation**: `accumulation_steps=4` → effective batch = 32
- **Gradient Clipping**: `max_norm=1.0` applied before each optimizer step
- **Encoder Freeze**: First 12 epochs encoder frozen; then unfrozen
- **Differential LR**: Encoder gets `lr * encoder_lr_factor = 4e-4 * 0.02 = 8e-6`
- **EMA**: Exponential moving average of weights (`decay=0.999`)
- **TensorBoard**: Loss, Dice, LR, sample predictions logged every epoch

### Optimizer

`AdamW` with two param groups:
- Encoder: `lr=8e-6`, `weight_decay=5e-4`
- Decoder: `lr=4e-4`, `weight_decay=5e-4`

### Scheduler

`CosineAnnealingLR(T_max=100, eta_min=1e-6)` — smooth monotonic decay from max LR to min LR over all 100 epochs. Steps once per epoch.

### Early Stopping

If validation Dice doesn't improve for `patience=35` consecutive epochs, training stops. Best model is saved immediately when a new best Dice is achieved.

### Checkpointing

Two checkpoint files maintained:
- `best_model.pth` — highest validation Dice
- `latest_model.pth` — last completed epoch (for resuming)

Both contain: `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `best_dice`, `epochs_no_improve`.

### Resume Logic

When `--resume` is passed:
1. Loads model weights (strict → falls back to non-strict if architecture changed)
2. Determines `start_epoch` from checkpoint
3. Sets correct encoder freeze state based on `start_epoch`
4. Builds optimizer/scheduler AFTER freeze state is set (param group count must match)
5. Restores optimizer, scheduler, and scaler states

---

## 📏 Metrics (`src/metrics.py`)

`SegmentationMetrics` accumulates per-sample scores then averages.

| Metric | Notes |
|---|---|
| `dice` | Skips TN samples (both pred + GT are zero) |
| `iou` | Jaccard, skips TN samples |
| `precision` | Returns `nan` if no positive predictions |
| `recall` | Returns `nan` if no positive GT |
| `accuracy` | Pixel-wise, all samples |
| `specificity` | True negative rate |
| `auc` | ROC-AUC using raw probabilities; skips all-negative samples |

`nan` values are excluded from averaging — not counted as 0.

---

## 🖼️ Inference Details (`config.yaml`)

| Setting | Value | Rationale |
|---|---|---|
| `threshold` | 0.65 | Raised from 0.5 → filters low-confidence FP pixels |
| `use_tta` | true | Test-Time Augmentation for robustness |
| `tta_transforms` | `[null, hflip, vflip]` | 3× inference, merged by mean |

---

## 🐳 Deployment

- **Docker**: `python:3.10-slim` base, installs from `requirements.txt`, exposes `:7860`
- **Entry**: `python app.py` → loads Gradio app
- **Gradio UI**: Accepts DICOM or standard image, runs full inference + Grad-CAM overlay
- **Cloudflare**: `cloudflared_config.json` suggests tunneled public access

---

## 🔁 Config Evolution (v1 → v2)

| Parameter | v1 | v2 | Reason |
|---|---|---|---|
| `encoder_lr_factor` | 0.1 | 0.02 | Prevent catastrophic forgetting of ImageNet features |
| `loss` | `dice_bce_iou` | `unified_focal` | Better FP suppression |
| `scheduler` | `one_cycle` | `cosine_annealing` | Smoother decay, no destabilizing warmup spike |
| `weight_decay` | 1e-4 | 5e-4 | Stronger L2 regularization against overfitting |
| `negative_ratio` | 0.30 | 0.10 | Cleaner positive learning signal |
| `accumulation_steps` | 8 | 4 | More optimizer updates per epoch (eff. batch 16→32) |
| `encoder_freeze_epochs` | 8 | 12 | Longer decoder warmup |
| `lr` | 2e-4 | 4e-4 | Scaled up for larger effective batch size |
| `decoder_dropout` | 0.3 | 0.2 | More decoder capacity |
| `ema_decay` | 0.9995 | 0.999 | Aligned with 2× fewer steps/epoch |
| `tversky_alpha` | 0.4 | 0.3 | Stronger FP penalty (precision was bottleneck) |
| `tversky_beta` | 0.6 | 0.7 | Push precision higher |
| `threshold` | 0.5 | 0.65 | Filter low-confidence FP pixels at inference |
| `rotation_limit` | 15 | 10 | Preserve anatomical realism |
| `coarse_dropout_prob` | 0.15 | 0.10 | Less training noise |

---

## 🔑 Key Design Decisions & Rationale

1. **Per-sample loss computation**: Avoids batch-composition-dependent gradient magnitudes. A batch with 7 negatives + 1 positive won't drown out the positive.

2. **Lung mask as input preprocessing**: Zero-out non-lung pixels before feeding to model. This reduces irrelevant context and also masks GT to only the lung region.

3. **Precomputed mask priority**: If a PNG segmentation mask exists, it overrides the bbox-derived mask. Allows refined/corrected masks without changing the CSV.

4. **Encoder freeze then unfreeze**: Standard transfer learning trick — decoder learns to decode ImageNet features first, then encoder is fine-tuned gently with a very low LR.

5. **Stratified patient-level split**: Ensures no patient appears in both train and val sets (avoids leakage), and that both sets have similar positive/negative ratios.

6. **Differential LR ratio (0.02)**: 50× slower encoder vs. decoder. Standard practice for fine-tuning pretrained networks in medical imaging.

7. **EMA at decay=0.999**: With ~100 steps/epoch (batch=8, ~800 train samples), gives an effective averaging window of ~1000 steps ≈ 10 epochs. Stabilizes the validation model.

---

## 🛠️ Development Toolchain

| Tool | Role |
|---|---|
| `uv` | Fast Python package manager (workspace-aware) |
| `ruff` | Linter, `line-length=100`, target `py310` |
| `black` | Formatter, `line-length=100`, target `py310` |
| `pytest` | Testing (minimal test coverage currently) |
| TensorBoard | Live training visualization |
| PlantUML | Thesis methodology diagrams |
| LaTeX | Thesis document compilation |

---

## 📌 Quick Reference: Running the Project

```bash
# Fresh training (from scratch)
bash scripts/run_train_fresh.sh

# Training with logging
bash scripts/run_train.sh

# Resume from latest checkpoint
bash scripts/run_resume_train.sh

# Precompute lung masks (needed before training)
uv run python scripts/precompute_lungmask.py

# Run Gradio app
uv run python app.py

# Evaluate a checkpoint
bash scripts/evaluate.sh
```
