# Segmentasi Citra Radiografi Dada untuk Pendeteksian Area Infeksi Pneumonia Menggunakan Model Deep Learning Attention UNet++

Proyek ini mengimplementasikan pipeline deep learning untuk deteksi dan segmentasi otomatis pneumonia dari citra X-ray dada (CXR) menggunakan arsitektur **Attention UNet++** dengan encoder **EfficientNet-B4** pretrained.

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Attention UNet++** | Nested skip connections dengan SCSE (Spatial & Channel Squeeze-and-Excitation) attention gates |
| **Pretrained Encoder** | ImageNet-pretrained EfficientNet-B4 backbone dengan differential learning rate |
| **Dual Mask Approach** | Lung segmentation mask untuk masking input + precomputed pneumonia mask untuk ground truth |
| **Auto Lung Segmentation** | Fallback otomatis menggunakan torchxrayvision PSPNet saat precomputed lung mask tidak tersedia |
| **AMP Training** | Automatic Mixed Precision untuk training lebih cepat |
| **Dice+BCE Loss** | Per-sample Dice Loss + weighted BCE untuk optimasi langsung metrik segmentasi |
| **OneCycleLR** | Super-convergence scheduler dengan built-in warmup |
| **Resume Training** | Lanjutkan training dari checkpoint terakhir dengan `--resume`, log dilanjutkan di file yang sama |
| **Comprehensive Metrics** | Per-sample Dice, IoU, Precision, Recall, Specificity, AUC (negatif-aware) |
| **Medical Augmentation** | CLAHE, GaussNoise, CoarseDropout, Elastic, Affine |
| **Grad-CAM** | Visualisasi area yang "diperhatikan" model |
| **Gradio Web App** | Interface interaktif untuk upload dan analisis X-ray |

## Arsitektur Model

```
Input (3, 512, 512)
    |
EfficientNet-B4 Encoder (pretrained ImageNet)
    |-- Stage 1: 64x64 features
    |-- Stage 2: 32x32 features
    |-- Stage 3: 16x16 features
    |-- Stage 4: 8x8 features
    |-- Stage 5: 4x4 features
    |
UNet++ Decoder with SCSE Attention Gates
    |-- Nested skip connections (dense connections)
    |-- SCSE: Spatial + Channel attention per decoder block
    |
Output (1, 512, 512) - Sigmoid probability map
```

### Training Strategy

```
Epoch 1-2: Encoder frozen, decoder-only training (LR = 3e-4)
Epoch 3+:  Encoder unfrozen, differential LR
             ├── Encoder: 3e-5 (10x lower, preserve ImageNet features)
             └── Decoder: 3e-4 (full LR)

Scheduler: OneCycleLR
             ├── Warmup: 10% of total steps
             ├── Peak: max_lr per param group
             └── Annealing: cosine decay to final_lr

Loss: 0.7 × DiceLoss (per-sample) + 0.3 × BCE (pos_weight=2.0)
```

## Struktur Project

```
.
├── config.yaml              # File konfigurasi sentral
├── pyproject.toml           # UV package management
├── README.md                # Dokumentasi ini
├── src/
│   ├── config.py            # YAML parser → dataclass
│   ├── dataset.py           # RSNA dataset: DICOM reader, bbox→mask, precomputed masks
│   ├── transforms.py        # Albumentations: augmentasi + normalisasi
│   ├── model.py             # Attention UNet++ (SMP wrapper)
│   ├── losses.py            # Per-sample Dice, Tversky, BCE, ComboLoss
│   ├── metrics.py           # Per-sample Dice, IoU, Precision, Recall, Specificity, AUC
│   ├── train.py             # Training loop: AMP, gradient accumulation, resume, early stopping
│   ├── evaluate.py          # Evaluasi + visualisasi overlay
│   ├── predict.py           # Inference: single image / batch + auto lung masking
│   ├── lung_segmentation.py # Auto lung segmentation (torchxrayvision PSPNet)
│   ├── explainability.py    # Grad-CAM heatmap generation
│   └── utils.py             # DICOM read, windowing, overlay, seeding, logging (JSON state)
├── app/
│   └── gradio_app.py        # Web interface (Gradio)
├── scripts/
│   ├── train.sh             # Script training
│   ├── evaluate.sh          # Script evaluasi
│   ├── predict.sh           # Script inference
│   ├── app.sh               # Script launch web app
│   └── precompute_lungmask.py  # Precompute lung mask batch (GPU-accelerated)
├── data/
│   ├── rsna-pneumonia-detection-challenge/
│   │   ├── stage_2_train_images/       # DICOM training images (~26K)
│   │   ├── stage_2_test_images/        # DICOM test images
│   │   ├── stage_2_train_labels.csv    # Bounding box annotations
│   │   └── stage_2_detailed_class_info.csv
│   └── lung_masks/
│       ├── lung_segmentation/          # Lung mask: masking input image
│       ├── pneumonia_ground_truth/     # Precomputed pneumonia mask: ground truth
│       ├── test/                       # Lung mask untuk test/inference images
│       ├── combined/                   # Visualisasi: lung + pneumonia overlay
│       └── visualizations/            # Visualisasi hasil
└── outputs/
    ├── checkpoints/         # best_model.pth & latest_model.pth
    ├── logs/                # Terminal logs + JSON state files
    ├── tensorboard/         # TensorBoard event files
    ├── predictions/         # Hasil inference
    ├── evaluation/          # Metrics JSON + visualisasi
    └── gradcam/             # Grad-CAM heatmaps
```

## Dataset

### RSNA Pneumonia Detection Challenge

Dataset terdiri dari ~26,000 citra X-ray dada dalam format DICOM dengan anotasi bounding box untuk area pneumonia.

### Precomputed Masks (`data/lung_masks/`)

| Folder | Isi | Penggunaan |
|--------|-----|------------|
| `lung_segmentation/` | Binary mask paru-paru per pasien | Masking input image (pixel non-paru = 0) |
| `pneumonia_ground_truth/` | Binary mask pneumonia per pasien | Ground truth training (lebih akurat dari bbox→mask) |
| `test/` | Binary mask paru-paru untuk test images | Masking input saat inference |
| `combined/` | Overlay lung + pneumonia | Referensi visualisasi |
| `visualizations/` | Hasil visualisasi | Referensi visualisasi |

**Dual Mask Approach**: Dataset class akan otomatis menggunakan precomputed pneumonia mask sebagai ground truth (override bbox→mask), dan lung segmentation mask untuk membatasi input hanya pada area paru-paru.

### Auto Lung Segmentation (Fallback)

Saat inference, jika precomputed lung mask tidak tersedia untuk suatu gambar, sistem secara otomatis menggunakan model **torchxrayvision PSPNet** untuk melakukan segmentasi paru-paru secara *on-the-fly*. Model ini mendeteksi 14 struktur anatomi dan mengekstrak mask Left Lung + Right Lung menjadi binary mask gabungan.

Modul ini digunakan di:
- `src/predict.py` — Inference via CLI
- `app/gradio_app.py` — Inference via Web App
- `src/explainability.py` — Grad-CAM generation

## Setup

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (recommended, min 8GB VRAM)
- [UV](https://docs.astral.sh/uv/) package manager

### Install Dependencies

```bash
# Install dengan UV
uv sync

# Aktifkan virtual environment
source .venv/bin/activate
```

## Penggunaan

### Precompute Lung Masks

Sebelum training, generate lung mask untuk seluruh dataset menggunakan PSPNet (GPU-accelerated batch processing):

```bash
# Lung mask untuk training data
uv run python -m scripts.precompute_lungmask \
  --input data/rsna-pneumonia-detection-challenge/stage_2_train_images/ \
  --output data/lung_masks/lung_segmentation/

# Lung mask untuk test data
uv run python -m scripts.precompute_lungmask \
  --input data/rsna-pneumonia-detection-challenge/stage_2_test_images/ \
  --output data/lung_masks/test/

# Dengan batch processing dan visualisasi
uv run python -m scripts.precompute_lungmask \
  --input data/new_images/ --batch-size 16 --visualize
```

Script ini mendukung resume otomatis (skip gambar yang sudah diproses).

### Training

```bash
# Training dari awal
uv run python -m src.train --config config.yaml

# Via script
./scripts/train.sh
```

Training akan otomatis:
- Load precomputed pneumonia masks sebagai ground truth
- Apply lung masking pada input
- Freeze encoder selama 2 epoch pertama (warmup)
- Menggunakan differential LR (encoder 3e-5, decoder 3e-4)
- OneCycleLR scheduler dengan 10% warmup
- Menggunakan AMP (jika GPU tersedia)
- Save best model ke `outputs/checkpoints/best_model.pth`
- Save latest model ke `outputs/checkpoints/latest_model.pth` (untuk resume)
- Log metrics ke TensorBoard di `outputs/tensorboard/`
- Log terminal output ke `outputs/logs/train_*.log`

### Checkpoint Strategy

Training menyimpan dua checkpoint terpisah dengan fungsi berbeda:

| Checkpoint | Kapan Disimpan | Fungsi |
|------------|----------------|--------|
| `best_model.pth` | Hanya saat val Dice meningkat | Evaluasi & inference (performa terbaik) |
| `latest_model.pth` | Setiap akhir epoch | Resume training (state paling mutakhir) |

> **Catatan**: Jika epoch terakhir juga merupakan best model, kedua file akan berisi model yang sama. Pemisahan ini memastikan `best_model.pth` tidak pernah tertimpa oleh epoch dengan performa lebih rendah.

### Resume Training

Jika training terputus (crash, listrik mati, dll), lanjutkan dari checkpoint terakhir:

```bash
# Resume dari latest_model.pth (default)
uv run python -m src.train --config config.yaml --resume

# Resume dari checkpoint spesifik
uv run python -m src.train --config config.yaml --resume outputs/checkpoints/best_model.pth
```

Resume akan merestore:
- Model weights & optimizer state
- Learning rate scheduler state
- AMP scaler state
- Epoch counter & best dice score
- Early stopping counter
- TensorBoard graph disambung tanpa putus

#### Log Tracking (JSON State)

Saat pertama kali training dimulai, sistem akan membuat file log baru dan mencatat path-nya di `outputs/logs/train_state.json`. Ketika `--resume` dijalankan, sistem membaca file JSON ini untuk mengetahui secara pasti file log mana yang harus dilanjutkan — sehingga semua output training (termasuk setelah beberapa kali resume) tetap berada dalam **satu file log yang sama**.

```
outputs/logs/
├── train_20260610_143022.log    ← Satu file log untuk seluruh sesi training
└── train_state.json             ← State tracker: {"current_log": "..."}
```

### Evaluasi

```bash
./scripts/evaluate.sh
```

Output:
- `outputs/evaluation/metrics.json` — Semua metrics (Dice, IoU, dll)
- `outputs/evaluation/evaluation_samples.png` — Visualisasi 16 sample random

### Inference

Single image:
```bash
./scripts/predict.sh path/to/xray.dcm
```

Batch directory:
```bash
uv run python -m src.predict --config config.yaml \
  --input path/to/images/ \
  --output outputs/predictions
```

Output per image:
- `{name}_pred.png` — Binary prediction mask
- `{name}_overlay.png` — Overlay merah pada area pneumonia

> **Auto Lung Masking**: Saat inference, jika precomputed lung mask tidak tersedia, model torchxrayvision PSPNet akan otomatis melakukan segmentasi paru-paru secara *on-the-fly* sebelum prediksi pneumonia.

### Grad-CAM Explainability

```bash
uv run python -m src.explainability \
  --config config.yaml \
  --input path/to/xray.dcm \
  --output outputs/gradcam
```

Output:
- `{name}_gradcam.png` — Overlay Grad-CAM pada citra
- `{name}_heatmap.png` — Heatmap raw

### Web App (Gradio)

```bash
./scripts/app.sh
```

Buka browser: http://localhost:7860

## Konfigurasi

File `config.yaml` mengatur seluruh pipeline. Key configurations:

| Section | Key | Default | Deskripsi |
|---------|-----|---------|-----------|
| `model` | `architecture` | UnetPlusPlus | Arsitektur model |
| `model` | `encoder_name` | timm-efficientnet-b4 | Backbone encoder |
| `model` | `decoder_attention_type` | scse | Attention: Spatial+Channel SE |
| `preprocessing` | `image_size` | [512, 512] | Target resize |
| `training` | `batch_size` | 4 | Batch size |
| `training` | `accumulation_steps` | 8 | Gradient accumulation (effective batch = 32) |
| `training` | `use_amp` | true | Mixed precision training |
| `training` | `encoder_freeze_epochs` | 2 | Epoch freeze encoder (warmup) |
| `training` | `early_stopping_patience` | 20 | Early stopping patience |
| `loss` | `type` | dice_bce | Loss function (per-sample Dice + BCE) |
| `loss` | `bce_weight` | 0.3 | Weight BCE dalam combo loss |
| `loss` | `pos_weight` | 2.0 | Weight untuk class pneumonia |
| `optimizer` | `type` | adamw | Optimizer |
| `optimizer` | `lr` | 3e-4 | Learning rate (decoder); encoder = lr × 0.1 |
| `optimizer` | `weight_decay` | 7e-4 | L2 regularization |
| `scheduler` | `type` | one_cycle | LR scheduler dengan warmup |

### Data Augmentation

| Augmentation | Parameter | Deskripsi |
|--------------|-----------|-----------|
| HorizontalFlip | p=0.5 | Flip horizontal |
| Affine | shift=0.1, scale=0.15, rotate=±15° | Geometric transforms |
| ElasticTransform | α=120, σ=6, p=0.1 | Elastic deformation |
| RandomBrightnessContrast | ±0.15, p=0.3 | Intensity variation |
| CLAHE | clip_limit=2.0, p=0.3 | Contrast enhancement (radiology) |
| GaussNoise | std=0.01-0.05, p=0.2 | Simulate imaging noise |
| CoarseDropout | 1-4 holes, 20-60px, p=0.2 | Cutout regularization |

## Metrik

Metrik dihitung **per-sample** untuk hasil yang akurat:

| Metrik | Handling Sample Negatif | Deskripsi |
|--------|------------------------|-----------|
| **Dice** | Skip true negative (pred=0 ∧ target=0) | Overlap coefficient |
| **IoU** | Skip true negative | Jaccard index |
| **Precision** | Hitung false positive | Positive predictive value |
| **Recall** | Hitung false negative | Sensitivity |
| **Accuracy** | Hitung semua sample | Pixel-wise accuracy |
| **Specificity** | Hitung semua sample | True negative rate |
| **AUC** | Skip sample tanpa positif | Area under ROC curve (pixel-level) |

> **Catatan**: False positive (model prediksi pneumonia di gambar normal) tetap dihitung dan dihukum (dice ≈ 0). Hanya true negative (pred=0 dan target=0) yang di-skip karena tidak informatif untuk kualitas segmentasi.

## Hasil

Metrics tersimpan di `outputs/evaluation/metrics.json`:

```json
{
  "dice": 0.XXXX,
  "iou": 0.XXXX,
  "precision": 0.XXXX,
  "recall": 0.XXXX,
  "accuracy": 0.XXXX,
  "specificity": 0.XXXX,
  "auc": 0.XXXX
}
```

## Referensi

```bibtex
@article{zhou2018unetplusplus,
  title={UNet++: A Nested U-Net Architecture for Medical Image Segmentation},
  author={Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  journal={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support},
  year={2018}
}

@inproceedings{rsna2018,
  title={RSNA Pneumonia Detection Challenge},
  organization={Radiological Society of North America},
  year={2018}
}

@misc{cohen2022torchxrayvision,
  title={TorchXRayVision: A library of chest X-ray datasets and models},
  author={Cohen, Joseph Paul and Viviano, Joseph D. and Berber, Paul and Morrison, Paul and Torabian, Parsa and Guarber, Matteo and Lungren, Matthew P. and Chaudhari, Akshay and Brooks, Rupert and Hashir, Mohammad and Bertrand, Hadrien},
  year={2022}
}
```

## Disclaimer

Proyek ini untuk tujuan **edukasi dan penelitian** saja. Tidak untuk penggunaan klinis tanpa validasi dan persetujuan regulasi yang sesuai.
