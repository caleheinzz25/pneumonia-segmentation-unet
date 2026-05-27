# Segmentasi Citra Radiografi Dada untuk Pendeteksian Area Infeksi Pneumonia Menggunakan Model Deep Learning Attention UNet++

Proyek ini mengimplementasikan pipeline deep learning untuk deteksi dan segmentasi otomatis pneumonia dari citra X-ray dada (CXR) menggunakan arsitektur **Attention UNet++** dengan encoder **EfficientNet-B4** pretrained.

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Attention UNet++** | Nested skip connections dengan SCSE (Spatial & Channel Squeeze-and-Excitation) attention gates |
| **Pretrained Encoder** | ImageNet-pretrained EfficientNet-B4 backbone |
| **Dual Mask Approach** | Lung segmentation mask untuk masking input + precomputed pneumonia mask untuk ground truth |
| **AMP Training** | Automatic Mixed Precision untuk training lebih cepat |
| **Advanced Loss** | Focal Tversky Loss + BCE untuk menangani class imbalance |
| **Comprehensive Metrics** | Dice, IoU, Precision, Recall, Specificity, AUC |
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
│   ├── losses.py            # Focal Tversky, Dice, BCE, ComboLoss
│   ├── metrics.py           # Dice, IoU, Precision, Recall, Specificity, AUC
│   ├── train.py             # Training loop: AMP, gradient accumulation, early stopping
│   ├── evaluate.py          # Evaluasi + visualisasi overlay
│   ├── predict.py           # Inference: single image / batch
│   ├── explainability.py    # Grad-CAM heatmap generation
│   └── utils.py             # DICOM read, windowing, overlay, seeding
├── app/
│   └── gradio_app.py        # Web interface
├── scripts/
│   ├── train.sh             # Script training
│   ├── evaluate.sh          # Script evaluasi
│   ├── predict.sh           # Script inference
│   └── app.sh               # Script launch web app
└── data/
    ├── rsna-pneumonia-detection-challenge/
    │   ├── stage_2_train_images/       # DICOM training images (~26K)
    │   ├── stage_2_test_images/        # DICOM test images
    │   ├── stage_2_train_labels.csv    # Bounding box annotations
    │   └── stage_2_detailed_class_info.csv
    └── lung_masks/
        ├── lung_segmentation/          # Lung mask: masking input image
        ├── pneumonia_ground_truth/     # Precomputed pneumonia mask: ground truth
        ├── combined/                   # Visualisasi: lung + pneumonia overlay
        └── visualizations/             # Visualisasi hasil
```

## Dataset

### RSNA Pneumonia Detection Challenge

Dataset terdiri dari ~26,000 citra X-ray dada dalam format DICOM dengan anotasi bounding box untuk area pneumonia.

### Precomputed Masks (`data/lung_masks/`)

| Folder | Isi | Penggunaan |
|--------|-----|------------|
| `lung_segmentation/` | Binary mask paru-paru per pasien | Masking input image (pixel non-paru = 0) |
| `pneumonia_ground_truth/` | Binary mask pneumonia per pasien | Ground truth training (lebih akurat dari bbox→mask) |
| `combined/` | Overlay lung + pneumonia | Referensi visualisasi |
| `visualizations/` | Hasil visualisasi | Referensi visualisasi |

**Dual Mask Approach**: Dataset class akan otomatis menggunakan precomputed pneumonia mask sebagai ground truth (override bbox→mask), dan lung segmentation mask untuk membatasi input hanya pada area paru-paru.

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

### Training

```bash
# Via script
./scripts/train.sh

# Atau langsung
python -m src.train --config config.yaml
```

Training akan otomatis:
- Load precomputed pneumonia masks sebagai ground truth
- Apply lung masking pada input
- Menggunakan AMP (jika GPU tersedia)
- Save best model ke `outputs/checkpoints/best_model.pth`
- Log metrics ke TensorBoard di `outputs/tensorboard/`

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
python -m src.predict --config config.yaml \
  --input path/to/images/ \
  --output outputs/predictions
```

Output per image:
- `{name}_pred.png` — Binary prediction mask
- `{name}_overlay.png` — Overlay merah pada area pneumonia

### Grad-CAM Explainability

```bash
python -m src.explainability \
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
| `preprocessing` | `window_level` | -600 | Lung window HU center |
| `preprocessing` | `window_width` | 1500 | Lung window HU width |
| `training` | `batch_size` | 4 | Batch size |
| `training` | `accumulation_steps` | 4 | Gradient accumulation (effective batch = 16) |
| `training` | `use_amp` | true | Mixed precision training |
| `training` | `early_stopping_patience` | 20 | Early stopping patience |
| `loss` | `type` | focal_tversky | Loss function |
| `loss` | `pos_weight` | 3.0 | Weight untuk class pneumonia |
| `optimizer` | `lr` | 3e-4 | Learning rate |
| `scheduler` | `type` | cosine_annealing | LR scheduler |

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
```

## Disclaimer

Proyek ini untuk tujuan **edukasi dan penelitian** saja. Tidak untuk penggunaan klinis tanpa validasi dan persetujuan regulasi yang sesuai.
