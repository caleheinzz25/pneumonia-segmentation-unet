# DETEKSI AREA INFEKSI PNEUMONIA PADA CITRA RONTGEN DADA MENGGUNAKAN ARSITEKTUR U-NET BERBASIS CONVOLUTIONAL NEURAL NETWORK (CNN)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/Web%20App-Gradio-orange.svg)](https://gradio.app)
[![Cloudflare Tunnel](https://img.shields.io/badge/Deployment-Cloudflare%20Tunnel-f38020.svg)](https://www.cloudflare.com/)

Sistem *Deep Learning* untuk deteksi dan segmentasi otomatis area infeksi pneumonia pada citra rontgen dada (*Chest X-Ray* / CXR) menggunakan arsitektur **U-Net dengan Atensi sCSE (*Spatial and Channel Squeeze-and-Excitation*)** dan *backbone* **EfficientNet-B3** *pretrained* ImageNet.

Proyek ini dikembangkan berdasarkan kerangka kerja **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) dan diintegrasikan dengan antarmuka web interaktif berbasis **Gradio** serta penyebaran publik melalui **Cloudflare Tunnel**.

---

## 📌 Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **U-Net + sCSE Attention** | Arsitektur segmentasi U-Net yang diperkuat modul atensi *Spatial and Channel Squeeze-and-Excitation* (sCSE) pada *skip connections* untuk menajamkan kontras fitur infeksi. |
| **Pretrained EfficientNet-B3** | *Backbone encoder* EfficientNet-B3 *pretrained* ImageNet dengan laju belajar diferensial (*encoder factor* `0.02` / 50x lebih lambat) untuk mencegah *catastrophic forgetting*. |
| **Dual Masking Strategy** | Pendekatan pemotongan organ paru (*lung segmentation*) otomatis berbasis model **PSPNet** (*TorchXRayVision*) untuk mengisolasi area analisis + masker *ground truth* pneumonia. |
| **Unified Focal Loss** | Fungsi kerugian kombinasi *Focal Loss* (bobot `0.5`) dan *Focal Tversky Loss* ($\alpha=0.3, \beta=0.7, \gamma=0.75$) untuk menekan *false positives* dan menangani ketimpangan data. |
| **Pelatihan Dua Tahap & AMP** | *Encoder* dibekukan pada 12 epoch pertama (*warmup* decoder), menggunakan *Automatic Mixed Precision* (AMP) dan *Gradient Accumulation* 4 langkah (*effective batch size* 32). |
| **Cosine Annealing Scheduler** | Penurunan laju belajar secara halus tanpa lonjakan *warmup* tiba-tiba ($T_{\text{max}} = 100$ epoch). |
| **Exponential Moving Average (EMA)** | Menjaga bobot *moving average* model ($\text{decay} = 0.999$) untuk evaluasi validasi yang lebih stabil. |
| **Test-Time Augmentation (TTA)** | Penggabungan prediksi saat inferensi dengan TTA (*horizontal & vertical flips*) untuk stabilitas batas segmentasi. |
| **Grad-CAM Explainability** | Generasi peta panas (*heatmap*) Grad-CAM pada lapisan konvolusi terakhir untuk transparansi keputusan klinis (*Explainable AI*). |
| **Analisis Keparahan & Lateralitas** | Kalkulasi rasio persentase infeksi ke dalam 5 tingkat keparahan (*Sangat Ringan* hingga *Sangat Berat*) serta lokasi paru (*Kiri, Kanan, Bilateral*). |
| **Gradio Web App & Cloudflare Tunnel** | Antarmuka interaktif medis yang dapat diakses secara publik via internet melalui enkripsi terowongan aman Cloudflare. |

---

## 📊 Hasil Evaluasi Kuantitatif

Model dievaluasi pada data validasi *RSNA Pneumonia Detection Challenge* (26.684 citra) dengan nilai ambang batas inferensi `0.65`:

| Metrik Evaluasi | Nilai Kuantitatif | Keterangan Klinis |
|-----------------|-------------------|-------------------|
| **Dice Coefficient** | **0,6234** (62,34%) | Mengukur derajat kesamaan/overlap antara hasil segmentasi model dan *ground truth*. |
| **Specificity (Spesifisitas)** | **0,9781** (97,81%) | Mengukur keandalan model dalam membedakan jaringan paru sehat (*true negative rate* tinggi). |
| **Recall (Sensitivitas)** | **0,7082** (70,82%) | Memastikan model mampu mendeteksi sebagian besar area infeksi pneumonia (*true positive rate*). |
| **Precision (Presisi)** | **0,6868** (68,68%) | Rasio ketepatan piksel yang diprediksi sebagai infeksi pneumonia. |
| **Intersection over Union (IoU)** | **0,5010** (50,10%) | Jaccard index tingkat piksel. |
| **Area Under ROC Curve (AUC)** | **0,9857** (98,57%) | Kemampuan pemisahan distribusi probabilistik kelas positif dan negatif. |
| **Accuracy (Akurasi Piksel)** | **0,9509** (95,09%) | Akurasi agregat klasifikasi piksel pada seluruh area citra rontgen. |

---

## 🏗️ Arsitektur Sistem & Model

```
                    ┌─────────────────────────────────────────┐
                    │      Citra Rontgen Dada (DICOM/PNG)     │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │   Segmentasi Paru-Paru (PSPNet Auto)    │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │     Input Masked Tensor (3, 512, 512)   │
                    └────────────────────┬────────────────────┘
                                         │
       ┌─────────────────────────────────┴─────────────────────────────────┐
       │                                                                   │
       ▼                                                                   ▼
┌─────────────────────────────┐                         ┌─────────────────────────────┐
│    EfficientNet-B3 Encoder  │                         │       U-Net Decoder         │
│     (Pretrained ImageNet)   │ ──── Skip Connections ─▶│  + sCSE Attention Modules   │
│  (LR Factor 0.02 / 50x slow)│   (Spatial & Channel)   │    (Channels: 256..16)      │
└─────────────────────────────┘                         └──────────────┬──────────────┘
                                                                       │
                                                                       ▼
                                                        ┌─────────────────────────────┐
                                                        │  Sigmoid Output (1,512,512) │
                                                        └──────────────┬──────────────┘
                                                                       │
                                         ┌─────────────────────────────┴─────────────────────────────┐
                                         │                                                           │
                                         ▼                                                           ▼
                      ┌────────────────────────────────────┐                      ┌────────────────────────────────────┐
                      │    Overlay Segmentasi Pneumonia    │                      │       Heatmap Grad-CAM (XAI)       │
                      │    + Keparahan & Lateralitas       │                      │      (Visualisasi Fokus Model)     │
                      └────────────────────────────────────┘                      └────────────────────────────────────┘
```

---

## 📁 Struktur Project

```
.
├── config.yaml                     # Berkas konfigurasi sentral (default: configs/config_v2.yaml)
├── configs/                        # Direktori versi konfigurasi eksperimen
│   ├── config_v1.yaml              # Model Baseline v1 (Dice+BCE, OneCycleLR)
│   ├── config_v2.yaml              # Model Produksi v2 (Unified Focal, Cosine)
│   ├── config_v3.yaml              # Model Eksperimen v3 (EfficientNet-B4, High-Precision Focal)
│   └── README.md                   # Dokumentasi varian konfigurasi
├── pyproject.toml                  # Manajemen dependen berbasis UV
├── README.md                       # Dokumentasi utama proyek
├── CODEBASE_NOTES.md               # Catatan arsitektur & keputusan teknis (v1, v2, v3)
├── app/
│   └── gradio_app.py               # Antarmuka web interaktif medis (Gradio)
├── src/
│   ├── config.py                   # YAML parser & pembentuk dataclass konfigurasi
│   ├── dataset.py                  # Dataset RSNA: pemroses DICOM & dual-masking
│   ├── transforms.py               # Augmentasi medis (Albumentations) & normalisasi
│   ├── model.py                    # Arsitektur U-Net + sCSE + Auxiliary Head
│   ├── losses.py                   # Unified Focal Loss (Focal + Focal Tversky)
│   ├── metrics.py                  # Komputasi metrik (Dice, IoU, Precision, Recall, Specificity, AUC)
│   ├── train.py                    # Loop pelatihan: AMP, accumulation, EMA, resume state
│   ├── evaluate.py                 # Evaluasi kuantitatif & visualisasi sampel
│   ├── predict.py                  # Inferensi CLI (single/batch) + auto lung masking
│   ├── lung_segmentation.py        # Segmentasi paru-paru otomatis (PSPNet)
│   ├── explainability.py           # Generasi peta panas Grad-CAM
│   └── utils.py                    # Windowing DICOM, logging state, overlay visual
├── scripts/
│   ├── train.sh                    # Script eksekusi pelatihan
│   ├── evaluate.sh                 # Script eksekusi evaluasi
│   ├── predict.sh                  # Script eksekusi inferensi
│   ├── app.sh                      # Script peluncuran web app
│   ├── precompute_all_masks.py     # Precompute lung mask + ground truth GT masks
│   └── precompute_lungmask.py     # Precompute lung mask batch (GPU)
├── data/
│   ├── rsna-pneumonia-detection-challenge/  # Dataset DICOM RSNA
│   └── lung_masks/                 # Masker paru-paru & ground truth pneumonia
├── outputs/
│   ├── checkpoints/                # Model terlatih (best_model.pth & latest_model.pth)
│   ├── evaluation/                 # metrics.json & visualisasi evaluasi
│   ├── gradcam/                    # Peta panas Grad-CAM
│   ├── logs/                       # Log eksekusi terminal & JSON state tracker
│   └── predictions/                # Hasil prediksi inferensi
└── doc/
    ├── skripsi.tex                 # Source naskah skripsi LaTeX
    ├── skripsi.pdf                 # Hasil kompilasi skripsi PDF (123 halaman)
    ├── Makefile                    # Automation build LaTeX PDF & Word
    ├── references.bib              # Referensi bibliografi (Harvard style)
    └── img/                        # Aset gambar & diagram naskah skripsi
```

---

## ⚙️ Instalasi & Persiapan

### Prasyarat
- **Python**: $\ge 3.10$
- **GPU**: NVIDIA GPU (direkomendasikan $\ge 8$ GB VRAM, seperti RTX 3070 / T4)
- **Paket Manager**: [UV](https://docs.astral.sh/uv/)

### Instalasi Dependensi

```bash
# Clone repository
git clone https://github.com/louiscalvin/pneumonia-segmentation-unet.git
cd pneumonia-segmentation-unet

# Sinkronkan virtual environment menggunakan UV
uv sync

# Aktifkan virtual environment
source .venv/bin/activate
```

---

## 🚀 Panduan Penggunaan

### 1. Precompute Masker Paru & Ground Truth

Sebelum melakukan pelatihan, generate masker paru-paru (*PSPNet*) dan masker *ground truth* pneumonia (dari anotasi *bounding box* RSNA):

```bash
# Generate seluruh masker (Lung Mask + Ground Truth + Visualisasi)
uv run python -m scripts.precompute_all_masks --visualize
```

### 2. Pelatihan Model (*Training*)

```bash
# Jalankan pelatihan menggunakan konfigurasi sentral
uv run python -m src.train --config config.yaml

# Atau menggunakan script bash
./scripts/train.sh
```

- Bobot model terbaik (*best val Dice*) otomatis tersimpan di `outputs/checkpoints/best_model.pth`.
- Checkpoint paling akhir untuk kelanjutan pelatihan tersimpan di `outputs/checkpoints/latest_model.pth`.

#### Melanjutkan Pelatihan (*Resume Training*)

```bash
# Lanjutkan dari checkpoint terbaru secara otomatis
uv run python -m src.train --config config.yaml --resume
```

### 3. Evaluasi Model

Evaluasi performa model terbaik pada data validasi:

```bash
uv run python -m src.evaluate --config config.yaml

# Atau via script
./scripts/evaluate.sh
```
Hasil evaluasi akan memperbarui berkas `outputs/evaluation/metrics.json`.

### 4. Inferensi Citra Rontgen (*Prediction*)

Predict citra X-ray tunggal atau direktori:

```bash
# Single image (DICOM / PNG / JPG)
uv run python -m src.predict --config config.yaml --input data/sample.dcm

# Batch directory
uv run python -m src.predict --config config.yaml --input data/test_images/ --output outputs/predictions
```

### 5. Generasi Grad-CAM Heatmap

```bash
uv run python -m src.explainability --config config.yaml --input data/sample.dcm --output outputs/gradcam
```

### 6. Menjalankan Web Application (Gradio)

Peluncuran antarmuka medis interaktif:

```bash
# Jalankan aplikasi Gradio
uv run python -m app.gradio_app

# Atau via script bash
./scripts/app.sh
```
Aplikasi dapat diakses melalui peramban web di `http://localhost:7860`.

---

## 🎛️ Konfigurasi Utama (`config.yaml`)

| Parameter | Nilai Default | Deskripsi |
|-----------|---------------|-----------|
| `model.architecture` | `Unet` | Arsitektur dasar model segmentasi. |
| `model.encoder_name` | `timm-efficientnet-b3` | *Backbone encoder* berbasis EfficientNet-B3. |
| `model.decoder_attention_type` | `scse` | Modul atensi *Spatial and Channel Squeeze-and-Excitation*. |
| `preprocessing.image_size` | `[512, 512]` | Resolusi input tensor citra rontgen. |
| `training.batch_size` | `8` | Ukuran batch per langkah GPU. |
| `training.accumulation_steps` | `4` | Akumulasi gradien (*effective batch size* = $8 \times 4 = 32$). |
| `training.encoder_freeze_epochs` | `12` | Jumlah epoch awal dengan *encoder* dibekukan (*warmup* decoder). |
| `training.early_stopping_patience` | `35` | Toleransi *early stopping* jika *val Dice* tidak meningkat. |
| `optimizer.type` | `adamw` | Optimizer AdamW dengan L2 *weight decay* `5.0e-4`. |
| `optimizer.lr` | `4.0e-4` | Laju belajar maksimum untuk *decoder*. |
| `optimizer.encoder_lr_factor` | `0.02` | Faktor pengali laju belajar *encoder* (50x lebih lambat = `8.0e-6`). |
| `scheduler.type` | `cosine_annealing` | Penurunan laju belajar harmonis hingga `1.0e-6`. |
| `loss.type` | `unified_focal` | Gabungan *Focal Loss* ($w=0.5$) dan *Focal Tversky Loss* ($\alpha=0.3, \beta=0.7$). |
| `inference.threshold` | `0.65` | Ambang batas nilai biner probabilistik piksel pneumonia. |

---

## 📄 Kompilasi Naskah Skripsi (LaTeX)

Naskah lengkap skripsi berbasis LaTeX tersedia pada direktori `doc/`.

```bash
cd doc

# Kompilasi naskah skripsi ke PDF
make pdf
```
Hasil kompilasi PDF akan dibuat di `doc/skripsi.pdf` (123 halaman).

---

## 📚 Referensi Akademis

```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages={234--241},
  year={2015},
  organization={Springer}
}

@article{roy2018scse,
  title={Concurrent Spatial and Channel Squeeze \& Excitation in Fully Convolutional Networks},
  author={Roy, Abhijit Guha and Navab, Nassir and Wachinger, Christian},
  journal={IEEE Transactions on Medical Imaging},
  volume={37},
  number={10},
  pages={2372--2384},
  year={2018}
}

@inproceedings{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={6105--6114},
  year={2019}
}

@inproceedings{selvaraju2017gradcam,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  pages={618--626},
  year={2017}
}

@misc{rsna2018pneumonia,
  title={RSNA Pneumonia Detection Challenge},
  author={{Radiological Society of North America}},
  year={2018},
  url={https://www.kaggle.com/c/rsna-pneumonia-detection-challenge}
}
```

---

## ⚠️ Penolakan Tanggung Jawab (*Disclaimer*)

Sistem ini dikembangkan secara murni untuk tujuan **akademis, edukasi, dan penelitian**. Sistem ini berfungsi sebagai alat bantu skrining awal (*triase*) dan **tidak ditujukan untuk menggantikan peran, diagnosis klinis, atau keputusan akhir dari dokter spesialis radiologi**.
