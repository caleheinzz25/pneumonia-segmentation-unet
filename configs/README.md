# ⚙️ Project Configurations Directory (`configs/`)

Direktori ini berisi seluruh versi file konfigurasi eksperimen proyek **Pneumonia Segmentation U-Net**.

---

## 📋 Daftar Versi Konfigurasi

| File | Nama Versi | Status | Deskripsi Ringkas & Strategi Utamnya |
|---|---|---|---|
| [`config_v1.yaml`](config_v1.yaml) | **Baseline Model (v1)** | *Archived / Baseline* | Model dasar awal: `timm-efficientnet-b3`, `scse` attention, `OneCycleLR`, `dice_bce` loss, `encoder_lr_factor: 0.1`, `negative_ratio: 0.30`, `threshold: 0.50`. |
| [`config_v2.yaml`](config_v2.yaml) | **Production Model (v2)** | *Active Production / Paper* | Model produksi utama (Skripsi): `Unified Focal Loss`, `Cosine Annealing`, `encoder_lr_factor: 0.02` (50x lebih lambat), `negative_ratio: 0.10`, `threshold: 0.65`. **Hasil: Dice 0.6234, Specificity 0.9781, Recall 0.7082**. |
| [`config_v3.yaml`](config_v3.yaml) | **Experimental Model (v3)** | *Next Experiment* | Eksperimen tahap berikutnya: `timm-efficientnet-b4`, `encoder_lr_factor: 0.015`, `Unified Focal Loss` dengan penalti FP lebih tinggi ($\alpha=0.25, \beta=0.75$), `epochs: 120`, `encoder_freeze_epochs: 15`, `threshold: 0.70`. |

---

## 🚀 Panduan Eksekusi Pelatihan per Versi

Untuk menjalankan pelatihan menggunakan konfigurasi tertentu, tentukan argumen `--config`:

```bash
# 1. Jalankan versi Baseline v1
uv run python -m src.train --config configs/config_v1.yaml

# 2. Jalankan versi Production v2 (Master Skripsi)
uv run python -m src.train --config configs/config_v2.yaml

# 3. Jalankan eksperimen baru v3
uv run python -m src.train --config configs/config_v3.yaml
```

---

## 🔍 Perbandingan Parameter Utama (v1 vs v2 vs v3)

| Parameter Konfigurasi | `config_v1.yaml` (Baseline) | `config_v2.yaml` (Production) | `config_v3.yaml` (Experiment) |
|---|---|---|---|
| `model.encoder_name` | `timm-efficientnet-b3` | `timm-efficientnet-b3` | `timm-efficientnet-b4` |
| `model.decoder_dropout` | `0.3` | `0.2` | `0.2` |
| `loss.type` | `dice_bce` | `unified_focal` | `unified_focal` |
| `loss.tversky_alpha` / `beta` | `0.4` / `0.6` | `0.3` / `0.7` | `0.25` / `0.75` |
| `scheduler.type` | `one_cycle` | `cosine_annealing` | `cosine_annealing` |
| `optimizer.encoder_lr_factor` | `0.10` (10x slower) | `0.02` (50x slower) | `0.015` (66x slower) |
| `training.encoder_freeze_epochs` | `8` | `12` | `15` |
| `training.epochs` | `100` | `100` | `120` |
| `data.negative_ratio` | `0.30` | `0.10` | `0.05` |
| `inference.threshold` | `0.50` | `0.65` | `0.70` |
