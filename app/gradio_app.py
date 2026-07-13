"""Gradio web application for pneumonia region detection and localization."""

import shutil
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.config import Config, load_config
from src.lung_segmentation import LungSegmenter
from src.model import build_model
from src.predict import predict_single
from src.utils import overlay_mask, read_dicom


# Supported file extensions
DICOM_EXTS = {".dcm", ".dicom"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ALL_EXTS = DICOM_EXTS | IMAGE_EXTS


def _load_image(file_path: str | Path) -> np.ndarray:
    """Load an image from any supported format and return a grayscale float32 array.

    Supports: DICOM (.dcm, .dicom), PNG, JPG, JPEG, BMP, TIFF, WEBP.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in DICOM_EXTS:
        # DICOM: use pydicom via read_dicom, already float32
        image = read_dicom(path)
        # Scale to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        return image

    # Standard image formats: try PIL first (handles more formats), fallback cv2
    try:
        pil_img = Image.open(path).convert("L")  # convert to grayscale
        image = np.array(pil_img, dtype=np.float32) / 255.0
        return image
    except Exception:
        pass

    # Fallback: OpenCV
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Format file tidak didukung atau file rusak: {path.name}")
    return image.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Severity classification based on infection ratio within the lung area
# ---------------------------------------------------------------------------
SEVERITY_LEVELS = [
    # (max_ratio, label, emoji, color_hex, description)
    (0.0, "Normal", "✅", "#22c55e",
     "Tidak terdeteksi area infeksi pneumonia."),
    (5.0, "Ringan (Mild)", "🟡", "#eab308",
     "Area infeksi kecil, kemungkinan pneumonia tahap awal."),
    (15.0, "Sedang (Moderate)", "🟠", "#f97316",
     "Area infeksi cukup signifikan, perlu evaluasi klinis lebih lanjut."),
    (30.0, "Berat (Severe)", "🔴", "#ef4444",
     "Area infeksi luas, kemungkinan pneumonia berat. Segera konsultasi dokter."),
    (100.0, "Kritis (Critical)", "🚨", "#dc2626",
     "Area infeksi sangat luas mencakup sebagian besar paru-paru. "
     "Memerlukan penanganan medis segera."),
]


def _classify_severity(infection_ratio: float) -> tuple[str, str, str, str]:
    """Return (label, emoji, color_hex, description) based on infection ratio."""
    for max_ratio, label, emoji, color, desc in SEVERITY_LEVELS:
        if infection_ratio <= max_ratio:
            return label, emoji, color, desc
    # fallback
    return SEVERITY_LEVELS[-1][1:]


def _spatial_distribution(pred_mask: np.ndarray) -> dict[str, float]:
    """Analyse the left/right and upper/lower distribution of infection."""
    h, w = pred_mask.shape
    total = pred_mask.sum()
    if total == 0:
        return {"left": 0.0, "right": 0.0, "upper": 0.0, "lower": 0.0}

    mid_w = w // 2
    mid_h = h // 2

    left = pred_mask[:, :mid_w].sum() / total * 100
    right = pred_mask[:, mid_w:].sum() / total * 100
    upper = pred_mask[:mid_h, :].sum() / total * 100
    lower = pred_mask[mid_h:, :].sum() / total * 100

    return {"left": left, "right": right, "upper": upper, "lower": lower}


def _confidence_stats(prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute confidence statistics over the probability map."""
    positive_probs = prob[prob >= threshold]
    if len(positive_probs) == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(positive_probs)),
        "max": float(np.max(positive_probs)),
        "min": float(np.min(positive_probs)),
        "std": float(np.std(positive_probs)),
    }


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a text label with a black background box on the top-left of the image."""
    annotated = img.copy()
    h_img, w_img = annotated.shape[:2]
    
    # Scale parameters based on image width (reference: 512px width)
    scale_factor = w_img / 512.0
    font_scale = max(0.6, 0.8 * scale_factor)
    thickness = max(1, int(2 * scale_factor))
    padding = int(12 * scale_factor)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = padding, h + padding
    
    # Draw black background rectangle
    cv2.rectangle(
        annotated,
        (x - 5, y - h - 5),
        (x + w + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )
    # Draw white text
    cv2.putText(
        annotated,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def _generate_kpi_html(detected: str, lung_ratio: float, severity_label: str, severity_emoji: str, severity_color: str) -> str:
    """Generate responsive HTML for dashboard KPI metrics cards."""
    status_color = "#ef4444" if detected == "Ya" else "#22c55e"
    status_text_color = "#ef4444" if detected == "Ya" else "#22c55e"
    status_icon = "⚠️" if detected == "Ya" else "✅"
    
    ratio_color = "#3b82f6"
    if lung_ratio > 0:
        ratio_color = severity_color

    return f"""
    <div class="kpi-container" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem; width: 100%;">
        <div class="kpi-card" style="background: var(--block-background-fill); border-left: 5px solid {status_color}; padding: 1.25rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); display: flex; flex-direction: column; justify-content: space-between; min-height: 100px; border: 1px solid var(--border-color-primary);">
            <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--body-text-color-subdued, #64748b);">Status Deteksi</span>
            <div style="font-size: 1.75rem; font-weight: 800; color: {status_text_color}; margin-top: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>{status_icon}</span> <span>{detected}</span>
            </div>
        </div>
        <div class="kpi-card" style="background: var(--block-background-fill); border-left: 5px solid {ratio_color}; padding: 1.25rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); display: flex; flex-direction: column; justify-content: space-between; min-height: 100px; border: 1px solid var(--border-color-primary);">
            <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--body-text-color-subdued, #64748b);">Rasio Infeksi Paru</span>
            <div style="font-size: 1.75rem; font-weight: 800; color: var(--body-text-color, #1e293b); margin-top: 0.5rem;">
                <span>{lung_ratio:.2f}%</span>
            </div>
        </div>
        <div class="kpi-card" style="background: var(--block-background-fill); border-left: 5px solid {severity_color}; padding: 1.25rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); display: flex; flex-direction: column; justify-content: space-between; min-height: 100px; border: 1px solid var(--border-color-primary);">
            <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--body-text-color-subdued, #64748b);">Tingkat Keparahan</span>
            <div style="font-size: 1.5rem; font-weight: 800; color: {severity_color}; margin-top: 0.5rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>{severity_emoji}</span> <span>{severity_label}</span>
            </div>
        </div>
    </div>
    """


def _default_kpi_placeholder() -> str:
    """Generate default placeholder HTML for KPI cards before run."""
    return """
    <div class="kpi-container" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem; width: 100%;">
        <div class="kpi-card" style="background: var(--block-background-fill); border-left: 5px solid #64748b; padding: 1.25rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued, #64748b); min-height: 100px; display: flex; flex-direction: column; justify-content: space-between;">
            <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Status Deteksi</span>
            <div style="font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem; color: var(--body-text-color-subdued, #64748b);">Menunggu analisis...</div>
        </div>
        <div class="kpi-card" style="background: var(--block-background-fill); border-left: 5px solid #64748b; padding: 1.25rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued, #64748b); min-height: 100px; display: flex; flex-direction: column; justify-content: space-between;">
            <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Rasio Infeksi Paru</span>
            <div style="font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem; color: var(--body-text-color-subdued, #64748b);">Menunggu analisis...</div>
        </div>
        <div class="kpi-card" style="background: var(--block-background-fill); border-left: 5px solid #64748b; padding: 1.25rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued, #64748b); min-height: 100px; display: flex; flex-direction: column; justify-content: space-between;">
            <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Tingkat Keparahan</span>
            <div style="font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem; color: var(--body-text-color-subdued, #64748b);">Menunggu analisis...</div>
        </div>
    </div>
    """


class PneumoniaDetectionApp:
    """Gradio app for pneumonia region detection and localization from chest X-rays."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model(config.model, device=self.device)

        # Load checkpoint
        checkpoint = torch.load(
            config.inference.model_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load lung segmenter for auto lung masking
        self.lung_segmenter = LungSegmenter(device=self.device)

        print(f"Model loaded on {self.device}")

    def _predict_from_file(
        self, file_path: str, threshold: float
    ) -> tuple[np.ndarray | None, str, str]:
        """Core prediction from a file path (any supported format)."""
        path = Path(file_path)
        ext = path.suffix.lower()
        file_type = "DICOM" if ext in DICOM_EXTS else ext.upper().lstrip(".")

        try:
            gray = _load_image(path)
        except Exception as e:
            return None, _default_kpi_placeholder(), f"❌ Gagal membaca file: {e}"

        # Save as temp PNG for predict_single
        with tempfile.NamedTemporaryFile(suffix=ext if ext in DICOM_EXTS else ".png", delete=False) as f:
            temp_path = f.name
            if ext in DICOM_EXTS:
                # Copy the original DICOM so predict_single can read it natively
                shutil.copy2(str(path), temp_path)
            else:
                cv2.imwrite(temp_path, (gray * 255).astype(np.uint8))

        try:
            prob, original, lung_mask = predict_single(
                self.model, temp_path, self.config, self.device,
                lung_segmenter=self.lung_segmenter,
            )
            pred_mask = (prob >= threshold).astype(np.float32)

            # --- Visual outputs -----------------------------------------------

            lung_mask_vis = (lung_mask * 255).astype(np.uint8)
            lung_mask_rgb = cv2.cvtColor(lung_mask_vis, cv2.COLOR_GRAY2RGB)

            lung_img = original * lung_mask
            lung_img_vis = (lung_img * 255).astype(np.uint8)
            lung_img_rgb = cv2.cvtColor(lung_img_vis, cv2.COLOR_GRAY2RGB)

            # Overlay: red for infection
            overlay = overlay_mask(
                original,
                pred_mask,
                color=tuple(self.config.inference.overlay_color),
                alpha=self.config.inference.overlay_alpha,
            )
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # Probability heatmap
            prob_colored = cv2.applyColorMap(
                (prob * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            prob_rgb = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)

            # --- Combine into 2x2 Grid with Labels -----------------------------
            img_a = _add_label(overlay_rgb, "A. Overlay Infeksi")
            img_b = _add_label(prob_rgb, "B. Heatmap Grad-CAM")
            img_c = _add_label(lung_mask_rgb, "C. Masker Paru")
            img_d = _add_label(lung_img_rgb, "D. Paru Terisolasi")

            top_row = np.hstack([img_a, img_b])
            bottom_row = np.hstack([img_c, img_d])
            grid_rgb = np.vstack([top_row, bottom_row])

            # --- Analysis stats -----------------------------------------------

            h_img, w_img = pred_mask.shape
            total_pixels = pred_mask.size
            infected_pixels = int(pred_mask.sum())
            total_ratio = infected_pixels / total_pixels * 100

            # Estimate lung area (non-zero pixels in input → rough lung area)
            lung_pixels = int(np.count_nonzero(original))
            lung_ratio = (
                infected_pixels / lung_pixels * 100 if lung_pixels > 0 else 0.0
            )

            # Severity
            severity_label, severity_emoji, severity_color, severity_desc = (
                _classify_severity(lung_ratio)
            )

            # Spatial distribution
            spatial = _spatial_distribution(pred_mask)

            # Confidence
            conf = _confidence_stats(prob, threshold)

            # --- Build markdown report ----------------------------------------

            info_lines = []

            # Header with severity
            info_lines.append(
                f"## {severity_emoji} Hasil Analisis: "
                f"<span style='color:{severity_color}'>{severity_label}</span>\n"
            )
            info_lines.append(f"> {severity_desc}\n")
            info_lines.append("---\n")

            # Detection summary
            detected = "Ya" if infected_pixels > 0 else "Tidak"
            info_lines.append("### 🔍 Ringkasan Deteksi\n")
            info_lines.append(f"| Parameter | Nilai |")
            info_lines.append(f"|---|---|")
            info_lines.append(f"| **Pneumonia Terdeteksi** | {detected} |")
            info_lines.append(f"| **Area Infeksi (total)** | "
                               f"{infected_pixels:,} piksel ({total_ratio:.2f}%) |")
            info_lines.append(
                f"| **Area Infeksi (paru-paru)** | {lung_ratio:.2f}% dari area paru |"
            )
            info_lines.append(
                f"| **Ukuran Gambar** | {w_img} × {h_img} piksel |"
            )
            info_lines.append(
                f"| **Format Input** | {file_type} |"
            )
            info_lines.append(
                f"| **Threshold** | {threshold} |\n"
            )

            # Severity detail
            info_lines.append("### 📊 Tingkat Keparahan\n")
            # Progress bar via markdown
            bar_len = 20
            filled = int(lung_ratio / 100 * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            info_lines.append(
                f"**`[{bar}]` {lung_ratio:.1f}%** dari area paru terinfeksi\n"
            )

            # Spatial
            if infected_pixels > 0:
                info_lines.append("### 🗺️ Distribusi Spasial Infeksi\n")
                info_lines.append(f"| Zona | Persentase |")
                info_lines.append(f"|---|---|")

                # Note: In an X-ray, left side of the image = patient's right lung
                info_lines.append(
                    f"| Paru Kanan (gambar kiri) | {spatial['left']:.1f}% |"
                )
                info_lines.append(
                    f"| Paru Kiri (gambar kanan) | {spatial['right']:.1f}% |"
                )
                info_lines.append(
                    f"| Zona Atas | {spatial['upper']:.1f}% |"
                )
                info_lines.append(
                    f"| Zona Bawah | {spatial['lower']:.1f}% |\n"
                )

                # Laterality
                if spatial["left"] > 70:
                    info_lines.append("📌 **Dominan di paru kanan**\n")
                elif spatial["right"] > 70:
                    info_lines.append("📌 **Dominan di paru kiri**\n")
                else:
                    info_lines.append("📌 **Bilateral** (kedua paru terpengaruh)\n")

            # Confidence stats
            if infected_pixels > 0:
                info_lines.append("### 🎯 Statistik Confidence Model\n")
                info_lines.append(f"| Metrik | Nilai |")
                info_lines.append(f"|---|---|")
                info_lines.append(
                    f"| Confidence Rata-rata | {conf['mean']:.4f} |"
                )
                info_lines.append(f"| Confidence Maksimum | {conf['max']:.4f} |")
                info_lines.append(f"| Confidence Minimum | {conf['min']:.4f} |")
                info_lines.append(
                    f"| Std. Deviasi | {conf['std']:.4f} |\n"
                )

            info = "\n".join(info_lines)
            
            # Generate KPI metrics HTML block
            kpi_html = _generate_kpi_html(detected, lung_ratio, severity_label, severity_emoji, severity_color)
            
            return grid_rgb, kpi_html, info

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def predict_from_upload(self, file, threshold: float):
        """Handle gr.File upload (DICOM or any image)."""
        if file is None:
            return None, _default_kpi_placeholder(), "⚠️ Silakan upload file X-ray terlebih dahulu."
        return self._predict_from_file(file, threshold)

    def predict_from_image(self, input_image: np.ndarray, threshold: float):
        """Handle gr.Image upload (drag-and-drop regular images)."""
        if input_image is None:
            return None, _default_kpi_placeholder(), "⚠️ Silakan upload gambar X-ray terlebih dahulu."

        # Convert numpy to a temp file
        if input_image.ndim == 3:
            gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = input_image
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, (gray * 255).astype(np.uint8))

        try:
            return self._predict_from_file(temp_path, threshold)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        
        # Define modern styles
        custom_css = """
        /* Header styling */
        .app-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .app-header h1 {
            font-size: 2.5rem !important;
            font-weight: 850 !important;
            background: linear-gradient(135deg, #0d9488, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem !important;
        }
        
        /* Sidebar styling */
        .sidebar-container {
            border-radius: 12px !important;
            padding: 1rem !important;
        }
        
        /* Result images styling */
        .result-image {
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        
        /* Stats cards custom animations */
        .kpi-card {
            transition: all 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08) !important;
        }
        
        /* Example container */
        .examples-container {
            margin-top: 1.5rem;
            border-top: 1px solid var(--border-color-primary);
            padding-top: 1rem;
        }
        
        /* Custom tabs styling */
        .nav-tabs {
            border-bottom: 2px solid var(--border-color-primary) !important;
        }
        """

        theme = gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        ).set(
            body_background_fill="*neutral_50",
            body_background_fill_dark="*neutral_950",
            block_background_fill="*neutral_100",
            block_background_fill_dark="*neutral_900",
            block_border_width="1px",
            block_border_color="*neutral_200",
            block_border_color_dark="*neutral_800",
            block_label_text_size="*text_sm",
            block_label_text_weight="600",
            button_primary_background_fill="*primary_600",
            button_primary_background_fill_dark="*primary_700",
            button_primary_background_fill_hover="*primary_500",
            button_primary_background_fill_hover_dark="*primary_600",
            button_primary_text_color="*white",
        )

        with gr.Blocks(
            title=self.config.app.title,
            theme=theme,
            css=custom_css,
        ) as demo:
            
            # --- Header ---
            with gr.Column(elem_classes="app-header"):
                gr.Markdown(
                    f"# 🫁 {self.config.app.title}\n"
                    f"{self.config.app.description}"
                )
            
            # --- Main Layout Grid ---
            with gr.Row():
                
                # --- Left Side: Input & Configuration ---
                with gr.Column(scale=4, elem_classes="sidebar-container"):
                    with gr.Group():
                        gr.Markdown("### 📥 Input Citra Medis")
                        with gr.Tabs():
                            with gr.TabItem("File Upload (DICOM / Image)"):
                                input_file = gr.File(
                                    label="Upload DICOM (.dcm, .dicom) atau format citra standar",
                                    file_types=[
                                        ".dcm", ".dicom",
                                        ".png", ".jpg", ".jpeg",
                                        ".bmp", ".tif", ".tiff", ".webp",
                                    ],
                                    type="filepath",
                                )
                                predict_btn_file = gr.Button(
                                    "🔬 Analisis File", variant="primary", size="lg"
                                )
                            with gr.TabItem("Drag & Drop Gambar"):
                                input_image = gr.Image(
                                    type="numpy",
                                    label="Tarik & letakkan gambar Chest X-ray di sini",
                                    sources=["upload"],
                                    height=250,
                                )
                                predict_btn_image = gr.Button(
                                    "🔬 Analisis Gambar", variant="primary", size="lg"
                                )
                    
                    with gr.Group():
                        gr.Markdown("### ⚙️ Pengaturan Model")
                        threshold_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=self.config.inference.threshold,
                            step=0.05,
                            label="🎚️ Threshold Deteksi",
                            info="Nilai lebih rendah: lebih sensitif (banyak deteksi). Nilai lebih tinggi: lebih spesifik (deteksi lebih pasti).",
                        )
                    
                    with gr.Accordion("📋 Panduan Interpretasi & Nilai Rujukan", open=False):
                        gr.Markdown(
                            """
                            **Tingkat Keparahan berdasarkan Rasio Infeksi Paru:**
                            - **0%**: Normal ✅ (Tidak terdeteksi area infeksi)
                            - **0.1% - 5.0%**: Ringan (Mild) 🟡
                            - **5.1% - 15.0%**: Sedang (Moderate) 🟠
                            - **15.1% - 30.0%**: Berat (Severe) 🔴
                            - **> 30.0%**: Kritis (Critical) 🚨
                            
                            *Rasio infeksi dihitung sebagai persentase area terinfeksi pneumonia terhadap estimasi total area paru-paru.*
                            """
                        )
                        
                    with gr.Accordion("⚙️ Spesifikasi Model Aktif", open=False):
                        gr.Markdown(
                            f"""
                            - **Arsitektur:** {self.config.model.architecture}
                            - **Backbone Encoder:** `{self.config.model.encoder_name}` (ImageNet Weights)
                            - **Attention Type:** `{self.config.model.decoder_attention_type}` (Spatial-Channel Squeeze & Excitation)
                            - **Decoder Channels:** `{list(self.config.model.decoder_channels)}`
                            - **Running Device:** `{self.device.upper()}`
                            """
                        )

                    # --- Examples Panel ---
                    gr.Markdown("### 📋 Contoh Citra X-Ray", elem_classes="examples-container")
                    examples = gr.Examples(
                        examples=[
                            ["data/pneumonia/chest_xray/test/NORMAL/IM-0001-0001.jpeg", self.config.inference.threshold],
                            ["data/pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg", self.config.inference.threshold],
                            ["data/pneumonia/chest_xray/test/PNEUMONIA/person108_bacteria_511.jpeg", self.config.inference.threshold],
                        ],
                        inputs=[input_file, threshold_slider],
                        label="Klik salah satu contoh gambar di bawah untuk mencoba langsung:",
                    )
                
                # --- Right Side: Dashboard & Visualizations ---
                with gr.Column(scale=6):
                    # --- KPI Metrics Row ---
                    output_kpis = gr.HTML(
                        value=_default_kpi_placeholder(),
                        label="Statistik Utama",
                    )
                    
                    # --- Output Visualization Grid ---
                    with gr.Row():
                        with gr.Column(scale=6):
                            output_grid = gr.Image(
                                type="numpy",
                                label="Hasil Analisis (A: Overlay, B: Heatmap, C: Masker Paru, D: Paru Terisolasi)",
                                elem_classes="result-image",
                                interactive=False,
                                height=600,
                            )
                        with gr.Column(scale=4):
                            output_info = gr.Markdown(
                                value="*Silakan unggah citra chest X-ray di panel kiri dan klik **Analisis** untuk melihat laporan diagnosis di sini.*",
                                label="Laporan Hasil Analisis",
                            )

            # --- Footer / Disclaimer ---
            gr.Markdown(
                """
                ---
                <div style="text-align: center; color: var(--body-text-color-subdued); font-size: 0.85rem; padding: 1rem 0;">
                    <strong>⚠️ Disclaimer Medis</strong><br>
                    Sistem ini merupakan purwarupa penelitian (thesis) berbasis deep learning untuk tujuan edukasi dan evaluasi akademis. 
                    Hasil analisis model AI tidak dapat dijadikan sebagai pengganti diagnosis klinis oleh dokter spesialis radiologi/paru-paru profesional.
                </div>
                """
            )

            # --- Click Handlers ---
            predict_btn_file.click(
                fn=self.predict_from_upload,
                inputs=[input_file, threshold_slider],
                outputs=[output_grid, output_kpis, output_info],
            )
            predict_btn_image.click(
                fn=self.predict_from_image,
                inputs=[input_image, threshold_slider],
                outputs=[output_grid, output_kpis, output_info],
            )

        return demo


def main():
    import os
    config = load_config("config.yaml")
    app = PneumoniaDetectionApp(config)
    demo = app.create_interface()
    port = int(os.environ.get("GRADIO_SERVER_PORT", config.app.port))
    demo.launch(
        server_name=config.app.host,
        server_port=port,
        share=config.app.share,
    )


if __name__ == "__main__":
    main()
