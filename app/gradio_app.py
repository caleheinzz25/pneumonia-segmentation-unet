"""Gradio web application for pneumonia segmentation."""

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


class PneumoniaSegmentationApp:
    """Gradio app for pneumonia segmentation from chest X-rays."""

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
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Core prediction from a file path (any supported format)."""
        path = Path(file_path)
        ext = path.suffix.lower()
        file_type = "DICOM" if ext in DICOM_EXTS else ext.upper().lstrip(".")

        try:
            gray = _load_image(path)
        except Exception as e:
            return None, None, f"❌ Gagal membaca file: {e}"

        # Save as temp PNG for predict_single
        with tempfile.NamedTemporaryFile(suffix=ext if ext in DICOM_EXTS else ".png", delete=False) as f:
            temp_path = f.name
            if ext in DICOM_EXTS:
                # Copy the original DICOM so predict_single can read it natively
                shutil.copy2(str(path), temp_path)
            else:
                cv2.imwrite(temp_path, (gray * 255).astype(np.uint8))

        try:
            prob, original = predict_single(
                self.model, temp_path, self.config, self.device,
                lung_segmenter=self.lung_segmenter,
            )
            pred_mask = (prob >= threshold).astype(np.float32)

            # --- Visual outputs -----------------------------------------------

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

            # Model info
            info_lines.append("---\n")
            info_lines.append("### ⚙️ Informasi Model\n")
            info_lines.append(f"| Parameter | Nilai |")
            info_lines.append(f"|---|---|")
            info_lines.append(
                f"| Arsitektur | {self.config.model.architecture} |"
            )
            info_lines.append(
                f"| Encoder | {self.config.model.encoder_name} |"
            )
            info_lines.append(
                f"| Attention | {self.config.model.decoder_attention_type} |"
            )
            info_lines.append(f"| Device | {self.device} |")

            info = "\n".join(info_lines)
            return overlay_rgb, prob_rgb, info

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def predict_from_upload(self, file, threshold: float):
        """Handle gr.File upload (DICOM or any image)."""
        if file is None:
            return None, None, "⚠️ Silakan upload file X-ray terlebih dahulu."
        return self._predict_from_file(file, threshold)

    def predict_from_image(self, input_image: np.ndarray, threshold: float):
        """Handle gr.Image upload (drag-and-drop regular images)."""
        if input_image is None:
            return None, None, "⚠️ Silakan upload gambar X-ray terlebih dahulu."

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
        custom_css = """
        /* Make result images large on desktop */
        .result-image img {
            min-height: 400px;
            object-fit: contain;
        }
        /* Mobile: stack columns vertically */
        @media (max-width: 768px) {
            .input-row { flex-direction: column !important; }
            .result-image img { min-height: 250px; }
        }
        """

        with gr.Blocks(
            title=self.config.app.title,
            theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
            css=custom_css,
        ) as demo:
            gr.Markdown(
                f"# 🫁 {self.config.app.title}\n"
                f"{self.config.app.description}"
            )

            # --- Top row: input controls ---
            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=2):
                    gr.Markdown(
                        "**Format yang didukung:** DICOM (`.dcm`), PNG, JPG, JPEG, BMP, TIFF, WEBP"
                    )
                    with gr.Tabs():
                        with gr.TabItem("📁 Upload File (DICOM & Gambar)"):
                            input_file = gr.File(
                                label="Upload file X-Ray",
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
                        with gr.TabItem("🖼️ Drag & Drop Gambar"):
                            input_image = gr.Image(
                                type="numpy",
                                label="Upload gambar X-Ray",
                                sources=["upload"],
                                height=250,
                            )
                            predict_btn_image = gr.Button(
                                "🔬 Analisis Gambar", variant="primary", size="lg"
                            )
                with gr.Column(scale=1):
                    threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=self.config.inference.threshold,
                        step=0.05,
                        label="🎚️ Threshold Deteksi",
                        info="Semakin rendah = lebih sensitif, semakin tinggi = lebih spesifik",
                    )

            # --- Middle: large result images (full width) ---
            gr.Markdown("### 📷 Hasil Visualisasi")
            with gr.Row():
                output_overlay = gr.Image(
                    type="numpy",
                    label="🖼️ Hasil Segmentasi",
                    elem_classes="result-image",
                    height=480,
                )
                output_heatmap = gr.Image(
                    type="numpy",
                    label="🌡️ Probability Heatmap",
                    elem_classes="result-image",
                    height=480,
                )

            # --- Bottom: analysis report ---
            output_info = gr.Markdown(
                value="*Upload gambar X-ray dan klik **Analisis** untuk memulai.*",
                label="📋 Laporan Analisis",
            )

            predict_btn_file.click(
                fn=self.predict_from_upload,
                inputs=[input_file, threshold_slider],
                outputs=[output_overlay, output_heatmap, output_info],
            )
            predict_btn_image.click(
                fn=self.predict_from_image,
                inputs=[input_image, threshold_slider],
                outputs=[output_overlay, output_heatmap, output_info],
            )

            gr.Markdown(
                """
                ---
                ### ⚠️ Disclaimer
                Tool ini hanya untuk tujuan **edukasi dan penelitian**.
                Tidak boleh digunakan sebagai pengganti diagnosis medis profesional.
                Selalu konsultasikan dengan tenaga kesehatan yang berkualifikasi
                untuk saran medis.
                """
            )

        return demo


def main():
    config = load_config("config.yaml")
    app = PneumoniaSegmentationApp(config)
    demo = app.create_interface()
    demo.launch(
        server_name=config.app.host,
        server_port=config.app.port,
        share=config.app.share,
    )


if __name__ == "__main__":
    main()
