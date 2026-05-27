"""Gradio web application for pneumonia segmentation."""

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from src.config import Config, load_config
from src.model import build_model
from src.predict import predict_single
from src.utils import overlay_mask


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

        print(f"Model loaded on {self.device}")

    def predict(self, input_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Run prediction on input image.

        Args:
            input_image: Input image as numpy array (H, W, 3) or (H, W)

        Returns:
            Tuple of (overlay image, probability heatmap, info text)
        """
        # Convert to grayscale if RGB
        if input_image.ndim == 3:
            gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = input_image

        # Normalize
        gray = gray.astype(np.float32) / 255.0

        # Save to temp for reuse of predict_single
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, (gray * 255).astype(np.uint8))

        try:
            prob, original = predict_single(self.model, temp_path, self.config, self.device)
            pred_mask = (prob >= self.config.inference.threshold).astype(np.float32)

            # Overlay
            overlay = overlay_mask(
                original,
                pred_mask,
                color=tuple(self.config.inference.overlay_color),
                alpha=self.config.inference.overlay_alpha,
            )

            # Convert BGR to RGB for Gradio
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # Probability heatmap
            prob_colored = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
            prob_rgb = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)

            # Stats
            infected_pixels = int(pred_mask.sum())
            total_pixels = pred_mask.size
            infection_ratio = infected_pixels / total_pixels * 100

            info = (
                f"**Infection Detected:** {'Yes' if infected_pixels > 0 else 'No'}\n\n"
                f"**Infected Area:** {infected_pixels:,} pixels ({infection_ratio:.2f}%)\n\n"
                f"**Threshold:** {self.config.inference.threshold}\n\n"
                f"**Model:** {self.config.model.architecture} with {self.config.model.encoder_name}"
            )

            return overlay_rgb, prob_rgb, info

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        with gr.Blocks(title=self.config.app.title) as demo:
            gr.Markdown(f"# {self.config.app.title}")
            gr.Markdown(self.config.app.description)

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        type="numpy",
                        label="Upload Chest X-Ray",
                        sources=["upload"],
                    )
                    predict_btn = gr.Button("Analyze", variant="primary")

                with gr.Column():
                    output_overlay = gr.Image(type="numpy", label="Segmentation Result")
                    output_heatmap = gr.Image(type="numpy", label="Probability Heatmap")
                    output_info = gr.Markdown(label="Analysis Info")

            predict_btn.click(
                fn=self.predict,
                inputs=input_image,
                outputs=[output_overlay, output_heatmap, output_info],
            )

            gr.Examples(
                examples=[],
                inputs=input_image,
                label="Example X-Rays",
            )

            gr.Markdown(
                """
                ### Disclaimer
                This tool is for educational and research purposes only.
                It should not be used as a substitute for professional medical diagnosis.
                Always consult a qualified healthcare provider for medical advice.
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
