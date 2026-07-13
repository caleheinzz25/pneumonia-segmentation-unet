"""Automatic lung segmentation using torchxrayvision PSPNet.

Provides:
- LungSegmentationModel: GPU-accelerated batch lung segmentation (from Lungmask project)
- LungSegmenter: Lightweight wrapper for on-the-fly inference fallback

Uses torchxrayvision's pretrained PSPNet which segments 14 anatomical
structures. We extract Left Lung + Right Lung for a combined binary mask.
"""

import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _postprocess_lung_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphological post-processing to clean up a raw lung segmentation mask.

    Performs closing (fill small holes) followed by dilation (expand slightly)
    to produce a smooth, connected binary mask.

    Args:
        mask: Grayscale mask array (H, W), dtype uint8, values 0 or 255.

    Returns:
        Post-processed grayscale mask, same shape and dtype as input.
    """
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
    return dilated


class LungSegmentationModel:
    """GPU-accelerated lung segmentation using torchxrayvision's pretrained PSPNet.

    Supports both single-image and batch inference for maximum GPU utilization.
    Ported from the Lungmask project (~/Projects/github/Lungmask).
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"  Loading torchxrayvision segmentation model on {self.device}...")

        # Load the pretrained segmentation model
        self.model = xrv.baseline_models.chestx_det.PSPNet()
        self.model = self.model.to(self.device)
        self.model.eval()

        # Find indices for left and right lung
        self.target_names = self.model.targets
        self.left_lung_idx = self.target_names.index("Left Lung") if "Left Lung" in self.target_names else None
        self.right_lung_idx = self.target_names.index("Right Lung") if "Right Lung" in self.target_names else None

        if self.left_lung_idx is None or self.right_lung_idx is None:
            print(f"  WARNING: Lung indices not found. Available targets: {self.target_names}")
            for i, name in enumerate(self.target_names):
                print(f"    [{i}] {name}")

        print(f"  Model loaded. Left Lung idx={self.left_lung_idx}, Right Lung idx={self.right_lung_idx}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a uint8 grayscale image for the model.

        Args:
            image: Grayscale image as uint8 (H, W).

        Returns:
            Preprocessed tensor (1, 1, 512, 512).
        """
        # torchxrayvision expects images normalized to [-1024, 1024]
        img = image.astype(np.float32)
        img = xrv.datasets.normalize(img, 255)  # normalize to [-1024, 1024]

        # Ensure correct shape: (1, H, W)
        if img.ndim == 2:
            img = img[None, ...]

        # Resize to 512x512 (model expects this)
        transform = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(512),
        ])
        img = transform(img)

        return torch.from_numpy(img).unsqueeze(0)  # (1, 1, 512, 512)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Generate a lung mask from a uint8 grayscale image.

        Args:
            image: Grayscale image as uint8 (H, W).

        Returns:
            Binary mask (uint8, 0 or 255) at the original image resolution.
        """
        original_h, original_w = image.shape[:2]

        # Preprocess
        tensor = self.preprocess(image).to(self.device)

        # Inference
        pred = self.model(tensor)  # (1, num_targets, H, W)

        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy()[0]  # (num_targets, H, W)

        # Combine left and right lung masks
        lung_mask = np.zeros(pred.shape[1:], dtype=np.float32)
        if self.left_lung_idx is not None:
            lung_mask = np.maximum(lung_mask, pred[self.left_lung_idx])
        if self.right_lung_idx is not None:
            lung_mask = np.maximum(lung_mask, pred[self.right_lung_idx])

        # Threshold
        binary_mask = (lung_mask > 0.5).astype(np.uint8) * 255

        # Resize back to original dimensions
        mask_pil = Image.fromarray(binary_mask, mode="L")
        mask_pil = mask_pil.resize((original_w, original_h), Image.NEAREST)

        return np.array(mask_pil)

    @torch.no_grad()
    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Process a batch of images for better GPU utilization.

        Args:
            images: List of uint8 grayscale images.

        Returns:
            List of binary masks (uint8, 0 or 255) at original resolutions.
        """
        original_shapes = [(img.shape[0], img.shape[1]) for img in images]

        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)  # (B, 1, 512, 512)

        # Inference
        pred = self.model(batch)
        pred = torch.sigmoid(pred).cpu().numpy()  # (B, num_targets, H, W)

        results = []
        for i in range(len(images)):
            lung_mask = np.zeros(pred.shape[2:], dtype=np.float32)
            if self.left_lung_idx is not None:
                lung_mask = np.maximum(lung_mask, pred[i, self.left_lung_idx])
            if self.right_lung_idx is not None:
                lung_mask = np.maximum(lung_mask, pred[i, self.right_lung_idx])

            binary_mask = (lung_mask > 0.5).astype(np.uint8) * 255
            mask_pil = Image.fromarray(binary_mask, mode="L")
            mask_pil = mask_pil.resize(
                (original_shapes[i][1], original_shapes[i][0]), Image.NEAREST
            )
            results.append(np.array(mask_pil))

        return results


class LungSegmenter:
    """Lightweight wrapper for on-the-fly lung segmentation during inference.

    Used as fallback in predict.py / gradio_app.py when precomputed
    lung masks are not available for a given image.
    """

    def __init__(self, device: str = "cpu", threshold: float = 0.5):
        """Initialize the lung segmenter.

        Args:
            device: Device to run inference on ('cpu' or 'cuda').
            threshold: Probability threshold for binary mask.
        """
        self.device = device
        self.threshold = threshold
        self.model = LungSegmentationModel(device=device)
        print(f"[LungSeg] Ready on {device}")

    @torch.no_grad()
    def segment(
        self,
        image: np.ndarray,
        target_h: int = 512,
        target_w: int = 512,
    ) -> np.ndarray:
        """Generate a binary lung mask from a grayscale chest X-ray.

        Args:
            image: Grayscale image as float32, range [0, 1] or [0, 255].
                   Shape (H, W).
            target_h: Output mask height.
            target_w: Output mask width.

        Returns:
            Binary float32 mask of shape (target_h, target_w), where 1.0
            indicates lung region and 0.0 indicates non-lung.
        """
        # Convert to uint8 for LungSegmentationModel
        img = image.astype(np.float32)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Get mask (uint8, 0 or 255)
        mask_uint8 = self.model.predict(img)

        # Convert to float32 binary
        mask = (mask_uint8 > 127).astype(np.float32)

        # Resize to target dimensions
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(
                mask, (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return mask
