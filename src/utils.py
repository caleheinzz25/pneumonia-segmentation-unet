"""Utility functions for DICOM reading, mask generation, and seeding."""

import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pydicom
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom(path: str | Path) -> np.ndarray:
    """Read a DICOM file and return pixel array as float32 numpy array."""
    dicom = pydicom.dcmread(str(path))
    pixel_array = dicom.pixel_array.astype(np.float32)

    # Apply rescale slope and intercept if present
    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    pixel_array = pixel_array * slope + intercept

    # Handle MONOCHROME1 where 0=white, max=black
    photometric = getattr(dicom, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    return pixel_array


def apply_window(
    image: np.ndarray,
    window_level: int = -600,
    window_width: int = 1500,
) -> np.ndarray:
    """Apply Hounsfield Unit windowing to DICOM image."""
    min_val = window_level - window_width // 2
    max_val = window_level + window_width // 2
    windowed = np.clip(image, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val)
    return windowed.astype(np.float32)


def bbox_to_mask(
    bboxes: list[dict[str, float]],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Convert bounding boxes to binary segmentation mask."""
    mask = np.zeros((image_height, image_width), dtype=np.float32)
    for bbox in bboxes:
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])
        if w > 0 and h > 0:
            mask[y : y + h, x : x + w] = 1.0
    return mask


def resize_image_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize image and mask to target size (width, height)."""
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized
    return image_resized, None


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image (H, W) to RGB (H, W, 3)."""
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay segmentation mask on image."""
    # Handle dtype conversion safely
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = image.copy()
    mask_bool = (mask > 0).astype(np.uint8)
    colored_mask = np.zeros_like(image)
    colored_mask[mask_bool == 1] = color

    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay
