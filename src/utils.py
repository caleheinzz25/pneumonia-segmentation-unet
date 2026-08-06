"""Utility functions for DICOM reading, mask generation, logging, and seeding."""

import atexit
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pydicom
import torch


class TeeStream:
    """Stream wrapper that writes to both a file and the original stream.

    Handles tqdm-style progress bars that use carriage return (``\r``) to
    overwrite the current line.  Only the final state of each line is
    written to the log file (i.e. when ``\n`` arrives), keeping it clean.
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self._line_buf = ""  # buffer for \r-overwritten content

    def write(self, data: str) -> int:
        # Always pass everything to the terminal immediately
        self.stream.write(data)

        # For the log file: only write completed lines
        for char in data:
            if char == "\r":
                # Carriage return → discard buffer (tqdm overwrite)
                self._line_buf = ""
            elif char == "\n":
                # Newline → this line is final, write it to the log
                self.log_file.write(self._line_buf + "\n")
                self.log_file.flush()
                self._line_buf = ""
            else:
                self._line_buf += char

        return len(data)

    def flush(self) -> None:
        # Only flush the terminal stream; do NOT dump _line_buf to the log.
        # Intermediate \r updates from tqdm call flush() constantly —
        # writing them would defeat the whole purpose of buffering.
        self.stream.flush()

    def close(self) -> None:
        """Flush any remaining buffered content and close the log file."""
        if self.log_file.closed:
            return
        if self._line_buf:
            self.log_file.write(self._line_buf + "\n")
            self._line_buf = ""
        self.log_file.flush()
        self.log_file.close()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self) -> bool:
        return self.stream.isatty()



def setup_logging(
    logs_dir: str = "outputs/logs",
    run_name: str = "run",
    resume: bool = False,
) -> Path:
    """Set up terminal logging to a file.

    Creates a new timestamped log file in ``logs_dir`` and redirects
    both ``sys.stdout`` and ``sys.stderr`` through a ``TeeStream`` so
    that all terminal output is simultaneously written to the log file.

    When ``resume=True``, appends to the original log file of the current
    run (tracked via a JSON state file) instead of creating a new one.

    Args:
        logs_dir: Directory where log files are stored.
        run_name: Prefix for the log filename (e.g. "train", "evaluate").
        resume: If True, append to the existing log file tracked in state.

    Returns:
        Path to the log file (new or existing).
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    state_file = logs_path / f"{run_name}_state.json"
    log_filepath = None

    if resume:
        # Read the exact log file path from state file
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    saved_log = state.get("current_log")
                    if saved_log:
                        potential_path = Path(saved_log)
                        if potential_path.exists():
                            log_filepath = potential_path
            except Exception as e:
                print(f"[WARNING] Could not read log state: {e}")

        # Fallback to the most recent if state file is missing
        if log_filepath is None:
            existing_logs = sorted(
                logs_path.glob(f"{run_name}_*.log"),
                key=lambda p: p.stat().st_mtime,
            )
            if existing_logs:
                log_filepath = existing_logs[-1]

    if log_filepath is None:
        # Create a new log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{run_name}_{timestamp}.log"
        log_filepath = logs_path / log_filename
        
        # Save this new log file path to the state file
        try:
            with open(state_file, "w") as f:
                json.dump({"current_log": str(log_filepath)}, f)
        except Exception as e:
            print(f"[WARNING] Could not write log state: {e}")

    mode = "a" if resume and log_filepath.exists() else "w"
    log_file = open(log_filepath, mode, encoding="utf-8")

    if mode == "a":
        log_file.write(f"\n{'=' * 80}\n")
        log_file.write(f"  [RESUME] Appending to log at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'=' * 80}\n\n")
        log_file.flush()

    stdout_tee = TeeStream(sys.__stdout__, log_file)
    stderr_tee = TeeStream(sys.__stderr__, log_file)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    # Ensure buffered content is flushed when the program exits
    atexit.register(stdout_tee.close)
    atexit.register(stderr_tee.close)

    action = "Resuming log" if mode == "a" else "Logging to"
    print(f"[LOG] {action} {log_filepath}")
    return log_filepath


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
    color: tuple[int, int, int] = (0, 140, 255),
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
