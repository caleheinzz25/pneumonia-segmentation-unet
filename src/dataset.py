"""Dataset classes for RSNA Pneumonia Detection Challenge."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import DataConfig, PreprocessingConfig
from src.utils import (
    apply_window,
    bbox_to_mask,
    grayscale_to_rgb,
    read_dicom,
    resize_image_mask,
)


class RSNADataset(Dataset):
    """RSNA Pneumonia Detection dataset.

    Supports:
    - Precomputed pneumonia masks (from mask_dir) as ground truth
    - Bounding box to mask conversion (fallback)
    - Lung masking (from lung_mask_dir) to focus input on lung regions only
    """

    def __init__(
        self,
        data_config: DataConfig,
        prep_config: PreprocessingConfig,
        patient_ids: list[str],
        transform=None,
        is_train: bool = True,
    ):
        self.data_config = data_config
        self.prep_config = prep_config
        self.patient_ids = patient_ids
        self.transform = transform
        self.is_train = is_train

        # Load labels CSV
        self.labels_df = pd.read_csv(data_config.train_labels_csv)
        self.labels_df["patientId"] = self.labels_df["patientId"].astype(str)

        # Filter to relevant patients
        self.labels_df = self.labels_df[
            self.labels_df["patientId"].isin(patient_ids)
        ].copy()

        # Group by patientId
        self.grouped = self.labels_df.groupby("patientId")

        # Build list of (patient_id, rows) tuples
        self.samples: list[tuple[str, pd.DataFrame]] = []
        for pid in patient_ids:
            if pid in self.grouped.groups:
                rows = self.grouped.get_group(pid)
                self.samples.append((pid, rows))

        self.train_dicom_dir = Path(data_config.train_dicom_dir)

        # Precomputed mask directories
        self.mask_dir = Path(data_config.mask_dir) if data_config.mask_dir else None
        self.lung_mask_dir = Path(data_config.lung_mask_dir) if data_config.lung_mask_dir else None

        # Check which patients have precomputed masks
        self.has_mask: dict[str, bool] = {}
        self.has_lung_mask: dict[str, bool] = {}

        if self.mask_dir and self.mask_dir.exists():
            for pid in patient_ids:
                self.has_mask[pid] = (self.mask_dir / f"{pid}.png").exists()
        if self.lung_mask_dir and self.lung_mask_dir.exists():
            for pid in patient_ids:
                self.has_lung_mask[pid] = (self.lung_mask_dir / f"{pid}.png").exists()

        # Log statistics
        if self.mask_dir and self.mask_dir.exists():
            num_masks = sum(self.has_mask.values())
            print(f"[Dataset] Precomputed pneumonia masks: {num_masks}/{len(patient_ids)} patients")
        if self.lung_mask_dir and self.lung_mask_dir.exists():
            num_lung = sum(self.has_lung_mask.values())
            print(f"[Dataset] Lung segmentation masks: {num_lung}/{len(patient_ids)} patients")

    def _load_precomputed_mask(
        self,
        patient_id: str,
        target_h: int,
        target_w: int,
    ) -> Optional[np.ndarray]:
        """Load precomputed pneumonia mask PNG.

        Returns binary float32 mask or None if not found.
        """
        if not self.mask_dir or not self.has_mask.get(patient_id, False):
            return None

        mask_path = self.mask_dir / f"{patient_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        mask = (mask > 127).astype(np.float32)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def _load_lung_mask(
        self,
        patient_id: str,
        target_h: int,
        target_w: int,
    ) -> Optional[np.ndarray]:
        """Load lung segmentation mask PNG.

        Returns binary float32 mask or None if not found.
        """
        if not self.lung_mask_dir or not self.has_lung_mask.get(patient_id, False):
            return None

        mask_path = self.lung_mask_dir / f"{patient_id}.png"
        lung_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if lung_mask is None:
            return None

        lung_mask = (lung_mask > 127).astype(np.float32)
        lung_mask = cv2.resize(
            lung_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        return lung_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        patient_id, rows = self.samples[idx]
        dicom_path = self.train_dicom_dir / f"{patient_id}.dcm"

        # Read DICOM
        image = read_dicom(dicom_path)
        h_orig, w_orig = image.shape

        # Apply lung windowing
        if self.prep_config.apply_lung_window:
            image = apply_window(
                image,
                window_level=self.prep_config.window_level,
                window_width=self.prep_config.window_width,
            )

        # Build mask from bounding boxes (fallback)
        bboxes = []
        for _, row in rows.iterrows():
            if pd.notna(row.get("x")) and pd.notna(row.get("y")):
                bboxes.append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                })

        if bboxes:
            mask = bbox_to_mask(bboxes, h_orig, w_orig)
        else:
            mask = np.zeros((h_orig, w_orig), dtype=np.float32)

        # Resize to target size
        target_w, target_h = self.prep_config.image_size[1], self.prep_config.image_size[0]
        image, mask = resize_image_mask(image, mask, (target_w, target_h))

        # Try to load precomputed pneumonia mask (overrides bbox mask)
        precomputed_mask = self._load_precomputed_mask(patient_id, target_h, target_w)
        if precomputed_mask is not None:
            mask = precomputed_mask

        # Load and apply lung mask to input image (focus on lung regions)
        lung_mask = self._load_lung_mask(patient_id, target_h, target_w)
        if lung_mask is not None:
            # Apply lung mask: zero out non-lung regions
            image = image * lung_mask
            # Also mask the ground truth to lung regions only
            mask = mask * lung_mask

        # Convert grayscale to RGB (3 channels for pretrained encoder)
        image = grayscale_to_rgb(image)

        # Apply augmentations / normalization
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Manual normalization if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Ensure mask is float tensor with shape (1, H, W)
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()

        return {
            "image": image,
            "mask": mask,
            "patient_id": patient_id,
        }


def get_train_val_split(
    data_config: DataConfig,
    val_split: float = 0.2,
    seed: int = 42,
    stratified: bool = True,
) -> tuple[list[str], list[str]]:
    """Split patients into train and validation sets.

    Uses patient-level split to avoid data leakage.
    """
    labels_df = pd.read_csv(data_config.train_labels_csv)
    labels_df["patientId"] = labels_df["patientId"].astype(str)

    # Determine if patient has pneumonia (Target=1) or not
    patient_targets = labels_df.groupby("patientId")["Target"].max().reset_index()
    patient_ids = patient_targets["patientId"].tolist()
    targets = patient_targets["Target"].tolist()

    if stratified:
        from sklearn.model_selection import train_test_split

        train_ids, val_ids = train_test_split(
            patient_ids,
            test_size=val_split,
            random_state=seed,
            stratify=targets,
        )
    else:
        np.random.seed(seed)
        np.random.shuffle(patient_ids)
        split_idx = int(len(patient_ids) * (1 - val_split))
        train_ids = patient_ids[:split_idx]
        val_ids = patient_ids[split_idx:]

    return train_ids, val_ids
