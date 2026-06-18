"""Data augmentation and preprocessing transforms using Albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import AugmentationConfig, PreprocessingConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_training_transforms(
    aug_config: AugmentationConfig,
    prep_config: PreprocessingConfig,
) -> A.Compose:
    """Get training augmentation pipeline."""
    transforms = []

    if aug_config.enabled:
        transforms.extend([
            A.HorizontalFlip(p=aug_config.horizontal_flip_prob),
            A.Affine(
                translate_percent={"x": (-aug_config.shift_limit, aug_config.shift_limit),
                                   "y": (-aug_config.shift_limit, aug_config.shift_limit)},
                scale=(1 - aug_config.scale_limit, 1 + aug_config.scale_limit),
                rotate=(-aug_config.rotation_limit, aug_config.rotation_limit),
                p=0.5,
            ),
            A.ElasticTransform(
                alpha=aug_config.elastic_alpha,
                sigma=aug_config.elastic_sigma,
                p=aug_config.elastic_transform_prob,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.brightness_limit,
                contrast_limit=aug_config.contrast_limit,
                p=aug_config.brightness_contrast_prob,
            ),
            # Medical-specific augmentations
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(20, 60),
                hole_width_range=(20, 60),
                fill="random",
                p=0.2,
            ),
        ])

    transforms.append(
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0)
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_validation_transforms(prep_config: PreprocessingConfig) -> A.Compose:
    """Get validation/test preprocessing pipeline (no augmentation)."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])
