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
    h, w = prep_config.image_size
    transforms = [A.Resize(height=h, width=w, interpolation=1)]

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
        ])

    transforms.append(
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0)
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_validation_transforms(prep_config: PreprocessingConfig) -> A.Compose:
    """Get validation/test preprocessing pipeline (no augmentation)."""
    h, w = prep_config.image_size
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])
