"""Configuration loader and validator."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    rsna_root: str
    train_dicom_dir: str
    test_dicom_dir: str
    train_labels_csv: str
    class_info_csv: str
    mask_dir: Optional[str] = None
    lung_mask_dir: Optional[str] = None
    rle_csv: Optional[str] = None


@dataclass
class PreprocessingConfig:
    image_size: list[int]
    window_level: int
    window_width: int
    apply_lung_window: bool
    normalize: bool


@dataclass
class ModelConfig:
    architecture: str
    encoder_name: str
    encoder_weights: Optional[str]
    in_channels: int
    classes: int
    activation: Optional[str]
    decoder_attention_type: Optional[str]


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    num_workers: int
    pin_memory: bool
    use_amp: bool
    accumulation_steps: int
    seed: int
    val_split: float
    k_folds: int
    fold: int
    stratified_split: bool
    early_stopping_patience: int
    encoder_freeze_epochs: int


@dataclass
class OptimizerConfig:
    type: str
    lr: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    type: str
    reduce_factor: float
    reduce_patience: int
    reduce_min_lr: float
    cosine_t_max: int
    cosine_eta_min: float


@dataclass
class LossConfig:
    type: str
    bce_weight: float
    tversky_alpha: float
    tversky_beta: float
    tversky_gamma: float
    pos_weight: float


@dataclass
class AugmentationConfig:
    enabled: bool
    horizontal_flip_prob: float
    rotation_limit: int
    shift_limit: float
    scale_limit: float
    elastic_transform_prob: float
    elastic_alpha: int
    elastic_sigma: int
    brightness_limit: float
    contrast_limit: float
    brightness_contrast_prob: float


@dataclass
class InferenceConfig:
    model_path: str
    threshold: float
    use_tta: bool
    tta_transforms: list[str]
    output_dir: str
    save_overlay: bool
    overlay_color: list[int]
    overlay_alpha: float


@dataclass
class EvaluationConfig:
    output_dir: str
    num_visualization_samples: int
    metrics: list[str]
    save_metrics_file: str


@dataclass
class ExplainabilityConfig:
    enabled: bool
    target_layer: Optional[str]
    output_dir: str
    colormap: str


@dataclass
class AppConfig:
    port: int
    host: str
    share: bool
    title: str
    description: str


@dataclass
class OutputConfig:
    root: str
    checkpoints_dir: str
    logs_dir: str
    tensorboard_dir: str


@dataclass
class Config:
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    augmentation: AugmentationConfig
    inference: InferenceConfig
    evaluation: EvaluationConfig
    explainability: ExplainabilityConfig
    app: AppConfig
    output: OutputConfig


def _to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a dataclass instance."""
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in field_types:
            expected = field_types[key]
            if isinstance(value, dict) and hasattr(expected, "__dataclass_fields__"):
                kwargs[key] = _to_dataclass(expected, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = _to_dataclass(Config, raw)

    # Validate dataset paths
    train_dir = Path(config.data.train_dicom_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Train DICOM directory not found: {train_dir}")

    labels_csv = Path(config.data.train_labels_csv)
    if not labels_csv.exists():
        raise FileNotFoundError(f"Train labels CSV not found: {labels_csv}")

    # Create output directories
    Path(config.output.root).mkdir(parents=True, exist_ok=True)
    Path(config.output.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    return config
