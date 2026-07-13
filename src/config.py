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
    test_lung_mask_dir: Optional[str] = None
    rle_csv: Optional[str] = None
    negative_ratio: Optional[float] = None


@dataclass
class PreprocessingConfig:
    image_size: list[int]
    window_level: int
    window_width: int
    apply_lung_window: bool
    normalize: bool
    mean: Optional[list[float]] = None
    std: Optional[list[float]] = None
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: list[int] = field(default_factory=lambda: [8, 8])


@dataclass
class ModelConfig:
    architecture: str
    encoder_name: str
    encoder_weights: Optional[str]
    in_channels: int
    classes: int
    activation: Optional[str]
    decoder_attention_type: Optional[str]
    decoder_channels: Optional[list[int]] = None
    decoder_dropout: float = 0.0
    auxiliary_head: bool = False
    auxiliary_head_weight: float = 0.0


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
    early_stopping_pvariance: Optional[int] = None  # in case early_stopping_patience is used
    early_stopping_patience: int = 30
    encoder_freeze_epochs: int = 3
    gradient_clip_val: Optional[float] = None
    use_ema: bool = False
    ema_decay: float = 0.999
    monitor_metric: str = "val_dice"
    monitor_mode: str = "max"
    save_top_k: int = 1


@dataclass
class OptimizerConfig:
    type: str
    lr: float
    weight_decay: float
    encoder_lr_factor: Optional[float] = None


@dataclass
class SchedulerConfig:
    type: str
    reduce_factor: float
    reduce_patience: int
    reduce_min_lr: float
    cosine_t_max: int
    cosine_eta_min: float
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4


@dataclass
class LossConfig:
    type: str
    bce_weight: float
    tversky_alpha: float
    tversky_beta: float
    tversky_gamma: float
    pos_weight: float
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


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
    coarse_dropout_prob: float = 0.0
    coarse_dropout_holes: int = 8
    coarse_dropout_max_size: int = 32
    grid_distortion_prob: float = 0.0
    grid_distortion_limit: float = 0.1


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
    tta_merge_mode: str = "mean"


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
