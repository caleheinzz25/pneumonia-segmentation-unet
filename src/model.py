"""Model definition: Attention UNet++ with pretrained encoder."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.config import ModelConfig


class AttentionUNetPlusPlus(nn.Module):
    """UNet++ with attention gates and pretrained encoder backbone.

    Uses segmentation-models-pytorch (SMP) UnetPlusPlus with:
    - Pretrained ImageNet encoder
    - SCSE attention in decoder (Spatial + Channel SE)
    - Deep supervision support (optional)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build model using SMP
        self.model = smp.create_model(
            arch="unetplusplus",
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.classes,
            activation=config.activation,
            decoder_attention_type=config.decoder_attention_type,
        )

        # Store encoder for Grad-CAM / feature extraction
        self.encoder = self.model.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output logits of shape (B, classes, H, W)
        """
        return self.model(x)

    def get_encoder_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate encoder features for deep supervision / analysis.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature maps from each encoder stage.
        """
        return self.encoder(x)

    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: ModelConfig, device: str = "cuda") -> AttentionUNetPlusPlus:
    """Build and move model to device.

    Args:
        config: Model configuration
        device: Target device

    Returns:
        Initialized model on target device
    """
    model = AttentionUNetPlusPlus(config)
    model = model.to(device)

    total_params = model.num_parameters
    print(f"Model: UNet++ with {config.encoder_name} encoder")
    print(f"Decoder attention: {config.decoder_attention_type}")
    print(f"Total trainable parameters: {total_params:,}")

    return model
