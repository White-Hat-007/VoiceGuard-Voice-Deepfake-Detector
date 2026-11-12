from __future__ import annotations

import torch
from torch import nn


class VoiceClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs)).squeeze(1)


def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("architecture") == "wav2vec2_mean_std_transfer_head":
        from .transfer import TransferClassifier

        model = TransferClassifier()
        model.encoder.load_state_dict(
            checkpoint["encoder_state_dict"], strict=True
        )
        model.classifier.load_state_dict(
            checkpoint["classifier_state_dict"], strict=True
        )
    else:
        model = VoiceClassifier()
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
