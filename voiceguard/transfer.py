from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from .audio import SAMPLE_RATE

TRANSFER_MODEL_ID = (
    "caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen"
)
TRANSFER_MODEL_REVISION = "71250c8366e164e292e19a0451d85747038d2ed3"
WINDOW_SAMPLES = 4 * SAMPLE_RATE
HIDDEN_SIZE = 768


class TransferClassifier(nn.Module):
    def __init__(self, encoder: Wav2Vec2Model | None = None) -> None:
        super().__init__()
        if encoder is None:
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
            config.apply_spec_augment = False
            encoder = Wav2Vec2Model(config)
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encoder(waveforms).last_hidden_state
        output_lengths = self.encoder._get_feat_extract_output_lengths(lengths)
        output_lengths = output_lengths.clamp(min=1, max=encoded.size(1))
        positions = torch.arange(encoded.size(1), device=encoded.device)
        mask = positions.unsqueeze(0) < output_lengths.unsqueeze(1)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        denominator = output_lengths.unsqueeze(1).to(encoded.dtype).clamp(min=1)
        mean = (encoded * weights).sum(dim=1) / denominator
        variance = (
            ((encoded - mean.unsqueeze(1)) ** 2) * weights
        ).sum(dim=1) / denominator
        standard_deviation = variance.clamp(min=1e-8).sqrt()
        pooled = torch.cat((mean, standard_deviation), dim=1)
        return self.classifier(pooled).squeeze(1)


def normalize_and_pad(
    waveform: np.ndarray,
    valid_samples: int | None = None,
) -> tuple[np.ndarray, int]:
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if valid_samples is None:
        valid_samples = min(audio.size, WINDOW_SAMPLES)
    valid_samples = min(max(int(valid_samples), 0), audio.size, WINDOW_SAMPLES)
    audio = audio[:WINDOW_SAMPLES].copy()
    if valid_samples == 0:
        raise ValueError("Audio window contains no valid samples.")
    if audio.size < WINDOW_SAMPLES:
        audio = np.pad(
            audio, (0, WINDOW_SAMPLES - audio.size), mode="constant"
        )
    valid_audio = audio[:valid_samples]
    mean = float(valid_audio.mean())
    standard_deviation = max(float(valid_audio.std()), 1e-7)
    audio[:valid_samples] = (valid_audio - mean) / standard_deviation
    if valid_samples < WINDOW_SAMPLES:
        audio[valid_samples:] = 0.0
    return audio, valid_samples


def load_transfer_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[TransferClassifier, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = TransferClassifier()
    model.encoder.load_state_dict(
        checkpoint["encoder_state_dict"], strict=True
    )
    model.classifier.load_state_dict(
        checkpoint["classifier_state_dict"], strict=True
    )
    model.to(device).eval()
    return model, checkpoint
