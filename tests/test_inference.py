import numpy as np
import torch

import voiceguard.inference as inference


class ConstantModel:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tensor([-10.0], device=features.device)


def test_real_explanation_does_not_claim_unflagged_windows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        inference, "prepare_audio", lambda waveform, sample_rate: (waveform, sample_rate)
    )
    monkeypatch.setattr(
        inference,
        "load_checkpoint",
        lambda _path, _device: (ConstantModel(), {"threshold": 0.5}),
    )
    monkeypatch.setattr(
        inference,
        "audio_windows",
        lambda _audio, _sample_rate: [
            (0.0, 0.5, np.ones(8000, dtype=np.float32))
        ],
    )
    monkeypatch.setattr(
        inference,
        "log_mel_spectrogram",
        lambda _segment: np.zeros((64, 2), dtype=np.float32),
    )

    result = inference.analyze_waveform(
        np.ones(8000, dtype=np.float32), 16_000, "unused.pt"
    )

    assert result["prediction"] == "Real"
    assert result["reasons"][1] == (
        "No overlapping audio window exceeds the threshold."
    )
