import numpy as np
import torch

from voiceguard.transfer import TransferClassifier, normalize_and_pad


class TinyEncoder(torch.nn.Module):
    def _get_feat_extract_output_lengths(
        self, lengths: torch.Tensor
    ) -> torch.Tensor:
        return torch.clamp(lengths // 2, min=1)

    def forward(self, waveforms: torch.Tensor):
        hidden = waveforms[:, :8:2].unsqueeze(-1).expand(-1, -1, 768)
        return type("EncoderOutput", (), {"last_hidden_state": hidden})()


def test_normalize_and_pad_preserves_valid_length() -> None:
    waveform = np.array([1.0, 3.0], dtype=np.float32)
    padded, valid_samples = normalize_and_pad(waveform)

    assert valid_samples == 2
    assert padded.shape == (64_000,)
    assert np.isclose(padded[:2].mean(), 0)
    assert np.all(padded[2:] == 0)


def test_transfer_classifier_pools_variable_length_features() -> None:
    model = TransferClassifier(TinyEncoder())
    logits = model(
        torch.randn(2, 64_000),
        torch.tensor([16, 8]),
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
