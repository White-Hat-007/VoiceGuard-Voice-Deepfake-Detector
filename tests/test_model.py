import torch

from voiceguard.network import VoiceClassifier
from voiceguard.training import choose_threshold


def test_classifier_returns_one_logit_per_audio_window() -> None:
    model = VoiceClassifier()
    logits = model(torch.zeros(3, 1, 64, 251))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_threshold_selection_balances_classes() -> None:
    labels = [0, 0, 0, 1]
    probabilities = [0.1, 0.2, 0.8, 0.7]

    assert choose_threshold(labels, probabilities) == 0.7
