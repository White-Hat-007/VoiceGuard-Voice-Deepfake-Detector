from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .audio import (
    HOP_SECONDS,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    audio_windows,
    load_audio,
    log_mel_spectrogram,
    prepare_audio,
)
from .network import VoiceClassifier, load_checkpoint
from .transfer import TransferClassifier, normalize_and_pad


def analyze_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    checkpoint_path: str | Path,
) -> dict:
    audio, sample_rate = prepare_audio(waveform, sample_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    threshold = float(checkpoint["threshold"])
    windows = audio_windows(audio, sample_rate)

    probabilities = []
    with torch.inference_mode():
        if isinstance(model, TransferClassifier):
            batch = []
            lengths = []
            for start, end, segment in windows:
                normalized, valid_samples = normalize_and_pad(
                    segment,
                    valid_samples=round((end - start) * sample_rate),
                )
                batch.append(normalized)
                lengths.append(valid_samples)
            waveform_tensor = torch.from_numpy(np.stack(batch)).to(device)
            length_tensor = torch.tensor(
                lengths, dtype=torch.long, device=device
            )
            logits = model(waveform_tensor, length_tensor)
            probabilities = torch.sigmoid(logits).cpu().tolist()
        else:
            for _, _, segment in windows:
                features = torch.from_numpy(
                    log_mel_spectrogram(segment)
                ).unsqueeze(0).unsqueeze(0)
                logits = model(features.to(device))
                probabilities.append(float(torch.sigmoid(logits).item()))

    overall_probability = float(np.mean(probabilities))
    suspicious = [
        {
            "start_seconds": round(start, 2),
            "end_seconds": round(end, 2),
            "fake_probability": round(probability, 4),
            "flagged": probability >= threshold,
        }
        for (start, end, _), probability in zip(windows, probabilities)
    ]
    flagged = [segment for segment in suspicious if segment["flagged"]]
    if overall_probability >= threshold:
        prediction = "Fake"
        reasons = [
            f"The mean model score ({overall_probability:.1%}) exceeds the "
            f"validation-selected threshold ({threshold:.1%}).",
            f"{len(flagged)} of {len(suspicious)} overlapping audio windows "
            "also exceed that threshold.",
        ]
        if flagged:
            strongest = max(flagged, key=lambda item: item["fake_probability"])
            reasons.append(
                "The strongest flagged region is "
                f"{strongest['start_seconds']:.2f}–"
                f"{strongest['end_seconds']:.2f} seconds "
                f"({strongest['fake_probability']:.1%} model score)."
            )
    else:
        prediction = "Real"
        window_reason = (
            f"{len(flagged)} of {len(suspicious)} overlapping audio windows "
            "exceed the threshold; review the flagged regions below."
            if flagged
            else "No overlapping audio window exceeds the threshold."
        )
        reasons = [
            f"The mean model score ({overall_probability:.1%}) is below the "
            f"validation-selected threshold ({threshold:.1%}).",
            window_reason,
        ]

    return {
        "prediction": prediction,
        "fake_probability": round(overall_probability, 4),
        "threshold": round(threshold, 4),
        "duration_seconds": round(len(audio) / SAMPLE_RATE, 3),
        "model_device": str(device),
        "reasons": reasons,
        "regions": suspicious,
        "explanation_limit": (
            "Scores and regions are model evidence, not proof of how audio "
            "was made. Performance depends on the training data and may not "
            "generalize to unseen voices, codecs, or synthesis methods."
        ),
    }


def analyze_file(
    audio_path: str | Path, checkpoint_path: str | Path
) -> dict:
    waveform, sample_rate = load_audio(audio_path)
    return analyze_waveform(waveform, sample_rate, checkpoint_path)


def detect_windows(
    waveform: np.ndarray,
    sample_rate: int,
) -> list[tuple[float, float, np.ndarray]]:
    audio, sample_rate = prepare_audio(waveform, sample_rate)
    return audio_windows(audio, sample_rate, WINDOW_SECONDS, HOP_SECONDS)


def classifier_from_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[VoiceClassifier | TransferClassifier, dict, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    return model, checkpoint, device
