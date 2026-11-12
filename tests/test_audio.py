import numpy as np

from voiceguard.audio import (
    SAMPLE_RATE,
    audio_windows,
    log_mel_spectrogram,
    prepare_audio,
)


def test_prepare_audio_mixes_channels_and_resamples() -> None:
    stereo = np.ones((8_000, 2), dtype=np.float32)
    audio, sample_rate = prepare_audio(stereo, 8_000)

    assert sample_rate == SAMPLE_RATE
    assert audio.ndim == 1
    assert len(audio) == 16_000


def test_audio_windows_cover_short_audio_and_end() -> None:
    short = np.ones(SAMPLE_RATE, dtype=np.float32)
    short_windows = audio_windows(short)
    assert len(short_windows) == 1
    assert short_windows[0][1] == 1.0
    assert short_windows[0][2].size == 4 * SAMPLE_RATE

    long = np.ones(7 * SAMPLE_RATE, dtype=np.float32)
    long_windows = audio_windows(long)
    assert long_windows[0][0] == 0
    assert long_windows[-1][1] == 7
    assert long_windows[-1][2].size == 4 * SAMPLE_RATE


def test_log_mel_has_expected_shape_and_finite_values() -> None:
    features = log_mel_spectrogram(
        np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
    )

    assert features.shape == (64, 1 + 4 * SAMPLE_RATE // 256)
    assert np.isfinite(features).all()
