from __future__ import annotations

import io
import subprocess
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 1.0
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
AUDIO_DECODER_STATS = {"soundfile": 0, "ffmpeg_fallback": 0}


def _decode_with_ffmpeg(source: str | Path | bytes) -> tuple[np.ndarray, int]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-err_detect",
        "ignore_err",
    ]
    if isinstance(source, bytes):
        command.extend(["-i", "pipe:0"])
        input_bytes = source
    else:
        command.extend(["-i", str(source)])
        input_bytes = None
    command.extend(
        [
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ]
    )
    result = subprocess.run(
        command,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "FFmpeg could not decode the audio"
            + (f": {detail}" if detail else ".")
        )
    if len(result.stdout) % 4:
        raise RuntimeError("FFmpeg returned incomplete float audio samples.")
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    return waveform, SAMPLE_RATE


@lru_cache(maxsize=2048)
def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        waveform, sample_rate = sf.read(
            path, dtype="float32", always_2d=True
        )
    except sf.LibsndfileError:
        waveform, sample_rate = _decode_with_ffmpeg(path)
        AUDIO_DECODER_STATS["ffmpeg_fallback"] += 1
        return prepare_audio(waveform, sample_rate)
    AUDIO_DECODER_STATS["soundfile"] += 1
    return prepare_audio(waveform, int(sample_rate))


def load_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        waveform, sample_rate = sf.read(
            io.BytesIO(audio_bytes), dtype="float32", always_2d=True
        )
    except sf.LibsndfileError:
        waveform, sample_rate = _decode_with_ffmpeg(audio_bytes)
        AUDIO_DECODER_STATS["ffmpeg_fallback"] += 1
        return prepare_audio(waveform, sample_rate)
    AUDIO_DECODER_STATS["soundfile"] += 1
    return prepare_audio(waveform, int(sample_rate))


def prepare_audio(
    waveform: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, int]:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    if waveform.size == 0:
        raise ValueError("Audio contains no samples.")

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError("Audio must be a mono or multi-channel waveform.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite sample values.")

    if sample_rate != SAMPLE_RATE:
        divisor = np.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32, copy=False)
    return audio, SAMPLE_RATE


def audio_windows(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> list[tuple[float, float, np.ndarray]]:
    window_size = int(round(sample_rate * window_seconds))
    hop_size = int(round(sample_rate * hop_seconds))
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("Window and hop durations must be positive.")
    if waveform.size == 0:
        raise ValueError("Audio contains no samples.")

    final_start = max(0, waveform.size - window_size)
    starts = list(range(0, final_start + 1, hop_size))
    if starts[-1] != final_start:
        starts.append(final_start)

    result = []
    for start in starts:
        segment = waveform[start : start + window_size]
        if segment.size < window_size:
            segment = np.pad(segment, (0, window_size - segment.size))
        end = min(start + window_size, waveform.size)
        result.append((start / sample_rate, end / sample_rate, segment))
    return result


def _mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
) -> np.ndarray:
    def to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(
        to_mel(np.array([0.0]))[0],
        to_mel(np.array([sample_rate / 2]))[0],
        n_mels + 2,
    )
    hz_points = to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = np.arange(
                left, center
            ) / (center - left)
        if right > center:
            filters[index, center:right] = (right - np.arange(
                center, right
            )) / (right - center)
    return filters


_MEL_FILTERBANK = _mel_filterbank()


def log_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32)
    frame_length = N_FFT
    padded = np.pad(audio, (frame_length // 2, frame_length // 2))
    frame_count = 1 + (padded.size - frame_length) // HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(
        padded, frame_length
    )[::HOP_LENGTH][:frame_count]
    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    mel_power = _MEL_FILTERBANK @ power.T
    log_mel = np.log(np.maximum(mel_power, 1e-10))
    log_mel = (log_mel - log_mel.mean()) / max(
        float(log_mel.std()), 1e-6
    )
    return log_mel.astype(np.float32)
