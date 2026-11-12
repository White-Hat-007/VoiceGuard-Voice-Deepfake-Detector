from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import imageio_ffmpeg
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audio import (
    AUDIO_DECODER_STATS,
    HOP_SECONDS,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    audio_windows,
    load_audio,
    log_mel_spectrogram,
)
from .network import VoiceClassifier

CLASSES = {"real": 0, "fake": 1}
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg"}


def collect_files(data_dir: Path, split: str) -> list[tuple[Path, int]]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

    samples = []
    for class_name, label in CLASSES.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class directory not found: {class_dir}"
            )
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        samples.extend((path, label) for path in files)

    labels = {label for _, label in samples}
    if labels != set(CLASSES.values()):
        raise ValueError(
            f"Split '{split}' must contain supported audio in both "
            "real/ and fake/ directories."
        )
    return samples


class AudioDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        training: bool,
        seed: int,
    ) -> None:
        self.samples = samples
        self.training = training
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        waveform, sample_rate = load_audio(path)
        window_size = int(WINDOW_SECONDS * sample_rate)
        if waveform.size > window_size:
            if self.training:
                start = self.rng.randint(0, waveform.size - window_size)
            else:
                start = (waveform.size - window_size) // 2
            waveform = waveform[start : start + window_size]
        elif waveform.size < window_size:
            waveform = np.pad(waveform, (0, window_size - waveform.size))
        features = log_mel_spectrogram(waveform)
        return (
            torch.from_numpy(features).unsqueeze(0),
            torch.tensor(label, dtype=torch.float32),
        )


def evaluate_files(
    model: VoiceClassifier,
    samples: list[tuple[Path, int]],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for path, label in samples:
            waveform, sample_rate = load_audio(path)
            windows = audio_windows(
                waveform,
                sample_rate,
                window_seconds=WINDOW_SECONDS,
                hop_seconds=HOP_SECONDS,
            )
            window_features = [
                log_mel_spectrogram(segment) for _, _, segment in windows
            ]
            sample_probabilities = []
            for offset in range(0, len(window_features), batch_size):
                batch = torch.from_numpy(
                    np.stack(window_features[offset : offset + batch_size])
                ).unsqueeze(1)
                logits = model(batch.to(device))
                sample_probabilities.extend(
                    torch.sigmoid(logits).cpu().numpy().tolist()
                )
            labels.append(label)
            probabilities.append(float(np.mean(sample_probabilities)))
    return np.asarray(labels, dtype=np.int64), np.asarray(
        probabilities, dtype=np.float64
    )


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(
        np.concatenate(([0.0], probabilities, [1.0]))
    )
    return float(
        max(
            candidates,
            key=lambda threshold: (
                balanced_accuracy_score(
                    labels, probabilities >= threshold
                ),
                -abs(float(threshold) - 0.5),
            ),
        )
    )


def metrics_at_threshold(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    predictions = probabilities >= threshold
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix_labels": ["real", "fake"],
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["real", "fake"],
            output_dict=True,
            zero_division=0,
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        split: collect_files(data_dir, split)
        for split in ("train", "val", "test")
    }
    if args.max_per_class is not None:
        if args.max_per_class < 1:
            raise ValueError("--max-per-class must be positive.")
        for split_index, (split, samples) in enumerate(splits.items()):
            rng = random.Random(args.seed + split_index)
            selected = []
            for label in CLASSES.values():
                class_samples = [item for item in samples if item[1] == label]
                rng.shuffle(class_samples)
                selected.extend(class_samples[: args.max_per_class])
            splits[split] = sorted(selected, key=lambda item: str(item[0]))
    dataset_hashes = {}
    seen_hashes = {}
    for split, samples in splits.items():
        dataset_hashes[split] = {}
        for path, _ in samples:
            digest = sha256_file(path)
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError(
                    "Identical audio files must not cross dataset splits: "
                    f"{seen_hashes[digest]} and {split} both contain "
                    f"{path.name}."
                )
            seen_hashes[digest] = split
            dataset_hashes[split][path] = digest
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "--require-cuda was set, but this PyTorch installation cannot "
            "access CUDA."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(args.seed)

    started_at = datetime.now(timezone.utc).isoformat()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = None

    train_loader = DataLoader(
        AudioDataset(splits["train"], training=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = VoiceClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_class_counts = {
        "real": sum(label == 0 for _, label in splits["train"]),
        "fake": sum(label == 1 for _, label in splits["train"]),
    }
    positive_weight = torch.tensor(
        train_class_counts["real"] / train_class_counts["fake"],
        dtype=torch.float32,
        device=device,
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best_auc = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    history_path = output_dir / "training_history.csv"

    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else ""))
    print(
        "Files: "
        + ", ".join(
            f"{split}={len(samples)}" for split, samples in splits.items()
        )
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        started = time.monotonic()
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)

        val_labels, val_scores = evaluate_files(
            model, splits["val"], device, args.batch_size
        )
        val_auc = float(roc_auc_score(val_labels, val_scores))
        val_threshold = choose_threshold(val_labels, val_scores)
        val_accuracy = float(
            accuracy_score(val_labels, val_scores >= val_threshold)
        )
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(seen, 1),
            "val_roc_auc": val_auc,
            "val_accuracy_at_val_threshold": val_accuracy,
            "val_threshold": val_threshold,
            "epoch_seconds": time.monotonic() - started,
        }
        history.append(record)
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=record.keys())
            writer.writeheader()
            writer.writerows(history)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss={record['train_loss']:.4f} "
            f"val_auc={val_auc:.4f} "
            f"val_accuracy={val_accuracy:.4f} "
            f"seconds={record['epoch_seconds']:.1f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": val_threshold,
                    "config": {
                        "sample_rate": SAMPLE_RATE,
                        "window_seconds": WINDOW_SECONDS,
                        "n_mels": 64,
                        "seed": args.seed,
                    },
                },
                output_dir / "checkpoint.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} stale epochs.")
                break

    checkpoint = torch.load(
        output_dir / "checkpoint.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    val_labels, val_scores = evaluate_files(
        model, splits["val"], device, args.batch_size
    )
    threshold = choose_threshold(val_labels, val_scores)
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    test_labels, test_scores = evaluate_files(
        model, splits["test"], device, args.batch_size
    )
    test_metrics = metrics_at_threshold(test_labels, test_scores, threshold)
    evaluation = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_threshold_from_validation": threshold,
        "validation_roc_auc": float(roc_auc_score(val_labels, val_scores)),
        "test_metrics": test_metrics,
        "test_predictions": [
            {
                "file": str(path.relative_to(data_dir)),
                "label": "fake" if label else "real",
                "fake_probability": float(score),
            }
            for (path, label), score in zip(splits["test"], test_scores)
        ],
        "note": (
            "Test metrics are measured only on the supplied test split. "
            "They are not a guarantee of accuracy on other datasets."
        ),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )

    dataset_manifest = {
        split: [
            {
                "file": str(path.relative_to(data_dir)),
                "label": "fake" if label else "real",
                "sha256": dataset_hashes[split][path],
            }
            for path, label in samples
        ]
        for split, samples in splits.items()
    }
    manifest = {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": __import__("platform").python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_compute_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "best_epoch_by_validation_roc_auc": best_epoch,
        "best_validation_roc_auc": best_auc,
        "seed": args.seed,
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "max_per_class_per_split": args.max_per_class,
        },
        "train_class_counts": train_class_counts,
        "dataset_files": dataset_manifest,
        "audio_decoder_usage": dict(AUDIO_DECODER_STATS),
        "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(),
    }
    provenance_path = data_dir / "acquisition_manifest.json"
    if provenance_path.is_file():
        manifest["dataset_acquisition"] = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Training evidence saved in: {output_dir.resolve()}")
    print(
        "Held-out test metrics: "
        + json.dumps(
            {
                key: test_metrics[key]
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
            },
            sort_keys=True,
        )
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a binary audio deepfake classifier."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/asvspoof2021_full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional deterministic per-class sample cap for each split.",
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.patience < 1
        or (args.max_per_class is not None and args.max_per_class < 1)
    ):
        raise SystemExit(
            "epochs, batch-size, patience, and any sample cap must be positive."
        )
    train(args)


if __name__ == "__main__":
    main()
