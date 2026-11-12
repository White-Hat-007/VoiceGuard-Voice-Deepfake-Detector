from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Model

from voiceguard.audio import (
    AUDIO_DECODER_STATS,
    SAMPLE_RATE,
    audio_windows,
    load_audio,
)
from voiceguard.training import CLASSES, collect_files, sha256_file
from voiceguard.transfer import (
    HIDDEN_SIZE,
    TRANSFER_MODEL_ID,
    TRANSFER_MODEL_REVISION,
    TransferClassifier,
    normalize_and_pad,
)


def select_samples(
    samples: list[tuple[Path, int]], maximum: int | None, seed: int
) -> list[tuple[Path, int]]:
    if maximum is None:
        return samples
    if maximum < 1:
        raise ValueError("--max-per-class must be positive.")
    rng = random.Random(seed)
    selected = []
    for label in CLASSES.values():
        group = [item for item in samples if item[1] == label]
        rng.shuffle(group)
        selected.extend(group[:maximum])
    return sorted(selected, key=lambda item: str(item[0]))


def extract_features(
    encoder: Wav2Vec2Model,
    samples: list[tuple[Path, int]],
    device: torch.device,
    batch_size: int,
    training: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    examples = []
    labels = []
    for path, label in samples:
        waveform, sample_rate = load_audio(path)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected audio resampled to {SAMPLE_RATE} Hz: {path}"
            )
        if waveform.size > 4 * SAMPLE_RATE:
            start = (
                rng.randint(0, waveform.size - 4 * SAMPLE_RATE)
                if training
                else (waveform.size - 4 * SAMPLE_RATE) // 2
            )
            waveform = waveform[start : start + 4 * SAMPLE_RATE]
        normalized, valid_samples = normalize_and_pad(waveform)
        examples.append((normalized, valid_samples))
        labels.append(label)

    features = np.empty(
        (len(examples), HIDDEN_SIZE * 2), dtype=np.float32
    )
    encoder.eval()
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset : offset + batch_size]
            waveforms = torch.from_numpy(
                np.stack([item[0] for item in batch])
            ).to(device)
            lengths = torch.tensor(
                [item[1] for item in batch],
                dtype=torch.long,
                device=device,
            )
            hidden = encoder(waveforms).last_hidden_state
            encoded_lengths = encoder._get_feat_extract_output_lengths(
                lengths
            ).clamp(min=1, max=hidden.size(1))
            positions = torch.arange(hidden.size(1), device=device)
            mask = positions.unsqueeze(0) < encoded_lengths.unsqueeze(1)
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            denominator = encoded_lengths.unsqueeze(1).to(
                hidden.dtype
            ).clamp(min=1)
            mean = (hidden * weights).sum(dim=1) / denominator
            variance = (
                ((hidden - mean.unsqueeze(1)) ** 2) * weights
            ).sum(dim=1) / denominator
            pooled = torch.cat(
                (mean, variance.clamp(min=1e-8).sqrt()), dim=1
            )
            features[offset : offset + len(batch)] = (
                pooled.float().cpu().numpy()
            )
            if (offset // batch_size + 1) % 50 == 0:
                print(
                    f"  extracted {min(offset + batch_size, len(examples))}"
                    f"/{len(examples)} audio embeddings"
                )
    return features, np.asarray(labels, dtype=np.int64)


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
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    predictions = scores >= threshold
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_encoder(source_checkpoint: Path, device: torch.device) -> Wav2Vec2Model:
    source = torch.load(
        source_checkpoint, map_location="cpu", weights_only=True
    )
    source_state = source["model_state_dict"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in source_state.items()
        if key.startswith("encoder.")
    }
    encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    result = encoder.load_state_dict(encoder_state, strict=False)
    if result.unexpected_keys or result.missing_keys != ["masked_spec_embed"]:
        raise RuntimeError(
            "The pinned ASVspoof2019 encoder checkpoint does not match the "
            "expected Wav2Vec2 architecture: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    del source, source_state, encoder_state
    gc.collect()
    encoder.to(device).eval()
    return encoder


def train(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source_checkpoint = Path(
        hf_hub_download(
            repo_id=TRANSFER_MODEL_ID,
            filename="best.pt",
            repo_type="model",
            revision=TRANSFER_MODEL_REVISION,
        )
    )
    splits = {
        split: select_samples(
            collect_files(data_dir, split), args.max_per_class, args.seed + i
        )
        for i, split in enumerate(("train", "val", "test"))
    }
    split_hashes = {}
    seen = {}
    for split, samples in splits.items():
        split_hashes[split] = {}
        for path, _ in samples:
            digest = sha256_file(path)
            if digest in seen and seen[digest] != split:
                raise ValueError(
                    f"Duplicate audio crosses {seen[digest]} and {split}: "
                    f"{path}"
                )
            seen[digest] = split
            split_hashes[split][path] = digest

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed(args.seed)
    started_at = datetime.now(timezone.utc).isoformat()
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Transfer initialization: {TRANSFER_MODEL_ID}")
    print(f"Transfer dataset: ASVspoof 2019 LA; revision {TRANSFER_MODEL_REVISION}")
    print(f"Fine-tuning device: {device}" + (f" ({gpu_name})" if gpu_name else ""))
    print(
        "Files: "
        + ", ".join(f"{key}={len(value)}" for key, value in splits.items())
    )
    encoder = _load_encoder(source_checkpoint, device)
    train_features, train_labels = extract_features(
        encoder,
        splits["train"],
        device,
        args.feature_batch_size,
        training=True,
        seed=args.seed,
    )
    val_features, val_labels = extract_features(
        encoder, splits["val"], device, args.feature_batch_size
    )
    model = TransferClassifier(encoder).to(device)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    best_auc = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    history_path = run_dir / "training_history.csv"

    class FrozenEmbeddingHead(nn.Module):
        def __init__(self, classifier: nn.Module) -> None:
            super().__init__()
            self.classifier = classifier

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.classifier(features).squeeze(1)

    head = FrozenEmbeddingHead(model.classifier).to(device)
    val_tensor = torch.from_numpy(val_features).to(device)
    val_target = torch.from_numpy(val_labels).to(device)
    positive_weight = torch.tensor(
        (train_labels == 0).sum() / max((train_labels == 1).sum(), 1),
        dtype=torch.float32,
        device=device,
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    for epoch in range(1, args.epochs + 1):
        model.classifier.train()
        started = time.monotonic()
        loss_sum = 0.0
        examples_seen = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0)
            examples_seen += labels.size(0)
        head.eval()
        with torch.inference_mode():
            val_scores = torch.sigmoid(head(val_tensor)).cpu().numpy()
        auc = float(roc_auc_score(val_labels, val_scores))
        threshold = choose_threshold(val_labels, val_scores)
        val_accuracy = float(
            accuracy_score(val_labels, val_scores >= threshold)
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(examples_seen, 1),
            "val_roc_auc": auc,
            "val_accuracy_at_val_threshold": val_accuracy,
            "val_threshold": threshold,
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
            f"val_auc={auc:.4f} val_accuracy={val_accuracy:.4f} "
            f"seconds={record['epoch_seconds']:.1f}"
        )
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "classifier_state_dict": model.classifier.state_dict(),
                    "threshold": threshold,
                },
                run_dir / "best_head.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} stale epochs.")
                break

    best = torch.load(run_dir / "best_head.pt", map_location=device, weights_only=True)
    model.classifier.load_state_dict(best["classifier_state_dict"])
    model.eval()
    with torch.inference_mode():
        val_scores = torch.sigmoid(head(val_tensor)).cpu().numpy()
        threshold = choose_threshold(val_labels, val_scores)

    test_features, test_labels = extract_features(
        encoder, splits["test"], device, args.feature_batch_size
    )
    test_tensor = torch.from_numpy(test_features).to(device)
    with torch.inference_mode():
        test_scores = torch.sigmoid(head(test_tensor)).cpu().numpy()
    metrics = metrics_at_threshold(test_labels, test_scores, threshold)
    evaluation = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_threshold_from_validation": threshold,
        "validation_roc_auc": float(roc_auc_score(val_labels, val_scores)),
        "test_metrics": metrics,
        "test_predictions": [
            {
                "file": str(path.relative_to(data_dir)),
                "label": "fake" if label else "real",
                "fake_probability": float(score),
            }
            for (path, label), score in zip(splits["test"], test_scores)
        ],
        "note": (
            "The evaluation split was not used for fitting or threshold "
            "selection. This is a sampled benchmark result, not a guarantee "
            "of performance on other data."
        ),
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    model_checkpoint = {
        "architecture": "wav2vec2_mean_std_transfer_head",
        "encoder_state_dict": {
            key: value.detach().cpu()
            for key, value in encoder.state_dict().items()
        },
        "classifier_state_dict": {
            key: value.detach().cpu()
            for key, value in model.classifier.state_dict().items()
        },
        "threshold": threshold,
        "config": {
            "sample_rate": SAMPLE_RATE,
            "window_seconds": 4,
            "pooling": "masked temporal mean and standard deviation",
            "pretrained_model_id": TRANSFER_MODEL_ID,
            "pretrained_revision": TRANSFER_MODEL_REVISION,
            "label_index": {"real": 0, "fake": 1},
        },
    }
    torch.save(model_checkpoint, run_dir / "checkpoint.pt")
    dataset_manifest = {
        split: [
            {
                "file": str(path.relative_to(data_dir)),
                "label": "fake" if label else "real",
                "sha256": split_hashes[split][path],
            }
            for path, label in samples
        ]
        for split, samples in splits.items()
    }
    manifest = {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "pretrained_model_id": TRANSFER_MODEL_ID,
        "pretrained_revision": TRANSFER_MODEL_REVISION,
        "pretrained_source_license": "MIT",
        "pretrained_training_dataset": "ASVspoof 2019 LA",
        "pretrained_model_card": (
            "https://huggingface.co/"
            "caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen"
        ),
        "training_dataset_provenance": (
            "ASVspoof 2021 DF sampled official train/validation/test splits."
        ),
        "device": str(device),
        "gpu_name": gpu_name,
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
            "feature_batch_size": args.feature_batch_size,
            "max_per_class_per_split": args.max_per_class,
        },
        "dataset_files": dataset_manifest,
        "audio_decoder_usage": dict(AUDIO_DECODER_STATS),
        "dataset_acquisition": json.loads(
            (data_dir / "acquisition_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Training evidence saved in: {run_dir.resolve()}")
    print(
        "Held-out test metrics: "
        + json.dumps(
            {
                key: metrics[key]
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                )
            },
            sort_keys=True,
        )
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer an ASVspoof2019 Wav2Vec2 detector and train its "
            "classifier head on ASVspoof2021 audio."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/wav2vec2_asvspoof2019_transfer"),
    )
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.feature_batch_size < 1
        or args.patience < 1
        or (args.max_per_class is not None and args.max_per_class < 1)
    ):
        parser.error("Training sizes and epoch counts must be positive.")
    train(args)


if __name__ == "__main__":
    main()
