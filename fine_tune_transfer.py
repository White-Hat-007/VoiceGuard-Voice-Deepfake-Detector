from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from voiceguard.audio import AUDIO_DECODER_STATS, SAMPLE_RATE, load_audio
from voiceguard.network import load_checkpoint
from voiceguard.training import (
    collect_files,
    sha256_file,
)
from voiceguard.transfer import (
    TRANSFER_MODEL_ID,
    TRANSFER_MODEL_REVISION,
    TransferClassifier,
    normalize_and_pad,
)
from train_transfer import (
    choose_threshold,
    metrics_at_threshold,
    select_samples,
)

WINDOW_SAMPLES = 4 * SAMPLE_RATE


class PreparedDataset(Dataset):
    def __init__(
        self,
        waveforms_path: Path,
        lengths_path: Path,
        labels: np.ndarray,
    ) -> None:
        self.waveforms = np.load(waveforms_path, mmap_mode="r")
        self.lengths = np.load(lengths_path, mmap_mode="r")
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def close(self) -> None:
        for array in (self.waveforms, self.lengths):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waveform = torch.from_numpy(np.array(self.waveforms[index]))
        return (
            waveform,
            torch.tensor(int(self.lengths[index]), dtype=torch.long),
            torch.tensor(float(self.labels[index]), dtype=torch.float32),
        )


def prepare_split(
    split: str,
    samples: list[tuple[Path, int]],
    cache_dir: Path,
) -> tuple[PreparedDataset, np.ndarray]:
    waveforms_path = cache_dir / f"{split}_waveforms.npy"
    lengths_path = cache_dir / f"{split}_lengths.npy"
    labels = np.asarray([label for _, label in samples], dtype=np.int64)
    waveforms = np.lib.format.open_memmap(
        waveforms_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(samples), WINDOW_SAMPLES),
    )
    lengths = np.lib.format.open_memmap(
        lengths_path, mode="w+", dtype=np.int32, shape=(len(samples),)
    )
    for index, (path, _) in enumerate(samples):
        waveform, sample_rate = load_audio(path)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate for {path}: {sample_rate}")
        if waveform.size > WINDOW_SAMPLES:
            start = (waveform.size - WINDOW_SAMPLES) // 2
            waveform = waveform[start : start + WINDOW_SAMPLES]
        normalized, valid = normalize_and_pad(waveform)
        waveforms[index] = normalized
        lengths[index] = valid
        if (index + 1) % 500 == 0:
            print(f"  prepared {split}: {index + 1}/{len(samples)} audio clips")
    waveforms.flush()
    lengths.flush()
    del waveforms, lengths
    dataset = PreparedDataset(waveforms_path, lengths_path, labels)
    load_audio.cache_clear()
    return dataset, labels


def _scores(
    model: TransferClassifier,
    batches: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    scores = []
    with torch.inference_mode():
        for waveforms, lengths, batch_labels in batches:
            waveforms = waveforms.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                logits = model(waveforms, lengths)
            scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
            labels.extend(batch_labels.numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(
        scores, dtype=np.float64
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    source_run = Path(args.source_run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_checkpoint = source_run / "checkpoint.pt"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Transfer checkpoint not found: {source_checkpoint}")

    splits = {
        split: select_samples(
            collect_files(data_dir, split), args.max_per_class, args.seed + index
        )
        for index, split in enumerate(("train", "val", "test"))
    }
    split_hashes = {}
    seen_hashes = {}
    for split, samples in splits.items():
        split_hashes[split] = {}
        for path, _ in samples:
            digest = sha256_file(path)
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError(
                    f"Duplicate file crosses {seen_hashes[digest]} and {split}: "
                    f"{path}"
                )
            seen_hashes[digest] = split
            split_hashes[split][path] = digest

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"Fine-tuning model: {TRANSFER_MODEL_ID}")
    print(f"Initialized from ASVspoof2019 transfer run: {source_run.resolve()}")
    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else ""))
    print(
        "Files: "
        + ", ".join(f"{key}={len(value)}" for key, value in splits.items())
    )

    model, source_metadata = load_checkpoint(str(source_checkpoint), device)
    if not isinstance(model, TransferClassifier):
        raise TypeError("Source checkpoint is not a Wav2Vec2 transfer model.")
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.encoder.layers[-args.unfrozen_layers :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    model.train()
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()

    parameter_groups = [
        {
            "params": [
                parameter
                for parameter in model.encoder.parameters()
                if parameter.requires_grad
            ],
            "lr": args.encoder_learning_rate,
        },
        {
            "params": list(model.classifier.parameters()),
            "lr": args.head_learning_rate,
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay
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
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    with tempfile.TemporaryDirectory(
        prefix="voiceguard-wav2vec-",
        dir=output_dir,
    ) as temporary:
        cache_dir = Path(temporary)
        prepared_train, _ = prepare_split(
            "train", splits["train"], cache_dir
        )
        prepared_val, val_labels = prepare_split(
            "val", splits["val"], cache_dir
        )
        prepared_test, test_labels = prepare_split(
            "test", splits["test"], cache_dir
        )
        train_loader = DataLoader(
            prepared_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_loader = DataLoader(
            prepared_val,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            prepared_test,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            started = time.monotonic()
            loss_sum = 0.0
            examples_seen = 0
            for step, (waveforms, lengths, labels) in enumerate(
                train_loader, start=1
            ):
                waveforms = waveforms.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bfloat16,
                ):
                    logits = model(waveforms, lengths)
                    loss = loss_function(logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=1.0,
                )
                optimizer.step()
                loss_sum += float(loss.item()) * labels.size(0)
                examples_seen += labels.size(0)
                if step % 100 == 0:
                    print(
                        f"  epoch {epoch:03d} step {step:04d}/"
                        f"{len(train_loader)} loss={loss.item():.4f}"
                    )

            val_labels_observed, val_scores = _scores(
                model, val_loader, device, use_bfloat16
            )
            if not np.array_equal(val_labels, val_labels_observed):
                raise RuntimeError("Validation loader changed label order.")
            val_auc = float(roc_auc_score(val_labels, val_scores))
            threshold = choose_threshold(val_labels, val_scores)
            val_accuracy = float(
                accuracy_score(val_labels, val_scores >= threshold)
            )
            record = {
                "epoch": epoch,
                "train_loss": loss_sum / max(examples_seen, 1),
                "val_roc_auc": val_auc,
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
                f"val_auc={val_auc:.4f} val_accuracy={val_accuracy:.4f} "
                f"seconds={record['epoch_seconds']:.1f}"
            )
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "encoder_state_dict": model.encoder.state_dict(),
                        "classifier_state_dict": model.classifier.state_dict(),
                        "threshold": threshold,
                    },
                    output_dir / "best_checkpoint.pt",
                )
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping after {args.patience} stale epochs.")
                    break

        best_state = torch.load(
            output_dir / "best_checkpoint.pt",
            map_location=device,
            weights_only=True,
        )
        model.encoder.load_state_dict(best_state["encoder_state_dict"])
        model.classifier.load_state_dict(best_state["classifier_state_dict"])
        model.eval()
        val_labels_observed, val_scores = _scores(
            model, val_loader, device, use_bfloat16
        )
        threshold = choose_threshold(val_labels_observed, val_scores)
        test_labels_observed, test_scores = _scores(
            model, test_loader, device, use_bfloat16
        )
        if not np.array_equal(test_labels, test_labels_observed):
            raise RuntimeError("Test loader changed label order.")

        model_checkpoint = {
            "architecture": "wav2vec2_mean_std_transfer_head",
            "encoder_state_dict": {
                key: value.detach().cpu()
                for key, value in model.encoder.state_dict().items()
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
        torch.save(model_checkpoint, output_dir / "checkpoint.pt")
        prepared_train.close()
        prepared_val.close()
        prepared_test.close()
        del train_loader, val_loader, test_loader
        del prepared_train, prepared_val, prepared_test

    test_metrics = metrics_at_threshold(
        test_labels_observed, test_scores, threshold
    )
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
            for (path, label), score in zip(
                splits["test"], test_scores
            )
        ],
        "note": (
            "Threshold is selected using validation only. The held-out test "
            "split is not used for training or model selection."
        ),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8"
    )
    split_hashes_manifest = {
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
        "pretrained_model_revision": TRANSFER_MODEL_REVISION,
        "pretrained_source_license": "MIT",
        "pretrained_training_dataset": "ASVspoof 2019 LA",
        "pretrained_model_card": (
            "https://huggingface.co/"
            "caa-speech-detection-asvspoof2019/wav2vec2-v2-unfrozen"
        ),
        "fine_tuning_source_run": str(source_run.resolve()),
        "fine_tuning_source_threshold": source_metadata["threshold"],
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
            "eval_batch_size": args.eval_batch_size,
            "encoder_learning_rate": args.encoder_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "unfrozen_encoder_layers": args.unfrozen_layers,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "max_per_class_per_split": args.max_per_class,
            "bfloat16_autocast": use_bfloat16,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "train_class_counts": train_class_counts,
        "dataset_files": split_hashes_manifest,
        "audio_decoder_usage": dict(AUDIO_DECODER_STATS),
        "dataset_acquisition": json.loads(
            (data_dir / "acquisition_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Fine-tuning evidence saved in: {output_dir.resolve()}")
    print(
        "Held-out test metrics: "
        + json.dumps(
            {
                key: test_metrics[key]
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
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the top Wav2Vec2 layers and classification head on "
            "the ASVspoof2021 training split."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("runs/wav2vec2_asvspoof2019_transfer"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/wav2vec2_asvspoof2019_finetuned"),
    )
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfrozen-layers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.eval_batch_size < 1
        or args.unfrozen_layers < 1
        or args.unfrozen_layers > 12
        or args.patience < 1
        or args.max_per_class < 1
    ):
        parser.error("Training sizes, layer count, and epoch counts must be positive.")
    train(args)


if __name__ == "__main__":
    main()
