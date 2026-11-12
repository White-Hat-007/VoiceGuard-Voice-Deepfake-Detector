from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_report(run_dir: Path) -> str:
    evaluation = json.loads(
        (run_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    metrics = evaluation["test_metrics"]
    training_rows = (run_dir / "training_history.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    epoch_count = max(0, len(training_rows) - 1)
    matrix = metrics["confusion_matrix"]
    return "\n".join(
        (
            "VOICEGUARD | HELD-OUT TEST RESULTS",
            "=" * 58,
            f"Device: {manifest.get('gpu_name') or manifest['device']}",
            f"Training epochs completed: {epoch_count}",
            f"Test files: {len(evaluation['test_predictions'])}",
            f"Validation-selected threshold: "
            f"{evaluation['selected_threshold_from_validation']:.4f}",
            "",
            "TEST METRICS (measured; benchmark-specific)",
            f"  Accuracy:  {metrics['accuracy']:.1%}",
            f"  Precision: {metrics['precision']:.1%}",
            f"  Recall:    {metrics['recall']:.1%}",
            f"  F1:        {metrics['f1']:.1%}",
            f"  ROC-AUC:   {metrics['roc_auc']:.1%}",
            "",
            "CONFUSION MATRIX (rows=actual, columns=predicted)",
            "                 predicted real  predicted fake",
            f"  actual real         {matrix[0][0]:>5}          {matrix[0][1]:>5}",
            f"  actual fake         {matrix[1][0]:>5}          {matrix[1][1]:>5}",
            "",
            "Test is held out from training; results may not generalize.",
            f"Run evidence: {run_dir.resolve()}",
        )
    )


def build_training_report(run_dir: Path) -> str:
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    with (run_dir / "training_history.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise ValueError(f"No training epochs recorded in {run_dir}.")
    best = max(history, key=lambda row: float(row["val_roc_auc"]))
    lines = [
        "VOICEGUARD | RECORDED TRAINING HISTORY",
        "=" * 58,
        f"Device: {manifest.get('gpu_name') or manifest['device']}",
        f"Training start (UTC): {manifest['started_at_utc']}",
        f"Training complete (UTC): {manifest['completed_at_utc']}",
        f"Epochs completed: {len(history)}",
        f"Best epoch by validation ROC-AUC: {int(best['epoch'])}",
        f"Best validation ROC-AUC: {float(best['val_roc_auc']):.1%}",
        "",
        "RECORDED EPOCH METRICS",
        "  epoch   train loss   validation ROC-AUC   validation accuracy",
    ]
    lines.extend(
        f"  {int(row['epoch']):>5}   {float(row['train_loss']):>10.4f}"
        f"   {float(row['val_roc_auc']):>18.1%}"
        f"   {float(row['val_accuracy_at_val_threshold']):>19.1%}"
        for row in history
    )
    lines.extend(
        (
            "",
            "Source: saved training_history.csv and run_manifest.json.",
            f"Run evidence: {run_dir.resolve()}",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print measured test metrics from a saved VoiceGuard run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/wav2vec2_25k_fresh_test"),
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Print recorded epoch history instead of held-out test results.",
    )
    args = parser.parse_args()
    if args.training:
        print(build_training_report(args.run_dir))
    else:
        print(build_report(args.run_dir))


if __name__ == "__main__":
    main()
