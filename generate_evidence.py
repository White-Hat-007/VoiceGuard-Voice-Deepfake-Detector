from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = "#101827"
PANEL = "#192538"
TEXT = "#f4f7fb"
MUTED = "#a9b5c7"
CYAN = "#53d5e0"
CORAL = "#ff806e"
GREEN = "#6bd6a3"


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color("#314159")


def _save_training_plot(run_dir: Path, history: list[dict]) -> Path:
    epochs = [int(row["epoch"]) for row in history]
    loss = [float(row["train_loss"]) for row in history]
    auc = [float(row["val_roc_auc"]) for row in history]
    accuracy = [
        float(row["val_accuracy_at_val_threshold"]) for row in history
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BACKGROUND)
    figure.suptitle(
        "VOICEGUARD  /  TRAINING RUN",
        color=TEXT,
        fontsize=18,
        fontweight="bold",
        x=0.07,
        ha="left",
        y=0.99,
    )
    for axis in axes:
        _style_axis(axis)
        axis.grid(alpha=0.16, color=MUTED)
    axes[0].plot(epochs, loss, color=CORAL, marker="o", linewidth=2.5)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[1].plot(epochs, auc, color=CYAN, marker="o", label="Validation ROC-AUC")
    axes[1].plot(
        epochs,
        accuracy,
        color=GREEN,
        marker="s",
        label="Validation accuracy",
    )
    axes[1].set_title("Validation performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(facecolor=PANEL, edgecolor="#314159", labelcolor=TEXT)
    figure.text(
        0.07,
        0.025,
        "Real run history · metrics are dataset-specific, not a general accuracy guarantee",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.08, 0.98, 0.92))
    path = run_dir / "proof_training.png"
    figure.savefig(path, dpi=180, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)
    return path


def _save_evaluation_plot(run_dir: Path, evaluation: dict, manifest: dict) -> Path:
    metrics = evaluation["test_metrics"]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, (axis_matrix, axis_metrics) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        facecolor=BACKGROUND,
        gridspec_kw={"width_ratios": (1.1, 1.4)},
    )
    _style_axis(axis_matrix)
    _style_axis(axis_metrics)

    axis_matrix.imshow(matrix, cmap="GnBu", vmin=0)
    axis_matrix.set_title("Held-out test confusion matrix")
    axis_matrix.set_xticks([0, 1], ["Real", "Fake"])
    axis_matrix.set_yticks([0, 1], ["Real", "Fake"])
    axis_matrix.set_xlabel("Predicted")
    axis_matrix.set_ylabel("Actual")
    for row in range(2):
        for column in range(2):
            axis_matrix.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="#101827",
                fontsize=18,
                fontweight="bold",
            )

    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    values = [metrics[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc")]
    colors = [CYAN, GREEN, CORAL, "#b49aff", "#f0c46e"]
    bars = axis_metrics.barh(names, values, color=colors, height=0.62)
    axis_metrics.set_xlim(0, 1.08)
    axis_metrics.set_xlabel("Score")
    axis_metrics.set_title("Held-out test metrics")
    axis_metrics.invert_yaxis()
    axis_metrics.grid(axis="x", alpha=0.16, color=MUTED)
    for bar, value in zip(bars, values):
        axis_metrics.text(
            min(value + 0.025, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            color=TEXT,
            fontweight="bold",
        )
    figure.suptitle(
        "VOICEGUARD  /  HELD-OUT TEST RESULTS",
        color=TEXT,
        fontsize=18,
        fontweight="bold",
        x=0.07,
        ha="left",
        y=0.99,
    )
    fake_count = sum(item["label"] == "fake" for item in evaluation["test_predictions"])
    real_count = len(evaluation["test_predictions"]) - fake_count
    figure.text(
        0.07,
        0.025,
        f"Test files: {len(evaluation['test_predictions']):,}  ·  "
        f"{real_count:,} real / {fake_count:,} fake  ·  "
        f"GPU: {manifest.get('gpu_name') or 'CPU'}  ·  "
        "Measured on this benchmark subset only",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.08, 0.98, 0.92))
    path = run_dir / "proof_evaluation.png"
    figure.savefig(path, dpi=180, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)
    return path


def _select_example(
    evaluation: dict, expected_label: str
) -> dict:
    candidates = [
        item
        for item in evaluation["test_predictions"]
        if item["label"] == expected_label
    ]
    if not candidates:
        raise ValueError(f"No test example found for class {expected_label!r}.")
    correct = [
        item
        for item in candidates
        if (item["fake_probability"] >= evaluation["selected_threshold_from_validation"])
        == (expected_label == "fake")
    ]
    return max(
        correct or candidates,
        key=lambda item: abs(
            item["fake_probability"]
            - evaluation["selected_threshold_from_validation"]
        ),
    )


def _save_detection_plot(
    run_dir: Path,
    label: str,
    result: dict,
    source: dict,
) -> Path:
    regions = result["regions"]
    starts = [item["start_seconds"] for item in regions]
    ends = [item["end_seconds"] for item in regions]
    scores = [item["fake_probability"] for item in regions]
    colors = [CORAL if item["flagged"] else CYAN for item in regions]

    figure, axis = plt.subplots(figsize=(12, 4.8), facecolor=BACKGROUND)
    _style_axis(axis)
    centers = [(start + end) / 2 for start, end in zip(starts, ends)]
    for start, end, color in zip(starts, ends, colors):
        axis.axvspan(start, end, ymin=0, ymax=0.08, color=color, alpha=0.8)
    axis.plot(centers, scores, color=CYAN, linewidth=1.5, alpha=0.75)
    axis.scatter(centers, scores, c=colors, s=90, zorder=3)
    axis.axhline(
        result["threshold"],
        color="#f0c46e",
        linestyle="--",
        linewidth=2,
        label=f"Decision threshold {result['threshold']:.1%}",
    )
    axis.set_xlim(0, max(result["duration_seconds"], 0.2))
    axis.set_ylim(0, 1)
    axis.set_xlabel("Time in audio (seconds)")
    axis.set_ylabel("Fake score")
    axis.set_title(
        f"Reference {label.upper()} sample  ·  model says {result['prediction'].upper()}",
        loc="left",
        pad=16,
        color=TEXT,
    )
    axis.grid(axis="y", alpha=0.16, color=MUTED)
    axis.legend(facecolor=PANEL, edgecolor="#314159", labelcolor=TEXT)
    axis.text(
        0.01,
        0.96,
        f"File-level score {result['fake_probability']:.1%}   |   "
        f"reference label {source['label'].upper()}   |   "
        f"{sum(item['flagged'] for item in regions)}/{len(regions)} "
        "regions above threshold",
        transform=axis.transAxes,
        va="top",
        color=TEXT,
        fontsize=10,
        bbox={"facecolor": PANEL, "edgecolor": "#314159", "pad": 8},
    )
    figure.text(
        0.1,
        0.02,
        "Color: coral = region above threshold · cyan = below threshold. "
        "Model score is not forensic proof.",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.05, 0.08, 0.98, 0.96))
    path = run_dir / f"proof_detection_{label}.png"
    figure.savefig(path, dpi=180, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)
    return path


def _save_terminal_screenshot(
    source: Path,
    output: Path,
    title: str,
    max_lines: int = 48,
    command: str | None = None,
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Captured console log not found: {source}")
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 17)
        title_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 22)
        label_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
        label_font = font

    raw_content = source.read_bytes()
    if raw_content.startswith((b"\xff\xfe", b"\xfe\xff")):
        content = raw_content.decode("utf-16")
    else:
        content = raw_content.decode("utf-8-sig", errors="replace")
    if command:
        content = f"PS D:\\PROJECTS\\Voice Deepfake Detector> {command}\n" + content
    expanded = []
    for line in content.splitlines():
        expanded.extend(textwrap.wrap(line, width=145) or [""])
    visible = expanded[-max_lines:]
    line_height = 24
    header_height = 76
    footer_height = 36
    width = 1680
    height = header_height + len(visible) * line_height + footer_height
    image = Image.new("RGB", (width, height), "#0b111a")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (18, 14, width - 18, height - 14),
        radius=12,
        fill="#101722",
        outline="#304154",
        width=2,
    )
    draw.rectangle((20, 16, width - 20, header_height), fill="#1b2635")
    draw.text((40, 27), "PowerShell  ›  VoiceGuard", font=title_font, fill="#ecf2fa")
    draw.text(
        (width - 520, 32),
        "CAPTURED COMMAND OUTPUT",
        font=label_font,
        fill="#75d9c0",
    )
    y = header_height + 12
    for line in visible:
        draw.text((40, y), line, font=font, fill="#d4deeb")
        y += line_height
    draw.text(
        (40, height - footer_height + 1),
        f"Source log: {source.name}  ·  Exact captured stdout/stderr",
        font=label_font,
        fill="#91a2b8",
    )
    image.save(output)
    return output


def _capture_command(arguments: list[str], output: Path) -> str:
    command = [sys.executable, *arguments]
    with output.open("wb") as stream:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{subprocess.list2cmdline(command)}. See {output}."
        )
    return subprocess.list2cmdline(command)


def generate(run_dir: Path, data_dir: Path) -> dict:
    history_path = run_dir / "training_history.csv"
    evaluation_path = run_dir / "evaluation.json"
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    for path in (history_path, evaluation_path, manifest_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required training output not found: {path}")

    import csv

    with history_path.open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    outputs = {}
    example_dir = run_dir / "example_audio"
    example_dir.mkdir(exist_ok=True)
    for expected in ("fake", "real"):
        example = _select_example(evaluation, expected)
        input_path = data_dir / example["file"]
        example_copy = example_dir / f"reference_{expected}{input_path.suffix}"
        shutil.copy2(input_path, example_copy)
        output_path = run_dir / f"detection_{expected}.json"
        command = _capture_command(
            [
                "-u",
                str(Path(__file__).with_name("detect.py")),
                str(example_copy.resolve()),
                "--model",
                str(checkpoint_path.resolve()),
                "--output",
                str(output_path.resolve()),
            ],
            run_dir / f"detection_console_{expected}.log",
        )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["reference_label"] = example["label"]
        result["evaluation_file_score"] = example["fake_probability"]
        result["input_audio_evidence"] = str(
            example_copy.relative_to(run_dir)
        )
        result["dataset_source"] = manifest.get("dataset_acquisition", {}).get(
            "dataset_card"
        )
        result["expected_label_match"] = result["prediction"].lower() == expected
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        outputs[f"detection_{expected}"] = str(output_path.name)
        outputs[f"example_audio_{expected}"] = str(
            example_copy.relative_to(run_dir)
        )
        outputs[f"proof_detection_{expected}"] = str(
            _save_detection_plot(run_dir, expected, result, example).name
        )
        outputs[f"terminal_detection_{expected}"] = str(
            _save_terminal_screenshot(
                run_dir / f"detection_console_{expected}.log",
                run_dir / f"proof_terminal_detection_{expected}.png",
                f"{expected.title()} detection terminal",
                command=command,
            ).name
        )
        outputs[f"detection_console_log_{expected}"] = (
            f"detection_console_{expected}.log"
        )

    outputs["proof_training"] = str(
        _save_training_plot(run_dir, history).name
    )
    outputs["proof_evaluation"] = str(
        _save_evaluation_plot(run_dir, evaluation, manifest).name
    )
    training_command = _capture_command(
        [
            "-u",
            str(Path(__file__).with_name("show_results.py")),
            "--run-dir",
            str(run_dir.resolve()),
            "--training",
        ],
        run_dir / "training_console.log",
    )
    outputs["terminal_training"] = str(
        _save_terminal_screenshot(
            run_dir / "training_console.log",
            run_dir / "proof_terminal_training.png",
            "Recorded training history terminal",
            command=training_command,
        ).name
    )
    outputs["training_console_log"] = "training_console.log"
    results_command = _capture_command(
        [
            "-u",
            str(Path(__file__).with_name("show_results.py")),
            "--run-dir",
            str(run_dir.resolve()),
        ],
        run_dir / "results_console.log",
    )
    outputs["terminal_results"] = str(
        _save_terminal_screenshot(
            run_dir / "results_console.log",
            run_dir / "proof_terminal_results.png",
            "Held-out results terminal",
            command=results_command,
        ).name
    )
    outputs["results_console_log"] = "results_console.log"
    proof_dir = Path(__file__).resolve().parent / "proofs"
    proof_dir.mkdir(exist_ok=True)
    for label in ("upload", "fake", "real"):
        screenshot_name = f"proof_ui_{label}.png"
        screenshot_path = proof_dir / screenshot_name
        if screenshot_path.is_file():
            shutil.copy2(screenshot_path, run_dir / screenshot_name)
            outputs[f"ui_screenshot_{label}"] = screenshot_name
    for name in outputs.values():
        if name.endswith(".png"):
            shutil.copy2(run_dir / name, proof_dir / name)

    outputs["evidence_bundle"] = "VoiceGuard_Evidence.zip"
    evidence_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir.resolve()),
        "data_directory": str(data_dir.resolve()),
        "evidence_files": outputs,
        "metrics_source": "evaluation.json",
        "training_curve_source": "training_history.csv",
        "detection_source": "checkpoint.pt + held-out test audio files",
        "terminal_screenshot_source": (
            "PNG renderings of actual subprocess stdout/stderr logs. The "
            "training terminal reports saved training history; detection "
            "and results terminals come from executed project CLIs."
        ),
        "ui_screenshot_source": (
            "Browser screenshots captured from the live Streamlit app after "
            "opening the upload screen and analyzing the included held-out "
            "fake and real reference clips."
        ),
        "note": (
            "Generated from the completed training run; test performance is "
            "specific to this benchmark subset and is not guaranteed to "
            "generalize."
        ),
    }
    (run_dir / "evidence_manifest.json").write_text(
        json.dumps(evidence_manifest, indent=2), encoding="utf-8"
    )
    bundle_path = run_dir / outputs["evidence_bundle"]
    included_files = [
        run_dir / "checkpoint.pt",
        run_dir / "training_console.log",
        run_dir / "training_history.csv",
        run_dir / "evaluation.json",
        run_dir / "run_manifest.json",
        run_dir / "evidence_manifest.json",
        run_dir / "detection_fake.json",
        run_dir / "detection_real.json",
        *(
            run_dir / name
            for name in outputs.values()
            if name.endswith(".png")
        ),
        *(
            run_dir / name
            for name in outputs.values()
            if name.endswith(".log")
        ),
        *(example_dir / path.name for path in example_dir.iterdir()),
    ]
    readme = Path(__file__).with_name("README.md")
    if readme.is_file():
        included_files.append(readme)
    included_files = list(dict.fromkeys(included_files))
    with zipfile.ZipFile(
        bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5
    ) as archive:
        for path in included_files:
            if path.is_file():
                archive.write(
                    path,
                    arcname=(
                        f"README.md"
                        if path.resolve() == readme.resolve()
                        else str(path.relative_to(run_dir))
                    ),
                )
    return evidence_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PNG proof charts and JSON detection examples."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/wav2vec2_25k_fresh_test"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    for name in generate(args.run_dir, args.data_dir)["evidence_files"].values():
        if name != "VoiceGuard_Evidence.zip":
            print(args.run_dir / name)
    print(args.run_dir / "VoiceGuard_Evidence.zip")
    print(args.run_dir / "evidence_manifest.json")


if __name__ == "__main__":
    main()
