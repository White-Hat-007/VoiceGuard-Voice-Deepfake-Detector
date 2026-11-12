import json

from generate_evidence import (
    _capture_command,
    _save_terminal_screenshot,
    _select_example,
)
from show_results import build_training_report


def test_select_example_prefers_correct_held_out_prediction() -> None:
    evaluation = {
        "selected_threshold_from_validation": 0.5,
        "test_predictions": [
            {"label": "fake", "fake_probability": 0.2},
            {"label": "fake", "fake_probability": 0.9},
        ],
    }

    assert _select_example(evaluation, "fake")["fake_probability"] == 0.9


def test_terminal_screenshot_uses_captured_log_text(tmp_path) -> None:
    source = tmp_path / "training_console.log"
    output = tmp_path / "proof_terminal_training.png"
    source.write_text("Device: cuda\nEpoch 001 loss=0.25\n", encoding="utf-8")

    _save_terminal_screenshot(source, output, "Training terminal")

    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_capture_command_saves_actual_stdout(tmp_path) -> None:
    output = tmp_path / "command.log"

    _capture_command(["-c", "print('captured output')"], output)

    assert output.read_text(encoding="utf-8").strip() == "captured output"


def test_training_report_reads_saved_history(tmp_path) -> None:
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "device": "cuda",
                "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "completed_at_utc": "2026-01-01T00:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "training_history.csv").write_text(
        "epoch,train_loss,val_roc_auc,val_accuracy_at_val_threshold\n"
        "1,0.42,0.71,0.65\n"
        "2,0.31,0.80,0.74\n",
        encoding="utf-8",
    )

    report = build_training_report(tmp_path)

    assert "Best epoch by validation ROC-AUC: 2" in report
    assert "Best validation ROC-AUC: 80.0%" in report
    assert "Epochs completed: 2" in report
