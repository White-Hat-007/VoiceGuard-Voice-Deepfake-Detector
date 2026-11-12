from __future__ import annotations

import argparse
import json
from pathlib import Path

from voiceguard.inference import analyze_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze an audio file with a trained VoiceGuard model."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "runs/wav2vec2_25k_fresh_test/checkpoint.pt"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyze_file(args.audio, args.model)
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"Detection result saved to: {args.output}")
    print(serialized)


if __name__ == "__main__":
    main()
