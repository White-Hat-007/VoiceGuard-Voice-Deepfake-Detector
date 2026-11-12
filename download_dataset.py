from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_ID = "MoaazTalab/ASVspoof_2021_DF_Balanced_Normalized"
DATASET_REVISION = "479fbf291c9da1fca8c6b20d68b84442b2699f6b"
DATASET_CARD = (
    "https://huggingface.co/datasets/"
    "MoaazTalab/ASVspoof_2021_DF_Balanced_Normalized"
)
UPSTREAM_CHALLENGE = "https://www.asvspoof.org/index2021.html"
TEST_SHARD_INDICES = (0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 62)
SPLIT_SHARDS = {
    "train": tuple(range(10)),
    "val": (0, 1),
    "test": TEST_SHARD_INDICES,
}
LABEL_NAMES = {0: "fake", 1: "real"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_test_shards(shard_indices: list[int]) -> tuple[int, ...]:
    if (
        not shard_indices
        or len(set(shard_indices)) != len(shard_indices)
        or any(index < 0 or index >= 63 for index in shard_indices)
    ):
        raise ValueError("Test shard indices must be unique values from 0 to 62.")
    return tuple(shard_indices)


def load_excluded_test_hashes(manifest_path: Path | None) -> set[str]:
    if manifest_path is None:
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        test_files = manifest["splits"]["test"]["selected_files"]
        return {
            item["sha256"]
            for class_files in test_files.values()
            for item in class_files
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Test acquisition manifest has an unexpected structure: "
            f"{manifest_path}"
        ) from error


def _sample_split(
    split: str,
    count_per_class: int,
    source_dir: Path,
    output_dir: Path,
    seed: int,
    shard_indices: tuple[int, ...],
    excluded_hashes: set[str] | None = None,
) -> dict:
    rng = random.Random(seed)
    reservoirs: dict[int, list[dict]] = {0: [], 1: []}
    seen_counts = {0: 0, 1: 0}
    seen_hashes = {0: set(), 1: set()}
    shard_records = []

    for shard_index in shard_indices:
        filename = f"data/{split if split != 'val' else 'validation'}"
        total_shards = (
            10 if split == "train" else 2 if split == "val" else 63
        )
        source_split = "validation" if split == "val" else split
        filename = (
            f"data/{source_split}-{shard_index:05d}-of-"
            f"{total_shards:05d}.parquet"
        )
        parquet_path = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=filename,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=source_dir,
            )
        )
        parquet_file = pq.ParquetFile(parquet_path)
        shard_label_counts = {0: 0, 1: 0}
        shard_row_offset = 0
        for batch in parquet_file.iter_batches(
            batch_size=256, columns=["audio", "label"]
        ):
            for row in batch.to_pylist():
                label = int(row["label"])
                if label not in LABEL_NAMES:
                    raise ValueError(
                        f"Unexpected label {label} in {parquet_path}."
                    )
                audio = row["audio"]
                audio_bytes = audio.get("bytes") if audio else None
                if not audio_bytes:
                    raise ValueError(
                        f"Missing audio bytes in {parquet_path}, "
                        f"row {shard_row_offset}."
                    )
                digest = _sha256(audio_bytes)
                shard_label_counts[label] += 1
                shard_row_offset += 1
                if excluded_hashes and digest in excluded_hashes:
                    continue
                if digest in seen_hashes[label]:
                    continue
                seen_hashes[label].add(digest)
                seen_counts[label] += 1
                record = {
                    "bytes": audio_bytes,
                    "sha256": digest,
                    "source_shard": filename,
                    "source_row": shard_row_offset - 1,
                    "source_path": audio.get("path"),
                }
                reservoir = reservoirs[label]
                if len(reservoir) < count_per_class:
                    reservoir.append(record)
                else:
                    replacement = rng.randrange(seen_counts[label])
                    if replacement < count_per_class:
                        reservoir[replacement] = record

        shard_records.append(
            {
                "file": filename,
                "rows": shard_row_offset,
                "label_counts": {
                    LABEL_NAMES[label]: count
                    for label, count in shard_label_counts.items()
                },
            }
        )
        print(
            f"{split}: read {filename} "
            f"(rows={shard_row_offset}, "
            f"fake={shard_label_counts[0]}, real={shard_label_counts[1]})"
        )

    for label, class_name in LABEL_NAMES.items():
        if len(reservoirs[label]) < count_per_class:
            raise ValueError(
                f"Only found {len(reservoirs[label])} unique {class_name} "
                f"items in selected {split} shards; requested "
                f"{count_per_class}. Add shards or lower the requested count."
            )

    written = {}
    for label, class_name in LABEL_NAMES.items():
        target_dir = output_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        records = sorted(reservoirs[label], key=lambda item: item["sha256"])
        written[class_name] = []
        for index, record in enumerate(records):
            output_path = target_dir / f"{class_name}_{index:05d}.flac"
            output_path.write_bytes(record["bytes"])
            written[class_name].append(
                {
                    "file": str(output_path.relative_to(output_dir)),
                    "sha256": record["sha256"],
                    "source_shard": record["source_shard"],
                    "source_row": record["source_row"],
                    "source_path": record["source_path"],
                }
            )
    return {
        "requested_per_class": count_per_class,
        "unique_candidates_seen": {
            LABEL_NAMES[label]: count for label, count in seen_counts.items()
        },
        "selected_per_class": {
            name: len(items) for name, items in written.items()
        },
        "selected_files": written,
        "source_shards": shard_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and deterministically sample the ASVspoof 2021 DF "
            "benchmark into VoiceGuard's train/val/test folders."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Reuse a local Hugging Face dataset cache from another data directory.",
    )
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--val-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument(
        "--test-shards",
        type=int,
        nargs="+",
        default=list(TEST_SHARD_INDICES),
        help=(
            "Official test parquet shard indices to sample; choose unused "
            "shards for a fresh final evaluation."
        ),
    )
    parser.add_argument(
        "--exclude-test-manifest",
        type=Path,
        help=(
            "Exclude audio hashes in a prior acquisition manifest from the "
            "new test sample."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(
        args.train_per_class, args.val_per_class, args.test_per_class
    ) < 1:
        parser.error("sample counts must all be positive.")
    try:
        test_shards = validate_test_shards(args.test_shards)
    except ValueError as error:
        parser.error(str(error))

    args.data_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir or args.data_dir / "_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    excluded_test_hashes = load_excluded_test_hashes(
        args.exclude_test_manifest
    )
    shard_selection = {
        **SPLIT_SHARDS,
        "test": test_shards,
    }
    started_at = datetime.now(timezone.utc).isoformat()
    split_sizes = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    split_evidence = {}
    for index, (split, count) in enumerate(split_sizes.items()):
        split_evidence[split] = _sample_split(
            split,
            count,
            source_dir,
            args.data_dir,
            args.seed + index,
            shard_selection[split],
            excluded_test_hashes if split == "test" else None,
        )

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_card": DATASET_CARD,
        "official_upstream_challenge": UPSTREAM_CHALLENGE,
        "upstream_license": "Open Data Commons Attribution License (ODC-By)",
        "mirror_license_note": (
            "The pinned Hugging Face mirror card does not declare a separate "
            "license. Its source is the ASVspoof 2021 DF challenge database; "
            "retain attribution and verify dataset terms for redistribution."
        ),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sampling_method": (
            "Deterministic per-class reservoir sampling without replacement "
            "from selected parquet shards of the official train, validation, "
            "and test splits."
        ),
        "source_split_shards": {
            split: list(shard_selection[split])
            for split in split_sizes
        },
        "excluded_prior_test_audio_hashes": (
            {
                "manifest": str(args.exclude_test_manifest.resolve()),
                "count": len(excluded_test_hashes),
            }
            if args.exclude_test_manifest
            else None
        ),
        "splits": split_evidence,
    }
    manifest_path = args.data_dir / "acquisition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset and acquisition evidence saved in {args.data_dir.resolve()}")
    print(f"Acquisition manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
