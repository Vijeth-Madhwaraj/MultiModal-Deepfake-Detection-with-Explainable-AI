"""Prepare the Celeb-DF datasets for the GradCAM3D project.

This script scans the raw Celeb-DF folders, groups videos into the two target
classes (real and fake), performs a stratified train/val/test split at the
video level, and copies or moves the files into:

    data/train/real
    data/train/fake
    data/val/real
    data/val/fake
    data/test/real
    data/test/fake

The script defaults to copying files so the original datasets stay untouched.
Use --move if you want to relocate the files instead.

By default, the script balances the dataset by downsampling both classes to the
same number of videos before creating the train/val/test split.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import random
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_SOURCES: Tuple[str, str] = (
    r"C:\Users\Madhw\Downloads\Celeb-DF",
    r"C:\Users\Madhw\Downloads\Celeb-DF-v2",
)
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_MANIFEST_PATH = Path(DEFAULT_OUTPUT_DIR) / "split_manifest.csv"
TARGET_SPLITS: Tuple[str, str, str] = ("train", "val", "test")
TARGET_CLASSES: Tuple[str, str] = ("real", "fake")


def set_seed(seed: int) -> None:
    """Set the random seed used by the dataset splitter."""

    random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_video_file(path: str | Path) -> bool:
    """Return True when a path points to a supported video file."""

    candidate = Path(path)
    return candidate.is_file() and candidate.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class VideoSample:
    """A single video and its target class."""

    source_root: Path
    source_class: str
    path: Path
    label: str


def collect_video_samples(source_roots: Sequence[str | Path]) -> List[VideoSample]:
    """Collect all labeled videos from the provided raw dataset roots."""

    samples: List[VideoSample] = []
    seen_paths: set[Path] = set()

    class_name_map: Dict[str, str] = {
        "real": "real",
        "celeb-real": "real",
        "youtube-real": "real",
        "fake": "fake",
        "celeb-synthesis": "fake",
    }

    for source_root_value in source_roots:
        source_root = Path(source_root_value)
        if not source_root.exists():
            raise FileNotFoundError(f"Source folder not found: {source_root}")

        for child in source_root.iterdir():
            if not child.is_dir():
                continue

            class_key = child.name.lower()
            if class_key not in class_name_map:
                continue

            target_label = class_name_map[class_key]
            for video_path in child.rglob("*"):
                if not is_video_file(video_path):
                    continue

                resolved_path = video_path.resolve()
                if resolved_path in seen_paths:
                    continue

                seen_paths.add(resolved_path)
                samples.append(
                    VideoSample(
                        source_root=source_root,
                        source_class=child.name,
                        path=video_path,
                        label=target_label,
                    )
                )

    if not samples:
        raise RuntimeError("No video files were found in the provided source folders.")

    return samples


def clear_split_outputs(output_dir: Path) -> None:
    """Remove any existing split folders so the dataset can be rebuilt cleanly."""

    for split_name in TARGET_SPLITS:
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)


def balance_samples(samples: Sequence[VideoSample], seed: int) -> List[VideoSample]:
    """Downsample both classes to the size of the minority class."""

    grouped: Dict[str, List[VideoSample]] = {"real": [], "fake": []}
    for sample in samples:
        grouped[sample.label].append(sample)

    real_count = len(grouped["real"])
    fake_count = len(grouped["fake"])

    if real_count == 0 or fake_count == 0:
        raise RuntimeError(
            "Both real and fake videos must be present before balancing the dataset."
        )

    target_count = min(real_count, fake_count)
    rng = random.Random(seed)

    balanced_samples: List[VideoSample] = []
    for class_label in TARGET_CLASSES:
        class_samples = grouped[class_label][:]
        rng.shuffle(class_samples)
        balanced_samples.extend(class_samples[:target_count])

    rng.shuffle(balanced_samples)
    return balanced_samples


def stratified_split(
    samples: Sequence[VideoSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[VideoSample]]:
    """Split samples by class while preserving the requested ratios."""

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("The split ratios must sum to 1.0.")

    grouped: Dict[str, List[VideoSample]] = {"real": [], "fake": []}
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    split_samples: Dict[str, List[VideoSample]] = {split: [] for split in TARGET_SPLITS}

    for class_label, class_samples in grouped.items():
        rng.shuffle(class_samples)
        total_count = len(class_samples)

        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        test_count = total_count - train_count - val_count

        train_slice = class_samples[:train_count]
        val_slice = class_samples[train_count : train_count + val_count]
        test_slice = class_samples[train_count + val_count : train_count + val_count + test_count]

        split_samples["train"].extend(train_slice)
        split_samples["val"].extend(val_slice)
        split_samples["test"].extend(test_slice)

    for split_name in split_samples:
        rng.shuffle(split_samples[split_name])

    return split_samples


def build_target_path(
    output_dir: Path,
    split_name: str,
    sample: VideoSample,
    index: int,
) -> Path:
    """Create a collision-resistant destination path for a video."""

    class_dir = output_dir / split_name / sample.label
    ensure_dir(class_dir)

    unique_name = (
        f"{sample.source_root.name}__"
        f"{sample.source_class}__"
        f"{index:06d}__"
        f"{sample.path.stem}{sample.path.suffix.lower()}"
    )
    return class_dir / unique_name


def copy_or_move_file(source_path: Path, destination_path: Path, move_files: bool) -> None:
    """Copy or move one file to the destination path."""

    ensure_dir(destination_path.parent)
    if move_files:
        shutil.move(str(source_path), str(destination_path))
    else:
        shutil.copy2(source_path, destination_path)


def write_manifest(manifest_path: Path, rows: Sequence[Dict[str, str]]) -> None:
    """Save a CSV manifest describing where every video was placed."""

    ensure_dir(manifest_path.parent)
    fieldnames = [
        "split",
        "class",
        "source_root",
        "source_class",
        "source_path",
        "target_path",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    source_roots: Sequence[str | Path],
    output_dir: str | Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    move_files: bool,
    manifest_path: str | Path,
) -> None:
    """Run the complete dataset splitting pipeline."""

    set_seed(seed)

    output_dir_path = Path(output_dir)
    ensure_dir(output_dir_path)
    clear_split_outputs(output_dir_path)
    for split_name in TARGET_SPLITS:
        for class_name in TARGET_CLASSES:
            ensure_dir(output_dir_path / split_name / class_name)

    all_samples = collect_video_samples(source_roots)
    balanced_samples = balance_samples(all_samples, seed=seed)
    split_samples = stratified_split(
        balanced_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    manifest_rows: List[Dict[str, str]] = []
    counters = {split_name: 0 for split_name in TARGET_SPLITS}

    for split_name in TARGET_SPLITS:
        for sample in split_samples[split_name]:
            counters[split_name] += 1
            destination_path = build_target_path(
                output_dir=output_dir_path,
                split_name=split_name,
                sample=sample,
                index=counters[split_name],
            )
            copy_or_move_file(sample.path, destination_path, move_files=move_files)

            manifest_rows.append(
                {
                    "split": split_name,
                    "class": sample.label,
                    "source_root": str(sample.source_root),
                    "source_class": sample.source_class,
                    "source_path": str(sample.path),
                    "target_path": str(destination_path),
                }
            )

    write_manifest(Path(manifest_path), manifest_rows)

    print("Dataset split complete.")
    for split_name in TARGET_SPLITS:
        split_rows = [row for row in manifest_rows if row["split"] == split_name]
        real_count = sum(1 for row in split_rows if row["class"] == "real")
        fake_count = sum(1 for row in split_rows if row["class"] == "fake")
        print(f"{split_name:5s} -> real: {real_count:5d} | fake: {fake_count:5d} | total: {len(split_rows):5d}")

    print(f"Manifest saved to: {manifest_path}")
    print(f"Files were {'moved' if move_files else 'copied'} into: {output_dir_path}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Split Celeb-DF videos into train/val/test folders.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help="One or more raw Celeb-DF dataset folders.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Destination data directory.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and balancing.")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the CSV manifest file.",
    )
    return parser


def main() -> None:
    """Entry point for dataset preparation."""

    parser = build_parser()
    args = parser.parse_args()

    prepare_dataset(
        source_roots=args.sources,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        move_files=args.move,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    main()
