import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# This utility lives in data_tools/, but its inputs and outputs remain at the
# repository root so moving the script does not change its default behavior.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
METADATA_DIR = PROJECT_DIR / "metadata"
SPLIT_DIR = PROJECT_DIR / "fakeavceleb_split"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

REAL_TYPES = {"RealVideo-RealAudio"}

SPLIT_FILENAMES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def build_metadata(dataset_dir):
    rows = []

    for video_path in dataset_dir.rglob("*"):
        if not video_path.is_file():
            continue

        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        relative_parts = video_path.relative_to(dataset_dir).parts
        category = relative_parts[0]

        label = 0 if category in REAL_TYPES else 1

        rows.append({
            "filepath": str(video_path.relative_to(PROJECT_DIR)),
            "filename": video_path.name,
            "generator": category,
            "label": label,
            "label_name": "real" if label == 0 else "fake",
            "source": relative_parts[-2] if len(relative_parts) > 1 else "",
        })

    if not rows:
        raise RuntimeError(f"No videos found in {dataset_dir}")

    return pd.DataFrame(rows).sort_values(["label", "generator", "filepath"])


def _split_one_label(label_df, seed):
    unique_sources = label_df["source"].nunique()

    if unique_sources >= 3 and len(label_df) >= 3:
        first_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )
        train_index, temp_index = next(
            first_split.split(label_df, groups=label_df["source"])
        )

        train_df = label_df.iloc[train_index]
        temp_df = label_df.iloc[temp_index]

        if temp_df["source"].nunique() >= 2 and len(temp_df) >= 2:
            second_split = GroupShuffleSplit(
                n_splits=1,
                test_size=0.50,
                random_state=seed,
            )
            valid_index, test_index = next(
                second_split.split(temp_df, groups=temp_df["source"])
            )
            valid_df = temp_df.iloc[valid_index]
            test_df = temp_df.iloc[test_index]
            return train_df, valid_df, test_df

    train_df, temp_df = train_test_split(
        label_df,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    return train_df, valid_df, test_df


def split_fakeavceleb(df, seed):
    labels = set(df["label"].unique())
    required_labels = {0, 1}

    if labels != required_labels:
        raise RuntimeError(
            "FakeAVCeleb split needs both real and fake samples. "
            f"Found labels: {sorted(labels)}"
        )

    split_parts = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for label, label_df in df.groupby("label"):
        if len(label_df) < 3:
            raise RuntimeError(
                f"Label {label} has only {len(label_df)} sample(s); "
                "need at least 3 so train, validation, and test can all contain it."
            )

        train_df, valid_df, test_df = _split_one_label(label_df, seed)
        split_parts["train"].append(train_df)
        split_parts["validation"].append(valid_df)
        split_parts["test"].append(test_df)

    return {
        split_name: (
            pd.concat(parts)
            .sample(frac=1, random_state=seed)
            .sort_values(["label", "generator", "filepath"])
            .reset_index(drop=True)
        )
        for split_name, parts in split_parts.items()
    }


def make_split_filename(row):
    source_path = Path(row["filepath"])
    safe_parent = "_".join(source_path.parent.parts)
    if safe_parent:
        return f"{safe_parent}_{source_path.name}"
    return source_path.name


def copy_split_videos(splits, split_dir):
    copied_splits = {}

    for split_name, split_df in splits.items():
        copied_rows = []

        for _, row in split_df.iterrows():
            source_path = PROJECT_DIR / row["filepath"]
            destination_dir = split_dir / split_name / row["label_name"]
            destination_dir.mkdir(parents=True, exist_ok=True)

            destination_path = destination_dir / make_split_filename(row)
            shutil.copy2(source_path, destination_path)

            copied_row = row.copy()
            copied_row["original_filepath"] = row["filepath"]
            copied_row["filepath"] = str(destination_path.relative_to(PROJECT_DIR))
            copied_rows.append(copied_row)

        copied_splits[split_name] = pd.DataFrame(copied_rows).reset_index(drop=True)

    return copied_splits


def main():
    parser = argparse.ArgumentParser(
        description="Create train, validation, and test splits for FakeAVCeleb."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to the FakeAVCeleb dataset folder."
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per label for faster experiments."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=SPLIT_DIR,
        help="Folder where split videos will be copied into train/real, train/fake, validation/real, validation/fake, test/real, and test/fake."
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write CSV files. Do not copy videos into separate real/fake folders."
    )

    args = parser.parse_args()

    METADATA_DIR.mkdir(exist_ok=True)

    df = build_metadata(args.dataset_dir)

    if args.max_per_class:
        df = (
            df.groupby("label", group_keys=False)
            .sample(n=args.max_per_class, random_state=args.seed)
            .sort_values(["label", "generator", "filepath"])
        )

    splits = split_fakeavceleb(df, args.seed)

    if not args.metadata_only:
        splits = copy_split_videos(splits, args.split_dir)

    df.to_csv(METADATA_DIR / "fakeavceleb_metadata.csv", index=False)
    for split_name, split_df in splits.items():
        split_df.to_csv(METADATA_DIR / SPLIT_FILENAMES[split_name], index=False)

    print("\nFakeAVCeleb split complete")
    print("==========================")
    print(f"Total      : {len(df)}")
    print(f"Train      : {len(splits['train'])}")
    print(f"Validation : {len(splits['validation'])}")
    print(f"Test       : {len(splits['test'])}")

    print("\nLabel distribution")
    print("0 = real, 1 = fake")
    print(pd.crosstab(df["label_name"], df["label"]))

    print("\nSplit label distribution")
    for split_name, split_df in splits.items():
        print(split_name.title())
        print(split_df["label_name"].value_counts().reindex(["real", "fake"], fill_value=0))

    print("\nCategory distribution")
    print(df["generator"].value_counts())


if __name__ == "__main__":
    main()
