#!/usr/bin/env python3
"""
split_data_limited.py

Create train.csv, valid.csv, and test.csv from real_songs.csv and fake_songs.csv,
while skipping metadata rows whose .mp3 file does not actually exist.

Behavior:
1. Preserve the original row order from each metadata CSV.
2. Keep only rows whose audio file exists on disk.
3. Keep only rows that satisfy the current filtering rule:
      duration >= 30 and no_vocal == False
4. Optionally limit how many usable REAL songs and how many usable FAKE songs are kept.
   For example, if you set --limit 100, the script keeps the first 100 usable real
   songs and the first 100 usable fake songs (in original order), then writes the
   train/valid/test CSV files according to the existing split column.

Examples:
    python split_data.py
        Use all usable songs.

    python split_data.py --limit 100
        Use only the first 100 usable real songs and first 100 usable fake songs.

    python split_data.py --data-dir ./dataset --limit 500
        Same as above, but with an explicit dataset folder.

Output files:
    train.csv
    valid.csv
    test.csv
"""

import argparse
import os
import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create train/valid/test CSV files with optional per-class limits."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./dataset",
        help="Path to the dataset directory containing metadata CSVs and audio folders.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional per-class limit. For example, --limit 100 means keep only the "
            "first 100 usable real songs and the first 100 usable fake songs. "
            "If omitted, all usable songs are kept."
        ),
    )
    return parser.parse_args()


def list_existing_mp3_stems(folder):
    """
    Return a set of filename stems (without .mp3) that actually exist in folder.
    Uses os.scandir for efficiency on large directories.
    """
    stems = set()
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".mp3"):
                stems.add(os.path.splitext(entry.name)[0])
    return stems


def prepare_split_df(meta_csv_path, audio_folder, target_value, split_name, limit=None):
    """
    Read a metadata CSV, keep only rows whose mp3 file actually exists,
    apply the current filtering rule, optionally limit the number of usable rows,
    and create filepath/target columns.

    Important:
    - Row order is preserved.
    - The optional limit is applied AFTER existence checking and AFTER the
      original filtering rule (duration/no_vocal), so the script keeps up to
      'limit' usable songs for that class.
    """
    df = pd.read_csv(meta_csv_path, low_memory=False)

    # Make filename robust in case pandas reads it as numeric/mixed type.
    df["filename"] = df["filename"].astype(str).str.strip()

    # Build a fast lookup table of audio files that really exist.
    existing_stems = list_existing_mp3_stems(audio_folder)
    print(f"{split_name}: found {len(existing_stems)} existing mp3 files in {audio_folder}")

    # Keep only rows whose corresponding mp3 file exists.
    before_exists = len(df)
    df = df[df["filename"].isin(existing_stems)].copy()
    after_exists = len(df)
    print(
        f"{split_name}: kept {after_exists}/{before_exists} rows after existence check, "
        f"removed {before_exists - after_exists} missing-file rows"
    )

    # Apply the original filtering rule from your current script.
    before_filter = len(df)
    df = df[(df["duration"] >= 30) & (df["no_vocal"] == False)].copy()
    after_filter = len(df)
    print(
        f"{split_name}: kept {after_filter}/{before_filter} rows after duration/no_vocal filter, "
        f"removed {before_filter - after_filter} rows"
    )

    # If a per-class limit is requested, keep only the first 'limit' usable songs.
    if limit is not None:
        df = df.head(limit).copy()
        print(f"{split_name}: limited to first {len(df)} usable rows")

    # Add training fields expected by the training pipeline.
    df["filepath"] = audio_folder + "/" + df["filename"] + ".mp3"
    df["target"] = target_value
    return df


def main():
    args = parse_args()
    data_dir = args.data_dir
    limit = args.limit

    if limit is not None and limit < 0:
        raise ValueError("--limit must be a non-negative integer or omitted.")

    real_audio_dir = f"{data_dir}/real_songs"
    fake_audio_dir = f"{data_dir}/fake_songs"

    real_df = prepare_split_df(
        meta_csv_path=f"{data_dir}/real_songs.csv",
        audio_folder=real_audio_dir,
        target_value=0,
        split_name="real",
        limit=limit,
    )

    fake_df = prepare_split_df(
        meta_csv_path=f"{data_dir}/fake_songs.csv",
        audio_folder=fake_audio_dir,
        target_value=1,
        split_name="fake",
        limit=limit,
    )

    # Preserve the original class order used by your current script:
    # all selected real rows first, then all selected fake rows.
    df = pd.concat([real_df, fake_df])

    # Split according to the existing 'split' column.
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"train: {len(train_df)}")
    print(f"valid: {len(valid_df)}")
    print(f"test:  {len(test_df)}")

    train_df.to_csv("train.csv", index=False)
    valid_df.to_csv("valid.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print("Saved train.csv, valid.csv, test.csv")


if __name__ == "__main__":
    main()
