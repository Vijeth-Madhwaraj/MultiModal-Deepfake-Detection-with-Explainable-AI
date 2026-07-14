import argparse
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
# =====================================
# CONFIGURATION
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent

METADATA_DIR = PROJECT_DIR / "metadata"
FEATURE_DIR = PROJECT_DIR / "features"

FEATURE_DIR.mkdir(exist_ok=True)
INPUT_CSV = METADATA_DIR / "test.csv"
OUTPUT_CSV = FEATURE_DIR / "test_features.csv"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


def extract_audio_from_video(video_path):
    temp_dir = tempfile.TemporaryDirectory()
    audio_path = Path(temp_dir.name) / "audio.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return temp_dir, audio_path


def extract_features(audio_path):
    """
    Extract handcrafted audio features from a WAV file.
    Returns a fixed-length feature vector.
    """

    temp_dir = None

    if Path(audio_path).suffix.lower() in VIDEO_EXTENSIONS:
        temp_dir, audio_path = extract_audio_from_video(audio_path)

    try:
        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        features = []
            # -----------------------------
        # MFCC
        # -----------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
            # -----------------------------
        # Chroma
        # -----------------------------

        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        )

        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
            # -----------------------------
        # Spectral Contrast
        # -----------------------------

        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=sr
        )

        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))
            # -----------------------------
        # Spectral Centroid
        # -----------------------------

        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )

        features.append(np.mean(centroid))
        features.append(np.std(centroid))
            # -----------------------------
        # Spectral Bandwidth
        # -----------------------------

        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )

        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))
        # -----------------------------
        # Spectral Roll-off
        # -----------------------------
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )

        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        # -----------------------------
        # Spectral Flatness
        # -----------------------------
        flatness = librosa.feature.spectral_flatness(y=audio)

        features.append(np.mean(flatness))
        features.append(np.std(flatness))

        # -----------------------------
        # RMS Energy
        # -----------------------------
        rms = librosa.feature.rms(y=audio)

        features.append(np.mean(rms))
        features.append(np.std(rms))

        # -----------------------------
        # Zero Crossing Rate
        # -----------------------------
        zcr = librosa.feature.zero_crossing_rate(audio)

        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        return np.array(features, dtype=np.float32)

    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

def get_feature_names():

    names = []

    # MFCC
    for i in range(40):
        names.append(f"mfcc_{i+1}_mean")

    for i in range(40):
        names.append(f"mfcc_{i+1}_std")

    # Chroma
    for i in range(12):
        names.append(f"chroma_{i+1}_mean")

    for i in range(12):
        names.append(f"chroma_{i+1}_std")

    # Spectral Contrast
    for i in range(7):
        names.append(f"contrast_{i+1}_mean")

    for i in range(7):
        names.append(f"contrast_{i+1}_std")

    names.extend([
        "centroid_mean",
        "centroid_std",
        "bandwidth_mean",
        "bandwidth_std",
        "rolloff_mean",
        "rolloff_std",
        "flatness_mean",
        "flatness_std",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "zcr_std"
    ])

    return names

def main():
    parser = argparse.ArgumentParser(
        description="Extract handcrafted audio features from metadata CSV files."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV,
        help="Metadata CSV containing a filepath column."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Feature CSV to write."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    print(f"Processing {len(df)} audio files...")

    feature_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        audio_path = PROJECT_DIR / row["filepath"]

        try:

            vector = extract_features(audio_path)

            feature_rows.append(
                list(vector) + [
                    row["label"],
                    row["generator"],
                    row["filename"]
                ]
            )

        except Exception as e:

            print(f"Failed: {audio_path}")
            print(e)

    columns = get_feature_names()

    columns.extend([
        "label",
        "generator",
        "filename"
    ])

    feature_df = pd.DataFrame(
        feature_rows,
        columns=columns
    )
    args.output_csv.parent.mkdir(exist_ok=True)

    feature_df.to_csv(args.output_csv, index=False)

    print("\nFeature extraction complete!")

    print(feature_df.head())

    print(f"\nSaved to:\n{args.output_csv}")

    print(f"\nSamples : {len(feature_df)}")
    print(f"Features: {len(get_feature_names())}")


if __name__ == "__main__":
    main()
