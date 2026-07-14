import argparse
import subprocess
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from extract_features import extract_features, get_feature_names


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"


def extract_audio(video_path, audio_path):
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


def predict_video(video_path):
    model_path = MODEL_DIR / "random_forest.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run python random_forest\\train.py first."
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing scaler: {scaler_path}. Run python random_forest\\train.py first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "extracted_audio.wav"
        extract_audio(video_path, audio_path)

        features = extract_features(audio_path)
        feature_df = pd.DataFrame([features], columns=get_feature_names())
        scaled_features = scaler.transform(feature_df)

    probability_fake = model.predict_proba(scaled_features)[0][1]
    prediction = int(model.predict(scaled_features)[0])
    label = "FAKE" if prediction == 1 else "REAL"

    return label, probability_fake


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from a video and test it with the Random Forest model."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the video file to test."
    )

    args = parser.parse_args()
    video_path = args.video.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    label, probability_fake = predict_video(video_path)

    print("\n==============================")
    print("Random Forest Video Prediction")
    print("==============================")
    print(f"Video           : {video_path}")
    print(f"Prediction      : {label}")
    print(f"Fake probability: {probability_fake:.4f}")
    print(f"Real probability: {1 - probability_fake:.4f}")


if __name__ == "__main__":
    main()
