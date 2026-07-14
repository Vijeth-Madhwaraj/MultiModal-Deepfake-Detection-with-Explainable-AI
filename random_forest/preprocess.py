from pathlib import Path
import logging

import librosa
import soundfile as sf
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "processed_audio"
LOG_DIR = PROJECT_DIR / "logs"

TARGET_SR = 16000

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a"
}

print(f"Project Directory: {PROJECT_DIR}")


# =====================================================
# LOGGING
# =====================================================

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# =====================================================
# AUDIO PREPROCESSING
# =====================================================

def preprocess_audio(input_path: Path, output_path: Path):
    """
    Load an audio file, preprocess it, and save it.
    """

    try:
        # Load audio
        audio, sr = librosa.load(
            input_path,
            sr=TARGET_SR,
            mono=True
        )

        # Skip empty files
        if len(audio) == 0:
            logging.warning(f"Empty audio: {input_path}")
            return False

        # Trim leading/trailing silence
        audio, _ = librosa.effects.trim(
            audio,
            top_db=20
        )

        # Normalize amplitude
        audio = librosa.util.normalize(audio)

        # Create output directory if needed
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save processed audio
        sf.write(
            output_path,
            audio,
            TARGET_SR
        )

        logging.info(f"Processed: {input_path}")

        return True

    except Exception as e:
        logging.error(f"{input_path} -> {e}")
        return False
def main():

    processed = 0
    failed = 0

    audio_files = []

    # Look through every folder in the project
    for folder in PROJECT_DIR.iterdir():

        if not folder.is_dir():
            continue

        # Ignore project folders
        if folder.name in [
            "venv",
            "processed_audio",
            "logs",
            "models",
            "results",
            "features",
            "metadata",
            "data",
            "__pycache__"
        ]:
            continue

        files = list(folder.rglob("*.wav"))

        print(f"{folder.name:<20} {len(files)}")

        audio_files.extend(files)

    print(f"\nTotal files found: {len(audio_files)}\n")

    for input_file in tqdm(audio_files):

        relative = input_file.relative_to(PROJECT_DIR)

        output_file = OUTPUT_DIR / relative

        success = preprocess_audio(
            input_file,
            output_file
        )

        if success:
            processed += 1
        else:
            failed += 1

    print("\n==========================")
    print(f"Processed : {processed}")
    print(f"Failed    : {failed}")
    print("==========================")
if __name__ == "__main__":
    main()