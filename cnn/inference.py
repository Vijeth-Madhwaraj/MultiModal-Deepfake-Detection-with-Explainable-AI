"""Inference entrypoint for the custom 3D CNN deepfake classifier.

This script loads a trained checkpoint and predicts whether a single input
video is Real or Fake. It also reports the confidence score for the predicted
class.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from configs.config import Config
from model import SemanticDeepfakeDetector, load_compatible_checkpoint
from video_dataset import FaceAlignmentProcessor, prepare_frame_for_model


def read_video_frames(video_path: str | Path, frame_size: Optional[int] = None) -> List[np.ndarray]:
    """Read a video file as a list of RGB frames, optionally resized to a square size."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_size is not None:
                frame_rgb = cv2.resize(frame_rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames could be decoded from video: {video_path}")

    return frames


def sample_clip(frames: List[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Extract a deterministic clip from a full video for inference."""

    if len(frames) >= clip_length:
        start_index = max((len(frames) - clip_length) // 2, 0)
        clip = frames[start_index : start_index + clip_length]
    else:
        clip = list(frames)
        last_frame = clip[-1]
        while len(clip) < clip_length:
            clip.append(last_frame)

    return clip[:clip_length]


def clip_to_tensor(frames: List[np.ndarray]) -> Tensor:
    """Convert preprocessed RGB frames to a tensor in (C, T, H, W) format."""

    clip_array = np.asarray(frames, dtype=np.float32)
    clip_array = np.transpose(clip_array, (3, 0, 1, 2))
    return torch.from_numpy(clip_array)


def load_checkpoint(model: SemanticDeepfakeDetector, checkpoint_path: str | Path, device: torch.device) -> None:
    """Load model weights from a checkpoint path.

    The function accepts either the full checkpoint dictionary created by
    train.py or a plain model state dict for compatibility with older runs.
    """

    load_compatible_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device, strict=False)


@torch.no_grad()
def predict_video(
    model: SemanticDeepfakeDetector,
    video_path: str | Path,
    device: torch.device,
    frame_size: int = Config.IMAGE_SIZE,
    clip_length: int = Config.CLIP_LENGTH,
    align_faces: bool = Config.ALIGN_FACES,
) -> Tuple[str, float]:
    """Predict the class label and confidence for a single video."""

    frames = read_video_frames(video_path, frame_size=None)
    clip_frames = sample_clip(frames, clip_length=clip_length)
    face_aligner = FaceAlignmentProcessor() if align_faces else None
    try:
        model_frames = [
            prepare_frame_for_model(
                frame_rgb,
                frame_size,
                align_faces=align_faces,
                face_aligner=face_aligner,
            )
            for frame_rgb in clip_frames
        ]
    finally:
        if face_aligner is not None:
            face_aligner.close()
    clip_tensor = clip_to_tensor(model_frames).unsqueeze(0).to(device)

    model.eval()
    logits = model(clip_tensor)
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_index = torch.max(probabilities, dim=1)

    class_names = {0: "Real", 1: "Fake"}
    predicted_label = class_names[int(predicted_index.item())]
    return predicted_label, float(confidence.item())


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for inference."""

    parser = argparse.ArgumentParser(description="Predict whether a video is Real or Fake.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(Path(Config.CHECKPOINT_DIR) / Config.CHECKPOINT_NAME),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=Config.DEVICE,
        help="Device to run inference on, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--align-faces",
        action="store_true",
        default=Config.ALIGN_FACES,
        help="Use landmark-based eye alignment before face crop/resize when landmarks are available.",
    )
    return parser


def main() -> None:
    """Run inference on one video and print the predicted label and confidence."""

    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    model = SemanticDeepfakeDetector(
        num_classes=Config.NUM_CLASSES,
        concept_vocabulary=Config.CONCEPT_NAMES,
        extra_unsupervised_concepts=Config.EXTRA_UNSUPERVISED_CONCEPTS,
    ).to(device)
    load_checkpoint(model, args.checkpoint_path, device)

    predicted_label, confidence = predict_video(
        model=model,
        video_path=args.video_path,
        device=device,
        align_faces=args.align_faces,
    )

    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
