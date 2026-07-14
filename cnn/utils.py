"""Shared utility helpers for the GradCAM3D project.

This module centralizes video decoding, tensor conversion, checkpoint loading,
heatmap visualization, and reproducibility helpers so the training,
inference, and Grad-CAM scripts can stay compact and consistent.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor


VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str | None = None) -> torch.device:
    """Resolve a PyTorch device from a string or fall back to the default GPU/CPU."""

    if device_name is not None:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_video_file(path: str | Path, extensions: Sequence[str] = VIDEO_EXTENSIONS) -> bool:
    """Return True when the path has a supported video extension."""

    candidate = Path(path)
    return candidate.is_file() and candidate.suffix.lower() in tuple(ext.lower() for ext in extensions)


def read_video_frames(video_path: str | Path, frame_size: int) -> List[np.ndarray]:
    """Decode a video file into resized RGB frames."""

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
            frame_rgb = cv2.resize(frame_rgb, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames could be decoded from video: {video_path}")

    return frames


def pad_frames(frames: List[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Pad a short clip by repeating its last frame."""

    if len(frames) >= clip_length:
        return frames[:clip_length]

    padded = list(frames)
    last_frame = padded[-1]
    while len(padded) < clip_length:
        padded.append(last_frame)
    return padded


def sample_random_clip(frames: List[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Sample a random contiguous clip from a frame list."""

    if len(frames) <= clip_length:
        return pad_frames(frames, clip_length)

    start_index = random.randint(0, len(frames) - clip_length)
    return frames[start_index : start_index + clip_length]


def sample_center_clip(frames: List[np.ndarray], clip_length: int) -> List[np.ndarray]:
    """Sample a deterministic center clip from a frame list."""

    if len(frames) <= clip_length:
        return pad_frames(frames, clip_length)

    start_index = max((len(frames) - clip_length) // 2, 0)
    return frames[start_index : start_index + clip_length]


def frames_to_tensor(frames: List[np.ndarray]) -> Tensor:
    """Convert RGB frames to a normalized float tensor with shape (C, T, H, W)."""

    clip_array = np.asarray(frames, dtype=np.float32) / 255.0
    clip_array = np.transpose(clip_array, (3, 0, 1, 2))
    return torch.from_numpy(clip_array)


def tensor_to_frames(clip: Tensor) -> List[np.ndarray]:
    """Convert a tensor in (C, T, H, W) format back to RGB frames."""

    if clip.ndim != 4 or clip.shape[0] != 3:
        raise ValueError("Expected a tensor of shape (3, T, H, W).")

    clip_np = clip.detach().cpu().numpy()
    clip_np = np.transpose(clip_np, (1, 2, 3, 0))
    clip_np = np.clip(clip_np * 255.0, 0, 255).astype(np.uint8)
    return [frame for frame in clip_np]


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize a heatmap to the [0, 1] range."""

    heatmap = np.maximum(heatmap, 0)
    max_value = float(np.max(heatmap))
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Convert a normalized heatmap into a BGR color map."""

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


def overlay_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a heatmap onto an RGB frame and return the blended RGB image."""

    heatmap_bgr = apply_colormap(heatmap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)


def write_video(frames_rgb: List[np.ndarray], output_path: str | Path, fps: float = 24.0) -> None:
    """Write RGB frames to an MP4 file."""

    if not frames_rgb:
        raise ValueError("No frames were provided for video writing.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        for frame_rgb in frames_rgb:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def load_checkpoint_state(checkpoint_path: str | Path, device: torch.device) -> dict:
    """Load a checkpoint dictionary from disk."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint}


def load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    """Load model weights from either a training checkpoint or a raw state dict."""

    checkpoint = load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
