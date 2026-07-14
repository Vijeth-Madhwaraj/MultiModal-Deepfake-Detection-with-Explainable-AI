"""Generic visualization helpers for attribution maps.

These utilities are shared by SHAP and can also be reused by future
explainability methods. The code intentionally mirrors the existing Grad-CAM
overlay style so SHAP outputs remain easy to inspect frame by frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def normalize_attribution(volume: np.ndarray) -> np.ndarray:
    """Normalize an attribution volume to the ``[0, 1]`` range."""

    volume = np.asarray(volume, dtype=np.float32)
    volume = np.maximum(volume, 0.0)
    min_value = float(np.min(volume))
    max_value = float(np.max(volume))
    if max_value > min_value:
        volume = (volume - min_value) / (max_value - min_value)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)
    return np.clip(volume, 0.0, 1.0)


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Convert a normalized heatmap to an RGB color overlay."""

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a heatmap over an RGB frame."""

    heatmap_rgb = apply_colormap(heatmap)
    return cv2.addWeighted(frame_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)


def resize_attribution_volume(attribution_volume: np.ndarray, target_depth: int) -> np.ndarray:
    """Resize a 3D attribution volume to match a target temporal depth."""

    attribution_volume = np.asarray(attribution_volume, dtype=np.float32)
    if attribution_volume.ndim != 3:
        raise ValueError(f"Expected attribution volume with shape (T, H, W), got {attribution_volume.shape}")

    source_depth, height, width = attribution_volume.shape
    if source_depth == target_depth:
        return attribution_volume

    source_positions = np.arange(source_depth, dtype=np.float32)
    target_positions = np.linspace(0, source_depth - 1, num=target_depth, dtype=np.float32)

    flattened = attribution_volume.reshape(source_depth, -1)
    resized_flat = np.empty((target_depth, flattened.shape[1]), dtype=np.float32)
    for spatial_idx in range(flattened.shape[1]):
        resized_flat[:, spatial_idx] = np.interp(target_positions, source_positions, flattened[:, spatial_idx])

    return resized_flat.reshape(target_depth, height, width)


def build_overlay_frames(
    frames: List[np.ndarray],
    attribution_volume: np.ndarray,
    alpha: float = 0.4,
) -> List[np.ndarray]:
    """Overlay one attribution slice per frame."""

    resized_volume = resize_attribution_volume(attribution_volume, target_depth=len(frames))
    output_frames: List[np.ndarray] = []

    for frame_rgb, attribution_slice in zip(frames, resized_volume):
        height, width = frame_rgb.shape[:2]
        attribution_slice = cv2.resize(attribution_slice, (width, height), interpolation=cv2.INTER_LINEAR)
        attribution_slice = normalize_attribution(attribution_slice)
        output_frames.append(overlay_heatmap(frame_rgb, attribution_slice, alpha=alpha))

    return output_frames


def write_video(frames_rgb: List[np.ndarray], output_path: str | Path, fps: float = 24.0) -> None:
    """Write a list of RGB frames to an MP4 file."""

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


def save_overlay_video(
    frames_rgb: List[np.ndarray],
    attribution_volume: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.4,
    fps: float = 24.0,
) -> None:
    """Build overlays and save them as a video."""

    overlay_frames = build_overlay_frames(frames_rgb, attribution_volume, alpha=alpha)
    write_video(overlay_frames, output_path=output_path, fps=fps)


def save_signed_overlay_video(
    frames_rgb: List[np.ndarray], signed_volume: np.ndarray, output_path: str | Path,
    target_name: str, alpha: float = 0.55, fps: float = 24.0,
) -> None:
    """Save a diverging SHAP video: red supports the target and blue opposes it."""
    volume = resize_attribution_volume(signed_volume, target_depth=len(frames_rgb))
    scale = float(np.percentile(np.abs(volume), 99.0)) or 1.0
    output_frames: List[np.ndarray] = []
    for frame_rgb, signed_slice in zip(frames_rgb, volume):
        height, width = frame_rgb.shape[:2]
        signed_slice = cv2.resize(signed_slice, (width, height), interpolation=cv2.INTER_LINEAR)
        normalized = np.clip(signed_slice / scale, -1.0, 1.0)
        magnitude = np.abs(normalized)[..., None]
        colors = np.zeros_like(frame_rgb, dtype=np.float32)
        colors[..., 0] = np.maximum(normalized, 0.0) * 255.0
        colors[..., 2] = np.maximum(-normalized, 0.0) * 255.0
        blend = np.clip(alpha * magnitude, 0.0, 1.0)
        overlay = np.clip(frame_rgb * (1.0 - blend) + colors * blend, 0, 255).astype(np.uint8)
        cv2.putText(overlay, f"SHAP for {target_name}: RED supports | BLUE opposes", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        output_frames.append(overlay)
    write_video(output_frames, output_path=output_path, fps=fps)
